"""
app.py — Streamlit web interface for PE Monitor.

Pages:
  1. 自選股管理  — add/remove stocks, set holdings
  2. 每日監測儀表板 — live P/E signals + charts
  3. 歷史報告    — browse past daily reports
"""

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pe_calculator import (
    SIGNAL_EMOJI,
    SIGNAL_LABEL,
    build_historical_pb_series,
    build_historical_pe_series,
    get_percentiles,
    _to_tz_naive,
)
from src.data_fetcher import fetch_price_history
from src.composite_signal import composite_color
from src.report_generator import list_report_dates, load_report, scan_all, scan_ticker
from src.stock_analyzer import TYPE_LABEL, analyze_suitability, ensure_watchlist_analyzed
from src.utils import (
    add_to_watchlist,
    detect_reason_type_mismatch,
    ensure_dirs,
    get_holding,
    load_config,
    remove_from_watchlist,
    remove_holding,
    save_config,
    upsert_holding,
)

TYPE_CHINESE = {
    "stable": "穩定型",
    "growth": "成長型",
    "cyclical": "景氣循環型",
    "etf": "ETF",
    "unknown": "未知",
}

st.set_page_config(
    page_title="PE Monitor",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentiment_badge(sentiment: dict | None) -> str:
    """Return a compact emoji + text badge for a news_sentiment dict."""
    if not sentiment or not sentiment.get("available"):
        return "📰 N/A"
    label = sentiment.get("label", "neutral")
    return {"positive": "📰🟢 正面", "neutral": "📰⚪ 中性", "negative": "📰🔴 負面"}.get(label, "📰 N/A")


def _strategy_d_badge(signal) -> str:
    """Return badge for Strategy D signal. None = feature disabled."""
    if signal is None:
        return ""
    return "📐 收斂中" if signal else "—"


def _render_regime_badge(config: dict) -> None:
    """
    Render the market regime badge at the top of the dashboard.

    Shows the current RISK_ON / NEUTRAL / RISK_OFF / UNKNOWN classification
    with VIX, SPY vs 200MA, and 20-day return details so the user can sanity-
    check the macro context before reading individual-stock signals.
    """
    from src.market_regime import get_market_regime, regime_color, regime_display
    data_dir = config.get("settings", {}).get("data_dir", "data")
    try:
        regime = get_market_regime(data_dir)
    except Exception as e:
        st.caption(f"大盤環境偵測失敗：{e}")
        return

    key = regime.get("regime", "UNKNOWN")
    bg = regime_color(key) or "#f0f0f0"
    display = regime_display(key)
    details: list[str] = []
    if regime.get("vix") is not None:
        details.append(f"VIX {regime['vix']}")
    if regime.get("spy_vs_200ma") is not None:
        details.append(f"SPY vs 200MA {regime['spy_vs_200ma']*100:+.1f}%")
    if regime.get("spy_20d_ret") is not None:
        details.append(f"20日 {regime['spy_20d_ret']*100:+.1f}%")
    detail_line = "　|　".join(details) if details else "—"

    reasons = regime.get("reasons", [])
    reasons_html = "；".join(reasons) if reasons else ""

    # Badge
    st.markdown(
        f"""
        <div style='background-color:{bg}; padding:10px 14px; border-radius:8px; margin-bottom:8px;'>
          <div style='font-size:16px; font-weight:600;'>大盤環境：{display}</div>
          <div style='font-size:13px; color:#555; margin-top:4px;'>{detail_line}</div>
          {f"<div style='font-size:12px; color:#666; margin-top:2px;'>{reasons_html}</div>" if reasons_html else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if key == "RISK_OFF":
        st.caption("⚠️ RISK OFF：個股買進類訊號會自動降一級（避免熊市接刀）。")


@st.cache_data(ttl=300)
def cached_scan_all(watchlist_key: str) -> list[dict]:
    """Cache scan results for 5 minutes (key changes when watchlist changes)."""
    config = load_config()
    return scan_all(config)


def pe_band_chart(ticker: str, config: dict, strategy_d_dates: list | None = None) -> go.Figure | None:
    """
    Dual-axis chart:
      Left  Y-axis  — P/E (or P/B) line + percentile colour bands
      Right Y-axis  — Stock closing price line
    """
    settings = config["settings"]
    data_dir = settings["data_dir"]
    years = settings.get("pe_history_years", 5)

    wl_entry = next((e for e in config["watchlist"] if e["ticker"] == ticker), {})
    metric = wl_entry.get("recommended_metric", "PE")

    if metric == "PB":
        series = build_historical_pb_series(ticker, years=years, data_dir=data_dir)
        metric_name = "P/B"
    else:
        series = build_historical_pe_series(ticker, years=years, data_dir=data_dir)
        metric_name = "P/E"

    if series is None or series.empty:
        return None

    pcts = get_percentiles(series)
    if not pcts:
        return None

    # --- Stock price series (right axis) ---
    price_df = fetch_price_history(ticker, years=years, data_dir=data_dir)
    price_series = None
    if not price_df.empty and "Close" in price_df.columns:
        price_series = pd.to_numeric(price_df["Close"], errors="coerce").dropna()
        price_series.index = _to_tz_naive(pd.to_datetime(price_series.index))
        # Align to the PE series date range
        if not series.index.empty:
            price_series = price_series[price_series.index >= series.index.min()]

    # --- Build figure with secondary y-axis ---
    fig = go.Figure()

    # Percentile colour bands (left axis)
    band_defs = [
        (10, 25, "rgba(0,200,100,0.12)",   "BUY 區間"),
        (25, 50, "rgba(100,180,255,0.10)",  "WATCH 區間"),
        (50, 75, "rgba(255,220,0,0.10)",    "CAUTION 區間"),
        (75, 90, "rgba(255,80,80,0.12)",    "SELL 區間"),
    ]
    for lo, hi, color, label in band_defs:
        if lo in pcts and hi in pcts:
            fig.add_hrect(
                y0=pcts[lo], y1=pcts[hi],
                fillcolor=color, line_width=0,
                annotation_text=label,
                annotation_position="left",
                annotation_font_size=10,
                annotation_font_color="gray",
            )

    # Percentile horizontal lines (left axis)
    line_styles = {
        10: ("dash",  "#999"),
        25: ("dot",   "#27ae60"),
        50: ("solid", "#2980b9"),
        75: ("dot",   "#e74c3c"),
        90: ("dash",  "#999"),
    }
    for p, (dash, color) in line_styles.items():
        if p in pcts:
            fig.add_hline(
                y=pcts[p], line_dash=dash, line_color=color,
                annotation_text=f"P{p}: {pcts[p]:.1f}",
                annotation_position="right",
                annotation_font_size=10,
            )

    # P/E curve (left axis, yaxis="y")
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=metric_name,
            yaxis="y",
            line=dict(color="#2c7bb6", width=2),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{metric_name}: %{{y:.1f}}<extra></extra>",
        )
    )

    # Stock price curve (right axis, yaxis="y2")
    if price_series is not None and not price_series.empty:
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode="lines",
                name="股價 (USD)",
                yaxis="y2",
                line=dict(color="#e67e22", width=1.5, dash="dot"),
                opacity=0.75,
                hovertemplate="%{x|%Y-%m-%d}<br>股價: $%{y:.2f}<extra></extra>",
            )
        )

    # Strategy D signal markers on price curve (right axis)
    if strategy_d_dates and price_series is not None and not price_series.empty:
        sig_idx = _to_tz_naive(pd.to_datetime(strategy_d_dates, errors="coerce").dropna())
        sig_idx = sig_idx[
            (sig_idx >= price_series.index.min()) &
            (sig_idx <= price_series.index.max())
        ]
        if len(sig_idx) > 0:
            sig_prices = price_series.reindex(sig_idx, method="nearest")
            fig.add_trace(
                go.Scatter(
                    x=sig_prices.index,
                    y=sig_prices.values,
                    mode="markers",
                    name="Strategy D",
                    yaxis="y2",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#9b59b6",
                        line=dict(color="white", width=1),
                    ),
                    hovertemplate="%{x|%Y-%m-%d}<br>Strategy D 訊號<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"{ticker} — 歷史 {metric_name} Band + 股價（{years} 年）",
        xaxis=dict(title="日期", showgrid=False),
        yaxis=dict(
            title=dict(text=metric_name, font=dict(color="#2c7bb6")),
            tickfont=dict(color="#2c7bb6"),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis2=dict(
            title=dict(text="股價 (USD)", font=dict(color="#e67e22")),
            tickfont=dict(color="#e67e22"),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        height=450,
        margin=dict(l=60, r=80, t=60, b=40),
        hovermode="x unified",
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Page 1 — 自選股管理
# ---------------------------------------------------------------------------

def page_watchlist_management(config: dict) -> None:
    st.header("自選股管理")

    # --- Type selection guide ---
    with st.expander("📖 股票類型選擇指引（點擊展開）", expanded=False):
        st.markdown("""
### 快速決策表

| 條件 | 建議類型 | 主要因子 | 調整上限 |
|------|----------|---------|---------|
| EPS 穩定成長，波動 < 30% | **穩定型** (stable) | P/CF（權重×2）+ PEG + Forward P/E | ±2 格 |
| EPS 快速成長，波動 30–60% | **成長型** (growth) | PEG（×2）+ Forward P/E（×2） | ±2 格 |
| EPS 時正時負，與景氣強相關 | **景氣循環型** (cyclical) | EV/EBITDA + P/CF | ±1 格 |
| 追蹤指數或一籃子股票 | **ETF** (etf) | Forward P/E | ±1 格 |

### 各類型使用說明

**🟢 穩定型 (stable)** — 大型消費、公用事業、金融（如 MSFT、JNJ、V）
- P/CF 在此類型權重 ×2，因現金流比帳面盈餘更難造假
- 所有補充因子均啟用，允許最大 ±2 格訊號調整

**🟡 成長型 (growth)** — 科技、生技、高成長平台（如 GOOGL、NVDA、META）
- P/CF **不參與**計算（成長股大量再投資壓低 OCF，此數字失真）
- PEG 和 Forward P/E 各自權重 ×2，是最重要的判斷依據
- 允許最大 ±2 格訊號調整

**🔴 景氣循環型 (cyclical)** — 能源、鋼鐵、航運、半導體設備（如 XOM、CLF）
- PEG / Forward P/E **不參與**（景氣谷底時這些數字最危險）
- 改用 EV/EBITDA（< 8 低估，> 15 高估）作為主要補充因子
- 保守設計，僅允許 ±1 格訊號調整

**📦 ETF (etf)** — 指數基金、行業 ETF（如 SPY、QQQ）
- 僅 Forward P/E 有意義（反映市場整體共識估值）
- 僅允許 ±1 格訊號調整

---
**💡 手動覆蓋時機：** 自動分類依 EPS 波動率判斷，但產業邏輯更重要。
手動設定後不會被自動重新分析覆蓋（會標記為「✏️ 手動」來源）。
        """)

    # --- Add new stock ---
    st.subheader("新增股票")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_ticker = st.text_input("股票代號（例：AAPL）", key="new_ticker").strip().upper()
    with col2:
        analyze_btn = st.button("分析適合度", use_container_width=True)

    if analyze_btn and new_ticker:
        with st.spinner(f"分析 {new_ticker} 中..."):
            result = analyze_suitability(new_ticker, config["settings"]["data_dir"])

        if result.get("type") == "unknown" and not result.get("name"):
            st.error(f"找不到 {new_ticker} 的資料，請確認股票代號")
        else:
            score = result.get("suitability_score", 0)
            score_stars = "⭐" * score + "☆" * (5 - score)
            st.markdown(
                f"""
                **{result.get('name', new_ticker)}** (`{new_ticker}`)

                | 項目 | 說明 |
                |------|------|
                | 類型 | {TYPE_LABEL.get(result.get('type','unknown'), '未知')} |
                | 建議指標 | {result.get('recommended_metric', 'PE')} |
                | 適合度 | {score_stars} ({score}/5) |
                | 原因 | {result.get('reason', '')} |
                """
            )
            st.session_state["pending_add"] = result

    if "pending_add" in st.session_state:
        pending = st.session_state["pending_add"]
        existing = [e["ticker"] for e in config["watchlist"]]
        if pending["ticker"] in existing:
            st.info(f"{pending['ticker']} 已在自選股清單中")
            del st.session_state["pending_add"]
        else:
            with st.form("confirm_add_form"):
                st.write(f"確認加入 **{pending['ticker']}**？")
                has_holding = st.checkbox("我持有這檔股票")
                cost = st.number_input("持倉成本（USD）", min_value=0.0, value=0.0, step=0.01)
                shares = st.number_input("持倉股數", min_value=0.0, value=0.0, step=1.0)
                buy_date = st.date_input("買入日期", value=date.today())
                submitted = st.form_submit_button("確認加入")
                if submitted:
                    entry = {
                        "ticker": pending["ticker"],
                        "name": pending.get("name", ""),
                        "type": pending.get("type", "unknown"),
                        "recommended_metric": pending.get("recommended_metric", "PE"),
                        "suitability_score": pending.get("suitability_score", 0),
                        "reason": pending.get("reason", ""),
                        "added_date": date.today().isoformat(),
                    }
                    add_to_watchlist(config, entry)
                    if has_holding and cost > 0 and shares > 0:
                        upsert_holding(config, pending["ticker"], cost, shares, buy_date.isoformat())
                    del st.session_state["pending_add"]
                    st.success(f"已加入 {pending['ticker']}")
                    st.rerun()

    # --- Current watchlist ---
    st.subheader("目前自選股")
    if not config.get("watchlist"):
        st.info("自選股清單為空，請在上方新增股票")
        return

    for entry in config["watchlist"]:
        ticker = entry["ticker"]
        holding = get_holding(config, ticker)
        with st.expander(
            f"**{ticker}** — {entry.get('name', '')}  |  "
            f"{TYPE_LABEL.get(entry.get('type', 'unknown'), '未知')}  |  "
            f"{'💼 持倉' if holding else ''}",
            expanded=False,
        ):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                # Static info
                type_source = entry.get("type_source", "auto")
                src_label = {
                    "auto":      "🤖 自動",
                    "hard_rule": "🛡️ 產業規則",   # P2-12 / P2-13
                    "env":       "⚙️ .env 預設",
                    "manual":    "✏️ 手動",
                }.get(type_source, "🤖 自動")
                small_cap_tag = "  ｜  🏷️ 小型股" if entry.get("small_cap") else ""
                st.caption(
                    f"類型來源：{src_label}{small_cap_tag}  ｜  加入日期：{entry.get('added_date', '—')}"
                )

                # Consistency check: reason text claims type X but entry.type is Y
                claimed_type = detect_reason_type_mismatch(entry)
                if claimed_type:
                    col_warn, col_apply = st.columns([4, 1])
                    with col_warn:
                        st.warning(
                            f"⚠️ 分類已人工覆寫：分析器建議 **{TYPE_CHINESE.get(claimed_type, claimed_type)}**，"
                            f"但目前設定為 **{TYPE_CHINESE.get(entry.get('type', 'unknown'), '未知')}**。"
                        )
                    with col_apply:
                        # P2-13: 一鍵套用分類器建議（不必先按「重新分析」）
                        if st.button("✅ 套用建議", key=f"apply_auto_{ticker}", use_container_width=True):
                            for e in config["watchlist"]:
                                if e["ticker"] == ticker:
                                    e["type"] = claimed_type
                                    e["type_source"] = "auto"
                                    # Recompute recommended_metric based on the new type
                                    e["recommended_metric"] = "PB" if claimed_type == "cyclical" else "PE"
                                    break
                            save_config(config)
                            st.cache_data.clear()
                            st.rerun()

                st.write(f"分析說明：{entry.get('reason', '—')}")

                if holding:
                    st.markdown(
                        f"持倉：{holding['shares']} 股 @ ${holding['cost']:.2f}  "
                        f"（買入：{holding.get('buy_date', '—')}）"
                    )

                # Type & metric edit form
                TYPE_OPTIONS = ["stable", "growth", "cyclical", "etf", "unknown"]
                TYPE_NAMES = {
                    "stable": "穩定型 ✅", "growth": "成長型 🟡",
                    "cyclical": "景氣循環型 ❌", "etf": "ETF 📦", "unknown": "未知 ❓",
                }
                TYPE_HELP = (
                    "stable：P/CF 主導，允許 ±2 格調整\n"
                    "growth：PEG + Forward P/E 主導，P/CF 不參與，允許 ±2 格調整\n"
                    "cyclical：EV/EBITDA 主導，PEG/FPE 停用，允許 ±1 格調整\n"
                    "etf：僅 Forward P/E，允許 ±1 格調整\n"
                    "unknown：等同原始行為（全因子等權重，±1 格）"
                )
                with st.form(f"type_form_{ticker}"):
                    st.markdown("**調整類型與指標**")
                    current_type = entry.get("type", "unknown")
                    new_type = st.selectbox(
                        "股票類型",
                        options=TYPE_OPTIONS,
                        format_func=lambda t: TYPE_NAMES.get(t, t),
                        index=TYPE_OPTIONS.index(current_type) if current_type in TYPE_OPTIONS else 4,
                        help=TYPE_HELP,
                        key=f"type_sel_{ticker}",
                    )
                    current_metric = entry.get("recommended_metric", "PE")
                    new_metric = st.selectbox(
                        "估值指標",
                        options=["PE", "PB"],
                        index=0 if current_metric == "PE" else 1,
                        help="PE：適合 EPS 穩定正值的公司。PB：適合 EPS 負值、重資產或金融業。",
                        key=f"metric_sel_{ticker}",
                    )
                    fc1, fc2 = st.columns(2)
                    save_type = fc1.form_submit_button("💾 儲存", use_container_width=True)
                    reanalyze_btn = fc2.form_submit_button("🔄 重新分析", use_container_width=True)

                    if save_type:
                        for e in config["watchlist"]:
                            if e["ticker"] == ticker:
                                e["type"] = new_type
                                e["recommended_metric"] = new_metric
                                e["type_source"] = "manual"
                                break
                        save_config(config)
                        st.cache_data.clear()
                        st.rerun()

                    if reanalyze_btn:
                        with st.spinner(f"重新分析 {ticker}..."):
                            res_a = analyze_suitability(ticker, config["settings"]["data_dir"])
                        st.session_state[f"pending_reanalyze_{ticker}"] = res_a

                # Show re-analysis result (outside form — buttons cannot nest inside forms)
                if f"pending_reanalyze_{ticker}" in st.session_state:
                    res_a = st.session_state[f"pending_reanalyze_{ticker}"]
                    score_a = res_a.get("suitability_score", 0)
                    st.info(
                        f"分析結果：**{TYPE_LABEL.get(res_a.get('type','unknown'), '未知')}**  "
                        f"｜ 指標：{res_a.get('recommended_metric', 'PE')}  "
                        f"｜ 適合度：{'⭐' * score_a + '☆' * (5 - score_a)}  \n"
                        f"{res_a.get('reason', '')}"
                    )
                    if st.button("✅ 接受分析結果並儲存", key=f"confirm_ra_{ticker}"):
                        for e in config["watchlist"]:
                            if e["ticker"] == ticker:
                                e["type"] = res_a.get("type", "unknown")
                                e["type_source"] = "auto"
                                e["recommended_metric"] = res_a.get("recommended_metric", "PE")
                                e["suitability_score"] = res_a.get("suitability_score", 0)
                                e["reason"] = res_a.get("reason", "")
                                break
                        save_config(config)
                        st.cache_data.clear()
                        del st.session_state[f"pending_reanalyze_{ticker}"]
                        st.rerun()

            with col_b:
                # Edit holding
                with st.form(f"holding_form_{ticker}"):
                    st.write("更新持倉")
                    h_cost = st.number_input(
                        "成本", min_value=0.0,
                        value=float(holding["cost"]) if holding else 0.0,
                        step=0.01, key=f"cost_{ticker}"
                    )
                    h_shares = st.number_input(
                        "股數", min_value=0.0,
                        value=float(holding["shares"]) if holding else 0.0,
                        step=1.0, key=f"shares_{ticker}"
                    )
                    save_holding = st.form_submit_button("儲存持倉")
                    remove_h = st.form_submit_button("移除持倉")
                    if save_holding and h_cost > 0 and h_shares > 0:
                        upsert_holding(config, ticker, h_cost, h_shares)
                        st.success("持倉已更新")
                        st.rerun()
                    if remove_h:
                        remove_holding(config, ticker)
                        st.success("持倉已移除")
                        st.rerun()

                if st.button(f"移除 {ticker}", key=f"remove_{ticker}", type="secondary"):
                    remove_from_watchlist(config, ticker)
                    st.success(f"已移除 {ticker}")
                    st.rerun()


# ---------------------------------------------------------------------------
# Page 2 — 每日監測儀表板
# ---------------------------------------------------------------------------

def page_dashboard(config: dict) -> None:
    st.header("每日監測儀表板")

    col_ts, col_refresh = st.columns([4, 1])
    with col_ts:
        st.caption(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col_refresh:
        if st.button("🔄 重新掃描", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if not config.get("watchlist"):
        st.info("自選股清單為空，請先至「自選股管理」新增股票")
        return

    watchlist_key = ",".join(sorted(e["ticker"] for e in config["watchlist"]))

    with st.spinner("掃描中..."):
        results = cached_scan_all(watchlist_key)

    # --- Market regime badge (top-level context for all signals below) ---
    _render_regime_badge(config)

    # --- Summary table ---
    rows = []
    for r in results:
        metric_label = r.get("metric_label") or r.get("recommended_metric", "PE")
        rows.append(
            {
                "代號": r["ticker"],
                "名稱": (r.get("name") or "")[:20],
                "收盤價": r.get("price"),
                "TTM EPS": r.get("ttm_eps"),
                metric_label: r.get("metric_value"),
                "歷史百分位": r.get("percentile_rank"),
                "訊號": r.get("signal_display", "N/A"),
                "新聞情緒": _sentiment_badge(r.get("news_sentiment")),
                "綜合訊號": r.get("composite_display", r.get("signal_display", "N/A")),
                "技術訊號": _strategy_d_badge(r.get("strategy_d_signal")),
                "持倉損益%": r.get("holding_pnl_pct"),
                "⚠️": " | ".join(filter(None, [
                    "EPS 過期" if r.get("eps_stale") else "",
                    "⚠️ OCF 負值" if (r.get("operating_cashflow") is not None and r["operating_cashflow"] < 0) else "",
                    f"🪤 價值陷阱({(r.get('value_trap') or {}).get('severity', 0)})" if (r.get("value_trap") or {}).get("is_trap") else "",
                    r.get("error") or "",
                ])),
            }
        )

    df_display = pd.DataFrame(rows)

    # Colour signal column
    def color_signal(val: str) -> str:
        mapping = {
            "🌟": "background-color: #a8e6a3",
            "🟢": "background-color: #d4edda",
            "🔵": "background-color: #cce5ff",
            "⚪": "",
            "🟡": "background-color: #fff3cd",
            "🔴": "background-color: #f8d7da",
            "🚨": "background-color: #f5a0a8",
        }
        for emoji, style in mapping.items():
            if emoji in str(val):
                return style
        return ""

    signal_cols = [c for c in ["訊號", "綜合訊號"] if c in df_display.columns]
    if signal_cols:
        styled = df_display.style.map(color_signal, subset=signal_cols)
        if "技術訊號" in df_display.columns:
            styled = styled.map(
                lambda v: "background-color: #e8d5f5" if "收斂中" in str(v) else "",
                subset=["技術訊號"],
            )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # --- Per-ticker details ---
    st.subheader("個股詳情 & P/E Band 走勢圖")
    for r in results:
        ticker = r["ticker"]
        signal_disp = r.get("signal_display", "N/A")
        with st.expander(f"{ticker}  {signal_disp}", expanded=False):
            if r.get("error"):
                st.warning(f"資料錯誤：{r['error']}")
                continue

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("收盤價", f"${r['price']:.2f}" if r.get("price") else "N/A")
            eps_val = f"{r['ttm_eps']:.2f}" if r.get("ttm_eps") is not None else "N/A"
            c2.metric("TTM EPS", eps_val)
            metric_label = r.get("metric_label") or r.get("recommended_metric", "PE")
            c3.metric(metric_label, f"{r['metric_value']:.2f}" if r.get("metric_value") is not None else "N/A")
            c4.metric("歷史百分位", f"{r['percentile_rank']:.1f}%" if r.get("percentile_rank") is not None else "N/A")

            # Composite signal row
            cn1, cn2, cn3 = st.columns(3)
            cn1.metric("PE 訊號", r.get("signal_display", "N/A"))
            cn2.metric("新聞情緒", _sentiment_badge(r.get("news_sentiment")))
            cn3.metric("綜合訊號", r.get("composite_display", r.get("signal_display", "N/A")))

            # Factor breakdown for composite signal
            factors: dict = r.get("composite_factors", {})
            if factors:
                _FACTOR_BADGE = {1: "🔼", 0: "➡️", -1: "🔽"}
                badges = "  ".join(
                    f"{_FACTOR_BADGE.get(v, '?')} **{k}**"
                    for k, v in factors.items()
                )
                sentiment_badge_label = r.get("news_sentiment", {}).get("label", "neutral")
                sentiment_vote = {"positive": 1, "neutral": 0, "negative": -1}.get(sentiment_badge_label, 0)
                badges = f"{_FACTOR_BADGE.get(sentiment_vote, '?')} **新聞情緒**  " + badges
                st.caption(f"因子明細（影響綜合訊號）：{badges}")

            # Type-consistency warning (P2-13): reason disagrees with stored type,
            # or a hard-rule classification exists but user manually overrode.
            wl_entry = next((e for e in config.get("watchlist", []) if e["ticker"] == ticker), None)
            if wl_entry:
                claimed_type = detect_reason_type_mismatch(wl_entry)
                if claimed_type:
                    current_disp = TYPE_CHINESE.get(wl_entry.get("type", "unknown"), "未知")
                    claimed_disp = TYPE_CHINESE.get(claimed_type, claimed_type)
                    st.warning(
                        f"⚠️ **分類可能需要更新**：分析器/產業規則建議 **{claimed_disp}**，"
                        f"目前設定為 **{current_disp}**。前往「自選股管理」可一鍵套用建議。"
                    )
                elif wl_entry.get("type_source") == "hard_rule":
                    st.caption(
                        f"🛡️ 類型由產業規則判定（{TYPE_CHINESE.get(wl_entry.get('type'), '?')}）"
                    )
                if wl_entry.get("small_cap"):
                    st.caption("🏷️ **小型股警示**：市值 < $2B，流動性與資料品質可能較差")

            # Value-trap warning (P1-7)
            vt = r.get("value_trap") or {}
            if vt.get("flags"):
                severity = vt.get("severity", 0)
                flags_html = "；".join(vt["flags"])
                if vt.get("is_trap"):
                    st.warning(f"🪤 **價值陷阱警告**（嚴重度 {severity}/4）：{flags_html}\n\n已自動將買進類訊號壓低到 WATCH。")
                else:
                    st.info(f"⚠️ 基本面惡化訊號（嚴重度 {severity}/4）：{flags_html}")

            # Position-sizing advice (P1-8)
            advice = r.get("position_advice") or {}
            if advice:
                _ACTION_EMOJI = {
                    "INITIAL": "🟢 建倉",
                    "ADD":     "➕ 加碼",
                    "HOLD":    "⏸️ 續抱",
                    "TRIM":    "✂️ 減碼",
                    "EXIT":    "🚪 出清",
                }
                act_label = _ACTION_EMOJI.get(advice.get("action"), advice.get("action", ""))
                size = advice.get("size_pct", 0.0)
                size_str = f"（{size:.0f}%）" if size else ""
                st.markdown(f"**建議動作**：{act_label}{size_str}　—　{advice.get('advice', '')}")

            # Supplementary valuation metrics (P/CF, PEG, Forward P/E)
            pcf = r.get("pcf_ratio")
            peg = r.get("peg_ratio")
            fpe = r.get("forward_pe")
            if any(v is not None for v in [pcf, peg, fpe]):
                st.markdown("**補充估值指標（參考用）**")
                supp_cols = st.columns(3)
                trailing_pe = r.get("metric_value") if r.get("metric_label", "").startswith("PE") else None
                trailing_pe_str = f"{trailing_pe:.1f}x" if trailing_pe else "N/A"

                supp_cols[0].metric(
                    "P/CF（市現率）",
                    f"{pcf:.1f}x" if pcf is not None else "N/A",
                    help=(
                        "股價 ÷ 每股營業現金流（Operating Cash Flow per Share）\n\n"
                        "比 P/E 更難被會計手法調整，適合重資產公司（晶片廠、電信）\n\n"
                        "• < 10  ➜ 偏低估\n"
                        "• 10–20 ➜ 合理\n"
                        "• > 20  ➜ 偏高估"
                    ),
                )
                supp_cols[1].metric(
                    "PEG 比率",
                    f"{peg:.2f}" if peg is not None else "N/A",
                    help=(
                        "P/E ÷ 預期 EPS 年成長率（分析師共識）\n\n"
                        "修正高本益比不一定貴的問題，考慮成長速度\n\n"
                        "• < 1.0 ➜ 相對低估（Peter Lynch 標準）\n"
                        "• ≈ 1.0 ➜ 合理定價\n"
                        "• > 2.0 ➜ 相對高估"
                    ),
                )
                supp_cols[2].metric(
                    "Forward P/E",
                    f"{fpe:.1f}x" if fpe is not None else "N/A",
                    help=(
                        f"股價 ÷ 未來 12 個月預估 EPS（分析師共識）\n\n"
                        f"目前 Trailing P/E：{trailing_pe_str}\n\n"
                        "• Forward < Trailing ➜ 市場預期盈餘成長（好兆頭）\n"
                        "• Forward > Trailing ➜ 市場預期盈餘衰退（警訊）\n"
                        "• S&P 500 長期均值約 15–17x"
                    ),
                )

                # Row 2: type-specific additional metrics
                stock_type_r = r.get("stock_type", r.get("type", "unknown"))
                ps_ratio = r.get("ps_ratio")
                rev_growth = r.get("revenue_growth")
                div_yield = r.get("dividend_yield")
                beta = r.get("beta")

                if any(v is not None for v in [ps_ratio, rev_growth, div_yield, beta]):
                    row2_cols = st.columns(3)
                    if stock_type_r == "growth":
                        row2_cols[0].metric(
                            "P/S（市銷率）",
                            f"{ps_ratio:.1f}x" if ps_ratio is not None else "N/A",
                            help=(
                                "股價 ÷ 每股年營收（Price-to-Sales）\n\n"
                                "適用於獲利尚不穩定的高成長公司\n\n"
                                "• < 4  ➜ 偏低估\n"
                                "• 4–20 ➜ 合理（視產業而異）\n"
                                "• > 20 ➜ 偏高估\n\n"
                                "⚠️ P/S 業界差異大：SaaS 平均約 10x，消費品約 2x"
                            ),
                        )
                        rev_str = f"{rev_growth*100:.1f}%" if rev_growth is not None else "N/A"
                        row2_cols[1].metric(
                            "營收成長率（YoY）",
                            rev_str,
                            help=(
                                "年營收年增率（來源：yfinance 分析師共識）\n\n"
                                "• > 15% ➜ 成長論點成立\n"
                                "• 5–15% ➜ 溫和成長\n"
                                "• < 5%  ➜ 成長論點減弱，建議重新評估類型"
                            ),
                        )
                        row2_cols[2].metric(
                            "Beta",
                            f"{beta:.2f}" if beta is not None else "N/A",
                            help=(
                                "相對大盤波動性（Beta = 1 等於大盤）\n\n"
                                "• < 0.8 ➜ 防禦型（低波動）\n"
                                "• 0.8–1.2 ➜ 市場中性\n"
                                "• > 1.5 ➜ 高波動／景氣循環特性"
                            ),
                        )
                    elif stock_type_r == "stable":
                        div_str = f"{div_yield*100:.2f}%" if div_yield is not None else "N/A"
                        row2_cols[0].metric(
                            "股息殖利率",
                            div_str,
                            help=(
                                "年化股息 ÷ 股價（來源：yfinance）\n\n"
                                "穩定型公司的重要現金回饋指標\n\n"
                                "• ≥ 2% ➜ 健康（穩定型正面信號）\n"
                                "• < 0.5% ➜ 過低（若歷史有配息需留意）"
                            ),
                        )
                        row2_cols[1].metric(
                            "Beta",
                            f"{beta:.2f}" if beta is not None else "N/A",
                            help=(
                                "相對大盤波動性\n\n"
                                "• < 0.8 ➜ 防禦型，符合穩定股特徵\n"
                                "• > 1.3 ➜ 偏成長／高波動，考慮重新評估類型"
                            ),
                        )
                        row2_cols[2].metric(
                            "P/S（市銷率）",
                            f"{ps_ratio:.1f}x" if ps_ratio is not None else "N/A",
                            help=(
                                "股價 ÷ 每股年營收\n\n"
                                "即使穩定型公司，高 P/S 仍代表估值偏貴\n\n"
                                "• < 4  ➜ 偏低估\n"
                                "• > 20 ➜ 偏高估"
                            ),
                        )
                    else:
                        row2_cols[0].metric(
                            "Beta",
                            f"{beta:.2f}" if beta is not None else "N/A",
                            help=(
                                "相對大盤波動性（Beta = 1 等於大盤）\n\n"
                                "• < 0.8 ➜ 防禦型\n"
                                "• > 1.5 ➜ 高波動／景氣循環特性"
                            ),
                        )
                        row2_cols[1].metric(
                            "P/S（市銷率）",
                            f"{ps_ratio:.1f}x" if ps_ratio is not None else "N/A",
                            help="股價 ÷ 每股年營收，輔助估值參考",
                        )

                # P3-14: Shiller / Normalized P/E (secondary reference)
                shiller = r.get("shiller") or {}
                if shiller.get("available"):
                    sh_cols = st.columns(3)
                    sh_pe = shiller.get("shiller_pe")
                    norm_pe = shiller.get("normalized_pe")
                    years = shiller.get("years_used", 0)
                    trailing_pe_val = r.get("metric_value") if (r.get("metric_label") or "").startswith("PE") else None
                    sh_cols[0].metric(
                        "Shiller PE（10 年 CAPE）",
                        f"{sh_pe:.1f}x" if sh_pe is not None else "N/A",
                        help=(
                            f"股價 ÷ 10 年平均 EPS（2.5%/年通膨調整後）\n\n"
                            f"實際使用 {years} 年資料\n\n"
                            "• 平滑景氣循環中 EPS 峰谷，適合週期股估值\n"
                            "• 若 Trailing P/E 低但 Shiller PE 高 ➜ 目前 EPS 為循環高點\n"
                            "• 若 Trailing P/E 高但 Shiller PE 合理 ➜ EPS 暫時受壓"
                        ),
                    )
                    sh_cols[1].metric(
                        "Normalized PE（未通膨調整）",
                        f"{norm_pe:.1f}x" if norm_pe is not None else "N/A",
                        help="同上，但不做通膨調整，純 10 年 EPS 平均",
                    )
                    if trailing_pe_val and sh_pe:
                        diff_pct = (trailing_pe_val - sh_pe) / sh_pe * 100
                        sh_cols[2].metric(
                            "Trailing vs Shiller 差",
                            f"{diff_pct:+.0f}%",
                            help=(
                                "Trailing P/E 相對 Shiller PE 的差距：\n\n"
                                "• 正值大 ➜ 目前 EPS 低於 10 年常態，或市場過度樂觀\n"
                                "• 負值大 ➜ 目前 EPS 高於 10 年常態，可能處於循環高點"
                            ),
                        )

                with st.expander("📖 補充指標解讀說明", expanded=False):
                    st.markdown("""
| 指標 | 公式 | 偏低（較佳） | 偏高（謹慎） |
|------|------|------------|------------|
| P/CF | 股價 ÷ 每股 OCF | < 10 | > 20 |
| PEG  | P/E ÷ EPS 成長率 | < 1.0 | > 2.0 |
| Forward P/E | 股價 ÷ 預估 EPS | < 15（S&P均值）| > 25 |
| P/S  | 股價 ÷ 每股年營收 | < 4（保守）| > 20 |
| 殖利率 | 年化股息 ÷ 股價 | ≥ 2%（穩定型）| — |
| Beta | 相對大盤波動性 | < 0.8（防禦）| > 1.5（高波動）|

**組合解讀：**
- P/CF 低 + PEG < 1 + Forward P/E < Trailing P/E → 多重低估，強化 BUY 訊號
- Trailing P/E 低但 P/CF 高 → 帳面 EPS 好看但現金流差，需謹慎
- PEG > 2 但 Forward P/E 低 → 成長預期大幅下修，請留意
- 成長股：P/S 高但營收成長 > 15% → 成長溢價合理
- 穩定股：殖利率 ≥ 2% + Beta < 0.8 → 符合穩定型特徵

> 以上數據來源：yfinance 分析師共識預估，非公司官方公告，**僅供參考**。
                    """)

            # OCF quality warning
            ocf = r.get("operating_cashflow")
            if ocf is not None and ocf < 0:
                st.warning(
                    f"⚠️ 營業現金流為負（OCF: ${ocf:,.0f}）｜ 帳面盈利品質存疑，"
                    "綜合訊號可靠度下降，建議搭配其他財務指標驗證"
                )

            # Strategy D technical signal row
            sd_signal = r.get("strategy_d_signal")
            if sd_signal is not None:
                sd_val = "📐 收斂訊號（Strategy D）" if sd_signal else "— 無訊號"
                sd1, sd2 = st.columns(2)
                sd1.metric("技術動能訊號", sd_val)
                if sd_signal:
                    sd2.caption("KD 低檔金叉 + MACD 柱狀圖收斂，動能翻轉前兆")
                if r.get("strategy_d_error"):
                    st.caption(f"⚠️ 技術訊號計算錯誤：{r['strategy_d_error']}")

            if r.get("eps_stale"):
                st.warning(f"⚠️ EPS 資料可能過期（最後報告：{r.get('last_report_date', '未知')}）")

            # Holding info
            if r.get("holding_cost") is not None:
                pnl = r.get("holding_pnl_pct")
                pnl_str = f"{pnl:+.2f}%" if pnl is not None else "N/A"
                pnl_color = "green" if (pnl or 0) >= 0 else "red"
                st.markdown(
                    f"持倉 {r['holding_shares']} 股 @ ${r['holding_cost']:.2f}  "
                    f"損益：<span style='color:{pnl_color};font-weight:bold'>{pnl_str}</span>",
                    unsafe_allow_html=True,
                )
                if r.get("signal") == "SELL":
                    st.info("📢 已進入出場區間，考慮減倉")

            # Percentile band details
            pcts = r.get("percentiles")
            if pcts:
                p_cols = st.columns(5)
                for i, p in enumerate([10, 25, 50, 75, 90]):
                    p_cols[i].metric(f"P{p}", f"{pcts.get(p, 0):.1f}")

            # News sentiment section
            sentiment = r.get("news_sentiment")
            news_status = r.get("news_status", "none")
            news_source = r.get("news_source", "none")

            st.markdown("**📰 最新新聞與情緒**")
            # Source / status caption
            source_msgs = {
                "rss_fallback": "📡 資料來源：Yahoo Finance RSS（備用）",
                "rate_limited": "📡 Finnhub 達速率上限，使用 RSS 備用資料",
                "invalid_key":  "❌ Finnhub API Key 無效，使用 RSS 備用資料",
                "failed":       None,
            }
            src_msg = source_msgs.get(news_status) or (
                source_msgs.get(news_source) if news_source != "finnhub" else None
            )
            if src_msg:
                st.caption(src_msg)

            if news_status == "failed" and news_source == "none":
                st.warning("⚠️ 無法取得新聞，綜合訊號僅依 PE 計算")
            elif sentiment and sentiment.get("available"):
                articles_display = r.get("news_articles", [])
                if articles_display:
                    for art in articles_display:
                        art_label = art.get("sentiment_label", "neutral")
                        badge = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}.get(art_label, "⚪")
                        pub_date = art.get("date_str", "")
                        source_name = art.get("source", "")
                        url = art.get("url", "")
                        headline = art.get("headline", "")
                        meta = f"{source_name} · {pub_date}" if source_name else pub_date
                        if url:
                            st.markdown(
                                f"{badge} [{headline}]({url})  "
                                f"<small style='color:gray'>{meta}</small>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"{badge} {headline}  "
                                f"<small style='color:gray'>{meta}</small>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption("📰 近期無相關新聞")
            else:
                api_key = config.get("settings", {}).get("finnhub_api_key", "").strip()
                if not api_key:
                    st.caption("💡 在 config.json 設定 `finnhub_api_key` 可啟用精準新聞情緒分析")
                else:
                    st.caption("📰 近期無相關新聞")

            # Chart
            fig = pe_band_chart(ticker, config, strategy_d_dates=r.get("strategy_d_dates", []))
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("歷史資料不足，無法繪製 Band 圖")


# ---------------------------------------------------------------------------
# Page 3 — 歷史報告
# ---------------------------------------------------------------------------

def page_history(config: dict) -> None:
    st.header("歷史報告")

    report_dir = config["settings"]["report_dir"]
    dates = list_report_dates(report_dir)

    if not dates:
        st.info("尚無歷史報告。請先執行 `python main.py` 或在儀表板按「重新掃描」")
        return

    fmt_dates = {d: f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates}
    selected_fmt = st.selectbox("選擇日期", list(fmt_dates.values()))
    selected = [k for k, v in fmt_dates.items() if v == selected_fmt][0]

    df = load_report(selected, report_dir)
    if df.empty:
        st.error("無法讀取報告")
        return

    st.write(f"報告日期：**{selected_fmt}**  |  共 {len(df)} 檔股票")

    def color_row(val: str) -> str:
        mapping = {
            "🟢": "background-color: #d4edda",
            "🔵": "background-color: #cce5ff",
            "🟡": "background-color: #fff3cd",
            "🔴": "background-color: #f8d7da",
        }
        for emoji, style in mapping.items():
            if emoji in str(val):
                return style
        return ""

    if "signal_display" in df.columns:
        styled = df.style.map(color_row, subset=["signal_display"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下載 CSV",
        data=csv,
        file_name=f"pe_monitor_{selected}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------

def page_settings(config: dict) -> None:
    st.header("⚙️ 策略設置")
    settings = config["settings"]

    # ── 基本面設置 ──────────────────────────────────────────────────────────
    st.subheader("基本面（PE/PB 百分位）")
    with st.form("fundamental_form"):
        entry_pct = st.slider(
            "進場百分位（低於此值 = BUY）", 5, 45,
            value=settings.get("entry_percentile", 25), step=5,
            help="當前 PE/PB 百分位低於此值時，訊號顯示為 BUY",
        )
        exit_pct = st.slider(
            "出場百分位（高於此值 = SELL）", 55, 95,
            value=settings.get("exit_percentile", 75), step=5,
            help="當前 PE/PB 百分位高於此值時，訊號顯示為 SELL",
        )
        if st.form_submit_button("儲存基本面設置"):
            settings["entry_percentile"] = entry_pct
            settings["exit_percentile"] = exit_pct
            save_config(config)
            st.cache_data.clear()
            st.success("已儲存，下次掃描時生效")

    st.markdown("---")

    # ── Strategy D 設置 ──────────────────────────────────────────────────────
    st.subheader("Strategy D 技術訊號（KD + MACD）")
    sd = settings.get("strategy_d", {})
    with st.form("strategy_d_form"):
        enabled = st.toggle(
            "啟用 Strategy D",
            value=sd.get("enabled", False),
            help="啟用後掃描時計算 KD+MACD 收斂訊號（需安裝 pandas-ta）",
        )

        st.markdown("**KD 金叉條件**")
        col1, col2 = st.columns(2)
        kd_window = col1.slider(
            "回顧視窗 n（根）",
            min_value=3, max_value=30,
            value=int(sd.get("kd_window", 10)),
            help="在最近 n 根 K 棒內，只要曾出現金叉即算符合條件",
        )
        kd_k_threshold = col2.slider(
            "K 值門檻 m",
            min_value=10, max_value=40,
            value=int(sd.get("kd_k_threshold", 20)),
            help="金叉時 K 值須低於 m（例如 20 = 超賣區，放寬可設 30）",
        )

        st.markdown("**MACD 柱狀圖收斂條件**")
        col3, col4 = st.columns(2)
        n_bars = col3.slider(
            "收斂根數",
            min_value=2, max_value=6,
            value=int(sd.get("n_bars", 3)),
            help="連續幾根柱狀圖為負且逐漸縮小（越大越嚴格）",
        )
        recovery_pct = col4.slider(
            "回彈比例門檻",
            min_value=0.3, max_value=0.9,
            value=float(sd.get("recovery_pct", 0.7)),
            step=0.05, format="%.2f",
            help="最新柱狀圖須從谷底回彈達此比例（例如 0.7 = 回彈 70%，越大越嚴格）",
        )

        if st.form_submit_button("儲存 Strategy D 設置"):
            settings["strategy_d"] = {
                "enabled": enabled,
                "kd_window": kd_window,
                "kd_k_threshold": kd_k_threshold,
                "n_bars": n_bars,
                "recovery_pct": recovery_pct,
            }
            save_config(config)
            st.cache_data.clear()
            st.success("已儲存，下次掃描時生效")
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    ensure_dirs(config)

    # Auto-analyze any watchlist stock that hasn't been classified yet
    # (e.g. stocks bootstrapped from .env with type="unknown").
    # Runs once per browser session to avoid re-fetching on every rerun.
    if "watchlist_analyzed" not in st.session_state:
        unanalyzed = [e for e in config.get("watchlist", []) if e.get("type", "unknown") == "unknown"]
        if unanalyzed:
            with st.spinner(f"首次啟動：分析 {len(unanalyzed)} 檔股票適合度..."):
                ensure_watchlist_analyzed(config)
                config = load_config()  # reload with updated entries
                st.cache_data.clear()
        st.session_state["watchlist_analyzed"] = True

    st.sidebar.title("📊 PE Monitor")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "頁面",
        ["自選股管理", "每日監測儀表板", "歷史報告", "⚙️ 策略設置"],
        index=1,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"自選股：{len(config.get('watchlist', []))} 檔  \n"
        f"持倉：{len(config.get('holdings', []))} 檔"
    )

    # News status indicator (shown after a scan has run)
    watchlist_key = ",".join(sorted(e["ticker"] for e in config.get("watchlist", [])))
    if watchlist_key and page == "每日監測儀表板":
        cached_results = cached_scan_all(watchlist_key) if config.get("watchlist") else []
        if cached_results:
            statuses = {r.get("news_status", "none") for r in cached_results}
            if "invalid_key" in statuses:
                st.sidebar.error("新聞情緒：🔴 Finnhub API Key 無效")
            elif "rate_limited" in statuses:
                st.sidebar.warning("新聞情緒：⚠️ Finnhub 達速率上限，已切換 RSS")
            elif "rss_fallback" in statuses:
                st.sidebar.warning("新聞情緒：⚠️ 使用 RSS 備用（建議設定 Finnhub Key）")
            elif "failed" in statuses or "none" in statuses:
                st.sidebar.error("新聞情緒：🔴 無法取得新聞資料")
            elif "no_articles" in statuses:
                st.sidebar.info("新聞情緒：📰 部分股票無近期新聞")
            else:
                st.sidebar.success("新聞情緒：✅ Finnhub 正常運作")

    # Strategy D toggle (always visible, gated by pandas-ta availability)
    st.sidebar.markdown("---")
    sd_settings = config["settings"].get("strategy_d", {})
    sd_currently_enabled = sd_settings.get("enabled", False)
    sd_toggled = st.sidebar.toggle(
        "Strategy D 技術訊號",
        value=sd_currently_enabled,
        help="啟用 KD+MACD 收斂訊號（需安裝 pandas-ta）",
    )
    if sd_toggled != sd_currently_enabled:
        config["settings"].setdefault("strategy_d", {})["enabled"] = sd_toggled
        save_config(config)
        st.cache_data.clear()
        st.rerun()

    if page == "自選股管理":
        page_watchlist_management(config)
    elif page == "每日監測儀表板":
        page_dashboard(config)
    elif page == "歷史報告":
        page_history(config)
    elif page == "⚙️ 策略設置":
        page_settings(config)


if __name__ == "__main__":
    main()
