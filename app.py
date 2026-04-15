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
from src.stock_analyzer import TYPE_LABEL, analyze_suitability
from src.utils import (
    add_to_watchlist,
    ensure_dirs,
    get_holding,
    load_config,
    remove_from_watchlist,
    remove_holding,
    save_config,
    upsert_holding,
)

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
                st.write(f"建議指標：**{entry.get('recommended_metric', 'PE')}**")
                st.write(f"分析說明：{entry.get('reason', '—')}")
                st.write(f"加入日期：{entry.get('added_date', '—')}")

                if holding:
                    st.markdown(
                        f"持倉：{holding['shares']} 股 @ ${holding['cost']:.2f}  "
                        f"（買入：{holding.get('buy_date', '—')}）"
                    )

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
                "⚠️": "EPS 過期" if r.get("eps_stale") else ("" if not r.get("error") else r["error"]),
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    ensure_dirs(config)

    st.sidebar.title("📊 PE Monitor")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "頁面",
        ["自選股管理", "每日監測儀表板", "歷史報告"],
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


if __name__ == "__main__":
    main()
