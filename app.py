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
)
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

@st.cache_data(ttl=300)
def cached_scan_all(watchlist_key: str) -> list[dict]:
    """Cache scan results for 5 minutes (key changes when watchlist changes)."""
    config = load_config()
    return scan_all(config)


def pe_band_chart(ticker: str, config: dict) -> go.Figure | None:
    """Return a Plotly figure showing historical P/E (or P/B) with percentile bands."""
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

    fig = go.Figure()

    # Band fills
    band_colors = [
        (10, 25, "rgba(0,200,100,0.10)"),
        (25, 50, "rgba(100,180,255,0.10)"),
        (50, 75, "rgba(255,220,0,0.10)"),
        (75, 90, "rgba(255,80,80,0.10)"),
    ]
    for lo, hi, color in band_colors:
        if lo in pcts and hi in pcts:
            fig.add_hrect(
                y0=pcts[lo], y1=pcts[hi],
                fillcolor=color, line_width=0,
            )

    # Percentile lines
    line_styles = {
        10: ("dash", "gray"),
        25: ("dot", "green"),
        50: ("solid", "blue"),
        75: ("dot", "red"),
        90: ("dash", "gray"),
    }
    for p, (dash, color) in line_styles.items():
        if p in pcts:
            fig.add_hline(
                y=pcts[p], line_dash=dash, line_color=color,
                annotation_text=f"P{p}: {pcts[p]:.1f}",
                annotation_position="right",
            )

    # Main series
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=metric_name,
            line=dict(color="steelblue", width=1.5),
        )
    )

    fig.update_layout(
        title=f"{ticker} — 歷史 {metric_name} Band（{years} 年）",
        xaxis_title="日期",
        yaxis_title=metric_name,
        height=400,
        margin=dict(l=50, r=80, t=50, b=30),
        showlegend=False,
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
                "持倉損益%": r.get("holding_pnl_pct"),
                "⚠️": "EPS 過期" if r.get("eps_stale") else ("" if not r.get("error") else r["error"]),
            }
        )

    df_display = pd.DataFrame(rows)

    # Colour signal column
    def color_signal(val: str) -> str:
        mapping = {
            "🟢": "background-color: #d4edda",
            "🔵": "background-color: #cce5ff",
            "⚪": "",
            "🟡": "background-color: #fff3cd",
            "🔴": "background-color: #f8d7da",
        }
        for emoji, style in mapping.items():
            if emoji in str(val):
                return style
        return ""

    if "訊號" in df_display.columns:
        styled = df_display.style.map(color_signal, subset=["訊號"])
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

            # Chart
            fig = pe_band_chart(ticker, config)
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

    if page == "自選股管理":
        page_watchlist_management(config)
    elif page == "每日監測儀表板":
        page_dashboard(config)
    elif page == "歷史報告":
        page_history(config)


if __name__ == "__main__":
    main()
