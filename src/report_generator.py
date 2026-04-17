"""
Generate daily scan reports and save them to reports/.

Pipeline (post-QVM refactor):
    1. Price & TTM EPS
    2. Percentiles (PE, PB) — still displayed in UI band chart
    3. Supplementary ratios (P/CF, PEG, Forward P/E, EV/EBITDA, OCF, CAPE)
    4. Strategy D (feeds into M)
    5. SMA200 (trend filter)
    6. V/Q/M scores
    7. News sentiment
    8. QVM composite → base signal → sentiment-adjusted composite signal
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_fetcher import (
    fetch_cashflow,
    fetch_info,
    fetch_price_history,
    get_latest_close,
    is_etf as _is_etf_check,
)
from src.pe_calculator import (
    SIGNAL_EMOJI,
    SIGNAL_LABEL,
    build_historical_pb_series,
    build_historical_pe_series,
    calc_cape_style_pe,
    calc_ttm_eps,
    current_percentile_rank,
    get_ev_ebitda,
    get_forward_pe,
    get_pb_ratio,
    get_pcf_ratio,
    get_peg_ratio,
    get_percentiles,
)
from src.factors import (
    compute_m_score,
    compute_q_score,
    compute_qvm,
    compute_v_score,
)
from src.factors.quality_factor import QualityInputs
from src.factors.value_factor import ValueInputs
from src.etf_signal import compute_etf_q_score, compute_etf_v_score
from src.etf_industry_map import get_damodaran_for_yfinance_industry
from src.external_data import get_industry_trailing_pe
from src.news_fetcher import fetch_news
from src.sentiment_analyzer import analyze_sentiment, score_articles_individually
from src.utils import days_since, get_holding


def _compute_sma200(ticker: str, data_dir: str, years: int) -> float | None:
    try:
        df = fetch_price_history(ticker, years=years, data_dir=data_dir)
        if df.empty or "Close" not in df.columns:
            return None
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 200:
            return None
        return float(close.tail(200).mean())
    except Exception:
        return None


def _compute_eps_std_pct(ticker: str, data_dir: str) -> float | None:
    """Recompute annual EPS YoY growth std-% (from stock_analyzer._get_annual_eps)."""
    try:
        from src.stock_analyzer import _get_annual_eps
        annual_eps = _get_annual_eps(ticker, data_dir)
        if len(annual_eps) < 2:
            return None
        growth_rates = []
        for i in range(len(annual_eps) - 1):
            current = annual_eps[i + 1]
            newer = annual_eps[i]
            if current != 0:
                growth_rates.append((newer - current) / abs(current))
        if not growth_rates:
            return None
        return float(np.std(growth_rates) * 100)
    except Exception:
        return None


def scan_ticker(ticker: str, config: dict) -> dict:
    """Run a full QVM scan for a single ticker. Returns a result dict."""
    settings = config["settings"]
    data_dir = settings["data_dir"]
    years = settings.get("pe_history_years", 5)

    wl_entry = next((e for e in config["watchlist"] if e["ticker"] == ticker), {})
    stock_type = wl_entry.get("type", "unknown")
    etf_subtype = wl_entry.get("etf_subtype")
    is_etf_ticker = stock_type == "etf" or _is_etf_check(ticker, data_dir)
    if is_etf_ticker and not etf_subtype:
        from src.stock_analyzer import classify_etf_subtype
        etf_subtype = classify_etf_subtype(ticker, data_dir)
    if is_etf_ticker and stock_type != "etf":
        stock_type = "etf"

    result: dict = {
        "ticker": ticker,
        "name": wl_entry.get("name", ""),
        "type": stock_type,
        "stock_type": stock_type,
        "etf_subtype": etf_subtype,
        "recommended_metric": wl_entry.get("recommended_metric", "PE"),
        "price": None,
        "ttm_eps": None,
        "metric_value": None,
        "metric_label": wl_entry.get("recommended_metric", "PE"),
        "percentile_rank": None,
        "signal": "N/A",
        "signal_display": "N/A",
        "percentiles": {},
        "last_report_date": None,
        "eps_stale": False,
        "error": None,
        "pcf_ratio": None,
        "peg_ratio": None,
        "forward_pe": None,
        "ev_ebitda": None,
        "operating_cashflow": None,
        "cape_pe": None,
        "sma200": None,
        # QVM
        "v_score": None,
        "q_score": None,
        "m_score": None,
        "qvm_raw": None,
        "qvm_weights": {},
        "qvm_gates": {},
        "v_details": {},
        "q_details": {},
        "m_details": {},
        # Composite (post-sentiment)
        "composite_signal": "N/A",
        "composite_display": "N/A",
        "composite_factors": {},
    }

    # ---- 1. Price ----
    price = get_latest_close(ticker, data_dir)
    if price is None:
        result["error"] = "無法取得股價"
        return result
    result["price"] = round(price, 2)

    # ---- 2. TTM EPS + PE percentile ----
    pe_percentile = None
    ttm_eps_val: float | None = None
    try:
        ttm_eps_val, report_date = calc_ttm_eps(ticker, data_dir)
        if ttm_eps_val is not None:
            result["ttm_eps"] = round(ttm_eps_val, 4)
            result["last_report_date"] = report_date
            if report_date and days_since(report_date) > 100:
                result["eps_stale"] = True
        if ttm_eps_val is not None and ttm_eps_val > 0:
            pe = price / ttm_eps_val
            result["metric_value"] = round(pe, 2)
            result["metric_label"] = "PE"
            pe_history = build_historical_pe_series(ticker, years=years, data_dir=data_dir)
            if not pe_history.empty:
                result["percentiles"] = get_percentiles(pe_history)
                pe_percentile = current_percentile_rank(pe, pe_history)
                result["percentile_rank"] = round(pe_percentile, 1)
        elif ttm_eps_val is not None and ttm_eps_val <= 0:
            result["metric_label"] = "PB (EPS 負值)"
    except Exception as e:
        print(f"[report_generator] PE stage {ticker}: {e}")

    # ---- 3. P/B percentile ----
    pb_percentile = None
    try:
        pb = get_pb_ratio(ticker, price, data_dir)
        if pb:
            pb_history = build_historical_pb_series(ticker, years=years, data_dir=data_dir)
            if not pb_history.empty:
                pb_percentile = current_percentile_rank(pb, pb_history)
            # If PE was unusable, surface P/B in the UI metric column
            if result["metric_value"] is None:
                result["metric_value"] = round(pb, 2)
                result["percentiles"] = get_percentiles(pb_history) if not pb_history.empty else {}
                if pb_percentile is not None:
                    result["percentile_rank"] = round(pb_percentile, 1)
    except Exception as e:
        print(f"[report_generator] PB stage {ticker}: {e}")

    # ---- 4. CAPE-like current value ----
    try:
        result["cape_pe"] = calc_cape_style_pe(ticker, price, data_dir, years=years)
    except Exception as e:
        print(f"[report_generator] CAPE stage {ticker}: {e}")

    # ---- 5. Supplementary metrics ----
    try:
        result["pcf_ratio"] = get_pcf_ratio(ticker, price, data_dir)
        result["peg_ratio"] = get_peg_ratio(ticker, data_dir)
        result["forward_pe"] = get_forward_pe(ticker, data_dir)
        result["ev_ebitda"] = get_ev_ebitda(ticker, data_dir)
        result["operating_cashflow"] = fetch_cashflow(ticker, data_dir).get("operating_cashflow")
    except Exception as e:
        print(f"[report_generator] supplementary stage {ticker}: {e}")

    # ---- 6. Strategy D (must precede M) ----
    sd_cfg = settings.get("strategy_d", {})
    sd_signal: bool | None = None
    if sd_cfg.get("enabled", False):
        try:
            from src.technical_signals import compute_strategy_d
            sd = compute_strategy_d(
                ticker,
                data_dir,
                kd_window=sd_cfg.get("kd_window", 10),
                n_bars=sd_cfg.get("n_bars", 3),
                recovery_pct=sd_cfg.get("recovery_pct", 0.7),
            )
            sd_signal = sd["signal"]
            result["strategy_d_signal"] = sd["signal"]
            result["strategy_d_dates"] = sd["signal_dates"]
            result["strategy_d_error"] = sd["error"]
        except Exception as e:
            result["strategy_d_signal"] = None
            result["strategy_d_dates"] = []
            result["strategy_d_error"] = str(e)
    else:
        result["strategy_d_signal"] = None
        result["strategy_d_dates"] = []
        result["strategy_d_error"] = None

    # ---- 7. SMA200 (trend filter) ----
    sma200 = _compute_sma200(ticker, data_dir, years)
    result["sma200"] = round(sma200, 2) if sma200 else None

    # ---- 8. V score (ETF or individual stock path) ----
    try:
        if is_etf_ticker:
            v_score, v_details = compute_etf_v_score(
                ticker, etf_subtype, price, data_dir=data_dir, years=years
            )
        else:
            # Industry PE (Damodaran) for "stock PE vs sector median" input
            info_for_industry = fetch_info(ticker, data_dir)
            yf_ind = info_for_industry.get("industry")
            damodaran_name = get_damodaran_for_yfinance_industry(yf_ind)
            industry_pe = get_industry_trailing_pe(damodaran_name, data_dir) if damodaran_name else None
            ttm_pe_val = result["metric_value"] if result.get("metric_label") == "PE" else None

            v_inputs = ValueInputs(
                ttm_pe_percentile=pe_percentile,
                pb_percentile=pb_percentile,
                cape_absolute=result["cape_pe"],
                forward_pe=result["forward_pe"],
                pcf_ratio=result["pcf_ratio"],
                ev_ebitda=result["ev_ebitda"],
                ttm_pe=ttm_pe_val,
                industry_pe=industry_pe,
                industry_name=damodaran_name,
            )
            v_score, v_details = compute_v_score(v_inputs)
        result["v_score"] = v_score
        result["v_details"] = v_details
    except Exception as e:
        print(f"[report_generator] V stage {ticker}: {e}")

    # ---- 9. Q score (ETF or individual stock path) ----
    try:
        if is_etf_ticker:
            q_score, q_details = compute_etf_q_score(ticker, data_dir=data_dir)
        else:
            info = fetch_info(ticker, data_dir)
            q_inputs = QualityInputs(
                gross_margin=info.get("grossMargins"),
                return_on_equity=info.get("returnOnEquity"),
                operating_margin=info.get("operatingMargins"),
                eps_growth_std_pct=_compute_eps_std_pct(ticker, data_dir),
                debt_to_equity=info.get("debtToEquity"),
            )
            q_score, q_details = compute_q_score(q_inputs)
        result["q_score"] = q_score
        result["q_details"] = q_details
    except Exception as e:
        print(f"[report_generator] Q stage {ticker}: {e}")

    # ---- 10. M score ----
    try:
        m_score, m_details = compute_m_score(
            ticker,
            data_dir,
            strategy_d_signal=sd_signal,
            years=years,
        )
        result["m_score"] = m_score
        result["m_details"] = m_details
    except Exception as e:
        print(f"[report_generator] M stage {ticker}: {e}")

    # ---- 11. News sentiment ----
    sent_label = "neutral"
    try:
        lookback_days = int(settings.get("news_days_lookback", 7))
        articles, news_source, news_status = fetch_news(ticker, config, data_dir)
        sentiment = analyze_sentiment(articles, lookback_days=lookback_days)
        scored_articles = score_articles_individually(articles, lookback_days=lookback_days)
        result["news_sentiment"] = sentiment
        result["news_articles"] = scored_articles[:3]
        result["news_source"] = news_source
        result["news_status"] = news_status
        if sentiment.get("available"):
            sent_label = sentiment["label"]
    except Exception as e:
        print(f"[report_generator] news stage {ticker}: {e}")
        result["news_sentiment"] = {
            "available": False, "score": 0.0, "label": "neutral",
            "count": 0, "article_count_by_label": {}, "latest_headline": "",
            "latest_date": "",
        }
        result["news_articles"] = []
        result["news_source"] = "none"
        result["news_status"] = "failed"

    # ---- 12. QVM composite (base signal + gates + sentiment overlay) ----
    try:
        qvm_result = compute_qvm(
            v_score=result["v_score"],
            q_score=result["q_score"],
            m_score=result["m_score"],
            stock_type=stock_type,
            etf_subtype=etf_subtype,
            sentiment_label=sent_label,
            operating_cashflow=result["operating_cashflow"],
            ttm_eps=result["ttm_eps"],
            price=price,
            sma200=sma200,
            is_etf=is_etf_ticker,
        )
        result["qvm_raw"] = qvm_result["qvm_raw"]
        result["qvm_weights"] = qvm_result["weights"]
        result["qvm_gates"] = qvm_result["gates"]

        base = qvm_result["base_signal"]
        result["signal"] = base
        if base in SIGNAL_EMOJI:
            result["signal_display"] = f"{SIGNAL_EMOJI[base]} {SIGNAL_LABEL[base]}"

        result["composite_signal"] = qvm_result["composite_signal"]
        result["composite_display"] = qvm_result["composite_display"]

        # Factor breakdown for UI (reused field name)
        breakdown: dict[str, float] = {}
        if result["v_score"] is not None:
            breakdown["V (Value)"] = result["v_score"]
        if result["q_score"] is not None:
            breakdown["Q (Quality)"] = result["q_score"]
        if result["m_score"] is not None:
            breakdown["M (Momentum)"] = result["m_score"]
        result["composite_factors"] = breakdown
    except Exception as e:
        print(f"[report_generator] QVM stage {ticker}: {e}")
        if not result["error"]:
            result["error"] = f"QVM 計算失敗: {e}"

    return result


def scan_all(config: dict) -> list[dict]:
    """Scan all tickers in the watchlist and return list of result dicts."""
    tickers = [e["ticker"] for e in config.get("watchlist", [])]
    results = []
    for ticker in tickers:
        r = scan_ticker(ticker, config)
        holding = get_holding(config, ticker)
        if holding:
            r["holding_cost"] = holding["cost"]
            r["holding_shares"] = holding["shares"]
            if r["price"] and holding["cost"] > 0:
                r["holding_pnl_pct"] = round((r["price"] - holding["cost"]) / holding["cost"] * 100, 2)
            else:
                r["holding_pnl_pct"] = None
        else:
            r["holding_cost"] = None
            r["holding_shares"] = None
            r["holding_pnl_pct"] = None
        results.append(r)
    return results


def save_daily_report(results: list[dict], report_dir: str = "reports") -> str:
    """Save today's scan to reports/daily_YYYYMMDD.csv and return the file path."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    path = Path(report_dir) / f"daily_{today}.csv"

    rows = []
    for r in results:
        rows.append(
            {
                "ticker": r["ticker"],
                "name": r["name"],
                "price": r.get("price"),
                "ttm_eps": r.get("ttm_eps"),
                "metric_label": r.get("metric_label", r.get("recommended_metric")),
                "metric_value": r.get("metric_value"),
                "percentile_rank": r.get("percentile_rank"),
                "signal": r.get("signal"),
                "signal_display": r.get("signal_display"),
                "eps_stale": r.get("eps_stale", False),
                "holding_cost": r.get("holding_cost"),
                "holding_shares": r.get("holding_shares"),
                "holding_pnl_pct": r.get("holding_pnl_pct"),
                "error": r.get("error"),
                "last_report_date": r.get("last_report_date"),
                "news_status": r.get("news_status", ""),
                "news_source": r.get("news_source", ""),
                "news_label": (r.get("news_sentiment") or {}).get("label", ""),
                "composite_signal": r.get("composite_signal", r.get("signal", "")),
                "composite_display": r.get("composite_display", r.get("signal_display", "")),
                "strategy_d_signal": r.get("strategy_d_signal"),
                "stock_type": r.get("stock_type", r.get("type", "")),
                "etf_subtype": r.get("etf_subtype"),
                "pcf_ratio": r.get("pcf_ratio"),
                "peg_ratio": r.get("peg_ratio"),
                "forward_pe": r.get("forward_pe"),
                "ev_ebitda": r.get("ev_ebitda"),
                "cape_pe": r.get("cape_pe"),
                "sma200": r.get("sma200"),
                "v_score": r.get("v_score"),
                "q_score": r.get("q_score"),
                "m_score": r.get("m_score"),
                "qvm_raw": r.get("qvm_raw"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _yesterday_date_str() -> str:
    from datetime import timedelta
    return (date.today() - timedelta(days=1)).strftime("%Y%m%d")


def compare_signals(
    today_results: list[dict],
    yesterday_df: pd.DataFrame,
) -> list[dict]:
    """Compare today's composite signals against yesterday's saved CSV."""
    if yesterday_df.empty:
        return []

    prev_signals: dict[str, str] = {}
    if "ticker" in yesterday_df.columns and "composite_signal" in yesterday_df.columns:
        for _, row in yesterday_df.iterrows():
            ticker_val = str(row["ticker"])
            sig = str(row.get("composite_signal", "") or "")
            if ticker_val and sig:
                prev_signals[ticker_val] = sig

    changes = []
    for r in today_results:
        ticker_val = r["ticker"]
        today_sig = r.get("composite_signal") or r.get("signal", "N/A")
        prev_sig = prev_signals.get(ticker_val)
        if prev_sig and prev_sig != today_sig and today_sig not in ("N/A", None, ""):
            changes.append(
                {
                    "ticker": ticker_val,
                    "name": r.get("name", ""),
                    "from_signal": prev_sig,
                    "to_signal": today_sig,
                    "current_price": r.get("price"),
                    "metric_label": r.get("metric_label") or r.get("recommended_metric", "PE"),
                    "metric_value": r.get("metric_value"),
                    "percentile_rank": r.get("percentile_rank"),
                }
            )
    return changes


def list_report_dates(report_dir: str = "reports") -> list[str]:
    """Return sorted list of available report dates (YYYYMMDD strings)."""
    p = Path(report_dir)
    if not p.exists():
        return []
    dates = []
    for f in p.glob("daily_*.csv"):
        name = f.stem
        dates.append(name.replace("daily_", ""))
    return sorted(dates, reverse=True)


def load_report(date_str: str, report_dir: str = "reports") -> pd.DataFrame:
    """Load a daily report CSV by date string (YYYYMMDD)."""
    path = Path(report_dir) / f"daily_{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")
