"""
Generate daily scan reports and save them to reports/.
"""

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.data_fetcher import get_latest_close, is_etf
from src.pe_calculator import (
    SIGNAL_EMOJI,
    SIGNAL_LABEL,
    build_historical_pb_series,
    build_historical_pe_series,
    calc_ttm_eps,
    classify_signal,
    current_percentile_rank,
    get_pb_ratio,
    get_percentiles,
)
from src.composite_signal import compute_composite
from src.news_fetcher import fetch_news
from src.sentiment_analyzer import analyze_sentiment, score_articles_individually
from src.utils import days_since, get_holding, load_config


def scan_ticker(ticker: str, config: dict) -> dict:
    """
    Run a full P/E (or P/B) scan for a single ticker.
    Returns a result dict with all display fields.
    """
    settings = config["settings"]
    data_dir = settings["data_dir"]
    years = settings.get("pe_history_years", 5)
    entry_pct = settings.get("entry_percentile", 25)
    exit_pct = settings.get("exit_percentile", 75)

    # Find watchlist entry
    wl_entry = next((e for e in config["watchlist"] if e["ticker"] == ticker), {})
    recommended_metric = wl_entry.get("recommended_metric", "PE")

    result = {
        "ticker": ticker,
        "name": wl_entry.get("name", ""),
        "type": wl_entry.get("type", "unknown"),
        "recommended_metric": recommended_metric,
        "price": None,
        "ttm_eps": None,
        "metric_value": None,  # actual PE or PB
        "metric_label": recommended_metric,
        "percentile_rank": None,
        "signal": "N/A",
        "signal_display": "N/A",
        "percentiles": {},
        "last_report_date": None,
        "eps_stale": False,
        "error": None,
    }

    # 1. Latest price
    skip_metric = False
    history = None
    price = get_latest_close(ticker, data_dir)
    if price is None:
        result["error"] = "無法取得股價"
        skip_metric = True
    else:
        result["price"] = round(price, 2)

    # 2. Compute metric
    if not skip_metric:
        if recommended_metric == "PB":
            pb = get_pb_ratio(ticker, price, data_dir)
            if pb is None:
                result["error"] = "無法取得 P/B 資料"
                skip_metric = True
            else:
                result["metric_value"] = round(pb, 2)
                history = build_historical_pb_series(ticker, years=years, data_dir=data_dir)
        else:
            ttm_eps, report_date = calc_ttm_eps(ticker, data_dir)

            # If EPS is negative → fallback to P/B
            if ttm_eps is not None and ttm_eps <= 0:
                result["metric_label"] = "PB (EPS 負值)"
                result["recommended_metric"] = "PB"
                pb = get_pb_ratio(ticker, price, data_dir)
                if pb:
                    result["metric_value"] = round(pb, 2)
                    history = build_historical_pb_series(ticker, years=years, data_dir=data_dir)
                    result["ttm_eps"] = round(ttm_eps, 4)
                    result["last_report_date"] = report_date
                else:
                    result["error"] = "EPS 為負，且無法取得 P/B"
                    skip_metric = True
            elif ttm_eps is None:
                result["error"] = "無法計算 TTM EPS"
                skip_metric = True
            else:
                pe = price / ttm_eps
                result["ttm_eps"] = round(ttm_eps, 4)
                result["metric_value"] = round(pe, 2)
                result["last_report_date"] = report_date
                if report_date and days_since(report_date) > 100:
                    result["eps_stale"] = True
                history = build_historical_pe_series(ticker, years=years, data_dir=data_dir)

    # 3. Percentiles & signal
    if not skip_metric and history is not None and not history.empty:
        result["percentiles"] = get_percentiles(history)
        rank = current_percentile_rank(result["metric_value"], history)
        result["percentile_rank"] = round(rank, 1)
        signal = classify_signal(rank, entry=entry_pct, exit_=exit_pct)
        result["signal"] = signal
        result["signal_display"] = f"{SIGNAL_EMOJI[signal]} {SIGNAL_LABEL[signal]}"

    # 4. News sentiment & composite signal (always runs, even for ETFs without P/E data)
    try:
        lookback_days = int(settings.get("news_days_lookback", 7))
        articles, news_source, news_status = fetch_news(ticker, config, data_dir)
        sentiment = analyze_sentiment(articles, lookback_days=lookback_days)
        scored_articles = score_articles_individually(articles, lookback_days=lookback_days)
        result["news_sentiment"] = sentiment
        result["news_articles"] = scored_articles[:3]
        result["news_source"] = news_source
        result["news_status"] = news_status
        # Composite signal
        if sentiment.get("available") and result["signal"] not in ("N/A", None):
            composite_key, composite_display = compute_composite(
                result["signal"], sentiment["label"]
            )
        else:
            composite_key = result["signal"]
            composite_display = result["signal_display"]
        result["composite_signal"] = composite_key
        result["composite_display"] = composite_display
    except Exception:
        result["news_sentiment"] = {"available": False, "score": 0.0, "label": "neutral",
                                    "count": 0, "article_count_by_label": {}, "latest_headline": "",
                                    "latest_date": ""}
        result["news_articles"] = []
        result["news_source"] = "none"
        result["news_status"] = "failed"
        result["composite_signal"] = result["signal"]
        result["composite_display"] = result["signal_display"]

    # 5. Strategy D 技術訊號（KD 低檔金叉 + MACD 柱狀圖收斂）
    sd_cfg = settings.get("strategy_d", {})
    if sd_cfg.get("enabled", False):
        from src.technical_signals import compute_strategy_d  # lazy import：未啟用時不需 pandas-ta
        sd = compute_strategy_d(
            ticker,
            data_dir,
            kd_window=sd_cfg.get("kd_window", 10),
            n_bars=sd_cfg.get("n_bars", 3),
            recovery_pct=sd_cfg.get("recovery_pct", 0.7),
        )
        result["strategy_d_signal"] = sd["signal"]
        result["strategy_d_dates"]  = sd["signal_dates"]
        result["strategy_d_error"]  = sd["error"]
    else:
        result["strategy_d_signal"] = None   # None = 功能未啟用
        result["strategy_d_dates"]  = []
        result["strategy_d_error"]  = None

    return result


def scan_all(config: dict) -> list[dict]:
    """Scan all tickers in the watchlist and return list of result dicts."""
    tickers = [e["ticker"] for e in config.get("watchlist", [])]
    results = []
    for ticker in tickers:
        r = scan_ticker(ticker, config)
        # Attach holding info
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
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def list_report_dates(report_dir: str = "reports") -> list[str]:
    """Return sorted list of available report dates (YYYYMMDD strings)."""
    p = Path(report_dir)
    if not p.exists():
        return []
    dates = []
    for f in p.glob("daily_*.csv"):
        name = f.stem  # daily_YYYYMMDD
        dates.append(name.replace("daily_", ""))
    return sorted(dates, reverse=True)


def load_report(date_str: str, report_dir: str = "reports") -> pd.DataFrame:
    """Load a daily report CSV by date string (YYYYMMDD)."""
    path = Path(report_dir) / f"daily_{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")
