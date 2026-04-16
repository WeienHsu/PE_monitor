"""
Determine stock suitability for P/E-based analysis.
"""

import numpy as np
import pandas as pd

from src.data_fetcher import fetch_info, fetch_quarterly_financials, is_etf


def _get_annual_eps(ticker: str, data_dir: str = "data") -> list[float]:
    """Return list of annual EPS values (most recent first)."""
    df = fetch_quarterly_financials(ticker, data_dir)
    if df.empty:
        return []

    net_income_row = None
    for name in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
        if name in df.index:
            net_income_row = df.loc[name]
            break
    if net_income_row is None:
        return []

    # Group quarters into years
    net_income_row = net_income_row.dropna().sort_index(ascending=False)
    # Convert index to Period year
    by_year: dict[int, float] = {}
    for dt, val in net_income_row.items():
        year = pd.Timestamp(dt).year
        by_year[year] = by_year.get(year, 0) + val

    if len(by_year) < 2:
        return list(by_year.values())

    # Sort by year descending
    sorted_vals = [v for _, v in sorted(by_year.items(), reverse=True)]
    return sorted_vals


def analyze_suitability(ticker: str, data_dir: str = "data") -> dict:
    """
    Analyse a ticker and return a suitability dict:
    {
        "ticker": str,
        "type": "stable" | "growth" | "cyclical" | "etf" | "unknown",
        "recommended_metric": "PE" | "PB",
        "reason": str,
        "suitability_score": int (1-5),
    }
    """
    result = {
        "ticker": ticker,
        "type": "unknown",
        "recommended_metric": "PE",
        "reason": "",
        "suitability_score": 3,
    }

    # ETF check first
    if is_etf(ticker, data_dir):
        info = fetch_info(ticker, data_dir)
        name = info.get("longName") or ticker
        result.update(
            {
                "name": name,
                "type": "etf",
                "recommended_metric": "PE",
                "reason": "ETF：成分股混合，P/E 為加權平均，僅供參考",
                "suitability_score": 3,
            }
        )
        return result

    info = fetch_info(ticker, data_dir)
    name = info.get("longName") or ticker
    result["name"] = name

    annual_eps = _get_annual_eps(ticker, data_dir)
    if len(annual_eps) < 2:
        result["reason"] = "EPS 資料不足，無法自動判斷類型"
        result["suitability_score"] = 2
        return result

    has_negative = any(v < 0 for v in annual_eps)

    # Calculate YoY growth rates
    growth_rates = []
    for i in range(len(annual_eps) - 1):
        current = annual_eps[i + 1]  # older year
        newer = annual_eps[i]       # more recent year
        if current == 0:
            continue
        growth_rates.append((newer - current) / abs(current))

    if not growth_rates:
        result["reason"] = "EPS 年增率無法計算（分母為零）"
        result["suitability_score"] = 2
        return result

    std_pct = float(np.std(growth_rates) * 100)

    if has_negative:
        result.update(
            {
                "type": "cyclical",
                "recommended_metric": "PB",
                "reason": f"EPS 波動率 {std_pct:.0f}%，且曾出現負值，不適合 P/E 區間法",
                "suitability_score": 2,
            }
        )
    elif std_pct > 60:
        result.update(
            {
                "type": "cyclical",
                "recommended_metric": "PB",
                "reason": f"EPS 波動率 {std_pct:.0f}%，景氣循環型，建議改用 P/B",
                "suitability_score": 2,
            }
        )
    elif std_pct > 30:
        result.update(
            {
                "type": "growth",
                "recommended_metric": "PE",
                "reason": f"EPS 波動率 {std_pct:.0f}%，成長型，P/E 區間僅供參考，建議搭配 PEG",
                "suitability_score": 3,
            }
        )
    else:
        result.update(
            {
                "type": "stable",
                "recommended_metric": "PE",
                "reason": f"EPS 波動率 {std_pct:.0f}%，穩定型，適合用 P/E 區間",
                "suitability_score": 5,
            }
        )

    return result


def ensure_watchlist_analyzed(config: dict) -> bool:
    """
    Run analyze_suitability() for any watchlist entry whose type is "unknown".

    Updates config entries in-place and saves config.json.
    Returns True if at least one entry was updated.
    """
    from src.utils import save_config  # avoid circular import at module level

    data_dir = config.get("settings", {}).get("data_dir", "data")
    updated = False
    for entry in config.get("watchlist", []):
        if entry.get("type", "unknown") == "unknown":
            result = analyze_suitability(entry["ticker"], data_dir)
            entry["name"] = result.get("name") or entry.get("name", "")
            entry["type"] = result.get("type", "unknown")
            entry["recommended_metric"] = result.get("recommended_metric", "PE")
            entry["suitability_score"] = result.get("suitability_score", 0)
            entry["reason"] = result.get("reason", "")
            updated = True
    if updated:
        save_config(config)
    return updated


TYPE_LABEL = {
    "stable": "穩定型 ✅",
    "growth": "成長型 🟡",
    "cyclical": "景氣循環型 ❌",
    "etf": "ETF 📦",
    "unknown": "未知 ❓",
}
