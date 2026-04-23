"""
Determine stock suitability for P/E-based analysis.

Classification flow (P2-12):
  1. ETF → type="etf"
  2. Industry / sector hard-classification (``_SECTOR_HARD_CLASSIFY``) —
     domain knowledge overrides fundamentals. Semis are structurally
     cyclical regardless of the current quarter's EPS trend.
  3. Multi-factor scoring — the legacy path, kept for everything that
     doesn't match a hard rule.

A ``type_source`` field is included in the result so downstream code
(app.py P2-13) can surface disagreements between the auto-classifier and
any manual override in ``config.json``.
"""

import numpy as np
import pandas as pd

from src.data_fetcher import (
    fetch_annual_financials,
    fetch_fundamental_extras,
    fetch_info,
    fetch_quarterly_financials,
    is_etf,
)


# ---------------------------------------------------------------------------
# P2-12: Industry / sector hard-classification
# ---------------------------------------------------------------------------
#
# yfinance exposes both ``info["sector"]`` (broad: "Technology") and
# ``info["industry"]`` (granular: "Semiconductors").  We match both, with
# industry taking precedence when it's more specific.
#
# Only include industries where domain knowledge is high-confidence — a
# fundamentals miss for Semis is less costly than mis-labelling "Software"
# as cyclical.
# ---------------------------------------------------------------------------

# industry (granular) → type
_INDUSTRY_HARD_CLASSIFY: dict[str, str] = {
    # Cyclical — capital-intensive, tied to capex cycles
    "Semiconductors": "cyclical",
    "Semiconductor Equipment & Materials": "cyclical",
    "Auto Manufacturers": "cyclical",
    "Auto Parts": "cyclical",
    "Steel": "cyclical",
    "Aluminum": "cyclical",
    "Copper": "cyclical",
    "Oil & Gas E&P": "cyclical",
    "Oil & Gas Integrated": "cyclical",
    "Oil & Gas Midstream": "cyclical",
    "Oil & Gas Refining & Marketing": "cyclical",
    "Oil & Gas Equipment & Services": "cyclical",
    "Airlines": "cyclical",
    "Marine Shipping": "cyclical",
    "Homebuilding & Construction": "cyclical",
    "Residential Construction": "cyclical",
    "Building Materials": "cyclical",
    # Stable — defensive demand / regulated
    "Utilities - Regulated Electric": "stable",
    "Utilities - Regulated Gas": "stable",
    "Utilities - Regulated Water": "stable",
    "Utilities - Diversified": "stable",
    "Beverages - Non-Alcoholic": "stable",
    "Beverages - Brewers": "stable",
    "Tobacco": "stable",
    "Packaged Foods": "stable",
    "Household & Personal Products": "stable",
    "Discount Stores": "stable",
    "Grocery Stores": "stable",
    "Drug Manufacturers - General": "stable",
    "Pharmaceutical Retailers": "stable",
    "Medical Devices": "stable",
    "Medical Instruments & Supplies": "stable",
    "Insurance - Life": "stable",
}

# sector (broad) → type, used as a fallback when industry isn't in the table.
# Weaker evidence — only very clear sectors.
_SECTOR_HARD_CLASSIFY: dict[str, str] = {
    "Energy": "cyclical",
    "Basic Materials": "cyclical",
    "Utilities": "stable",
}

# Small-cap threshold (USD market cap). Flagged as ``small_cap=True`` in result.
_SMALL_CAP_THRESHOLD = 2_000_000_000  # $2B


def hard_classify_from_info(info: dict) -> tuple[str | None, str | None]:
    """
    Return (type, rule_source) if industry/sector is in the hard-classify table,
    else (None, None).  ``rule_source`` is one of "industry" or "sector".
    """
    industry = (info.get("industry") or "").strip()
    if industry and industry in _INDUSTRY_HARD_CLASSIFY:
        return _INDUSTRY_HARD_CLASSIFY[industry], "industry"

    sector = (info.get("sector") or "").strip()
    if sector and sector in _SECTOR_HARD_CLASSIFY:
        return _SECTOR_HARD_CLASSIFY[sector], "sector"

    return None, None


_NI_ROW_NAMES = ["Net Income", "NetIncome", "Net Income Common Stockholders"]


def _extract_net_income_row(df: pd.DataFrame) -> pd.Series | None:
    """Return the Net Income row from an income-statement DataFrame, or None."""
    if df is None or df.empty:
        return None
    for name in _NI_ROW_NAMES:
        if name in df.index:
            row = df.loc[name]
            # If the DataFrame has duplicates (rare), take the first
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.dropna()
    return None


def _get_annual_eps(ticker: str, data_dir: str = "data") -> list[float]:
    """
    Return list of annual Net Income values (most recent first).

    Prefers ``stock.income_stmt`` (5 years of real annual reports).  Falls
    back to grouping quarterly data into years only when the annual
    statement is unavailable — and in that fallback case, discards years
    with fewer than 4 quarters to avoid partial-year artifacts.

    Fixes the P0-2 follow-up: the old code aggregated quarterly data,
    which for most yfinance tickers yields only 1-2 annual points →
    ``len(growth_rates) < 2`` → classifier always reports "EPS資料不足".
    """
    # 1. Preferred path: annual income statement (5 years)
    annual_df = fetch_annual_financials(ticker, data_dir)
    ni_annual = _extract_net_income_row(annual_df)
    if ni_annual is not None and len(ni_annual) >= 2:
        sorted_series = ni_annual.sort_index(ascending=False)
        return [float(v) for v in sorted_series.values]

    # 2. Fallback: quarterly aggregation, with partial-year filter
    q_df = fetch_quarterly_financials(ticker, data_dir)
    ni_q = _extract_net_income_row(q_df)
    if ni_q is None:
        # Use whatever annual data we have, even if < 2 points
        if ni_annual is not None:
            sorted_series = ni_annual.sort_index(ascending=False)
            return [float(v) for v in sorted_series.values]
        return []

    ni_q = ni_q.sort_index(ascending=False)
    by_year: dict[int, tuple[float, int]] = {}  # year → (sum, n_quarters)
    for dt, val in ni_q.items():
        year = pd.Timestamp(dt).year
        cur_sum, cur_n = by_year.get(year, (0.0, 0))
        by_year[year] = (cur_sum + float(val), cur_n + 1)

    # Keep only years with all 4 quarters (avoid Q4-only years looking "small")
    full_years = {y: s for y, (s, n) in by_year.items() if n >= 4}
    if len(full_years) >= 2:
        return [v for _, v in sorted(full_years.items(), reverse=True)]

    # Last resort: include partial years if that's all we have. Classifier
    # will still mark std_pct=None (insufficient-data path) when only 1 annual.
    return [s for _, (s, _n) in sorted(by_year.items(), reverse=True)]


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
        "type_source": "auto",   # P2-12: "hard_rule" | "auto" | "manual"
        "small_cap": False,      # P2-12: market cap < $2B
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
                "type_source": "hard_rule",
            }
        )
        return result

    info = fetch_info(ticker, data_dir)
    name = info.get("longName") or ticker
    result["name"] = name

    # Small-cap flag (independent of type)
    mc = info.get("marketCap")
    try:
        if mc is not None and float(mc) < _SMALL_CAP_THRESHOLD:
            result["small_cap"] = True
    except (TypeError, ValueError):
        pass

    # P2-12: industry / sector hard-classification — skips multi-factor scoring.
    hard_type, rule_source = hard_classify_from_info(info)
    if hard_type is not None:
        industry = info.get("industry") or info.get("sector") or ""
        type_labels = {"stable": "穩定型", "growth": "成長型", "cyclical": "景氣循環型"}
        label = type_labels.get(hard_type, hard_type)
        metric = "PB" if hard_type == "cyclical" else "PE"
        # Suitability: stable 5, cyclical 2, growth 3 (matches downstream expectations)
        score_map = {"stable": 5, "growth": 3, "cyclical": 2}
        result.update({
            "type": hard_type,
            "recommended_metric": metric,
            "reason": f"產業「{industry}」依規則判定為{label}（跳過多因子評分）",
            "suitability_score": score_map.get(hard_type, 3),
            "type_source": "hard_rule",
        })
        return result

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

    # Need at least 2 growth rates to compute a meaningful std.
    # yfinance free tier often returns only 4-5 quarters → 1-2 annual EPS points,
    # yielding 0-1 growth rate → np.std returns 0, which previously caused every
    # stock to be mis-classified as "stable". Explicitly mark as insufficient data.
    if len(growth_rates) < 2:
        std_pct = None
    else:
        std_pct = float(np.std(growth_rates) * 100)

    # --- Multi-factor scoring ---
    # Each dimension contributes points toward growth, stable, or cyclical.
    extras = fetch_fundamental_extras(ticker, data_dir)
    beta = extras.get("beta")
    rev_growth = extras.get("revenue_growth")   # float, e.g. 0.15 = 15%
    div_yield = extras.get("dividend_yield")    # float, e.g. 0.025 = 2.5%

    growth_score = 0
    stable_score = 0
    cyclical_score = 0

    # EPS volatility dimension (most weight: ±2).
    # Skip voting entirely when EPS data is insufficient to avoid false bias.
    if std_pct is None:
        if has_negative:
            cyclical_score += 2   # negative EPS is still a strong signal even without volatility
    elif has_negative or std_pct > 60:
        cyclical_score += 2
    elif std_pct > 30:
        growth_score += 2
    else:
        stable_score += 2

    # Revenue growth dimension
    if rev_growth is not None:
        if rev_growth > 0.20:
            growth_score += 2
        elif rev_growth > 0.10:
            growth_score += 1
        elif rev_growth < 0:
            cyclical_score += 1
        elif rev_growth < 0.02:
            stable_score += 1

    # Beta dimension
    if beta is not None:
        if beta < 0.8:
            stable_score += 1
        elif beta >= 1.5:
            cyclical_score += 1
        elif beta >= 1.2:
            growth_score += 1

    # Dividend yield dimension
    if div_yield is not None and div_yield >= 0.02:
        stable_score += 1

    # Classify by highest score; tiebreak: stable > growth > cyclical (conservative)
    if stable_score >= growth_score and stable_score >= cyclical_score:
        best_type = "stable"
    elif growth_score >= cyclical_score:
        best_type = "growth"
    else:
        best_type = "cyclical"

    # Build factor summary for reason string
    if std_pct is None:
        factor_parts = ["EPS資料不足"]
    else:
        factor_parts = [f"EPS波動率{std_pct:.0f}%"]
    if rev_growth is not None:
        factor_parts.append(f"營收成長{rev_growth*100:.0f}%")
    if beta is not None:
        factor_parts.append(f"Beta={beta:.1f}")
    if div_yield is not None and div_yield > 0:
        factor_parts.append(f"殖利率{div_yield*100:.1f}%")
    factor_summary = "，".join(factor_parts)

    score_summary = f"（成長{growth_score}分／穩定{stable_score}分／循環{cyclical_score}分）"

    type_configs = {
        "growth": {
            "recommended_metric": "PE",
            "reason": f"{factor_summary}，多因子評分{score_summary}判定為成長型，建議搭配 PEG",
            "suitability_score": 3,
        },
        "stable": {
            "recommended_metric": "PE",
            "reason": f"{factor_summary}，多因子評分{score_summary}判定為穩定型，適合 P/E 區間",
            "suitability_score": 5,
        },
        "cyclical": {
            "recommended_metric": "PB",
            "reason": f"{factor_summary}，多因子評分{score_summary}判定為景氣循環型，建議改用 P/B",
            "suitability_score": 2,
        },
    }
    result.update({"type": best_type, **type_configs[best_type]})

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
            # Prefer the result's type_source (hard_rule > auto) over a blunt "auto".
            entry["type_source"] = result.get("type_source", "auto")
            entry["recommended_metric"] = result.get("recommended_metric", "PE")
            entry["suitability_score"] = result.get("suitability_score", 0)
            entry["reason"] = result.get("reason", "")
            if result.get("small_cap"):
                entry["small_cap"] = True
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
