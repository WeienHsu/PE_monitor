"""
ETF-specific V and Q inputs for the QVM composite.

Because ETFs don't have consolidated EPS / book value / margins, their
V-factor is sourced from subtype-appropriate substitutes, and their
Q-factor is a light structural proxy (AUM + expense ratio).

Subtypes (as classified in stock_analyzer.classify_etf_subtype):
    broad       — total-market / blend (SPY, VOO, VTI, 0050)
    sector      — sector or industry (XLK, SOXX)
    dividend    — dividend-focused equity ETF (0056, SCHD)
    commodity   — physical commodities (GLD, SLV)
    bond        — bond / treasury ETF (TLT, IEF, AGG)

Phase 2 implements the three subtypes whose inputs are already obtainable
from yfinance (dividend, commodity, bond) and uses price-percentile as a
placeholder for broad/sector. Phase 3 replaces those placeholders with
Shiller CAPE (broad) and Damodaran industry PE (sector).
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.data_fetcher import fetch_info, fetch_price_history
from src.external_data import (
    get_industry_trailing_pe,
    get_shiller_cape_percentile,
)
from src.etf_industry_map import get_industry_for_etf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_percentile(ticker: str, price: float, years: int, data_dir: str) -> Optional[float]:
    """Return the current price's percentile rank within the last `years` years."""
    df = fetch_price_history(ticker, years=years, data_dir=data_dir)
    if df.empty or "Close" not in df.columns:
        return None
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    rank = (close < price).sum() / len(close) * 100
    return float(rank)


def _dividend_yield_percentile(ticker: str, years: int, data_dir: str) -> Optional[float]:
    """Return the current trailing 12-month dividend yield's percentile vs own history.

    Higher yield = cheaper, so the caller must invert the returned percentile
    if it wants a "cheapness rank" in the usual direction.
    """
    info = fetch_info(ticker, data_dir)
    df = fetch_price_history(ticker, years=years, data_dir=data_dir)
    if df.empty or "Close" not in df.columns:
        return None

    yield_now = info.get("yield") or info.get("trailingAnnualDividendYield")
    if yield_now is None:
        return None
    yield_now = float(yield_now)
    # Some yfinance fields return percent (e.g. 9.93); others decimal (0.0993).
    # Normalise to decimal.
    if yield_now > 1:
        yield_now = yield_now / 100.0
    if yield_now <= 0:
        return None

    try:
        import yfinance as yf
        divs = yf.Ticker(ticker).dividends
        if divs is None or (hasattr(divs, "empty") and divs.empty):
            return None
        # Squeeze DataFrame → Series if needed
        if isinstance(divs, pd.DataFrame):
            divs = divs.iloc[:, 0]

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_convert("UTC").tz_localize(None)
        # Strip time-of-day so dividend dates align with trading days
        close.index = close.index.normalize()

        divs.index = pd.to_datetime(divs.index)
        if divs.index.tz is not None:
            divs.index = divs.index.tz_convert("UTC").tz_localize(None)
        divs.index = divs.index.normalize()
        # Combine multiple dividends on the same day (rare but possible)
        divs = divs.groupby(divs.index).sum()

        # Reindex to the full trading-day index, zero-fill non-ex-div days
        divs_daily = divs.reindex(close.index, fill_value=0)
        ttm_div = divs_daily.rolling(252, min_periods=252).sum()
        yld_series = (ttm_div / close).dropna()
        if yld_series.empty:
            return None
        rank = (yld_series < yield_now).sum() / len(yld_series) * 100
        return float(rank)
    except Exception:
        return None


def _treasury_10y_percentile(years: int, data_dir: str) -> Optional[tuple[float, float]]:
    """Return (current ^TNX yield, percentile rank 0-100 over `years` years).

    Higher yields = cheaper bonds. Returns None if the fetch fails.
    """
    try:
        df = fetch_price_history("^TNX", years=years, data_dir=data_dir)
        if df.empty or "Close" not in df.columns:
            return None
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if close.empty:
            return None
        current = float(close.iloc[-1])
        rank = (close < current).sum() / len(close) * 100
        return (current, float(rank))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API — V score
# ---------------------------------------------------------------------------

def compute_etf_v_score(
    ticker: str,
    subtype: Optional[str],
    price: float,
    data_dir: str = "data",
    years: int = 5,
) -> tuple[Optional[float], dict]:
    """Compute the V-factor score (0-100, higher = cheaper) for an ETF.

    Returns (score, details). Details is a dict with the same shape as
    individual-stock V details: {"components": {name: {raw, score, kind}}, "score": value}
    so the UI can treat both uniformly.
    """
    components: dict[str, dict] = {}

    def add(name: str, raw, score: Optional[float], kind: str) -> None:
        if score is None:
            return
        components[name] = {"raw": raw, "score": float(score), "kind": kind}

    subtype = subtype or "broad"

    if subtype == "broad":
        # Primary input: Shiller CAPE percentile (155-year history).
        # High CAPE = expensive → invert the rank so high rank → low V.
        cape_value, cape_rank = get_shiller_cape_percentile(data_dir=data_dir)
        if cape_rank is not None:
            add(
                "Shiller CAPE percentile",
                f"{cape_value:.1f} (rank {cape_rank:.0f})",
                100.0 - cape_rank,
                "percentile",
            )
        else:
            # Fall back to own-price percentile if the scrape fails
            pp = _price_percentile(ticker, price, years, data_dir)
            if pp is not None:
                add("Price percentile (5Y)", round(pp, 1), 100.0 - pp, "percentile")
        # Secondary: ETF's own trailing P/E vs absolute bands
        info = fetch_info(ticker, data_dir)
        pe = info.get("trailingPE")
        if pe and pe > 0:
            cheap, exp = 12.0, 35.0
            if pe <= cheap:
                s = 100.0
            elif pe >= exp:
                s = 0.0
            else:
                s = 100.0 * (1.0 - (pe - cheap) / (exp - cheap))
            add("ETF trailing P/E", round(float(pe), 2), s, "absolute")

    elif subtype == "sector":
        # Primary input: ETF trailing P/E vs Damodaran industry trailing P/E.
        # Ratio < 1 → ETF cheaper than industry average → high V.
        info = fetch_info(ticker, data_dir)
        etf_pe = info.get("trailingPE")
        industry = get_industry_for_etf(ticker)
        industry_pe = get_industry_trailing_pe(industry, data_dir) if industry else None
        if etf_pe and etf_pe > 0 and industry_pe and industry_pe > 0:
            ratio = float(etf_pe) / float(industry_pe)
            # ratio ≤ 0.7 → 100; ratio ≥ 1.3 → 0; linear
            if ratio <= 0.7:
                s = 100.0
            elif ratio >= 1.3:
                s = 0.0
            else:
                s = 100.0 * (1.0 - (ratio - 0.7) / (1.3 - 0.7))
            add(
                f"ETF PE / {industry} PE",
                f"{etf_pe:.1f} / {industry_pe:.1f} = {ratio:.2f}",
                s,
                "absolute",
            )
        else:
            # No industry mapping or no ETF PE — fall back to price percentile
            pp = _price_percentile(ticker, price, years, data_dir)
            if pp is not None:
                add("Price percentile (5Y)", round(pp, 1), 100.0 - pp, "percentile")
        # Secondary: ETF trailing PE vs absolute band (same as broad)
        if etf_pe and etf_pe > 0:
            cheap, exp = 12.0, 35.0
            if etf_pe <= cheap:
                s = 100.0
            elif etf_pe >= exp:
                s = 0.0
            else:
                s = 100.0 * (1.0 - (etf_pe - cheap) / (exp - cheap))
            add("ETF trailing P/E", round(float(etf_pe), 2), s, "absolute")

    elif subtype == "dividend":
        # Higher yield = cheaper. Invert the percentile so high yield → high V.
        yp = _dividend_yield_percentile(ticker, years, data_dir)
        if yp is not None:
            # yp = rank of "current yield" among historical yields.
            # If current yield is high relative to history, rank is high,
            # which means cheap → V should be high. No inversion needed.
            add("Dividend yield percentile (5Y)", round(yp, 1), yp, "percentile")

    elif subtype == "commodity":
        # Low price = cheap. Invert price percentile: low percentile → high V.
        pp = _price_percentile(ticker, price, years, data_dir)
        if pp is not None:
            add("Price percentile (5Y)", round(pp, 1), 100.0 - pp, "percentile")

    elif subtype == "bond":
        # Higher 10Y treasury yield → bonds cheaper → V should be high.
        tny = _treasury_10y_percentile(years, data_dir)
        if tny is not None:
            current, rank = tny
            add(
                "10Y Treasury yield percentile",
                f"{current:.2f}% (rank {rank:.0f})",
                rank,
                "percentile",
            )
        # Also include own price percentile as a secondary input (low price = cheap)
        pp = _price_percentile(ticker, price, years, data_dir)
        if pp is not None:
            add("Price percentile (5Y)", round(pp, 1), 100.0 - pp, "percentile")

    else:  # unknown subtype — fall through to price percentile
        pp = _price_percentile(ticker, price, years, data_dir)
        if pp is not None:
            add("Price percentile (5Y)", round(pp, 1), 100.0 - pp, "percentile")

    if not components:
        return None, {"components": {}, "score": None}

    scores = [c["score"] for c in components.values()]
    v_score = sum(scores) / len(scores)
    return round(v_score, 2), {
        "components": components,
        "score": round(v_score, 2),
    }


# ---------------------------------------------------------------------------
# Public API — Q score (structural proxy)
# ---------------------------------------------------------------------------

def compute_etf_q_score(
    ticker: str,
    data_dir: str = "data",
) -> tuple[Optional[float], dict]:
    """Return (q_score, details) for an ETF using AUM + expense ratio proxies.

    - totalAssets ≥ $500M   → 100 pts; else scaled linearly to $50M → 0
    - expenseRatio ≤ 0.10%  → 100 pts; ≥ 0.75% → 0; linear in between
    """
    info = fetch_info(ticker, data_dir)
    components: dict[str, dict] = {}

    # AUM — stability proxy
    aum = info.get("totalAssets")
    if aum and aum > 0:
        aum_m = aum / 1e6
        if aum_m >= 500:
            s = 100.0
        elif aum_m <= 50:
            s = 0.0
        else:
            s = 100.0 * (aum_m - 50) / (500 - 50)
        components["AUM (stability)"] = {
            "raw": f"${aum_m:.0f}M",
            "score": round(s, 1),
            "kind": "absolute",
        }

    # Expense ratio — cost proxy
    er = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
    if er and er > 0:
        # yfinance returns a decimal (e.g., 0.0003 for 0.03%) or a percent;
        # normalise: if > 1 assume it's already in %.
        er_pct = er if er > 1 else er * 100
        if er_pct <= 0.10:
            s = 100.0
        elif er_pct >= 0.75:
            s = 0.0
        else:
            s = 100.0 * (1.0 - (er_pct - 0.10) / (0.75 - 0.10))
        components["Expense ratio"] = {
            "raw": f"{er_pct:.2f}%",
            "score": round(s, 1),
            "kind": "absolute",
        }

    if not components:
        return None, {"components": {}, "score": None}

    scores = [c["score"] for c in components.values()]
    q_score = sum(scores) / len(scores)
    return round(q_score, 2), {
        "components": components,
        "score": round(q_score, 2),
    }
