"""
Calculate TTM P/E, P/B and historical percentile bands.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

def _to_tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone info from a DatetimeIndex regardless of whether it is tz-aware."""
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


from src.data_fetcher import (
    fetch_cashflow,
    fetch_info,
    fetch_price_history,
    fetch_quarterly_financials,
    fetch_shares_outstanding,
)


# ---------------------------------------------------------------------------
# TTM EPS
# ---------------------------------------------------------------------------

def calc_ttm_eps(ticker: str, data_dir: str = "data") -> tuple[float | None, str | None]:
    """
    Return (ttm_eps, last_report_date_str).
    TTM EPS = sum of last 4 quarters' Net Income / diluted shares outstanding.
    Returns (None, None) on failure.
    """
    df = fetch_quarterly_financials(ticker, data_dir)
    if df.empty:
        return None, None

    # Locate Net Income row (yfinance uses various names)
    net_income_row = None
    for name in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
        if name in df.index:
            net_income_row = df.loc[name]
            break

    if net_income_row is None:
        return None, None

    # Sort by date descending, take last 4 quarters
    net_income_row = net_income_row.sort_index(ascending=False)
    if len(net_income_row) < 4:
        return None, None

    last_4 = net_income_row.iloc[:4]
    last_report_date = last_4.index[0]

    ttm_net_income = last_4.sum()

    shares = fetch_shares_outstanding(ticker, data_dir)
    if not shares:
        return None, None

    ttm_eps = ttm_net_income / shares
    report_date_str = last_report_date.strftime("%Y-%m-%d") if hasattr(last_report_date, "strftime") else str(last_report_date)
    return float(ttm_eps), report_date_str


# ---------------------------------------------------------------------------
# P/B
# ---------------------------------------------------------------------------

def get_pb_ratio(ticker: str, price: float, data_dir: str = "data") -> float | None:
    """Return P/B = price / book value per share."""
    info = fetch_info(ticker, data_dir)
    bvps = info.get("bookValue")
    if bvps and bvps > 0:
        return price / bvps
    return None


# ---------------------------------------------------------------------------
# Supplementary valuation metrics: P/CF, PEG, Forward P/E
# ---------------------------------------------------------------------------

def get_pcf_ratio(ticker: str, price: float, data_dir: str = "data") -> float | None:
    """
    Return P/CF = price / (operating cash flow per share).
    Uses TTM operating cash flow (total) ÷ shares outstanding.
    Returns None if cash flow is unavailable or non-positive.
    """
    cf_data = fetch_cashflow(ticker, data_dir)
    ocf = cf_data.get("operating_cashflow")
    if not ocf or ocf <= 0:
        return None
    shares = fetch_shares_outstanding(ticker, data_dir)
    if not shares or shares <= 0:
        return None
    ocf_per_share = ocf / shares
    if ocf_per_share <= 0:
        return None
    return round(price / ocf_per_share, 2)


def get_peg_ratio(ticker: str, data_dir: str = "data") -> float | None:
    """
    Return analyst consensus PEG ratio from yfinance .info["pegRatio"].
    Returns None if unavailable or non-positive.
    """
    cf_data = fetch_cashflow(ticker, data_dir)
    peg = cf_data.get("peg_ratio")
    if peg is None or peg <= 0:
        return None
    return round(float(peg), 2)


def get_forward_pe(ticker: str, data_dir: str = "data") -> float | None:
    """
    Return forward P/E from yfinance .info["forwardPE"].
    Returns None if unavailable or non-positive.
    """
    cf_data = fetch_cashflow(ticker, data_dir)
    fpe = cf_data.get("forward_pe")
    if fpe is None or fpe <= 0:
        return None
    return round(float(fpe), 2)


def get_ev_ebitda(ticker: str, data_dir: str = "data") -> float | None:
    """
    Return EV/EBITDA from yfinance .info["enterpriseToEbitda"].
    Returns None if unavailable or non-positive.
    Used primarily for cyclical stock composite signal.
    """
    cf_data = fetch_cashflow(ticker, data_dir)
    ev_ebitda = cf_data.get("ev_ebitda")
    if ev_ebitda is None or ev_ebitda <= 0:
        return None
    return round(float(ev_ebitda), 2)


# ---------------------------------------------------------------------------
# Historical PE/PB bands
# ---------------------------------------------------------------------------

def build_historical_pe_series(
    ticker: str,
    years: int = 5,
    data_dir: str = "data",
) -> pd.Series:
    """
    Compute daily TTM P/E for the past `years` years using rolling quarterly EPS.
    Returns a pd.Series indexed by date.
    Saves result to data/{ticker}_pe_series.csv.
    """
    cache = Path(data_dir) / f"{ticker}_pe_series.csv"
    if cache.exists():
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            if df.empty:
                cache.unlink(missing_ok=True)
            else:
                s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
                if not s.empty:
                    return s
                cache.unlink(missing_ok=True)
        except Exception:
            cache.unlink(missing_ok=True)

    prices = fetch_price_history(ticker, years=years, data_dir=data_dir)
    if prices.empty:
        return pd.Series(dtype=float)

    df_fin = fetch_quarterly_financials(ticker, data_dir)
    if df_fin.empty:
        return pd.Series(dtype=float)

    # Build quarterly EPS series
    net_income_row = None
    for name in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
        if name in df_fin.index:
            net_income_row = df_fin.loc[name]
            break
    if net_income_row is None:
        return pd.Series(dtype=float)

    shares = fetch_shares_outstanding(ticker, data_dir)
    if not shares:
        return pd.Series(dtype=float)

    # Build per-quarter EPS
    quarterly_eps = (net_income_row / shares).sort_index()
    quarterly_eps = quarterly_eps.dropna()
    if quarterly_eps.empty:
        return pd.Series(dtype=float)

    # fetch_price_history guarantees flat columns; use to_numeric as safety net
    if "Close" not in prices.columns:
        return pd.Series(dtype=float)
    price_series = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    if price_series.empty:
        return pd.Series(dtype=float)

    # Normalise both indices to tz-naive for safe date comparison
    price_series.index = _to_tz_naive(pd.to_datetime(price_series.index))
    quarterly_eps = quarterly_eps.copy()
    quarterly_eps.index = _to_tz_naive(pd.to_datetime(quarterly_eps.index))

    # For each trading day, compute TTM EPS = sum of 4 most recent quarters
    pe_series = {}
    for dt in price_series.index:
        past_quarters = quarterly_eps[quarterly_eps.index <= dt].sort_index(ascending=False)
        if len(past_quarters) < 4:
            continue
        ttm_eps = float(past_quarters.iloc[:4].sum())
        if ttm_eps <= 0:
            continue
        try:
            price_val = price_series.loc[dt]
            if isinstance(price_val, pd.Series):
                price_val = price_val.iloc[0]
            pe_series[dt] = float(price_val) / ttm_eps
        except Exception:
            continue

    result = pd.Series(pe_series, name="PE")
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()

    if not result.empty:
        result.to_csv(cache, header=True)

    return result


def build_historical_pb_series(
    ticker: str,
    years: int = 5,
    data_dir: str = "data",
) -> pd.Series:
    """
    Compute daily P/B for the past `years` years.
    Returns a pd.Series indexed by date.
    """
    cache = Path(data_dir) / f"{ticker}_pb_series.csv"
    if cache.exists():
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            if df.empty:
                cache.unlink(missing_ok=True)
            else:
                s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
                if not s.empty:
                    return s
                cache.unlink(missing_ok=True)
        except Exception:
            cache.unlink(missing_ok=True)

    info = fetch_info(ticker, data_dir)
    bvps = info.get("bookValue")
    if not bvps or bvps <= 0:
        return pd.Series(dtype=float)

    prices = fetch_price_history(ticker, years=years, data_dir=data_dir)
    if prices.empty:
        return pd.Series(dtype=float)

    if "Close" not in prices.columns:
        return pd.Series(dtype=float)
    price_series = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    if price_series.empty:
        return pd.Series(dtype=float)
    price_series.index = _to_tz_naive(pd.to_datetime(price_series.index))
    pb_series = (price_series / float(bvps)).rename("PB")
    pb_series.to_csv(cache, header=True)
    return pb_series


def get_percentiles(series: pd.Series) -> dict:
    """Return key percentile values from a P/E or P/B series."""
    if series.empty:
        return {}
    clean = series.dropna()
    return {
        10: float(np.percentile(clean, 10)),
        25: float(np.percentile(clean, 25)),
        35: float(np.percentile(clean, 35)),
        50: float(np.percentile(clean, 50)),
        65: float(np.percentile(clean, 65)),
        75: float(np.percentile(clean, 75)),
        90: float(np.percentile(clean, 90)),
    }


def current_percentile_rank(value: float, series: pd.Series) -> float:
    """Return the percentile rank (0-100) of `value` within `series`."""
    if series.empty:
        return 50.0
    clean = series.dropna()
    rank = (clean < value).sum() / len(clean) * 100
    return float(rank)


def calc_cape_style_pe(
    ticker: str,
    price: float,
    data_dir: str = "data",
    years: int = 5,
) -> float | None:
    """Return CAPE-like P/E = current price / (average TTM EPS over past `years`).

    This is a stock-level analogue of Shiller's CAPE:
        CAPE = current price / N-year average of TTM earnings.

    We require at least 4 × years quarters of net-income history. Short history
    or missing shares data → None.
    """
    df = fetch_quarterly_financials(ticker, data_dir)
    if df.empty:
        return None

    net_income_row = None
    for name in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
        if name in df.index:
            net_income_row = df.loc[name]
            break
    if net_income_row is None:
        return None

    net_income_row = net_income_row.dropna().sort_index(ascending=False)
    needed = 4 * years
    if len(net_income_row) < needed:
        return None

    shares = fetch_shares_outstanding(ticker, data_dir)
    if not shares or shares <= 0:
        return None

    # Average quarterly EPS over `needed` quarters, then annualise by ×4
    avg_quarterly_ni = float(net_income_row.iloc[:needed].mean())
    avg_annual_eps = (avg_quarterly_ni * 4.0) / float(shares)
    if avg_annual_eps <= 0:
        return None
    return round(price / avg_annual_eps, 2)


def classify_signal(percentile_rank: float, entry: int = 25, exit_: int = 75) -> str:
    """Map a percentile rank to a signal label."""
    if percentile_rank < entry:
        return "BUY"
    elif percentile_rank < entry + 10:
        return "WATCH"
    elif percentile_rank < exit_ - 10:
        return "NEUTRAL"
    elif percentile_rank < exit_:
        return "CAUTION"
    else:
        return "SELL"


SIGNAL_EMOJI = {
    "BUY": "🟢",
    "WATCH": "🔵",
    "NEUTRAL": "⚪",
    "CAUTION": "🟡",
    "SELL": "🔴",
}

SIGNAL_LABEL = {
    "BUY": "BUY ZONE",
    "WATCH": "WATCH ZONE",
    "NEUTRAL": "NEUTRAL",
    "CAUTION": "CAUTION",
    "SELL": "SELL ZONE",
}
