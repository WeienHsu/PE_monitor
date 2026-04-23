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
    fetch_annual_financials,
    fetch_cashflow,
    fetch_fundamental_extras,
    fetch_info,
    fetch_price_history,
    fetch_quarterly_balance_sheet,
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


def get_ps_ratio(ticker: str, data_dir: str = "data") -> float | None:
    """
    Return trailing P/S (Price-to-Sales) from yfinance .info.
    Critical for high-growth or pre-profit companies where P/E is unreliable.
    Returns None if unavailable or non-positive.
    """
    extras = fetch_fundamental_extras(ticker, data_dir)
    ps = extras.get("ps_ratio")
    if ps is None or ps <= 0:
        return None
    return round(float(ps), 2)


# ---------------------------------------------------------------------------
# Historical PE/PB bands
# ---------------------------------------------------------------------------

def _build_historical_shares_series(ticker: str, data_dir: str) -> pd.Series:
    """
    Build a quarterly shares-outstanding series from the quarterly balance sheet.
    Returns an empty Series if shares data is unavailable.
    """
    bs = fetch_quarterly_balance_sheet(ticker, data_dir)
    if bs.empty:
        return pd.Series(dtype=float)
    shares_row = None
    for name in ["Ordinary Shares Number", "Share Issued", "Common Stock"]:
        if name in bs.index:
            shares_row = bs.loc[name]
            break
    if shares_row is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(shares_row, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return pd.Series(dtype=float)
    s.index = _to_tz_naive(pd.to_datetime(s.index))
    return s.sort_index()


def build_historical_pe_series(
    ticker: str,
    years: int = 5,
    data_dir: str = "data",
) -> pd.Series:
    """
    Compute daily TTM P/E for the past `years` years using rolling quarterly EPS.

    Each quarter's EPS is computed with THAT QUARTER'S shares outstanding
    (from the balance sheet), not the current share count. This corrects
    survivorship-style bias where heavy buyback names (AAPL, GOOGL) used to
    divide historical net income by the smaller current share count, which
    inflated historical EPS and deflated historical P/E.

    When a quarter has no balance-sheet shares data, falls back to the current
    share count for that quarter only (preserves legacy behaviour per-quarter).

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

    current_shares = fetch_shares_outstanding(ticker, data_dir)
    historical_shares = _build_historical_shares_series(ticker, data_dir)

    # Normalise net_income index to tz-naive before aligning
    net_income_row = pd.to_numeric(net_income_row, errors="coerce").dropna()
    if net_income_row.empty:
        return pd.Series(dtype=float)
    net_income_row.index = _to_tz_naive(pd.to_datetime(net_income_row.index))
    net_income_row = net_income_row.sort_index()

    # Align per-quarter shares to net-income quarters.
    # For each quarter: use historical shares if available (exact date match
    # or most recent prior report); else fall back to current_shares.
    if not historical_shares.empty:
        aligned = historical_shares.reindex(
            historical_shares.index.union(net_income_row.index).sort_values()
        ).ffill()
        aligned = aligned.reindex(net_income_row.index)
        if current_shares:
            aligned = aligned.fillna(current_shares)
        elif aligned.isna().all():
            return pd.Series(dtype=float)
        else:
            # Back-fill earliest missing quarters with the first known value
            aligned = aligned.bfill()
    else:
        if not current_shares:
            return pd.Series(dtype=float)
        aligned = pd.Series(current_shares, index=net_income_row.index)

    aligned = aligned[aligned > 0]
    if aligned.empty:
        return pd.Series(dtype=float)

    # Per-quarter EPS = NI_q / shares_q
    quarterly_eps = (net_income_row / aligned).dropna().sort_index()
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


def _build_historical_bvps_series(ticker: str, data_dir: str) -> pd.Series:
    """
    Build a quarterly BVPS series (date → BVPS) from the quarterly balance sheet.
    BVPS = Stockholders Equity / Ordinary Shares Number for each quarter.
    Returns an empty Series if balance sheet is unavailable.
    """
    bs = fetch_quarterly_balance_sheet(ticker, data_dir)
    if bs.empty:
        return pd.Series(dtype=float)

    # Equity row: try common yfinance names
    equity_row = None
    for name in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
        if name in bs.index:
            equity_row = bs.loc[name]
            break
    if equity_row is None:
        return pd.Series(dtype=float)

    # Shares row: Ordinary Shares Number → Share Issued fallback
    shares_row = None
    for name in ["Ordinary Shares Number", "Share Issued", "Common Stock"]:
        if name in bs.index:
            shares_row = bs.loc[name]
            break
    if shares_row is None:
        return pd.Series(dtype=float)

    # Align by date; compute BVPS per quarter
    equity_row = pd.to_numeric(equity_row, errors="coerce").dropna()
    shares_row = pd.to_numeric(shares_row, errors="coerce").dropna()
    common_dates = equity_row.index.intersection(shares_row.index)
    if len(common_dates) == 0:
        return pd.Series(dtype=float)

    bvps = (equity_row.loc[common_dates] / shares_row.loc[common_dates]).sort_index()
    bvps = bvps[bvps > 0]  # filter out negative-equity quarters (very rare, meaningless for P/B)
    bvps.index = _to_tz_naive(pd.to_datetime(bvps.index))
    return bvps


def build_historical_pb_series(
    ticker: str,
    years: int = 5,
    data_dir: str = "data",
) -> pd.Series:
    """
    Compute daily P/B for the past `years` years using time-varying quarterly BVPS.

    For each trading day, P/B = price / (most recent reported BVPS up to that day).
    This fixes the legacy behaviour where a single current BVPS was divided into all
    historical prices (which made the historical P/B series equivalent to the price
    series — percentile ranks were meaningless).

    Falls back to current BVPS × price series only if the balance-sheet data is
    completely unavailable (rare; typically some ADRs).

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

    prices = fetch_price_history(ticker, years=years, data_dir=data_dir)
    if prices.empty or "Close" not in prices.columns:
        return pd.Series(dtype=float)
    price_series = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    if price_series.empty:
        return pd.Series(dtype=float)
    price_series.index = _to_tz_naive(pd.to_datetime(price_series.index))

    # Try time-varying BVPS first (correct approach)
    bvps_quarterly = _build_historical_bvps_series(ticker, data_dir)

    if not bvps_quarterly.empty:
        # Forward-fill quarterly BVPS to each trading day
        # (reindex with union of dates, ffill, then select price dates)
        combined_idx = price_series.index.union(bvps_quarterly.index).sort_values()
        bvps_daily = bvps_quarterly.reindex(combined_idx).ffill()
        bvps_daily = bvps_daily.reindex(price_series.index)
        # Drop days before the first reported BVPS (no ffill source)
        valid_mask = bvps_daily.notna() & (bvps_daily > 0)
        if valid_mask.any():
            pb_series = (price_series[valid_mask] / bvps_daily[valid_mask]).rename("PB")
            pb_series.to_csv(cache, header=True)
            return pb_series

    # Fallback: current BVPS (legacy behaviour, preserved for ADRs / missing data)
    info = fetch_info(ticker, data_dir)
    bvps = info.get("bookValue")
    if not bvps or bvps <= 0:
        return pd.Series(dtype=float)
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


# Type-adaptive percentile thresholds (entry, exit) per stock type.
# Rationale:
#   stable   — standard 25/75 bands; EPS is predictable, PE-reversion works
#   growth   — 15/60 (more aggressive entry, earlier exit). Growth stocks
#              spend long stretches at elevated valuations; demanding a 25th-
#              percentile entry means you rarely get to buy. Earlier exit (60)
#              because growth PE can collapse quickly when the story breaks.
#   cyclical — 35/85 (on P/B). Cyclical low-P/B is usually AT THE PEAK of
#              earnings (denominator is inflated); you want to wait until
#              the market has REALLY written the stock off. High P/B near
#              cycle trough is normal (low BVPS); don't sell prematurely.
#   etf      — standard 25/75
#   unknown  — standard 25/75
_TYPE_THRESHOLDS: dict[str, tuple[int, int]] = {
    "stable":   (25, 75),
    "growth":   (15, 60),
    "cyclical": (35, 85),
    "etf":      (25, 75),
    "unknown":  (25, 75),
}


def get_type_thresholds(stock_type: str | None) -> tuple[int, int]:
    """Return (entry_percentile, exit_percentile) for a stock type."""
    if not stock_type:
        return _TYPE_THRESHOLDS["unknown"]
    return _TYPE_THRESHOLDS.get(stock_type.lower(), _TYPE_THRESHOLDS["unknown"])


def classify_signal(
    percentile_rank: float,
    entry: int = 25,
    exit_: int = 75,
    *,
    stock_type: str | None = None,
) -> str:
    """
    Map a percentile rank to a signal label.

    When `stock_type` is provided, it overrides `entry`/`exit_` with type-
    adaptive thresholds (see _TYPE_THRESHOLDS). This prevents e.g. growth
    stocks from almost never triggering BUY with the stable-stock 25/75 bands.

    The WATCH/CAUTION buffer is the min(10, (exit_-entry)/4) bps on each side
    so the zones remain reasonable when the band is narrow.
    """
    if stock_type is not None:
        entry, exit_ = get_type_thresholds(stock_type)

    buffer = max(1, min(10, (exit_ - entry) // 4))
    if percentile_rank < entry:
        return "BUY"
    elif percentile_rank < entry + buffer:
        return "WATCH"
    elif percentile_rank < exit_ - buffer:
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


# ---------------------------------------------------------------------------
# P3-14: Shiller / Normalized P/E
# ---------------------------------------------------------------------------
#
# Shiller PE (aka CAPE, cyclically-adjusted P/E) divides today's price by the
# inflation-adjusted average of the last 10 years of annual EPS. The aim is
# to smooth out business-cycle earnings distortions — a stock might look
# cheap on trailing P/E because EPS is at a cyclical peak, or expensive
# because EPS is at a cyclical trough. Averaging 10 years of real (inflation
# adjusted) EPS gives a more stable denominator.
#
# We report it as a SECONDARY reference number — the primary entry/exit
# signal still uses trailing P/E vs historical percentile band. Shiller PE is
# most useful for cyclical/industrial names and as a sanity check on growth
# names trading at optically low trailing P/E (earnings may be peak).
#
# Simplification: we use a flat 2.5%/yr constant as the inflation assumption
# rather than wiring in live CPI data. For the 10-year windows we deal with,
# the difference between constant 2.5% and actual CPI is small (~5 pp over
# a decade) and well within the noise of annual EPS volatility.

_SHILLER_DEFAULT_YEARS = 10
_SHILLER_INFLATION_RATE = 0.025  # 2.5% annualised


def _extract_annual_net_income(df: pd.DataFrame) -> pd.Series | None:
    """Pull the Net Income row out of an annual income-statement DataFrame."""
    if df is None or df.empty:
        return None
    for name in [
        "Net Income",
        "NetIncome",
        "Net Income Common Stockholders",
        "Net Income Continuous Operations",
    ]:
        if name in df.index:
            return df.loc[name]
    return None


def calc_shiller_pe(
    ticker: str,
    price: float | None = None,
    data_dir: str = "data",
    years: int = _SHILLER_DEFAULT_YEARS,
    inflation_rate: float = _SHILLER_INFLATION_RATE,
) -> dict:
    """
    Compute Shiller / normalized P/E from the last ``years`` of annual EPS.

    Returns
    -------
    dict:
        {
            "shiller_pe":        float | None,     # inflation-adjusted
            "normalized_pe":     float | None,     # no inflation adjust
            "normalized_eps":    float | None,     # mean of adjusted EPS
            "years_used":        int,              # how many annual points
            "available":         bool,
            "reason":            str,              # explanation when unavailable
        }

    The function degrades gracefully: if yfinance only provides 3-5 annual
    statements (as is typical on the free tier), we still compute a
    "normalized PE" using whatever's available and flag the window in
    ``years_used`` — the caller can decide whether to trust it.
    """
    default: dict = {
        "shiller_pe": None,
        "normalized_pe": None,
        "normalized_eps": None,
        "years_used": 0,
        "available": False,
        "reason": "",
    }

    annual_df = fetch_annual_financials(ticker, data_dir)
    net_income = _extract_annual_net_income(annual_df)
    if net_income is None or net_income.empty:
        default["reason"] = "無年度淨利資料"
        return default

    # Share count — use current value; historical per-year would be better but
    # yfinance rarely provides historical share counts for free.
    shares = fetch_shares_outstanding(ticker, data_dir)
    if not shares or shares <= 0:
        default["reason"] = "無法取得股數"
        return default

    if price is None:
        info = fetch_info(ticker, data_dir)
        price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price or price <= 0:
        default["reason"] = "無法取得股價"
        return default

    # Convert annual Net Income to EPS per year, sort newest → oldest, cap at `years`
    try:
        ni_sorted = net_income.sort_index(ascending=False).iloc[:years]
    except Exception:
        default["reason"] = "年度資料索引錯誤"
        return default

    # Adjust each year's EPS to today's dollars using compound inflation
    # (year 0 = most recent; no adjustment for year 0). NaN and non-numeric
    # cells are skipped — yfinance's free-tier annual statements can have
    # gaps (e.g. restatement years) and a single NaN would poison np.mean.
    today = pd.Timestamp.today()
    adjusted_eps: list[float] = []
    unadjusted_eps: list[float] = []
    for ts, ni in ni_sorted.items():
        try:
            ni_val = float(ni)
        except (TypeError, ValueError):
            continue
        # Skip NaN / inf — they represent missing data in the yfinance frame
        if not np.isfinite(ni_val):
            continue
        eps_val = ni_val / shares
        unadjusted_eps.append(eps_val)
        try:
            year_ts = pd.Timestamp(ts)
            age_years = max(0.0, (today - year_ts).days / 365.25)
        except Exception:
            age_years = 0.0
        inflation_factor = (1.0 + inflation_rate) ** age_years
        adjusted_eps.append(eps_val * inflation_factor)

    if not adjusted_eps:
        default["reason"] = "無有效 EPS 資料"
        return default

    normalized_eps = float(np.mean(adjusted_eps))
    normalized_eps_unadjusted = float(np.mean(unadjusted_eps))

    # Guard against div-by-zero or negative normalized EPS (makes no sense
    # to quote a negative PE — return unavailable with reason).
    if normalized_eps <= 0:
        default["reason"] = "10 年平均 EPS 為負（週期性虧損）"
        default["years_used"] = len(adjusted_eps)
        default["normalized_eps"] = round(normalized_eps, 4)
        return default

    return {
        "shiller_pe": round(price / normalized_eps, 2),
        "normalized_pe": round(price / normalized_eps_unadjusted, 2) if normalized_eps_unadjusted > 0 else None,
        "normalized_eps": round(normalized_eps, 4),
        "years_used": len(adjusted_eps),
        "available": True,
        "reason": "",
    }
