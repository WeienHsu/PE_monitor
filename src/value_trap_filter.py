"""
Value-trap filter — flags stocks that *look* cheap on P/E but are
experiencing fundamental deterioration.

A cheap stock without underlying business quality is a value trap: the P/E
keeps compressing because earnings are falling faster than the price. We want
to demote BUY-class signals in those cases, not ride them down.

Checks (each triggers one flag; severity = count of triggered flags):
    1. Revenue YoY negative for the last 4 consecutive quarters
    2. Free cash flow turned negative (latest TTM FCF < 0)
    3. Gross margin dropped > 300 bps across the latest 2 reported quarters
    4. Operating cash flow negative for the last 2 consecutive quarters

Integration point: `compute_multi_factor_composite()` may demote BUY-side
signals when severity >= 2. The caller owns the override policy.
"""

from __future__ import annotations

import pandas as pd

from src.data_fetcher import (
    fetch_cashflow,
    fetch_quarterly_cashflow,
    fetch_quarterly_financials,
)


def _find_row(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _sorted_numeric(series: pd.Series | None) -> pd.Series:
    """Return the series sorted descending by date (most recent first), numeric, dropna."""
    if series is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return s
    try:
        s.index = pd.to_datetime(s.index)
        s = s.sort_index(ascending=False)
    except Exception:
        pass
    return s


def _check_revenue_decline(fin_df: pd.DataFrame) -> str | None:
    """
    Flag if revenue YoY is negative for 4 consecutive quarters.

    Needs at least 8 quarters (4 current + 4 year-ago comparison).
    """
    rev_row = _find_row(fin_df, ["Total Revenue", "Revenue", "TotalRevenue"])
    if rev_row is None:
        return None
    rev = _sorted_numeric(rev_row)
    if len(rev) < 8:
        return None
    # Check 4 most recent quarters vs their year-ago counterparts
    for i in range(4):
        curr = rev.iloc[i]
        year_ago = rev.iloc[i + 4]
        if year_ago <= 0:
            return None  # data issue — avoid false flag
        yoy = (curr - year_ago) / year_ago
        if yoy >= 0:
            return None
    return "營收連續 4 季 YoY 下滑"


def _check_negative_fcf(ticker: str, data_dir: str) -> str | None:
    """Flag if the most recent reported TTM free cash flow is negative."""
    cf = fetch_cashflow(ticker, data_dir)
    fcf = cf.get("free_cashflow")
    if fcf is None:
        return None
    if fcf < 0:
        return "自由現金流為負"
    return None


def _check_gross_margin_drop(fin_df: pd.DataFrame) -> str | None:
    """
    Flag if gross margin dropped > 300 bps across the latest 2 reported quarters.

    Gross margin = Gross Profit / Total Revenue.
    """
    gp_row = _find_row(fin_df, ["Gross Profit", "GrossProfit"])
    rev_row = _find_row(fin_df, ["Total Revenue", "Revenue", "TotalRevenue"])
    if gp_row is None or rev_row is None:
        return None
    gp = _sorted_numeric(gp_row)
    rev = _sorted_numeric(rev_row)
    if len(gp) < 2 or len(rev) < 2:
        return None
    # Align to common dates
    common = gp.index.intersection(rev.index)
    if len(common) < 2:
        return None
    common_sorted = sorted(common, reverse=True)
    latest, prev = common_sorted[0], common_sorted[1]
    if rev.loc[latest] <= 0 or rev.loc[prev] <= 0:
        return None
    gm_latest = gp.loc[latest] / rev.loc[latest]
    gm_prev = gp.loc[prev] / rev.loc[prev]
    if gm_latest - gm_prev < -0.03:  # 300 bps = 0.03
        drop_bps = int((gm_prev - gm_latest) * 10000)
        return f"毛利率 2 季下滑 {drop_bps} bps"
    return None


def _check_ocf_negative_2q(ticker: str, data_dir: str) -> str | None:
    """Flag if operating cash flow is negative for the last 2 consecutive quarters."""
    cf_df = fetch_quarterly_cashflow(ticker, data_dir)
    if cf_df.empty:
        return None
    ocf_row = _find_row(cf_df, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    if ocf_row is None:
        return None
    ocf = _sorted_numeric(ocf_row)
    if len(ocf) < 2:
        return None
    if ocf.iloc[0] < 0 and ocf.iloc[1] < 0:
        return "營業現金流連 2 季為負"
    return None


def check_value_trap(ticker: str, data_dir: str = "data") -> dict:
    """
    Return value-trap diagnosis for a ticker.

    Returns
    -------
    {
        'is_trap':  bool,           # True iff severity >= 2
        'severity': int,            # 0..4 — count of triggered flags
        'flags':    list[str],      # human-readable reasons
    }

    A severity >= 2 is the threshold used by composite_signal to override
    BUY-class signals. ETFs / funds should not be passed here.
    """
    flags: list[str] = []

    fin_df = fetch_quarterly_financials(ticker, data_dir)

    # Only run financial-statement checks if we have data
    if not fin_df.empty:
        r = _check_revenue_decline(fin_df)
        if r:
            flags.append(r)
        gm = _check_gross_margin_drop(fin_df)
        if gm:
            flags.append(gm)

    fcf = _check_negative_fcf(ticker, data_dir)
    if fcf:
        flags.append(fcf)

    ocf = _check_ocf_negative_2q(ticker, data_dir)
    if ocf:
        flags.append(ocf)

    severity = len(flags)
    return {
        "is_trap": severity >= 2,
        "severity": severity,
        "flags": flags,
    }
