"""
Aggregate forward-return observations into summary metrics per signal class.

All functions are pure: they take a DataFrame produced by
``compute_forward_returns.forward_returns_batch`` and return plain dicts /
DataFrames. No file I/O here so the functions are easy to unit-test.
"""

from math import sqrt

import numpy as np
import pandas as pd


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float | None:
    """Annualised Sharpe assuming zero risk-free rate.  Returns None when
    stdev is zero (all-identical returns)."""
    if returns.empty:
        return None
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0:
        return None
    # returns are already period-return-to-horizon; if horizon_days H, and
    # we treat each observation as a period, the annualisation factor is
    # √(252/H). The caller passes ``periods_per_year`` = 252/horizon_days
    # effectively — default 252 gives a raw sqrt(N) scaling which is a
    # reasonable baseline when observations are overlapping.
    return float((mean / std) * sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float | None:
    """Compute max drawdown on the equity curve of cumulative returns."""
    if returns.empty:
        return None
    # Treat each observation as sequential; compound to build an equity curve
    curve = (1 + returns).cumprod()
    peak = curve.cummax()
    dd = (curve - peak) / peak
    return float(dd.min())  # most negative value


def metrics_by_signal(
    df: pd.DataFrame,
    signal_col: str = "composite_signal",
    horizon_days: int = 60,
) -> pd.DataFrame:
    """
    Aggregate per-signal metrics:
        count, win_rate, mean_return, median_return,
        stdev_return, alpha_mean, sharpe, max_drawdown

    Returns an empty DataFrame if the input is empty.
    """
    if df is None or df.empty or "forward_return" not in df.columns:
        return pd.DataFrame()
    if signal_col not in df.columns:
        return pd.DataFrame()

    # Annualisation factor: 252 / horizon means each period is one horizon
    periods_per_year = max(1, int(round(252 / max(horizon_days, 1))))

    rows = []
    for sig, grp in df.groupby(signal_col):
        rets = grp["forward_return"].dropna()
        alphas = grp["alpha"].dropna() if "alpha" in grp.columns else pd.Series(dtype=float)
        if rets.empty:
            continue
        rows.append(
            {
                "signal": str(sig),
                "count": int(len(rets)),
                "win_rate": float((rets > 0).mean()),
                "mean_return": float(rets.mean()),
                "median_return": float(rets.median()),
                "stdev_return": float(rets.std(ddof=0)),
                "alpha_mean": float(alphas.mean()) if not alphas.empty else None,
                "sharpe": _sharpe(rets, periods_per_year=periods_per_year),
                "max_drawdown": _max_drawdown(rets.reset_index(drop=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_return", ascending=False).reset_index(drop=True)


def overall_summary(df: pd.DataFrame) -> dict:
    """High-level numbers across ALL signals — sanity check."""
    if df is None or df.empty or "forward_return" not in df.columns:
        return {"count": 0, "mean_return": None, "win_rate": None, "alpha_mean": None}
    rets = df["forward_return"].dropna()
    alphas = df["alpha"].dropna() if "alpha" in df.columns else pd.Series(dtype=float)
    return {
        "count": int(len(rets)),
        "mean_return": float(rets.mean()) if not rets.empty else None,
        "win_rate": float((rets > 0).mean()) if not rets.empty else None,
        "alpha_mean": float(alphas.mean()) if not alphas.empty else None,
    }
