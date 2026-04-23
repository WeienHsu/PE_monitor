"""
Given a (signal_date, ticker) pair, compute the N-trading-day forward return
using cached yfinance price history.

We deliberately cache to ``data/{ticker}_prices.csv`` (populated by the main
app) rather than re-fetching — backtests should be reproducible and not
flooding the yfinance API.
"""

from pathlib import Path

import pandas as pd

from src.data_fetcher import fetch_price_history


def _load_prices(ticker: str, data_dir: str = "data", years: int = 5) -> pd.Series:
    """Return a Close price Series indexed by tz-naive trading date."""
    df = fetch_price_history(ticker, years=years, data_dir=data_dir)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["Close"].dropna()
    # Strip timezone for cross-date arithmetic
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_convert("UTC").tz_localize(None)
    close.index = pd.to_datetime(close.index).normalize()
    # Deduplicate any accidental repeat indices (yfinance occasionally gives these)
    close = close[~close.index.duplicated(keep="last")]
    return close.sort_index()


def forward_return(
    ticker: str,
    signal_date: str | pd.Timestamp,
    horizon_days: int = 60,
    data_dir: str = "data",
) -> float | None:
    """
    Return the fractional return (e.g. 0.07 for +7%) from the first trading
    day on/after ``signal_date`` to ``horizon_days`` trading days later.

    Returns None when:
      - price cache is empty or the signal date is after the last available bar
      - fewer than ``horizon_days`` bars exist after the signal date
    """
    close = _load_prices(ticker, data_dir=data_dir)
    if close.empty:
        return None

    target = pd.Timestamp(signal_date).normalize()
    # Find first trading day on or after `target`
    future_idx = close.index[close.index >= target]
    if len(future_idx) < horizon_days + 1:
        return None

    start_price = float(close.loc[future_idx[0]])
    end_price = float(close.loc[future_idx[horizon_days]])
    if start_price <= 0:
        return None
    return (end_price - start_price) / start_price


def forward_returns_batch(
    signals_df: pd.DataFrame,
    horizon_days: int = 60,
    data_dir: str = "data",
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Add ``forward_return`` and ``benchmark_return`` columns to a signals
    DataFrame (output of ``load_historical_signals``).

    ``alpha = forward_return - benchmark_return`` is added as well.

    Rows where either return cannot be computed are dropped.
    """
    if signals_df is None or signals_df.empty:
        return signals_df.copy() if signals_df is not None else pd.DataFrame()

    # Pre-load benchmark once so we don't thrash the cache
    bench_prices = _load_prices(benchmark, data_dir=data_dir)

    def _bench_return(signal_date: pd.Timestamp) -> float | None:
        if bench_prices.empty:
            return None
        future = bench_prices.index[bench_prices.index >= signal_date]
        if len(future) < horizon_days + 1:
            return None
        sp = float(bench_prices.loc[future[0]])
        ep = float(bench_prices.loc[future[horizon_days]])
        if sp <= 0:
            return None
        return (ep - sp) / sp

    out = signals_df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    fwd_vals: list[float | None] = []
    bench_vals: list[float | None] = []
    for _, row in out.iterrows():
        fr = forward_return(
            str(row["ticker"]),
            row["date"],
            horizon_days=horizon_days,
            data_dir=data_dir,
        )
        br = _bench_return(row["date"])
        fwd_vals.append(fr)
        bench_vals.append(br)
    out["forward_return"] = fwd_vals
    out["benchmark_return"] = bench_vals
    out["alpha"] = out.apply(
        lambda r: (r["forward_return"] - r["benchmark_return"])
        if pd.notna(r["forward_return"]) and pd.notna(r["benchmark_return"])
        else None,
        axis=1,
    )
    return out.dropna(subset=["forward_return"]).reset_index(drop=True)
