"""
Simple walk-forward backtest for the QVM signal.

Usage
-----
    python -m src.backtest --ticker AAPL --start 2020-01-01
    python -m src.backtest --ticker SPY  --start 2021-01-01 --out data/bt_spy.csv

What it does
------------
Loads ≥ 5 years of daily prices, then for every trading day t (from --start
onward) recomputes:

  * M-factor  — 12-1 momentum percentile vs a 5-year rolling window on THIS
                ticker (fully point-in-time from the price series).
  * V-factor  — current snapshot (from src.report_generator.scan_ticker),
                held constant across the backtest window.
  * Q-factor  — current snapshot (ditto).
  * SMA200   — rolling 200-day mean of close up to t (point-in-time).

Then applies the full QVM pipeline (compute_qvm) and labels the day with a
5-level signal: BUY / WATCH / NEUTRAL / CAUTION / SELL.

For each date we also record the forward 1M / 3M / 6M log return (close(t+h) /
close(t) - 1). Aggregated by signal:

  * Count of days at that signal
  * Average forward return at each horizon
  * Win rate  (pct of days with positive forward 1M return)
  * Max drawdown of a buy-and-hold strategy that stays invested only while
    the signal is ≥ a given level (BUY-only, WATCH-or-better, etc.)

Known limitations
-----------------
* **V and Q are not point-in-time**: yfinance does not expose historical
  fundamentals deep enough for a true PIT recomputation, so V/Q are held at
  their CURRENT values. Treat the backtest's BUY/SELL counts as reliable but
  the absolute return numbers as indicative rather than tradeable P&L.
* Results reflect the past 5 years only (yfinance history limit). Deeper
  regime changes (2008-09, dotcom) are not covered.
* No transaction costs, slippage, or position sizing. Returns are close-to-close.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_fetcher import fetch_price_history, is_etf
from src.factors.momentum_factor import _compute_12_1_series
from src.factors.qvm_composite import compute_qvm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIGNAL_ORDER = ["BUY", "WATCH", "NEUTRAL", "CAUTION", "SELL"]
SIGNAL_RANK = {s: i for i, s in enumerate(SIGNAL_ORDER)}   # 0 = best


def _tz_naive_close(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    close.index = idx
    return close


def _forward_return(close: pd.Series, days: int) -> pd.Series:
    """Return (close(t+days) / close(t)) - 1 as a series indexed by t."""
    return close.shift(-days) / close - 1.0


def _max_drawdown(equity: pd.Series) -> float:
    """Return the most negative peak-to-trough drawdown in `equity`."""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# Snapshot fetch (V & Q held constant)
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    v_score: Optional[float]
    q_score: Optional[float]
    stock_type: str
    etf_subtype: Optional[str]
    is_etf_flag: bool
    operating_cashflow: Optional[float]
    ttm_eps: Optional[float]


def fetch_snapshot(ticker: str, data_dir: str) -> Snapshot:
    """Call the live pipeline once to get V, Q, stock_type, and gate inputs.

    Swallows failures and returns Nones so the backtest can still run on
    M-only when fundamentals are unavailable.
    """
    row: dict = {}
    try:
        from src.report_generator import scan_ticker
        from src.utils import load_config
        try:
            config = load_config()
            config["settings"]["data_dir"] = data_dir
        except Exception:
            config = {
                "settings": {"data_dir": data_dir, "pe_history_years": 5},
                "watchlist": [],
            }
        row = scan_ticker(ticker, config)
    except Exception as e:
        print(f"[backtest] scan_ticker({ticker}) failed: {e}", file=sys.stderr)

    is_etf_flag = bool(row.get("is_etf")) or (row.get("stock_type") == "etf") \
        or is_etf(ticker, data_dir=data_dir)
    return Snapshot(
        v_score=row.get("v_score"),
        q_score=row.get("q_score"),
        stock_type=row.get("stock_type") or ("etf" if is_etf_flag else "unknown"),
        etf_subtype=row.get("etf_subtype"),
        is_etf_flag=is_etf_flag,
        operating_cashflow=row.get("operating_cashflow"),
        ttm_eps=row.get("ttm_eps"),
    )


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def replay_signals(
    ticker: str,
    start: pd.Timestamp,
    data_dir: str = "data",
    window_years: int = 5,
    snap: Optional[Snapshot] = None,
) -> pd.DataFrame:
    """Walk through each trading day from `start` to latest; return per-day rows.

    Columns: date, close, m_score, qvm_raw, base_signal, sma200,
             fwd_1m, fwd_3m, fwd_6m
    """
    prices = fetch_price_history(ticker, years=max(window_years + 1, 6), data_dir=data_dir)
    if prices.empty or "Close" not in prices.columns:
        raise RuntimeError(f"no price history for {ticker}")
    close = _tz_naive_close(prices)

    if snap is None:
        snap = fetch_snapshot(ticker, data_dir)

    # Precompute full 12-1 momentum series once; then percentile-rank on an
    # expanding window ending at each date.
    mom_full = _compute_12_1_series(close)

    # SMA200 — point-in-time rolling mean
    sma200 = close.rolling(window=200, min_periods=200).mean()

    # Forward returns
    fwd_1m = _forward_return(close, 21)
    fwd_3m = _forward_return(close, 63)
    fwd_6m = _forward_return(close, 126)

    records: list[dict] = []
    start_ts = pd.Timestamp(start)
    # We need ≥ 253 trading days of history (for 12-1 momentum) before the
    # first valid backtest date. Align to the first date both in close and
    # mom_full that is ≥ start_ts.
    eligible_dates = mom_full.index[mom_full.index >= start_ts]

    for dt in eligible_dates:
        mom_val = float(mom_full.loc[dt])
        # Percentile rank vs all momentum observations up to and including dt
        window_start = dt - pd.DateOffset(years=window_years)
        window_mom = mom_full.loc[window_start:dt]
        if len(window_mom) < 21:
            continue
        rank = float((window_mom < mom_val).sum()) / len(window_mom) * 100.0
        m_score = round(rank, 2)

        price_t = float(close.loc[dt])
        sma_t = float(sma200.loc[dt]) if pd.notna(sma200.loc[dt]) else None

        result = compute_qvm(
            v_score=snap.v_score,
            q_score=snap.q_score,
            m_score=m_score,
            stock_type=snap.stock_type,
            etf_subtype=snap.etf_subtype,
            is_etf=snap.is_etf_flag,
            operating_cashflow=snap.operating_cashflow,
            ttm_eps=snap.ttm_eps,
            price=price_t,
            sma200=sma_t,
        )

        records.append({
            "date": dt,
            "close": price_t,
            "m_score": m_score,
            "qvm_raw": result["qvm_raw"],
            "base_signal": result["base_signal"],
            "sma200": sma_t,
            "fwd_1m": float(fwd_1m.loc[dt]) if dt in fwd_1m.index and pd.notna(fwd_1m.loc[dt]) else None,
            "fwd_3m": float(fwd_3m.loc[dt]) if dt in fwd_3m.index and pd.notna(fwd_3m.loc[dt]) else None,
            "fwd_6m": float(fwd_6m.loc[dt]) if dt in fwd_6m.index and pd.notna(fwd_6m.loc[dt]) else None,
        })

    if not records:
        return pd.DataFrame(columns=[
            "close", "m_score", "qvm_raw", "base_signal", "sma200",
            "fwd_1m", "fwd_3m", "fwd_6m",
        ])
    df = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(results: pd.DataFrame, snap: Snapshot) -> dict:
    """Build aggregated stats from a per-day result frame."""
    if results.empty:
        return {"error": "empty backtest"}

    by_signal_rows = []
    for sig in SIGNAL_ORDER:
        sub = results[results["base_signal"] == sig]
        if sub.empty:
            continue
        wins = sub["fwd_1m"].dropna()
        win_rate = float((wins > 0).mean()) if not wins.empty else float("nan")
        by_signal_rows.append({
            "signal": sig,
            "days": len(sub),
            "days_pct": round(len(sub) / len(results) * 100, 1),
            "avg_fwd_1m": round(sub["fwd_1m"].mean() * 100, 2) if sub["fwd_1m"].notna().any() else None,
            "avg_fwd_3m": round(sub["fwd_3m"].mean() * 100, 2) if sub["fwd_3m"].notna().any() else None,
            "avg_fwd_6m": round(sub["fwd_6m"].mean() * 100, 2) if sub["fwd_6m"].notna().any() else None,
            "win_rate_1m_pct": round(win_rate * 100, 1) if not np.isnan(win_rate) else None,
        })
    by_signal = pd.DataFrame(by_signal_rows)

    # Strategy equity curves: hold only when signal ≥ threshold, else cash.
    close = results["close"]
    daily_ret = close.pct_change().fillna(0.0)
    strat_rows = []
    for thresh in SIGNAL_ORDER:  # BUY-only, BUY-or-WATCH, ...
        thresh_rank = SIGNAL_RANK[thresh]
        in_market = results["base_signal"].map(lambda s: SIGNAL_RANK.get(s, 99) <= thresh_rank)
        strat_ret = daily_ret.where(in_market.shift(1).fillna(False), 0.0)
        equity = (1.0 + strat_ret).cumprod()
        if equity.empty or equity.iloc[-1] <= 0:
            total_return = float("nan")
        else:
            total_return = float(equity.iloc[-1] - 1.0)
        strat_rows.append({
            "hold_while": f"signal ≥ {thresh}",
            "time_in_market_pct": round(in_market.mean() * 100, 1),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(_max_drawdown(equity) * 100, 2),
        })
    strategies = pd.DataFrame(strat_rows)

    # Buy-and-hold baseline for comparison
    bh_equity = (1.0 + daily_ret).cumprod()
    bh_return = float(bh_equity.iloc[-1] - 1.0) if not bh_equity.empty else float("nan")
    bh_dd = _max_drawdown(bh_equity)

    return {
        "span": f"{results.index.min().date()} → {results.index.max().date()}",
        "days": len(results),
        "v_score_snapshot": snap.v_score,
        "q_score_snapshot": snap.q_score,
        "stock_type": snap.stock_type,
        "etf_subtype": snap.etf_subtype,
        "buy_and_hold_total_pct": round(bh_return * 100, 2),
        "buy_and_hold_max_dd_pct": round(bh_dd * 100, 2),
        "by_signal": by_signal,
        "strategies": strategies,
    }


def print_report(ticker: str, summary: dict) -> None:
    print(f"\n=== QVM Backtest: {ticker} ===")
    if "error" in summary:
        print("ERROR:", summary["error"])
        return
    print(f"Window:          {summary['span']}  ({summary['days']} trading days)")
    print(f"Stock type:      {summary['stock_type']}"
          + (f" / {summary['etf_subtype']}" if summary["etf_subtype"] else ""))
    print(f"V snapshot:      {summary['v_score_snapshot']}")
    print(f"Q snapshot:      {summary['q_score_snapshot']}")
    print(f"Buy & hold:      {summary['buy_and_hold_total_pct']}%  (max DD {summary['buy_and_hold_max_dd_pct']}%)")

    print("\n--- Forward returns by signal ---")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary["by_signal"].to_string(index=False))

    print("\n--- Strategy: hold while signal is at-or-better than threshold ---")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary["strategies"].to_string(index=False))

    print("\nNote: V and Q are held at current snapshot values (no historical PIT "
          "fundamentals). Signal frequencies are reliable; absolute return numbers "
          "are indicative, not tradeable P&L.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward QVM backtest")
    ap.add_argument("--ticker", required=True, help="Yahoo ticker (e.g. AAPL, 0050.TW)")
    ap.add_argument("--start", default="2020-01-01", help="Backtest start date (YYYY-MM-DD)")
    ap.add_argument("--data-dir", default="data", help="Cache directory (default: data)")
    ap.add_argument("--window-years", type=int, default=5,
                    help="Rolling percentile window for momentum (default 5Y)")
    ap.add_argument("--out", default=None, help="Optional path to save per-day CSV")
    args = ap.parse_args(argv)

    snap = fetch_snapshot(args.ticker, args.data_dir)
    results = replay_signals(
        args.ticker,
        start=pd.Timestamp(args.start),
        data_dir=args.data_dir,
        window_years=args.window_years,
        snap=snap,
    )
    summary = summarise(results, snap)
    print_report(args.ticker, summary)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out)
        print(f"\nPer-day results written to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
