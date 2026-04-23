"""
CLI entry point for the P3-15 backtest framework.

Usage
-----
    python -m tests.backtest.run_backtest [--years N] [--horizon D]
                                          [--signal SIG] [--benchmark TKR]
                                          [--report-dir reports]
                                          [--data-dir data]

Output
------
Prints two tables:
  1. Overall summary (count, mean return, win rate, alpha vs SPY)
  2. Per-signal breakdown (one row per composite_signal value)

Non-zero exit if no signals can be loaded (useful for CI smoke tests).

Caveat: initial repos have only a day or two of reports; the framework
prints a warning but still runs.
"""

import argparse
import sys
from datetime import date, timedelta

from .backtest_metrics import metrics_by_signal, overall_summary
from .compute_forward_returns import forward_returns_batch
from .load_historical_signals import load_historical_signals


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PE Monitor composite-signal backtest")
    p.add_argument("--years", type=int, default=3,
                   help="Lookback window in years (default 3).")
    p.add_argument("--horizon", type=int, default=60,
                   help="Forward-return horizon in trading days (default 60).")
    p.add_argument("--signal", type=str, default=None,
                   help="Filter to a single composite_signal value (e.g. BUY).")
    p.add_argument("--benchmark", type=str, default="SPY",
                   help="Benchmark ticker for alpha (default SPY).")
    p.add_argument("--report-dir", type=str, default="reports",
                   help="Directory containing daily_*.csv reports.")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Directory containing cached price history.")
    return p.parse_args(argv)


def _print_summary(s: dict) -> None:
    print("\n=== Overall ===")
    if s.get("count", 0) == 0:
        print("  no observations")
        return
    print(f"  observations : {s['count']}")
    if s.get("mean_return") is not None:
        print(f"  mean return  : {s['mean_return'] * 100:+.2f}%")
    if s.get("win_rate") is not None:
        print(f"  win rate     : {s['win_rate'] * 100:.1f}%")
    if s.get("alpha_mean") is not None:
        print(f"  alpha vs bmk : {s['alpha_mean'] * 100:+.2f}%")


def _print_by_signal(df) -> None:
    print("\n=== By Signal ===")
    if df is None or df.empty:
        print("  no signal-level stats available")
        return
    # Pretty-print without pandas dependency on display width
    cols = ["signal", "count", "win_rate", "mean_return",
            "alpha_mean", "sharpe", "max_drawdown"]
    widths = {c: max(len(c), 8) for c in cols}
    hdr = " | ".join(c.rjust(widths[c]) for c in cols)
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and not _is_finite(v)):
                s = "N/A"
            elif c in ("win_rate", "mean_return", "alpha_mean", "max_drawdown"):
                s = f"{v * 100:+.2f}%"
            elif c == "sharpe":
                s = f"{v:+.2f}"
            elif c == "count":
                s = str(int(v))
            else:
                s = str(v)
            vals.append(s.rjust(widths[c]))
        print("  " + " | ".join(vals))


def _is_finite(x: float) -> bool:
    try:
        import math
        return math.isfinite(x)
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=int(args.years * 365.25))
    signals = load_historical_signals(
        report_dir=args.report_dir,
        start_date=start_dt.isoformat(),
        end_date=end_dt.isoformat(),
    )

    if signals.empty:
        print(f"[warn] no signals loaded from {args.report_dir}/ — "
              f"is the report directory populated?", file=sys.stderr)
        return 1

    print(f"[info] loaded {len(signals)} signal rows spanning "
          f"{signals['date'].min().date()} → {signals['date'].max().date()}")

    if args.signal:
        signals = signals[signals["composite_signal"] == args.signal]
        print(f"[info] filtered to composite_signal={args.signal}: "
              f"{len(signals)} rows")

    if signals.empty:
        print("[warn] no signals match filter — exiting", file=sys.stderr)
        return 1

    enriched = forward_returns_batch(
        signals,
        horizon_days=args.horizon,
        data_dir=args.data_dir,
        benchmark=args.benchmark,
    )
    if enriched.empty:
        print("[warn] no signals have enough forward price history for "
              f"the {args.horizon}-day horizon — extend reports further "
              "back in time or reduce --horizon", file=sys.stderr)
        return 1

    print(f"[info] {len(enriched)} of {len(signals)} signals had enough "
          f"forward data for {args.horizon}-day returns")

    _print_summary(overall_summary(enriched))
    _print_by_signal(metrics_by_signal(enriched, horizon_days=args.horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
