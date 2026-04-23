"""
Load historical signals from ``reports/daily_*.csv`` into a single DataFrame.

The output is a long-form frame:
    date, ticker, signal, composite_signal, metric_label, metric_value,
    percentile_rank, stock_type, name

Designed to be consumed by ``compute_forward_returns`` and
``backtest_metrics``. No mutation of the source CSVs.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


# Columns we actually care about for backtesting. Missing columns are filled
# with None so we never hard-fail on schema drift across report vintages.
_WANTED_COLUMNS = [
    "ticker",
    "name",
    "signal",
    "signal_display",
    "composite_signal",
    "composite_display",
    "metric_label",
    "metric_value",
    "percentile_rank",
    "stock_type",
    "price",
    "news_label",
]


def _parse_report_date(filename: str) -> str | None:
    """Extract YYYY-MM-DD from ``daily_YYYYMMDD.csv``-style filenames."""
    stem = Path(filename).stem  # daily_20260418
    if not stem.startswith("daily_"):
        return None
    yyyymmdd = stem.replace("daily_", "")
    try:
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def load_historical_signals(
    report_dir: str = "reports",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Walk ``report_dir`` and return a long-form DataFrame of every
    (date × ticker) signal observation.

    Parameters
    ----------
    report_dir  : directory containing ``daily_YYYYMMDD.csv`` files.
    start_date  : inclusive lower bound, ISO format (``2025-01-01``); optional.
    end_date    : inclusive upper bound, ISO format; optional.

    Returns an empty DataFrame when no reports are found, so callers always
    get a usable object.
    """
    p = Path(report_dir)
    if not p.exists():
        return pd.DataFrame(columns=["date"] + _WANTED_COLUMNS)

    rows: list[dict] = []
    for f in sorted(p.glob("daily_*.csv")):
        iso = _parse_report_date(f.name)
        if iso is None:
            continue
        if start_date and iso < start_date:
            continue
        if end_date and iso > end_date:
            continue
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:
            continue
        if df.empty:
            continue
        for _, r in df.iterrows():
            row: dict = {"date": iso}
            for col in _WANTED_COLUMNS:
                row[col] = r.get(col) if col in df.columns else None
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date"] + _WANTED_COLUMNS)

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    # Drop rows missing a signal — backtest is meaningless without it
    out = out[out["composite_signal"].notna() | out["signal"].notna()]
    out = out.reset_index(drop=True)
    return out


def list_signal_types(df: pd.DataFrame) -> list[str]:
    """Return sorted unique composite_signal values in the frame."""
    if df.empty or "composite_signal" not in df.columns:
        return []
    vals = df["composite_signal"].dropna().astype(str).unique().tolist()
    return sorted(vals)
