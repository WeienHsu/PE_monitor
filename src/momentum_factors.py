"""
Momentum / technical-confirmation factors (P2-11).

These factors don't generate their own BUY/SELL signals — they feed the
multi-factor composite as confirmatory / disqualifying votes:

  * Volume factor  — 5-day avg volume / 20-day avg volume
    > 1.5   → +1  (new interest, signal more reliable)
    0.5-1.5 → 0
    < 0.5   → -1  (quiet tape, signal may be noise)

  * 52-week position — current price / 52-week high
    ≥ 0.95  → +1  (near highs — momentum confirmed; for BUY signals, avoid
                   thinking this is a discount)
    0.7-0.95 → 0
    < 0.7   → -1  (structural weakness / downtrend — don't catch the knife)

Votes are meant to be small ±1 adjustments consumed by
``compute_multi_factor_composite`` alongside PEG, forward P/E, etc.
"""

from __future__ import annotations

import pandas as pd

from src.data_fetcher import fetch_price_history


# ---------------------------------------------------------------------------
# Vote thresholds — module constants for easy tuning / testing
# ---------------------------------------------------------------------------

_VOL_HIGH = 1.5    # ratio at/above which volume vote becomes +1
_VOL_LOW = 0.5     # ratio at/below which volume vote becomes -1

_POS_HIGH = 0.95   # fraction of 52w high at/above which position vote is +1
_POS_LOW = 0.70    # fraction of 52w high below which position vote is -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recent_close_and_volume(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract numeric Close and Volume series, dropping any NaNs."""
    close = pd.to_numeric(df.get("Close"), errors="coerce").dropna() if "Close" in df.columns else pd.Series(dtype=float)
    volume = pd.to_numeric(df.get("Volume"), errors="coerce").dropna() if "Volume" in df.columns else pd.Series(dtype=float)
    return close, volume


# ---------------------------------------------------------------------------
# Volume factor
# ---------------------------------------------------------------------------

def get_volume_factor(ticker: str, data_dir: str = "data") -> dict:
    """
    Return {'ratio': float|None, 'vote': -1|0|+1, 'available': bool}.

    ratio = avg volume over the last 5 trading days / avg volume over the
    previous 20 trading days (days 6..25 back).  Requires ≥ 25 days of
    volume history — otherwise returns ``available=False, vote=0``.
    """
    result = {"ratio": None, "vote": 0, "available": False}

    df = fetch_price_history(ticker, years=1, data_dir=data_dir)
    if df.empty:
        return result

    _, volume = _recent_close_and_volume(df)
    if len(volume) < 25:
        return result

    recent5 = volume.iloc[-5:].mean()
    prior20 = volume.iloc[-25:-5].mean()
    if prior20 == 0 or pd.isna(prior20):
        return result

    ratio = float(recent5 / prior20)
    if ratio > _VOL_HIGH:
        vote = 1
    elif ratio < _VOL_LOW:
        vote = -1
    else:
        vote = 0

    return {"ratio": round(ratio, 3), "vote": vote, "available": True}


# ---------------------------------------------------------------------------
# 52-week position factor
# ---------------------------------------------------------------------------

def get_52w_position(ticker: str, data_dir: str = "data") -> dict:
    """
    Return {'position': float|None, 'high': float|None, 'low': float|None,
            'vote': -1|0|+1, 'available': bool}.

    position = latest close / max(close) over the most recent 252 trading
    days (~1 calendar year).  Requires ≥ 50 trading days of data.
    """
    result = {"position": None, "high": None, "low": None, "vote": 0, "available": False}

    df = fetch_price_history(ticker, years=1, data_dir=data_dir)
    if df.empty:
        return result

    close, _ = _recent_close_and_volume(df)
    if len(close) < 50:
        return result

    window = close.iloc[-252:] if len(close) > 252 else close
    high = float(window.max())
    low = float(window.min())
    latest = float(close.iloc[-1])
    if high <= 0:
        return result

    position = latest / high  # 0..1 where 1 = at 52w high
    if position >= _POS_HIGH:
        vote = 1
    elif position < _POS_LOW:
        vote = -1
    else:
        vote = 0

    return {
        "position": round(position, 3),
        "high": round(high, 2),
        "low": round(low, 2),
        "vote": vote,
        "available": True,
    }


# ---------------------------------------------------------------------------
# Convenience: both factors in one call
# ---------------------------------------------------------------------------

def get_momentum_factors(ticker: str, data_dir: str = "data") -> dict:
    """Return both factors under a single dict for convenience."""
    return {
        "volume": get_volume_factor(ticker, data_dir),
        "position_52w": get_52w_position(ticker, data_dir),
    }
