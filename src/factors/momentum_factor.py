"""
Momentum factor (M): price-trend score on 0-100.

12-1 momentum is the Jegadeesh-Titman (1993) classic:
    return over the past 12 months, **excluding** the most recent 1 month.
    Excluding t-1 month reduces short-term reversal noise.

The 12-1 value is compared to the stock's own 5-year distribution of rolling
12-1 values → percentile rank → M_base (0-100, higher = stronger uptrend).

Strategy D adds a +10 bonus (capped at 100) when its KD+MACD convergence
signal is active, reflecting additional technical evidence of a turn.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.data_fetcher import fetch_price_history


def _to_tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def _compute_12_1_series(close: pd.Series) -> pd.Series:
    """12-1 momentum series:
        mom(t) = close(t-21) / close(t-252) - 1

    (21 trading days ≈ 1 month; 252 ≈ 12 months.)
    Requires at least 253 trading days of history.
    """
    if len(close) < 253:
        return pd.Series(dtype=float)
    # Shift and divide vectorised
    mom = close.shift(21) / close.shift(252) - 1.0
    return mom.dropna()


def compute_m_score(
    ticker: str,
    data_dir: str = "data",
    strategy_d_signal: Optional[bool] = None,
    years: int = 5,
) -> tuple[Optional[float], dict]:
    """Compute M score (0-100, higher = stronger momentum).

    Returns (m_score, details). m_score is None if history is insufficient.
    """
    prices = fetch_price_history(ticker, years=years, data_dir=data_dir)
    if prices.empty or "Close" not in prices.columns:
        return None, {"components": {}, "score": None, "error": "no price history"}

    close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    if len(close) < 253:
        return None, {"components": {}, "score": None, "error": "< 253 trading days"}

    close.index = _to_tz_naive(pd.to_datetime(close.index))

    mom_series = _compute_12_1_series(close)
    if mom_series.empty:
        return None, {"components": {}, "score": None, "error": "12-1 series empty"}

    current_mom = float(mom_series.iloc[-1])

    # Percentile rank of current momentum vs own 5Y distribution
    rank = float((mom_series < current_mom).sum()) / len(mom_series) * 100.0
    m_base = round(rank, 2)

    components: dict[str, dict] = {
        "12-1 Momentum": {
            "raw": round(current_mom, 4),
            "percentile": round(rank, 2),
            "score": m_base,
        },
    }

    # Strategy D bonus
    if strategy_d_signal is True:
        bonus = 10.0
        components["Strategy D Bonus"] = {
            "raw": True,
            "score": bonus,
        }
    else:
        bonus = 0.0

    m_score = min(100.0, m_base + bonus)
    return round(m_score, 2), {
        "components": components,
        "score": round(m_score, 2),
    }
