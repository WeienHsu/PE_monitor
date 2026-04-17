"""Unit tests for M-factor: 12-1 momentum math (no network)."""

import numpy as np
import pandas as pd

from src.factors.momentum_factor import _compute_12_1_series


def _make_series(prices: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=idx)


def test_12_1_series_short_history_returns_empty():
    s = _make_series([100.0] * 100)
    assert _compute_12_1_series(s).empty


def test_12_1_series_constant_prices_yields_zero_momentum():
    s = _make_series([100.0] * 300)
    mom = _compute_12_1_series(s)
    assert not mom.empty
    assert np.isclose(mom.iloc[-1], 0.0)


def test_12_1_series_positive_when_prices_rising():
    # Linear ramp: by day 300, price = 300; momentum window uses t-21 / t-252
    s = _make_series(list(range(100, 400)))
    mom = _compute_12_1_series(s)
    assert not mom.empty
    # Latest momentum: close at t-21 vs close at t-252 → positive
    assert mom.iloc[-1] > 0


def test_12_1_series_negative_when_prices_falling():
    s = _make_series(list(range(400, 100, -1)))
    mom = _compute_12_1_series(s)
    assert not mom.empty
    assert mom.iloc[-1] < 0
