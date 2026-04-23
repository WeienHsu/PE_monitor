"""Unit tests for momentum_factors.py (P2-11)."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.composite_signal import compute_multi_factor_composite
from src.momentum_factors import get_52w_position, get_momentum_factors, get_volume_factor


def _df(close: list[float], volume: list[float]) -> pd.DataFrame:
    n = len(close)
    assert n == len(volume)
    dates = pd.date_range(end="2026-04-18", periods=n, freq="D")
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


# ---------------------------------------------------------------------------
# Volume factor
# ---------------------------------------------------------------------------

class TestVolumeFactor:
    def test_high_volume_surge_votes_positive(self):
        # 20 days at volume 1M, then 5 days at 2M → ratio = 2.0
        volume = [1_000_000] * 20 + [2_000_000] * 5
        close = [100.0] * 25
        df = _df(close, volume)
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_volume_factor("TEST")
        assert r["available"] is True
        assert r["vote"] == 1
        assert r["ratio"] == pytest.approx(2.0, abs=0.01)

    def test_quiet_tape_votes_negative(self):
        # 20 days at volume 2M, then 5 days at 600k → ratio = 0.3
        volume = [2_000_000] * 20 + [600_000] * 5
        close = [100.0] * 25
        df = _df(close, volume)
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_volume_factor("TEST")
        assert r["vote"] == -1
        assert r["ratio"] < 0.5

    def test_normal_volume_neutral(self):
        volume = [1_000_000] * 25
        close = [100.0] * 25
        df = _df(close, volume)
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_volume_factor("TEST")
        assert r["vote"] == 0
        assert r["ratio"] == pytest.approx(1.0, abs=0.01)

    def test_insufficient_history_unavailable(self):
        df = _df([100.0] * 10, [1e6] * 10)
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_volume_factor("TEST")
        assert r["available"] is False
        assert r["vote"] == 0

    def test_empty_df_unavailable(self):
        with patch("src.momentum_factors.fetch_price_history", return_value=pd.DataFrame()):
            r = get_volume_factor("TEST")
        assert r["available"] is False


# ---------------------------------------------------------------------------
# 52-week position factor
# ---------------------------------------------------------------------------

class TestPosition52W:
    def test_near_high_votes_positive(self):
        # ramp up to 200, current at 195 → position ≈ 0.975
        close = list(range(100, 200)) + [195.0] * 100
        df = _df(close, [1e6] * len(close))
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_52w_position("TEST")
        assert r["available"] is True
        assert r["vote"] == 1
        assert r["position"] >= 0.95

    def test_far_below_high_votes_negative(self):
        # high was 200, current at 120 → position = 0.6
        close = [200.0] + [180.0] * 50 + [150.0] * 50 + [120.0] * 50
        df = _df(close, [1e6] * len(close))
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_52w_position("TEST")
        assert r["vote"] == -1
        assert r["position"] < 0.7

    def test_middle_range_neutral(self):
        # high 200, current 160 → position = 0.8
        close = [200.0] + [190.0] * 50 + [180.0] * 50 + [160.0] * 50
        df = _df(close, [1e6] * len(close))
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_52w_position("TEST")
        assert r["vote"] == 0
        assert 0.7 <= r["position"] < 0.95

    def test_insufficient_history_unavailable(self):
        df = _df([100.0] * 20, [1e6] * 20)
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_52w_position("TEST")
        assert r["available"] is False


# ---------------------------------------------------------------------------
# Combined helper
# ---------------------------------------------------------------------------

class TestGetMomentumFactors:
    def test_returns_both_keys(self):
        close = list(range(100, 200))
        df = _df(close, [1e6] * len(close))
        with patch("src.momentum_factors.fetch_price_history", return_value=df):
            r = get_momentum_factors("TEST")
        assert set(r.keys()) == {"volume", "position_52w"}


# ---------------------------------------------------------------------------
# Composite integration
# ---------------------------------------------------------------------------

class TestCompositeIntegration:
    def test_volume_surge_lifts_buy_at_cap(self):
        """In cyclical (cap=1), volume+1 alone should lift NEUTRAL by one step."""
        key, _, factors = compute_multi_factor_composite(
            "NEUTRAL", "neutral",
            stock_type="cyclical",
            volume_vote=1,
        )
        assert factors.get("Volume") == 1
        # Moves from NEUTRAL upward to WATCH
        assert key == "WATCH"

    def test_52w_position_negative_hurts_buy(self):
        """Structural weakness (-1) on a BUY should downgrade by one step."""
        key, _, factors = compute_multi_factor_composite(
            "BUY", "neutral",
            stock_type="cyclical",
            position_52w_vote=-1,
        )
        assert factors.get("52w Position") == -1
        # cyclical cap=1 → BUY demotes by one step to CAUTIOUS_BUY
        assert key == "CAUTIOUS_BUY"

    def test_growth_type_uses_momentum_factors(self):
        """Growth config also includes volume + 52w position."""
        key, _, factors = compute_multi_factor_composite(
            "WATCH", "neutral",
            stock_type="growth",
            volume_vote=1,
            position_52w_vote=1,
        )
        assert "Volume" in factors
        assert "52w Position" in factors

    def test_none_votes_are_ignored(self):
        """None votes mean 'no data' — should not enter factors dict."""
        key, _, factors = compute_multi_factor_composite(
            "NEUTRAL", "neutral",
            stock_type="stable",
            volume_vote=None,
            position_52w_vote=None,
        )
        assert "Volume" not in factors
        assert "52w Position" not in factors
