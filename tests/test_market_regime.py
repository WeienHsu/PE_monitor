"""Unit tests for market_regime.py — regime classification logic."""

from unittest.mock import patch

import pandas as pd

from src import market_regime


def _mock_spy_df(last_price: float, ma200: float, ret_20d: float) -> pd.DataFrame:
    """
    Build a synthetic SPY DataFrame where:
      - The last 200 values average to `ma200`
      - The last row's Close is `last_price`
      - The value 21 rows before the last is last_price / (1 + ret_20d)
    """
    # Start with 250 rows of ma200
    prices = [ma200] * 250
    prices[-1] = last_price
    prices[-21] = last_price / (1 + ret_20d)
    # Adjust the rest of the tail so the last 200 still average ma200.
    # Simplest: set all of [-200:-21] and [-20:-1] to ma200, then adjust
    # one earlier row to correct the residual. Given the two modifications
    # above, compute the residual and fold it into index -199.
    desired_sum = ma200 * 200
    modified_slice = prices[-200:]
    actual_sum = sum(modified_slice)
    # Correction value: apply to index -199 in modified_slice (== -199 absolute index from end)
    diff = desired_sum - actual_sum
    prices[-199] = ma200 + diff  # might push one bar off the mean — harmless, only affects MA
    dates = pd.date_range(end="2026-04-18", periods=len(prices), freq="D")
    return pd.DataFrame({"Close": prices}, index=dates)


def _mock_vix_df(vix_level: float) -> pd.DataFrame:
    dates = pd.date_range(end="2026-04-18", periods=60, freq="D")
    return pd.DataFrame({"Close": [vix_level] * len(dates)}, index=dates)


class TestRegimeClassification:
    def test_risk_on_calm_and_uptrend(self, tmp_path):
        """VIX<20 AND SPY>200MA+3% → RISK_ON."""
        def _fake_fetch(ticker, years=1, data_dir="data"):
            if ticker == "^VIX":
                return _mock_vix_df(15.0)
            return _mock_spy_df(last_price=500.0, ma200=470.0, ret_20d=0.02)

        with patch("src.market_regime.fetch_price_history", side_effect=_fake_fetch):
            result = market_regime.get_market_regime(
                data_dir=str(tmp_path), force_refresh=True
            )
        assert result["regime"] == "RISK_ON"
        assert result["vix"] == 15.0

    def test_risk_off_high_vix(self, tmp_path):
        """VIX > 30 → RISK_OFF regardless of SPY."""
        def _fake_fetch(ticker, years=1, data_dir="data"):
            if ticker == "^VIX":
                return _mock_vix_df(35.0)
            return _mock_spy_df(last_price=500.0, ma200=470.0, ret_20d=0.0)

        with patch("src.market_regime.fetch_price_history", side_effect=_fake_fetch):
            result = market_regime.get_market_regime(
                data_dir=str(tmp_path), force_refresh=True
            )
        assert result["regime"] == "RISK_OFF"

    def test_risk_off_spy_below_200ma(self, tmp_path):
        """SPY < 200MA - 3% → RISK_OFF (trend-break signal)."""
        def _fake_fetch(ticker, years=1, data_dir="data"):
            if ticker == "^VIX":
                return _mock_vix_df(22.0)
            # SPY 5% below 200MA
            return _mock_spy_df(last_price=450.0, ma200=475.0, ret_20d=-0.02)

        with patch("src.market_regime.fetch_price_history", side_effect=_fake_fetch):
            result = market_regime.get_market_regime(
                data_dir=str(tmp_path), force_refresh=True
            )
        assert result["regime"] == "RISK_OFF"

    def test_neutral_mid_vix_flat_spy(self, tmp_path):
        """VIX 20-30 + SPY near 200MA → NEUTRAL."""
        def _fake_fetch(ticker, years=1, data_dir="data"):
            if ticker == "^VIX":
                return _mock_vix_df(23.0)
            return _mock_spy_df(last_price=475.0, ma200=475.0, ret_20d=0.005)

        with patch("src.market_regime.fetch_price_history", side_effect=_fake_fetch):
            result = market_regime.get_market_regime(
                data_dir=str(tmp_path), force_refresh=True
            )
        assert result["regime"] == "NEUTRAL"

    def test_unknown_when_both_fetches_fail(self, tmp_path):
        def _fake_fetch(ticker, years=1, data_dir="data"):
            return pd.DataFrame()

        with patch("src.market_regime.fetch_price_history", side_effect=_fake_fetch):
            result = market_regime.get_market_regime(
                data_dir=str(tmp_path), force_refresh=True
            )
        assert result["regime"] == "UNKNOWN"


class TestRegimeDemotion:
    """Integration: composite signal is demoted one level in RISK_OFF."""

    def test_buy_demoted_to_cautious_buy(self):
        from src.composite_signal import compute_multi_factor_composite
        # base PE=BUY, sentiment=neutral → BUY. With no extras, should be BUY.
        key, _, factors = compute_multi_factor_composite(
            "BUY", "neutral",
            stock_type="stable",
            market_regime="RISK_OFF",
        )
        # BUY → CAUTIOUS_BUY in RISK_OFF
        assert key == "CAUTIOUS_BUY"
        assert factors.get("Market Regime") == -1

    def test_strong_buy_demoted_to_buy(self):
        from src.composite_signal import compute_multi_factor_composite
        # PE=BUY, sentiment=positive → STRONG_BUY base, then demoted to BUY
        key, _, _ = compute_multi_factor_composite(
            "BUY", "positive",
            stock_type="stable",
            market_regime="RISK_OFF",
        )
        assert key == "BUY"

    def test_no_demotion_in_risk_on(self):
        from src.composite_signal import compute_multi_factor_composite
        key, _, factors = compute_multi_factor_composite(
            "BUY", "positive",
            stock_type="stable",
            market_regime="RISK_ON",
        )
        # Should be STRONG_BUY (matrix) unchanged
        assert key == "STRONG_BUY"
        assert "Market Regime" not in factors

    def test_sell_side_unaffected(self):
        from src.composite_signal import compute_multi_factor_composite
        key, _, factors = compute_multi_factor_composite(
            "SELL", "negative",
            stock_type="stable",
            market_regime="RISK_OFF",
        )
        # STRONG_SELL stays STRONG_SELL — we don't touch SELL-side
        assert key == "STRONG_SELL"
        assert "Market Regime" not in factors

    def test_none_regime_is_legacy(self):
        from src.composite_signal import compute_multi_factor_composite
        key, _, factors = compute_multi_factor_composite(
            "BUY", "neutral",
            stock_type="stable",
            market_regime=None,
        )
        assert key == "BUY"
        assert "Market Regime" not in factors
