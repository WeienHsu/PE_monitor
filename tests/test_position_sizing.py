"""Unit tests for position_sizing.py"""

from src.position_sizing import suggest_position


class TestNoHoldingBuySide:
    def test_strong_buy_no_position_initial_33(self):
        r = suggest_position("STRONG_BUY", percentile_rank=5.0, holding_shares=None)
        assert r["action"] == "INITIAL"
        assert r["size_pct"] == 33.0

    def test_buy_no_position_initial_25(self):
        r = suggest_position("BUY", percentile_rank=20.0, holding_shares=0)
        assert r["action"] == "INITIAL"
        assert r["size_pct"] == 25.0

    def test_cautious_buy_no_position_initial_15(self):
        r = suggest_position("CAUTIOUS_BUY", percentile_rank=22.0, holding_shares=None)
        assert r["action"] == "INITIAL"
        assert r["size_pct"] == 15.0

    def test_watch_no_position_hold(self):
        r = suggest_position("WATCH", percentile_rank=30.0, holding_shares=None)
        assert r["action"] == "HOLD"
        assert r["size_pct"] == 0.0


class TestExistingHoldingBuySide:
    def test_strong_buy_deep_discount_add_25(self):
        """STRONG_BUY at percentile<15 → add 25% to existing position."""
        r = suggest_position("STRONG_BUY", percentile_rank=10.0, holding_shares=100)
        assert r["action"] == "ADD"
        assert r["size_pct"] == 25.0

    def test_strong_buy_not_deep_add_15(self):
        r = suggest_position("STRONG_BUY", percentile_rank=20.0, holding_shares=100)
        assert r["action"] == "ADD"
        assert r["size_pct"] == 15.0

    def test_buy_with_holding_add_15(self):
        r = suggest_position("BUY", percentile_rank=22.0, holding_shares=100)
        assert r["action"] == "ADD"
        assert r["size_pct"] == 15.0

    def test_cautious_buy_with_holding_hold(self):
        """Don't ladder into weak-conviction buy signals."""
        r = suggest_position("CAUTIOUS_BUY", percentile_rank=22.0, holding_shares=100)
        assert r["action"] == "HOLD"


class TestSellSide:
    def test_strong_sell_with_position_exit(self):
        r = suggest_position("STRONG_SELL", percentile_rank=95.0, holding_shares=100)
        assert r["action"] == "EXIT"
        assert r["size_pct"] == 100.0

    def test_sell_with_position_trim_half(self):
        r = suggest_position("SELL", percentile_rank=80.0, holding_shares=100)
        assert r["action"] == "TRIM"
        assert r["size_pct"] == 50.0

    def test_cautious_sell_with_position_trim_quarter(self):
        r = suggest_position("CAUTIOUS_SELL", percentile_rank=78.0, holding_shares=100)
        assert r["action"] == "TRIM"
        assert r["size_pct"] == 25.0

    def test_sell_no_position_hold(self):
        r = suggest_position("SELL", percentile_rank=80.0, holding_shares=None)
        assert r["action"] == "HOLD"


class TestMiddleZones:
    def test_neutral_hold(self):
        r = suggest_position("NEUTRAL", percentile_rank=50.0, holding_shares=None)
        assert r["action"] == "HOLD"

    def test_caution_hold(self):
        r = suggest_position("CAUTION", percentile_rank=70.0, holding_shares=100)
        assert r["action"] == "HOLD"

    def test_unknown_signal_defaults_to_hold(self):
        r = suggest_position("SOMETHING_WEIRD", percentile_rank=50.0, holding_shares=None)
        assert r["action"] == "HOLD"


class TestNonePercentileFallback:
    def test_percentile_rank_none_does_not_crash(self):
        """percentile_rank can be None (insufficient history)."""
        r = suggest_position("STRONG_BUY", percentile_rank=None, holding_shares=100)
        # Without a rank, can't detect "deep discount" → default ADD size
        assert r["action"] == "ADD"
        assert r["size_pct"] == 15.0
