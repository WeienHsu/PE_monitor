"""
Unit tests for composite_signal.py — focusing on the P3-16 extreme-negative
sentiment filter (2-level asymmetric drop for CAUTION / NEUTRAL / WATCH).

The regime and value-trap post-filters are covered in test_market_regime.py
and test_value_trap_filter.py respectively — this file adds coverage for the
third post-filter in the chain.
"""

from src.composite_signal import compute_multi_factor_composite


class TestExtremeSentimentFilter:
    """P3-16: sentiment_score < -0.6 demotes CAUTION/NEUTRAL/WATCH one extra level."""

    def test_caution_demoted_to_cautious_sell_on_extreme_negative(self):
        """PE=NEUTRAL + sentiment=negative → CAUTION (matrix).  With
        sentiment_score=-0.7 we demote one more level in _SIGNAL_ORDER —
        which places us at CAUTIOUS_SELL (between CAUTION and SELL)."""
        key, _, factors = compute_multi_factor_composite(
            "NEUTRAL",
            "negative",
            stock_type="stable",
            sentiment_score=-0.7,
        )
        # Matrix: (NEUTRAL, negative) → CAUTION → demoted to CAUTIOUS_SELL
        assert key == "CAUTIOUS_SELL"
        assert factors.get("Extreme Sentiment") == -1

    def test_neutral_demoted_to_caution(self):
        """PE=NEUTRAL + sentiment=neutral → NEUTRAL.  Extreme score -0.8
        (edge case: label is neutral but aggregate score dragged negative by
        a couple of strongly negative headlines) demotes to CAUTION."""
        key, _, factors = compute_multi_factor_composite(
            "NEUTRAL",
            "neutral",
            stock_type="stable",
            sentiment_score=-0.8,
        )
        assert key == "CAUTION"
        assert factors.get("Extreme Sentiment") == -1

    def test_watch_demoted_to_neutral(self):
        """PE=WATCH + sentiment=neutral → WATCH → demoted to NEUTRAL."""
        key, _, factors = compute_multi_factor_composite(
            "WATCH",
            "neutral",
            stock_type="stable",
            sentiment_score=-0.65,
        )
        assert key == "NEUTRAL"
        assert factors.get("Extreme Sentiment") == -1

    def test_buy_class_not_affected(self):
        """BUY-class signals are NOT in the apply set — they're handled by
        value-trap and regime filters instead.  A -0.7 score here should NOT
        demote an already-BUY signal (even though it might feel counter-
        intuitive, the valuation signal dominates)."""
        key, _, factors = compute_multi_factor_composite(
            "BUY",
            "neutral",
            stock_type="stable",
            sentiment_score=-0.7,
        )
        # (BUY, neutral) → BUY; no extreme-sentiment demotion
        assert key == "BUY"
        assert "Extreme Sentiment" not in factors

    def test_sell_class_not_affected(self):
        """SELL and below are not in the apply set — they're already at or near
        the floor, further demotion is rarely meaningful."""
        key, _, factors = compute_multi_factor_composite(
            "CAUTION",
            "negative",
            stock_type="stable",
            sentiment_score=-0.9,
        )
        # Matrix: (CAUTION, negative) → SELL.  SELL not in apply-set → stays.
        assert key == "SELL"
        assert "Extreme Sentiment" not in factors

    def test_not_triggered_just_above_threshold(self):
        """sentiment_score == -0.6 exactly is NOT extreme — needs to be strictly
        less than -0.6.  (Boundary inclusive on the non-trigger side.)"""
        key, _, factors = compute_multi_factor_composite(
            "NEUTRAL",
            "negative",
            stock_type="stable",
            sentiment_score=-0.6,
        )
        # Stays CAUTION; no extra demotion
        assert key == "CAUTION"
        assert "Extreme Sentiment" not in factors

    def test_asymmetric_positive_never_promotes(self):
        """Extreme positive sentiment does NOT promote — asymmetric by design
        to avoid FOMO amplification at market tops."""
        key, _, factors = compute_multi_factor_composite(
            "NEUTRAL",
            "positive",
            stock_type="stable",
            sentiment_score=0.85,
        )
        # Matrix: (NEUTRAL, positive) → WATCH.  No further promotion.
        assert key == "WATCH"
        assert "Extreme Sentiment" not in factors

    def test_none_score_is_legacy(self):
        """sentiment_score=None → filter disabled, nothing changes."""
        key, _, factors = compute_multi_factor_composite(
            "NEUTRAL",
            "negative",
            stock_type="stable",
            sentiment_score=None,
        )
        assert key == "CAUTION"
        assert "Extreme Sentiment" not in factors

    def test_chained_with_regime_and_value_trap(self):
        """Chained post-filters: RISK_OFF + extreme negative sentiment should
        both apply.  Base (BUY, neutral) → BUY, demoted by RISK_OFF to
        CAUTIOUS_BUY.  CAUTIOUS_BUY is not in the extreme-sentiment apply-set,
        so no further drop. This is intentional — BUY-class is regime's
        domain.
        """
        key, _, factors = compute_multi_factor_composite(
            "BUY",
            "neutral",
            stock_type="stable",
            market_regime="RISK_OFF",
            sentiment_score=-0.7,
        )
        assert key == "CAUTIOUS_BUY"
        assert factors.get("Market Regime") == -1
        # Extreme-sentiment does not stack on BUY-class demotions
        assert "Extreme Sentiment" not in factors

    def test_chained_regime_lands_in_watch_then_extreme_sentiment_drops(self):
        """Regime demotes (BUY, positive) → STRONG_BUY → BUY (only once).
        To land in WATCH we need cap-reduced base: (WATCH, neutral) → WATCH
        + RISK_OFF → NEUTRAL.  Then extreme sentiment hits NEUTRAL → CAUTION."""
        key, _, factors = compute_multi_factor_composite(
            "WATCH",
            "neutral",
            stock_type="stable",
            market_regime="RISK_OFF",
            sentiment_score=-0.75,
        )
        # WATCH → NEUTRAL (regime), NEUTRAL → CAUTION (extreme sentiment)
        assert key == "CAUTION"
        assert factors.get("Market Regime") == -1
        assert factors.get("Extreme Sentiment") == -1
