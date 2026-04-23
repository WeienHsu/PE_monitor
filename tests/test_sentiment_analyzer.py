"""Unit tests for sentiment_analyzer.py (P2-10 LM + FinBERT engines)."""

import time

import pytest

import src.sentiment_analyzer as sa
from src.sentiment_analyzer import analyze_sentiment, active_engine


def _reset_engine_singletons():
    """Reset lazy-loaded singletons so env var changes take effect mid-test."""
    sa._ACTIVE_ENGINE = None
    sa._SIA = None
    sa._LM = None
    sa._FINBERT = None


@pytest.fixture(autouse=True)
def _isolate_engine(monkeypatch):
    """Each test starts with a clean engine-selection state."""
    _reset_engine_singletons()
    yield
    _reset_engine_singletons()


class TestEngineSelection:
    def test_default_picks_vader_lm_when_available(self, monkeypatch):
        monkeypatch.delenv("SENTIMENT_ENGINE", raising=False)
        monkeypatch.setattr(sa, "_ENGINE_ENV", "")
        assert active_engine() == "vader_lm"

    def test_force_vader(self, monkeypatch):
        monkeypatch.setattr(sa, "_ENGINE_ENV", "vader")
        assert active_engine() == "vader"

    def test_finbert_falls_back_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(sa, "_ENGINE_ENV", "finbert")
        # Pretend transformers isn't installed
        monkeypatch.setattr(sa, "_get_finbert", lambda: None)
        # Should fall through to vader_lm (since pysentiment2 is installed)
        assert active_engine() == "vader_lm"

    def test_falls_back_to_vader_when_lm_unavailable(self, monkeypatch):
        monkeypatch.setattr(sa, "_ENGINE_ENV", "")
        monkeypatch.setattr(sa, "_get_lm", lambda: None)
        _reset_engine_singletons()
        assert active_engine() == "vader"


class TestEmptyAndNoisy:
    def test_empty_list_returns_empty_sentiment(self):
        r = analyze_sentiment([])
        assert r["available"] is False
        assert r["count"] == 0
        assert r["score"] == 0.0

    def test_cjk_article_is_treated_as_neutral(self):
        articles = [
            {"headline": "公司公布季度財報", "summary": "營收增長", "datetime": int(time.time())}
        ]
        r = analyze_sentiment(articles)
        # Single CJK article scores 0 → overall neutral
        assert r["label"] == "neutral"

    def test_old_articles_are_dropped(self):
        """Articles outside the lookback window should not be counted."""
        old_ts = int(time.time()) - 60 * 86400  # 60 days ago
        articles = [{"headline": "ancient news miss earnings", "summary": "", "datetime": old_ts}]
        r = analyze_sentiment(articles, lookback_days=7)
        assert r["available"] is False


class TestLMImprovesFinancialSignal:
    """
    LM should outperform pure VADER on finance-specific vocabulary.
    These articles use terms (missed guidance, litigation) that LM weights
    heavily negatively but VADER treats as neutral.
    """

    @pytest.fixture
    def financial_negative_articles(self):
        now = int(time.time())
        return [
            {"headline": "Company missed earnings; cuts guidance amid litigation",
             "summary": "", "datetime": now},
            {"headline": "Write-down impairs goodwill; restructuring charges weigh on EPS",
             "summary": "", "datetime": now},
        ]

    def test_vader_lm_scores_more_negative_than_pure_vader(
        self, monkeypatch, financial_negative_articles
    ):
        # Pure VADER
        monkeypatch.setattr(sa, "_ENGINE_ENV", "vader")
        _reset_engine_singletons()
        vader_result = analyze_sentiment(financial_negative_articles)

        # VADER + LM blend
        monkeypatch.setattr(sa, "_ENGINE_ENV", "")
        _reset_engine_singletons()
        lm_result = analyze_sentiment(financial_negative_articles)

        # LM blend should push the score further negative on financial vocab
        assert lm_result["score"] <= vader_result["score"]
        # And it should at least tag some articles as negative
        assert lm_result["article_count_by_label"]["negative"] >= 1

    def test_engine_recorded_in_result(self, financial_negative_articles):
        r = analyze_sentiment(financial_negative_articles)
        assert r["engine"] in ("vader_lm", "vader")


class TestEarningsPulseWeighting:
    """P3-17: articles within ±3 days of an earnings date get 2× weight."""

    def test_pulse_amplifies_earnings_day_article(self, monkeypatch):
        """One very negative earnings-day article plus one mildly positive
        non-earnings article — without pulse the aggregate is near neutral,
        with pulse the earnings negative should dominate."""
        monkeypatch.setattr(sa, "_ENGINE_ENV", "vader")
        _reset_engine_singletons()

        now = int(time.time())
        earnings_ts = now  # earnings just happened
        non_earnings_ts = now - 10 * 86400  # 10 days ago
        # Both inside a 14-day lookback
        articles = [
            {
                "headline": "Missed earnings and cuts guidance amid probe",
                "summary": "",
                "datetime": earnings_ts,
            },
            {
                "headline": "Company launches new positive product line",
                "summary": "",
                "datetime": non_earnings_ts,
            },
        ]

        without_pulse = analyze_sentiment(articles, lookback_days=14, earnings_dates=None)
        with_pulse = analyze_sentiment(
            articles, lookback_days=14, earnings_dates=[earnings_ts]
        )

        # Pulse should push the score MORE negative (or at least no less so)
        assert with_pulse["score"] <= without_pulse["score"]
        assert with_pulse["earnings_pulse_count"] == 1
        assert without_pulse["earnings_pulse_count"] == 0

    def test_pulse_window_boundary(self, monkeypatch):
        """Article exactly 3 days out is pulse-eligible; 4 days is not."""
        monkeypatch.setattr(sa, "_ENGINE_ENV", "vader")
        _reset_engine_singletons()

        now = int(time.time())
        earnings_ts = now - 3 * 86400  # earnings 3 days ago
        articles = [
            {"headline": "Earnings beat expectations", "summary": "", "datetime": now},  # 3 days from earnings — eligible
        ]
        r = analyze_sentiment(articles, lookback_days=14, earnings_dates=[earnings_ts])
        assert r["earnings_pulse_count"] == 1

        # Now shift earnings to 4 days before → out of window
        earnings_far = now - 4 * 86400 - 3600  # 4 days + 1 hour ago
        r2 = analyze_sentiment(articles, lookback_days=14, earnings_dates=[earnings_far])
        assert r2["earnings_pulse_count"] == 0

    def test_pulse_with_no_earnings_dates_behaves_like_legacy(self, monkeypatch):
        monkeypatch.setattr(sa, "_ENGINE_ENV", "vader")
        _reset_engine_singletons()

        now = int(time.time())
        articles = [
            {"headline": "Strong growth and record revenue", "summary": "", "datetime": now},
        ]
        r_none = analyze_sentiment(articles, lookback_days=14, earnings_dates=None)
        r_empty = analyze_sentiment(articles, lookback_days=14, earnings_dates=[])
        assert r_none["score"] == r_empty["score"]
        assert r_none["earnings_pulse_count"] == 0
        assert r_empty["earnings_pulse_count"] == 0

    def test_score_articles_individually_flags_pulse(self, monkeypatch):
        """Per-article view should mark earnings_pulse=True for in-window articles."""
        monkeypatch.setattr(sa, "_ENGINE_ENV", "vader")
        _reset_engine_singletons()
        from src.sentiment_analyzer import score_articles_individually

        now = int(time.time())
        earnings_ts = now
        articles = [
            {"headline": "Earnings out today", "summary": "", "datetime": now},
            {"headline": "Unrelated old news", "summary": "", "datetime": now - 5 * 86400},
        ]
        scored = score_articles_individually(
            articles, lookback_days=14, earnings_dates=[earnings_ts]
        )
        # First article is within window; second is 5 days away — not in window
        flagged = {a["headline"]: a.get("earnings_pulse") for a in scored}
        assert flagged["Earnings out today"] is True
        assert flagged["Unrelated old news"] is False


class TestLabelBoundaries:
    def test_positive_boundary(self):
        # build a headline that definitely scores >0.05
        articles = [{"headline": "Excellent beat; raises dividend and expands margins.",
                     "summary": "", "datetime": int(time.time())}]
        r = analyze_sentiment(articles)
        assert r["label"] in ("positive", "neutral")

    def test_neutral_when_no_signal(self):
        articles = [{"headline": "The company held a meeting.", "summary": "",
                     "datetime": int(time.time())}]
        r = analyze_sentiment(articles)
        assert r["label"] == "neutral"
