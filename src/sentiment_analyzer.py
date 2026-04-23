"""
Sentiment analysis for news articles.

Engines (selectable via env var ``SENTIMENT_ENGINE``):

  * ``vader``       — classic VADER (legacy default; general-purpose English)
  * ``vader_lm``    — blended VADER + Loughran-McDonald financial dictionary
                      (default when ``pysentiment2`` is installed; financial
                      terms like "miss", "litigation", "guidance cut" get the
                      weight they deserve)
  * ``finbert``     — FinBERT transformer model (opt-in; heaviest, most
                      accurate on earnings-style text). Falls back to VADER+LM
                      if ``transformers`` / model download is unavailable.

VADER compound score ranges:
  >= +0.05  → positive
  <= -0.05  → negative
  otherwise → neutral

Time weighting: articles from today get weight 1.0; articles from
``lookback_days`` ago get weight ~0.1 (exponential decay).
"""

import math
import os
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Engine selection & lazy loaders
# ---------------------------------------------------------------------------

# VADER weight when blending with LM. LM weight = 1 - VADER_WEIGHT_IN_BLEND.
# Rationale: the plan specified 0.4 VADER / 0.6 LM — LM is financial-specific
# and should dominate, but VADER still catches general polarity cues.
_VADER_WEIGHT_IN_BLEND = 0.4
_LM_WEIGHT_IN_BLEND = 1.0 - _VADER_WEIGHT_IN_BLEND

_ENGINE_ENV = os.environ.get("SENTIMENT_ENGINE", "").strip().lower()


def _get_sia():
    """Return a VADER SentimentIntensityAnalyzer, downloading lexicon if needed."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()


def _get_lm():
    """
    Return a pysentiment2 Loughran-McDonald analyzer, or None if unavailable.

    pysentiment2 is an optional dependency. When not installed, the analyzer
    gracefully degrades to VADER-only scoring without errors.
    """
    try:
        import pysentiment2 as ps  # type: ignore
        return ps.LM()
    except Exception:
        return None


def _get_finbert():
    """
    Return a callable(text) → compound score in [-1, 1] using FinBERT, or None.

    Loading the FinBERT pipeline downloads ~400 MB on first use. We cache a
    single pipeline per process.  When transformers / torch aren't installed,
    returns None — caller should fall back to VADER+LM.
    """
    try:
        from transformers import pipeline  # type: ignore
        clf = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=256,
        )

        def _score(text: str) -> float:
            if not text:
                return 0.0
            out = clf(text[:1000])[0]
            label = str(out.get("label", "")).lower()
            conf = float(out.get("score", 0.0))
            if "pos" in label:
                return conf
            if "neg" in label:
                return -conf
            return 0.0

        return _score
    except Exception:
        return None


# Singletons — instantiated on first use
_SIA = None
_LM = None
_FINBERT = None
_ACTIVE_ENGINE: str | None = None


def _sia() -> object:
    global _SIA
    if _SIA is None:
        _SIA = _get_sia()
    return _SIA


def _lm() -> object | None:
    """Return the LM analyzer, caching the result even when it's None."""
    global _LM
    if _LM is None:
        _LM = _get_lm() or False  # False so we don't retry import on every call
    return _LM if _LM is not False else None


def _finbert() -> object | None:
    global _FINBERT
    if _FINBERT is None:
        _FINBERT = _get_finbert() or False
    return _FINBERT if _FINBERT is not False else None


def _resolve_engine() -> str:
    """
    Decide which engine to actually use based on env var + what's importable.

    Precedence:
      1. SENTIMENT_ENGINE=finbert → try FinBERT, fall back to vader_lm/vader
      2. SENTIMENT_ENGINE=vader   → pure VADER (legacy)
      3. default / SENTIMENT_ENGINE=vader_lm → VADER+LM if LM importable, else VADER
    """
    global _ACTIVE_ENGINE
    if _ACTIVE_ENGINE is not None:
        return _ACTIVE_ENGINE

    if _ENGINE_ENV == "finbert":
        if _finbert() is not None:
            _ACTIVE_ENGINE = "finbert"
            return _ACTIVE_ENGINE
        # fall through to vader_lm
    if _ENGINE_ENV == "vader":
        _ACTIVE_ENGINE = "vader"
        return _ACTIVE_ENGINE
    # default or vader_lm
    if _lm() is not None:
        _ACTIVE_ENGINE = "vader_lm"
    else:
        _ACTIVE_ENGINE = "vader"
    return _ACTIVE_ENGINE


def active_engine() -> str:
    """Return the sentiment engine actually in use (for UI / debug display)."""
    return _resolve_engine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cjk(text: str) -> bool:
    """Return True if text contains CJK (Chinese/Japanese/Korean) characters."""
    return any(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF for c in text)


def _vader_score(text: str) -> float:
    """VADER compound score for plain-English text (0.0 for CJK / empty)."""
    if not text or _has_cjk(text):
        return 0.0
    return _sia().polarity_scores(text)["compound"]


def _lm_score(text: str) -> float:
    """
    Loughran-McDonald financial-dictionary polarity in [-1, 1].

    LM's raw output is ``{'Positive': int, 'Negative': int, 'Polarity': float,
    'Subjectivity': float, ...}``.  We use ``Polarity`` directly: it's already
    normalised to [-1, 1] by design.  Returns 0.0 when LM isn't available.
    """
    lm = _lm()
    if lm is None or not text or _has_cjk(text):
        return 0.0
    try:
        tokens = lm.tokenize(text)
        scores = lm.get_score(tokens)
        polarity = scores.get("Polarity", 0.0)
        # Clamp to guard against any outlier numerics
        if polarity is None:
            return 0.0
        return max(-1.0, min(1.0, float(polarity)))
    except Exception:
        return 0.0


def _combined_score(text: str) -> float:
    """
    Return a combined polarity in [-1, 1] for the currently active engine.
    """
    engine = _resolve_engine()
    if not text:
        return 0.0

    if engine == "finbert":
        scorer = _finbert()
        if scorer is not None and not _has_cjk(text):
            try:
                return max(-1.0, min(1.0, float(scorer(text))))
            except Exception:
                pass
        # fall through to vader_lm below on any failure

    if engine == "vader_lm":
        return (
            _VADER_WEIGHT_IN_BLEND * _vader_score(text)
            + _LM_WEIGHT_IN_BLEND * _lm_score(text)
        )

    # legacy vader
    return _vader_score(text)


def _score_article(headline: str, summary: str) -> float:
    """
    Return a compound sentiment score for one article.
    Headline weighted 0.7, summary 0.3.
    Returns the combined compound float in [-1, 1].

    CJK headlines are skipped (engines we support are English-only); returns
    0.0 rather than producing a misleading near-zero score.
    """
    headline = headline or ""
    summary = summary or ""

    if _has_cjk(headline):
        return 0.0

    headline_score = _combined_score(headline)
    summary_score = _combined_score(summary) if summary and not _has_cjk(summary) else 0.0
    return 0.7 * headline_score + 0.3 * summary_score


def _time_weight(unix_ts: int, now_ts: float, lookback_days: int) -> float:
    """
    Exponential decay: weight = exp(-k * age_in_days)
    Calibrated so that age=0 → 1.0 and age=lookback_days → 0.1.
    """
    if not unix_ts:
        return 0.1
    age_days = max(0.0, (now_ts - unix_ts) / 86400)
    k = math.log(10) / max(lookback_days, 1)  # ln(10)/lookback
    return math.exp(-k * age_days)


# P3-17: Earnings-pulse window (days around an earnings announcement during
# which news coverage is amplified). Using a symmetric ±3 day window lets us
# catch pre-earnings rumour cycles and post-earnings analyst reactions.
_EARNINGS_PULSE_WINDOW_DAYS = 3
_EARNINGS_PULSE_MULTIPLIER = 2.0


def _earnings_pulse_multiplier(
    article_ts: int,
    earnings_dates: list[int] | None,
) -> float:
    """
    Return ``_EARNINGS_PULSE_MULTIPLIER`` (2.0) when ``article_ts`` falls within
    ±``_EARNINGS_PULSE_WINDOW_DAYS`` of any earnings date, else 1.0.

    Rationale (P3-17): real earnings surprises move stocks more than generic
    news; doubling the weight of articles in the earnings window keeps the
    aggregated score responsive to what actually matters fundamentally.

    This factor stacks multiplicatively on top of the exponential decay weight,
    so a fresh earnings-day headline ends up at 2× its decay-only weight, while
    a stale earnings-day headline still benefits proportionally.
    """
    if not earnings_dates or not article_ts:
        return 1.0
    window_sec = _EARNINGS_PULSE_WINDOW_DAYS * 86400
    for ed_ts in earnings_dates:
        if abs(article_ts - ed_ts) <= window_sec:
            return _EARNINGS_PULSE_MULTIPLIER
    return 1.0


def _label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_EMPTY_SENTIMENT = {
    "score": 0.0,
    "label": "neutral",
    "count": 0,
    "article_count_by_label": {"positive": 0, "neutral": 0, "negative": 0},
    "latest_headline": "",
    "latest_date": "",
    "available": False,
    "engine": "",
    "earnings_pulse_count": 0,
}


def analyze_sentiment(
    articles: list[dict],
    lookback_days: int = 7,
    earnings_dates: list[int] | None = None,
) -> dict:
    """
    Score a list of article dicts and return an aggregated sentiment result.

    Each article must have at minimum:
      {"headline": str, "summary": str, "datetime": int (unix ts)}

    Parameters
    ----------
    articles        : list of article dicts from news_fetcher.
    lookback_days   : exponential-decay horizon for time weighting.
    earnings_dates  : optional list of unix timestamps marking known earnings
                      announcements (past + upcoming). When provided, articles
                      within ±3 days of any listed date receive 2× additional
                      weight — real earnings news drives price discovery more
                      than generic coverage, so this keeps the aggregate
                      responsive to what moves markets (P3-17).

    Returns
    -------
    {
        "score": float,          # -1.0 to +1.0 time-weighted aggregate
        "label": str,            # "positive" | "neutral" | "negative"
        "count": int,            # articles analyzed
        "article_count_by_label": {"positive": int, "neutral": int, "negative": int},
        "latest_headline": str,
        "latest_date": str,      # "YYYY-MM-DD" or ""
        "available": bool,       # False when no articles were provided
        "engine": str,           # which sentiment engine was used
        "earnings_pulse_count": int,  # articles amplified by earnings pulse
    }
    """
    if not articles:
        return dict(_EMPTY_SENTIMENT)

    now_ts = time.time()
    cutoff_ts = now_ts - lookback_days * 86400

    scored: list[tuple[float, float]] = []  # (score, weight)
    label_counts = {"positive": 0, "neutral": 0, "negative": 0}
    pulse_count = 0

    # Sort by recency descending to find latest headline easily
    sorted_articles = sorted(articles, key=lambda a: a.get("datetime", 0), reverse=True)
    latest_headline = ""
    latest_date = ""

    for i, art in enumerate(sorted_articles):
        ts = art.get("datetime", 0) or 0
        # Skip articles older than lookback window
        if ts and ts < cutoff_ts:
            continue

        raw_score = _score_article(art.get("headline", ""), art.get("summary", ""))
        base_weight = _time_weight(ts, now_ts, lookback_days)
        pulse = _earnings_pulse_multiplier(ts, earnings_dates)
        if pulse > 1.0:
            pulse_count += 1
        weight = base_weight * pulse
        scored.append((raw_score, weight))
        label_counts[_label(raw_score)] += 1

        if i == 0:
            latest_headline = art.get("headline", "")
            raw_ts = ts
            if raw_ts:
                try:
                    latest_date = datetime.fromtimestamp(raw_ts).strftime("%Y-%m-%d")
                except Exception:
                    latest_date = ""

    if not scored:
        return dict(_EMPTY_SENTIMENT)

    # Weighted average
    total_weight = sum(w for _, w in scored)
    if total_weight == 0:
        agg_score = 0.0
    else:
        agg_score = sum(s * w for s, w in scored) / total_weight

    return {
        "score": round(agg_score, 4),
        "label": _label(agg_score),
        "count": len(scored),
        "article_count_by_label": label_counts,
        "latest_headline": latest_headline,
        "latest_date": latest_date,
        "available": True,
        "engine": _resolve_engine(),
        "earnings_pulse_count": pulse_count,
    }


def score_articles_individually(
    articles: list[dict],
    lookback_days: int = 7,
    earnings_dates: list[int] | None = None,
) -> list[dict]:
    """
    Return the same articles list with 'sentiment_label' + 'date_str' +
    'earnings_pulse' (bool, True when in ±3-day earnings window) keys added.

    Used by the dashboard to show per-headline emoji badges and earnings-pulse
    indicators.
    """
    now_ts = time.time()
    cutoff_ts = now_ts - lookback_days * 86400
    result = []
    for art in articles:
        ts = art.get("datetime", 0) or 0
        if ts and ts < cutoff_ts:
            continue
        s = _score_article(art.get("headline", ""), art.get("summary", ""))
        date_str = ""
        if ts:
            try:
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            except Exception:
                pass
        pulse = _earnings_pulse_multiplier(ts, earnings_dates) > 1.0
        result.append(
            {
                **art,
                "sentiment_label": _label(s),
                "date_str": date_str,
                "earnings_pulse": pulse,
            }
        )
    # Return up to 10 most recent
    result.sort(key=lambda a: a.get("datetime", 0), reverse=True)
    return result[:10]
