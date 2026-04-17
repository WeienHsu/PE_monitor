"""
Sentiment analysis for news articles.

Language routing:
  - Chinese text (>20% CJK characters) → SnowNLP (offline, no GPU)
  - English text → VADER (offline, no GPU)

VADER compound score: >= +0.05 positive, <= -0.05 negative, else neutral
SnowNLP sentiment: 0–1 (mapped to -1 to +1 for consistency)

Time weighting: articles from today get weight 1.0; articles from
`lookback_days` ago get weight ~0.1 (exponential decay).
"""

import math
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# VADER bootstrap
# ---------------------------------------------------------------------------

def _get_sia():
    """Return a SentimentIntensityAnalyzer, downloading vader_lexicon if needed."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()


_SIA = None


def _sia() -> object:
    global _SIA
    if _SIA is None:
        _SIA = _get_sia()
    return _SIA


# ---------------------------------------------------------------------------
# SnowNLP bootstrap (optional — graceful fallback to VADER if not installed)
# ---------------------------------------------------------------------------

_SNOWNLP_AVAILABLE: bool | None = None


def _snownlp_available() -> bool:
    global _SNOWNLP_AVAILABLE
    if _SNOWNLP_AVAILABLE is None:
        try:
            import snownlp  # noqa: F401
            _SNOWNLP_AVAILABLE = True
        except ImportError:
            _SNOWNLP_AVAILABLE = False
    return _SNOWNLP_AVAILABLE


def _score_chinese(text: str) -> float:
    """Return compound score in [-1, 1] for Chinese text via SnowNLP."""
    if not text:
        return 0.0
    try:
        from snownlp import SnowNLP
        # SnowNLP returns 0–1; convert to -1 to +1
        return (SnowNLP(text).sentiments - 0.5) * 2.0
    except Exception:
        return 0.0


def _cjk_ratio(text: str) -> float:
    """Return fraction of characters that are CJK (Chinese/Japanese/Korean)."""
    if not text:
        return 0.0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    return cjk / len(text)


def _is_chinese(text: str) -> bool:
    """Return True if text is predominantly Chinese (>20% CJK chars)."""
    return _cjk_ratio(text) > 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_article(headline: str, summary: str) -> float:
    """
    Return a compound sentiment score for one article in [-1, 1].
    Headline weighted 0.7, summary 0.3.
    Routes to SnowNLP for Chinese text, VADER for English.
    """
    headline = headline or ""
    summary = summary or ""

    # Detect language from headline (longer text → summary, else headline)
    detect_text = summary if len(summary) > len(headline) else headline
    use_chinese = _is_chinese(detect_text) and _snownlp_available()

    if use_chinese:
        h_score = _score_chinese(headline)
        s_score = _score_chinese(summary)
    else:
        analyzer = _sia()
        h_score = analyzer.polarity_scores(headline)["compound"]
        s_score = analyzer.polarity_scores(summary)["compound"]

    return 0.7 * h_score + 0.3 * s_score


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
}


def analyze_sentiment(articles: list[dict], lookback_days: int = 7) -> dict:
    """
    Score a list of article dicts and return an aggregated sentiment result.

    Each article must have at minimum:
      {"headline": str, "summary": str, "datetime": int (unix ts)}

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
    }
    """
    if not articles:
        return dict(_EMPTY_SENTIMENT)

    now_ts = time.time()
    cutoff_ts = now_ts - lookback_days * 86400

    scored: list[tuple[float, float]] = []  # (score, weight)
    label_counts = {"positive": 0, "neutral": 0, "negative": 0}

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
        weight = _time_weight(ts, now_ts, lookback_days)
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
    }


def score_articles_individually(articles: list[dict], lookback_days: int = 7) -> list[dict]:
    """
    Return the same articles list with an additional 'sentiment_label' key per article.
    Used by the dashboard to show per-headline emoji badges.
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
        result.append({**art, "sentiment_label": _label(s), "date_str": date_str})
    # Return up to 10 most recent
    result.sort(key=lambda a: a.get("datetime", 0), reverse=True)
    return result[:10]
