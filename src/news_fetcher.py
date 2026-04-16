"""
Fetch company news from Finnhub (primary) or Yahoo Finance RSS (fallback).
Results are cached to data/{ticker}_news.json with a 1-hour TTL.

Status codes returned in the result metadata:
  "ok"           — Finnhub returned articles successfully
  "rss_fallback" — Using Yahoo Finance RSS (no key set, or Finnhub failed)
  "rate_limited" — Finnhub returned HTTP 429; switched to RSS
  "invalid_key"  — Finnhub returned HTTP 401/403; switched to RSS
  "failed"       — Both sources failed
  "no_articles"  — Fetch succeeded but returned 0 articles
"""

import hashlib
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache helpers (mirrors data_fetcher.py pattern)
# ---------------------------------------------------------------------------

NEWS_CACHE_TTL_HOURS = 1


def _news_cache_path(data_dir: str, ticker: str) -> Path:
    return Path(data_dir) / f"{ticker}_news.json"


def _is_stale(path: Path, max_age_hours: int = NEWS_CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=max_age_hours)


def _load_cache(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Finnhub fetcher
# ---------------------------------------------------------------------------

def _fetch_finnhub(ticker: str, api_key: str, days: int) -> tuple[list[dict], str]:
    """
    Returns (articles, status).
    status: "ok" | "rate_limited" | "invalid_key" | "failed"
    """
    try:
        import finnhub  # type: ignore
    except ImportError:
        return [], "failed"

    try:
        client = finnhub.Client(api_key=api_key)
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = client.company_news(ticker, _from=from_date, to=to_date)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            return [], "rate_limited"
        if "401" in err_str or "403" in err_str:
            return [], "invalid_key"
        return [], "failed"

    articles = []
    for item in (raw or []):
        articles.append({
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "datetime": item.get("datetime", 0),  # unix timestamp
            "source": item.get("source", ""),
            "url": item.get("url", ""),
        })
    return articles, "ok"


# ---------------------------------------------------------------------------
# Yahoo Finance RSS fallback
# ---------------------------------------------------------------------------

def _fetch_rss(ticker: str) -> list[dict]:
    """
    Fetch news from Yahoo Finance RSS feed for the given ticker.
    Returns a list of article dicts (same schema as Finnhub).
    """
    try:
        import feedparser  # type: ignore
    except ImportError:
        return []

    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        articles = []
        for entry in (feed.entries or [])[:50]:
            # Convert published tuple → unix timestamp
            pub_ts = 0
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_ts = int(time.mktime(entry.published_parsed))
            articles.append({
                "headline": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "datetime": pub_ts,
                "source": "Yahoo Finance",
                "url": entry.get("link", ""),
            })
        return articles
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def _normalize_headline(headline: str) -> str:
    """Lowercase and strip all non-alphanumeric characters for stable comparison."""
    return re.sub(r"[^a-z0-9]", "", headline.lower())


def _deduplicate_articles(
    articles: list[dict], similarity_threshold: float = 0.85
) -> list[dict]:
    """
    Remove duplicate and near-duplicate articles.

    Two-pass strategy:
      Pass 1 — exact dedup via MD5 hash of normalized headline.
      Pass 2 — near-dedup via difflib.SequenceMatcher ratio >= similarity_threshold.
    """
    from difflib import SequenceMatcher

    seen_hashes: set[str] = set()
    seen_normalized: list[str] = []
    unique: list[dict] = []

    for art in articles:
        norm = _normalize_headline(art.get("headline", ""))
        if not norm:
            unique.append(art)
            continue

        # Pass 1: exact hash check
        h = hashlib.md5(norm.encode()).hexdigest()
        if h in seen_hashes:
            continue

        # Pass 2: fuzzy similarity against already-accepted headlines
        is_dup = any(
            SequenceMatcher(None, norm, existing).ratio() >= similarity_threshold
            for existing in seen_normalized
        )
        if not is_dup:
            seen_hashes.add(h)
            seen_normalized.append(norm)
            unique.append(art)

    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news(ticker: str, config: dict, data_dir: str = "data") -> tuple[list[dict], str, str]:
    """
    Fetch news for *ticker* using Finnhub (primary) or Yahoo Finance RSS (fallback).

    Returns
    -------
    (articles, news_source, news_status)
      articles    — list of dicts with keys: headline, summary, datetime, source, url
      news_source — "finnhub" | "rss_fallback" | "none"
      news_status — "ok" | "rss_fallback" | "rate_limited" | "invalid_key" |
                    "failed" | "no_articles"
    """
    cache_path = _news_cache_path(data_dir, ticker)

    # --- Try cache first ---
    if not _is_stale(cache_path):
        cached = _load_cache(cache_path)
        if cached is not None:
            return (
                cached.get("articles", []),
                cached.get("news_source", "none"),
                cached.get("news_status", "ok"),
            )

    settings = config.get("settings", {})
    api_key = settings.get("finnhub_api_key", "").strip()
    days = int(settings.get("news_days_lookback", 7))

    articles: list[dict] = []
    news_source: str = "none"
    news_status: str = "failed"

    # --- Try Finnhub ---
    if api_key:
        articles, finnhub_status = _fetch_finnhub(ticker, api_key, days)
        if finnhub_status == "ok":
            news_source = "finnhub"
            news_status = "ok" if articles else "no_articles"
        else:
            # Fallback to RSS and preserve the specific Finnhub error
            news_status = finnhub_status  # rate_limited | invalid_key | failed
            rss_articles = _fetch_rss(ticker)
            if rss_articles:
                articles = rss_articles
                news_source = "rss_fallback"
                # Keep news_status as the Finnhub error code so the UI can show it
            else:
                news_source = "none"
    else:
        # No key — go straight to RSS
        rss_articles = _fetch_rss(ticker)
        if rss_articles:
            articles = rss_articles
            news_source = "rss_fallback"
            news_status = "rss_fallback"
        else:
            news_source = "none"
            news_status = "failed"

    # Deduplicate before caching
    if articles:
        articles = _deduplicate_articles(articles)

    # Final no_articles check
    if not articles and news_status == "ok":
        news_status = "no_articles"

    # --- Persist cache ---
    payload = {
        "articles": articles,
        "news_source": news_source,
        "news_status": news_status,
        "cached_at": datetime.now().isoformat(),
    }
    _save_cache(cache_path, payload)

    return articles, news_source, news_status
