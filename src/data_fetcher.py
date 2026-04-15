"""
Fetch stock price and financial data via yfinance.
Results are cached to data/{ticker}_*.csv to avoid repeated API calls.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def _cache_path(data_dir: str, ticker: str, suffix: str) -> Path:
    return Path(data_dir) / f"{ticker}_{suffix}"


def _is_stale(path: Path, max_age_hours: int = 6) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=max_age_hours)


def fetch_info(ticker: str, data_dir: str = "data") -> dict:
    """Fetch yfinance .info dict; light cache (6 h)."""
    cache = _cache_path(data_dir, ticker, "info.json")
    if not _is_stale(cache):
        with open(cache, "r") as f:
            return json.load(f)
    try:
        info = yf.Ticker(ticker).info
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(info, f)
        return info
    except Exception as e:
        print(f"[data_fetcher] fetch_info {ticker} failed: {e}")
        return {}


def fetch_quarterly_financials(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Return quarterly income statement (rows = metrics, cols = dates).
    Cached to {ticker}_quarterly_financials.csv (refreshed every 12 h).
    """
    cache = _cache_path(data_dir, ticker, "quarterly_financials.csv")
    if not _is_stale(cache, max_age_hours=12):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        t = yf.Ticker(ticker)
        df = t.quarterly_income_stmt  # rows=metrics, cols=dates
        if df is None or df.empty:
            df = t.quarterly_financials
        if df is None or df.empty:
            return pd.DataFrame()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as e:
        print(f"[data_fetcher] fetch_quarterly_financials {ticker} failed: {e}")
        return pd.DataFrame()


def fetch_price_history(
    ticker: str,
    years: int = 5,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Return daily OHLCV history for the past `years` years.
    Cached to {ticker}_history.csv (refreshed every 6 h).
    """
    cache = _cache_path(data_dir, ticker, "price_history.csv")
    if not _is_stale(cache, max_age_hours=6):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        end = date.today()
        start = end - timedelta(days=365 * years + 30)
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as e:
        print(f"[data_fetcher] fetch_price_history {ticker} failed: {e}")
        return pd.DataFrame()


def fetch_shares_outstanding(ticker: str, data_dir: str = "data") -> float | None:
    """Return most recent diluted shares outstanding."""
    info = fetch_info(ticker, data_dir)
    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
    if shares and shares > 0:
        return float(shares)
    # fallback: from balance sheet
    try:
        t = yf.Ticker(ticker)
        bs = t.quarterly_balance_sheet
        if bs is not None and "Ordinary Shares Number" in bs.index:
            val = bs.loc["Ordinary Shares Number"].iloc[0]
            if pd.notna(val):
                return float(val)
    except Exception:
        pass
    return None


def get_latest_close(ticker: str, data_dir: str = "data") -> float | None:
    """Return today's (or most recent) closing price."""
    df = fetch_price_history(ticker, years=1, data_dir=data_dir)
    if df.empty:
        return None
    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close_col = ("Close", ticker)
        if close_col in df.columns:
            return float(df[close_col].dropna().iloc[-1])
        # Try first level
        try:
            return float(df["Close"].iloc[-1].item())
        except Exception:
            pass
    if "Close" in df.columns:
        return float(df["Close"].dropna().iloc[-1])
    return None


def is_etf(ticker: str, data_dir: str = "data") -> bool:
    """Return True if yfinance classifies this ticker as an ETF."""
    info = fetch_info(ticker, data_dir)
    quote_type = (info.get("quoteType") or "").lower()
    return quote_type == "etf"
