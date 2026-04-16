"""
Fetch stock price and financial data via yfinance.
Results are cached to data/{ticker}_*.csv to avoid repeated API calls.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def _cache_path(data_dir: str, ticker: str, suffix: str) -> Path:
    return Path(data_dir) / f"{ticker}_{suffix}"


def _is_stale(path: Path, max_age_hours: int = 6) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=max_age_hours)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise yfinance MultiIndex columns to simple flat names.

    yfinance >= 0.2.x returns columns like:
        MultiIndex([('Close', 'GOOGL'), ('High', 'GOOGL'), ...])
    or (newer):
        MultiIndex([('Price', 'Close'), ('Price', 'High'), ...])

    We keep only the first level that contains the metric name.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    # First level is the metric (Close, High, …); second is ticker or vice-versa.
    # Heuristic: the metric level contains standard OHLCV names.
    ohlcv = {"Close", "Open", "High", "Low", "Volume", "Adj Close",
              "Dividends", "Stock Splits"}
    l0 = set(df.columns.get_level_values(0))
    l1 = set(df.columns.get_level_values(1))
    if len(l0 & ohlcv) >= len(l1 & ohlcv):
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = df.columns.get_level_values(1)
    # Drop duplicate columns that may arise (e.g. two 'GOOGL' columns)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _is_valid_price_df(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame has a usable numeric Close column."""
    if df.empty or "Close" not in df.columns:
        return False
    close_numeric = pd.to_numeric(df["Close"].dropna().head(10), errors="coerce")
    return close_numeric.notna().any()


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
    Return daily OHLCV history for the past `years` years with flat column names.
    Cached to {ticker}_price_history.csv (refreshed every 6 h).

    Columns are always flat strings: Close, Open, High, Low, Volume.
    MultiIndex columns from yfinance are normalised before saving.
    """
    cache = _cache_path(data_dir, ticker, "price_history.csv")
    if not _is_stale(cache, max_age_hours=6):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if _is_valid_price_df(df):
            return df
        # Cached file is mangled (MultiIndex saved as multi-row header).
        # Delete it so we re-fetch cleanly.
        cache.unlink(missing_ok=True)

    try:
        end = date.today()
        start = end - timedelta(days=365 * years + 30)
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalise MultiIndex → flat column names before saving
        df = _flatten_columns(df)
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
    if df.empty or "Close" not in df.columns:
        return None
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    return float(close.iloc[-1]) if not close.empty else None


def is_etf(ticker: str, data_dir: str = "data") -> bool:
    """Return True if yfinance classifies this ticker as an ETF."""
    info = fetch_info(ticker, data_dir)
    quote_type = (info.get("quoteType") or "").lower()
    return quote_type == "etf"


def fetch_cashflow(ticker: str, data_dir: str = "data") -> dict:
    """
    Return supplementary valuation fields needed for P/CF, PEG, and Forward P/E.

    Pulls from yfinance .info first (fast, shares cache with fetch_info).
    Falls back to the annual .cashflow statement if operating cash flow is missing.

    Keys returned (any may be None if unavailable):
        operating_cashflow  — TTM operating cash flow (total dollars)
        free_cashflow       — TTM free cash flow (total dollars)
        peg_ratio           — analyst consensus PEG ratio
        forward_pe          — forward 12-month P/E (analyst consensus)
        forward_eps         — forward 12-month EPS (analyst consensus)
    """
    cache = _cache_path(data_dir, ticker, "cashflow.json")
    if not _is_stale(cache, max_age_hours=12):
        try:
            with open(cache, "r") as f:
                return json.load(f)
        except Exception:
            pass

    result: dict = {
        "operating_cashflow": None,
        "free_cashflow": None,
        "peg_ratio": None,
        "forward_pe": None,
        "forward_eps": None,
    }

    try:
        info = fetch_info(ticker, data_dir)

        peg = info.get("pegRatio") or info.get("trailingPegRatio")
        if peg and peg > 0:
            result["peg_ratio"] = float(peg)

        fpe = info.get("forwardPE")
        if fpe and fpe > 0:
            result["forward_pe"] = float(fpe)

        feps = info.get("forwardEps")
        if feps and feps != 0:
            result["forward_eps"] = float(feps)

        ocf = info.get("operatingCashflow")
        if ocf and ocf != 0:
            result["operating_cashflow"] = float(ocf)

        fcf = info.get("freeCashflow")
        if fcf and fcf != 0:
            result["free_cashflow"] = float(fcf)

        # Fallback: read from annual .cashflow statement if .info is empty
        if result["operating_cashflow"] is None:
            t = yf.Ticker(ticker)
            cf = t.cashflow  # rows=metrics, cols=dates; annual
            if cf is not None and not cf.empty:
                for row_name in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                    if row_name in cf.index:
                        val = cf.loc[row_name].iloc[0]
                        if pd.notna(val):
                            result["operating_cashflow"] = float(val)
                        break
    except Exception as e:
        print(f"[data_fetcher] fetch_cashflow {ticker} failed: {e}")

    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f)
    except Exception:
        pass

    return result
