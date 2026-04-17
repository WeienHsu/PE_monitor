"""
Fetch stock price and financial data via yfinance.
Results are cached to data/{ticker}_*.csv to avoid repeated API calls.

Resilience features:
  - Retry: each network call retries up to 2 times on transient errors.
  - Stale-cache fallback: if yfinance fails and a cached file exists (even
    if stale), the stale cache is returned with a warning rather than
    raising an exception.
  - Earnings-aware TTL: if the next earnings date is within 1 day, the
    fundamentals cache TTL is reduced to 1 h to avoid stale-EPS scenarios.
"""

import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def _write_json_atomic(path: Path, data: object) -> None:
    """Write JSON to a temp file then rename — prevents corrupt caches on interruption."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json_safe(path: Path) -> object | None:
    """Read JSON; delete and return None if the file is corrupt."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        path.unlink(missing_ok=True)
        return None

_RETRY_ATTEMPTS = 2
_RETRY_DELAY_S = 2.0


def _retry(fn, *args, attempts: int = _RETRY_ATTEMPTS, delay: float = _RETRY_DELAY_S, **kwargs):
    """Call fn(*args, **kwargs) up to `attempts` times on exception."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(delay)
    raise last_exc


def _cache_path(data_dir: str, ticker: str, suffix: str) -> Path:
    return Path(data_dir) / f"{ticker}_{suffix}"


def _is_stale(path: Path, max_age_hours: int = 6) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=max_age_hours)


# ---------------------------------------------------------------------------
# Earnings-aware cache TTL
# ---------------------------------------------------------------------------

def get_next_earnings_date(ticker: str, data_dir: str = "data") -> date | None:
    """Return the next/most-recent earnings date, or None if unavailable."""
    cache = _cache_path(data_dir, ticker, "earnings_date.json")
    if not _is_stale(cache, max_age_hours=24):
        raw = _read_json_safe(cache)
        if raw and raw.get("date"):
            try:
                return datetime.strptime(raw["date"], "%Y-%m-%d").date()
            except Exception:
                pass

    earnings_date: date | None = None
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is not None and not cal.empty:
            # calendar index contains "Earnings Date" (or similar)
            for label in ("Earnings Date", "earnings_date", "Earnings"):
                if label in cal.index:
                    val = cal.loc[label]
                    raw_date = val.iloc[0] if hasattr(val, "iloc") else val
                    if pd.notna(raw_date):
                        earnings_date = pd.Timestamp(raw_date).date()
                    break
    except Exception:
        pass

    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        _write_json_atomic(cache, {"date": earnings_date.isoformat() if earnings_date else None})
    except Exception:
        pass

    return earnings_date


def _fundamentals_ttl_hours(ticker: str, data_dir: str, default_hours: int = 12) -> int:
    """Return a reduced TTL (1 h) if earnings are within 1 day, else default."""
    try:
        ed = get_next_earnings_date(ticker, data_dir)
        if ed and abs((ed - date.today()).days) <= 1:
            return 1
    except Exception:
        pass
    return default_hours


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
    """Fetch yfinance .info dict; light cache (6 h). Falls back to stale cache on error."""
    cache = _cache_path(data_dir, ticker, "info.json")
    if not _is_stale(cache):
        data = _read_json_safe(cache)
        if data is not None:
            return data
        # File was corrupt — fall through to re-fetch
    try:
        info = _retry(lambda: yf.Ticker(ticker).info)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        _write_json_atomic(cache, info)
        return info
    except Exception as e:
        if cache.exists():
            print(f"[data_fetcher] fetch_info {ticker} network failed ({e}); using stale cache")
            data = _read_json_safe(cache)
            if data is not None:
                return data
        print(f"[data_fetcher] fetch_info {ticker} failed: {e}")
        return {}


def fetch_quarterly_financials(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Return quarterly income statement (rows = metrics, cols = dates).
    Cached to {ticker}_quarterly_financials.csv.
    TTL is 1 h when earnings are within 1 day, otherwise 12 h.
    Falls back to stale cache on network error.
    """
    ttl = _fundamentals_ttl_hours(ticker, data_dir, default_hours=12)
    cache = _cache_path(data_dir, ticker, "quarterly_financials.csv")
    if not _is_stale(cache, max_age_hours=ttl):
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    try:
        def _fetch():
            t = yf.Ticker(ticker)
            df = t.quarterly_income_stmt
            if df is None or df.empty:
                df = t.quarterly_financials
            return df

        df = _retry(_fetch)
        if df is None or df.empty:
            return pd.DataFrame()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as e:
        if cache.exists():
            print(f"[data_fetcher] fetch_quarterly_financials {ticker} network failed ({e}); using stale cache")
            try:
                return pd.read_csv(cache, index_col=0, parse_dates=True)
            except Exception:
                pass
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

        def _download():
            return yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=True,
            )

        df = _retry(_download)
        if df is None or df.empty:
            return pd.DataFrame()
        df = _flatten_columns(df)
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as e:
        if cache.exists():
            print(f"[data_fetcher] fetch_price_history {ticker} network failed ({e}); using stale cache")
            try:
                stale_df = pd.read_csv(cache, index_col=0, parse_dates=True)
                if _is_valid_price_df(stale_df):
                    return stale_df
            except Exception:
                pass
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
    ttl = _fundamentals_ttl_hours(ticker, data_dir, default_hours=12)
    cache = _cache_path(data_dir, ticker, "cashflow.json")
    if not _is_stale(cache, max_age_hours=ttl):
        data = _read_json_safe(cache)
        if data is not None:
            return data
    result: dict = {
        "operating_cashflow": None,
        "free_cashflow": None,
        "peg_ratio": None,
        "forward_pe": None,
        "forward_eps": None,
        "ev_ebitda": None,
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

        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda and ev_ebitda > 0:
            result["ev_ebitda"] = float(ev_ebitda)

        # Fallback: read from annual .cashflow statement if .info is empty
        if result["operating_cashflow"] is None:
            def _fetch_cf():
                t = yf.Ticker(ticker)
                return t.cashflow

            cf = _retry(_fetch_cf)
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
        _write_json_atomic(cache, result)
    except Exception:
        pass

    return result
