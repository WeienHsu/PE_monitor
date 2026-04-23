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


def fetch_annual_financials(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Return ANNUAL income statement (rows = metrics, cols = fiscal year end dates).
    Cached to {ticker}_annual_financials.csv (refreshed every 24 h — annual
    statements change infrequently).

    Why this exists separate from fetch_quarterly_financials:
      yfinance free tier returns 4-6 quarters from quarterly_income_stmt.
      Grouping those into calendar years yields only 1-2 annual EPS points,
      which is insufficient to compute EPS growth-rate volatility for the
      stock-type classifier (P0-2 follow-up). The annual statement returns
      5 years of fully-formed annual net income — exactly what the classifier
      needs.
    """
    cache = _cache_path(data_dir, ticker, "annual_financials.csv")
    if not _is_stale(cache, max_age_hours=24):
        try:
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        except Exception:
            cache.unlink(missing_ok=True)
    try:
        t = yf.Ticker(ticker)
        df = t.income_stmt  # rows=metrics, cols=dates; annual
        if df is None or df.empty:
            df = t.financials  # fallback
        if df is None or df.empty:
            return pd.DataFrame()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as e:
        print(f"[data_fetcher] fetch_annual_financials {ticker} failed: {e}")
        return pd.DataFrame()


def fetch_quarterly_balance_sheet(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Return quarterly balance sheet (rows = metrics, cols = dates).
    Cached to {ticker}_quarterly_balance_sheet.csv (refreshed every 12 h).

    Used primarily for historical BVPS (Stockholders Equity / Ordinary Shares Number)
    to build a properly time-varying P/B series, and for historical share count
    to fix survivorship-style bias in historical P/E.
    """
    cache = _cache_path(data_dir, ticker, "quarterly_balance_sheet.csv")
    if not _is_stale(cache, max_age_hours=12):
        try:
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        except Exception:
            cache.unlink(missing_ok=True)
    try:
        t = yf.Ticker(ticker)
        df = t.quarterly_balance_sheet
        if df is None or df.empty:
            return pd.DataFrame()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as e:
        print(f"[data_fetcher] fetch_quarterly_balance_sheet {ticker} failed: {e}")
        return pd.DataFrame()


def fetch_quarterly_cashflow(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Return quarterly cash-flow statement (rows = metrics, cols = dates).
    Cached to {ticker}_quarterly_cashflow.csv (refreshed every 12 h).

    Used by value_trap_filter to detect consecutive quarters of negative
    operating / free cash flow.
    """
    cache = _cache_path(data_dir, ticker, "quarterly_cashflow.csv")
    if not _is_stale(cache, max_age_hours=12):
        try:
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        except Exception:
            cache.unlink(missing_ok=True)
    try:
        t = yf.Ticker(ticker)
        df = t.quarterly_cashflow
        if df is None or df.empty:
            df = t.quarterly_cash_flow  # alt attribute in some yfinance versions
        if df is None or df.empty:
            return pd.DataFrame()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
        return df
    except Exception as e:
        print(f"[data_fetcher] fetch_quarterly_cashflow {ticker} failed: {e}")
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


def fetch_earnings_dates(ticker: str, data_dir: str = "data") -> list[int]:
    """
    Return a sorted list of unix timestamps for known earnings dates
    (recent + upcoming). Cached 24 h to {ticker}_earnings_dates.json.

    Used by P3-17 earnings-pulse sentiment weighting: articles whose datetime
    falls within ±3 days of any of these timestamps get an extra 2× weight
    factor in the sentiment aggregation.

    Sources (in order of preference):
      1. ``yf.Ticker.earnings_dates`` DataFrame — gives up to ~8 past + a few
         upcoming dates with actual announcement datetimes.
      2. ``info['earningsTimestamp']`` — next earnings (unix ts).
      3. Quarterly income statement column dates + ~45 day offset — fallback
         estimate when neither of the above are available.

    Returns an empty list when all sources fail (graceful degradation — the
    pulse-weighting filter treats an empty list as "no amplification").
    """
    cache = _cache_path(data_dir, ticker, "earnings_dates.json")
    if not _is_stale(cache, max_age_hours=24):
        try:
            with open(cache, "r") as f:
                data = json.load(f)
            return [int(t) for t in data if t]
        except Exception:
            cache.unlink(missing_ok=True)

    dates: set[int] = set()

    # 1. Preferred: yfinance earnings_dates DataFrame (index = datetime)
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            for idx in ed.index[:12]:  # cap to last 12 points
                try:
                    ts = int(pd.Timestamp(idx).timestamp())
                    dates.add(ts)
                except Exception:
                    pass
    except Exception:
        pass

    # 2. Secondary: info dict (already cached)
    if not dates:
        info = fetch_info(ticker, data_dir)
        for key in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"):
            ts = info.get(key)
            if ts:
                try:
                    dates.add(int(ts))
                except (TypeError, ValueError):
                    pass

    # 3. Fallback: quarter-end + ~45 days (rough announcement timing)
    if not dates:
        try:
            qf = fetch_quarterly_financials(ticker, data_dir)
            if not qf.empty:
                for col in list(qf.columns)[:5]:
                    try:
                        d = pd.Timestamp(col)
                        est = d + pd.Timedelta(days=45)
                        dates.add(int(est.timestamp()))
                    except Exception:
                        pass
        except Exception:
            pass

    result = sorted(dates)
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f)
    except Exception:
        pass
    return result


def fetch_fundamental_extras(ticker: str, data_dir: str = "data") -> dict:
    """
    Return supplementary fundamental fields for enhanced classification and P/S ratio.
    All values come from the cached .info dict — zero extra API calls.

    Keys returned (any may be None):
        beta              — market beta
        revenue_growth    — YoY revenue growth rate (0.15 = 15%)
        dividend_yield    — trailing dividend yield (0.025 = 2.5%)
        ps_ratio          — trailing price-to-sales ratio
        market_cap        — market cap in USD
    """
    info = fetch_info(ticker, data_dir)
    result: dict = {
        "beta": None,
        "revenue_growth": None,
        "dividend_yield": None,
        "ps_ratio": None,
        "market_cap": None,
    }

    beta = info.get("beta")
    if beta is not None:
        try:
            result["beta"] = float(beta)
        except (TypeError, ValueError):
            pass

    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        try:
            result["revenue_growth"] = float(rev_growth)
        except (TypeError, ValueError):
            pass

    # Dividend yield: yfinance changed `dividendYield` unit in late 2024
    # from decimal (0.005 = 0.5%) to percent (0.5 = 0.5%), which caused
    # AAPL to display as 39%, MSFT as 87% etc. in the legacy code.
    #
    # Strategy:
    #   1. Prefer `trailingAnnualDividendYield` — its unit (decimal) has never changed.
    #   2. Fall back to `dividendYield` and assume the current percent format
    #      (divide by 100). Users on very old yfinance may get underestimated
    #      yields, but no false giant yields that break classification.
    #   3. Sanity-cap at 50% regardless (real yields almost never exceed 20%).
    div_decimal: float | None = None
    ty_yield = info.get("trailingAnnualDividendYield")
    if ty_yield is not None:
        try:
            val = float(ty_yield)
            if 0 <= val <= 0.5:
                div_decimal = val
        except (TypeError, ValueError):
            pass
    if div_decimal is None:
        # Fall back to dividendYield assuming current yfinance percent format
        dy = info.get("dividendYield")
        if dy is not None:
            try:
                val = float(dy) / 100.0
                if 0 <= val <= 0.5:
                    div_decimal = val
            except (TypeError, ValueError):
                pass
    if div_decimal is not None:
        result["dividend_yield"] = div_decimal

    # Try both common key names across yfinance versions
    ps = info.get("priceToSalesTrailing12Months") or info.get("priceToSales")
    if ps is not None:
        try:
            val = float(ps)
            if val > 0:
                result["ps_ratio"] = val
        except (TypeError, ValueError):
            pass

    mc = info.get("marketCap")
    if mc is not None:
        try:
            val = float(mc)
            if val > 0:
                result["market_cap"] = val
        except (TypeError, ValueError):
            pass

    return result
