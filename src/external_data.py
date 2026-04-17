"""
External reference data for the V factor:
  - Shiller CAPE (S&P 500, 1871-present) from multpl.com
  - Damodaran US industry trailing P/E (annual) from stern.nyu.edu

Both sources are light-touch scrapes with local caching.

Shiller CAPE cache TTL: 30 days (the published series updates monthly)
Damodaran cache TTL: 365 days (Damodaran republishes in January each year)

All network I/O is best-effort. If a fetch fails, the function returns
None/empty and upstream callers degrade gracefully (ETF V just uses the
price-percentile fallback already shipped in Phase 2).
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

SHILLER_URL = "https://www.multpl.com/shiller-pe/table/by-month"
DAMODARAN_URL = (
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/pedata.html"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (PE_monitor research bot)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(data_dir: str, filename: str) -> Path:
    return Path(data_dir) / filename


def _is_stale(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(days=max_age_days)


# ---------------------------------------------------------------------------
# Shiller CAPE
# ---------------------------------------------------------------------------

def fetch_shiller_cape_series(data_dir: str = "data", ttl_days: int = 30) -> pd.Series:
    """Return Shiller P/E (CAPE) monthly series from multpl.com.

    Indexed by month-end date, values are CAPE multiples. Empty on failure.
    """
    cache = _cache_path(data_dir, "_shiller_cape.csv")
    if not _is_stale(cache, max_age_days=ttl_days):
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            if not df.empty:
                return pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        except Exception:
            cache.unlink(missing_ok=True)

    try:
        r = requests.get(SHILLER_URL, timeout=30, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        tbl = soup.find("table")
        if tbl is None:
            return pd.Series(dtype=float)
        rows = tbl.find_all("tr")
        data = []
        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
            if len(cells) < 2:
                continue
            try:
                dt = pd.to_datetime(cells[0])
                val = float(cells[1].replace(",", ""))
                data.append((dt, val))
            except (ValueError, TypeError):
                continue
        if not data:
            return pd.Series(dtype=float)
        s = pd.Series({d: v for d, v in data}, name="CAPE").sort_index()
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        s.to_csv(cache, header=True)
        return s
    except Exception as e:
        print(f"[external_data] fetch_shiller_cape failed: {e}")
        return pd.Series(dtype=float)


def get_shiller_cape_percentile(
    value: Optional[float] = None,
    data_dir: str = "data",
    years: Optional[int] = None,
) -> tuple[Optional[float], Optional[float]]:
    """Return (current_CAPE, percentile_rank_of_current_or_given_value).

    If `value` is None, uses the latest available CAPE. If `years` is given,
    percentile is computed against the trailing `years`-year window instead
    of the full 1871-present history (default: full history).
    """
    s = fetch_shiller_cape_series(data_dir)
    if s.empty:
        return (None, None)
    current = float(s.iloc[-1]) if value is None else float(value)
    if years is not None:
        cutoff = s.index.max() - pd.DateOffset(years=years)
        s = s[s.index >= cutoff]
        if s.empty:
            return (current, None)
    rank = (s < current).sum() / len(s) * 100
    return (current, float(rank))


# ---------------------------------------------------------------------------
# Damodaran industry P/E
# ---------------------------------------------------------------------------

def fetch_damodaran_industry_pe(data_dir: str = "data", ttl_days: int = 365) -> dict[str, dict]:
    """Return dict keyed by industry name with PE sub-fields.

    Sub-fields returned (best-effort, may be missing for some rows):
        current_pe, trailing_pe, forward_pe, peg_ratio, expected_growth
    """
    cache = _cache_path(data_dir, "_damodaran_pe.json")
    if not _is_stale(cache, max_age_days=ttl_days):
        try:
            with open(cache, "r") as f:
                return json.load(f)
        except Exception:
            cache.unlink(missing_ok=True)

    result: dict[str, dict] = {}
    try:
        r = requests.get(DAMODARAN_URL, timeout=30, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        tbl = soup.find("table")
        if tbl is None:
            return result
        rows = tbl.find_all("tr")
        # Header parsing — find indices of the columns we want
        header_cells = [c.get_text(" ", strip=True).lower() for c in rows[0].find_all(["th", "td"])]
        def idx(keyword: str) -> Optional[int]:
            for i, h in enumerate(header_cells):
                if keyword in h:
                    return i
            return None

        name_i = idx("industry")
        trailing_i = idx("trailing pe")
        current_i = idx("current pe")
        forward_i = idx("forward pe")
        peg_i = idx("peg")
        growth_i = idx("expected")

        if name_i is None:
            return result

        def _num(cell: str) -> Optional[float]:
            cell = cell.replace(",", "").replace("$", "").replace("%", "").strip()
            if not cell or cell.upper() in ("NA", "N/A", "-"):
                return None
            try:
                return float(cell)
            except ValueError:
                return None

        for row in rows[1:]:
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if len(cells) <= name_i:
                continue
            industry = cells[name_i].strip()
            if not industry or industry.lower().startswith("total"):
                continue
            row_data = {
                "trailing_pe": _num(cells[trailing_i]) if trailing_i is not None and len(cells) > trailing_i else None,
                "current_pe": _num(cells[current_i]) if current_i is not None and len(cells) > current_i else None,
                "forward_pe": _num(cells[forward_i]) if forward_i is not None and len(cells) > forward_i else None,
                "peg_ratio": _num(cells[peg_i]) if peg_i is not None and len(cells) > peg_i else None,
                "expected_growth": _num(cells[growth_i]) if growth_i is not None and len(cells) > growth_i else None,
            }
            result[industry] = row_data

        if result:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            with open(cache, "w") as f:
                json.dump(result, f, indent=2)
        return result
    except Exception as e:
        print(f"[external_data] fetch_damodaran_industry_pe failed: {e}")
        return result


def get_industry_trailing_pe(industry: str, data_dir: str = "data") -> Optional[float]:
    """Convenience: return trailing P/E for a Damodaran industry name.

    Industry names in the scraped table sometimes contain tab/newline noise
    (e.g. "Software\n\t(Internet)"). We look up against a whitespace-normalised
    key so callers can pass a clean name like "Software (Internet)".
    """
    import re
    data = fetch_damodaran_industry_pe(data_dir)
    if not data:
        return None

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    target = _norm(industry)
    for key, rec in data.items():
        if _norm(key) == target:
            return rec.get("trailing_pe") or rec.get("current_pe")
    return None
