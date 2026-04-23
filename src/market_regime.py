"""
Market regime filter — classifies the broad US-equity environment into
RISK_ON / NEUTRAL / RISK_OFF so individual-stock signals can be demoted
when the macro tape is hostile.

Inputs (via yfinance, cached through fetch_price_history):
    ^VIX  — CBOE Volatility Index (spot)
    SPY   — S&P 500 ETF (200-day MA trend filter)

Rules (conservative; only demotes BUY-side, never promotes):
    RISK_OFF : VIX > 30  OR  (SPY < 200DMA AND 20-day return < -5%)
    NEUTRAL  : VIX 20–30  OR  SPY within ±3% of 200DMA
    RISK_ON  : VIX < 20  AND  SPY > 200DMA

Design notes
------------
* We deliberately do NOT upgrade BUY signals in RISK_ON to avoid FOMO —
  risk appetite is asymmetric in drawdowns vs rallies.
* Cached per-day (24 h) to avoid re-fetching ^VIX on every scan; the
  regime is a slow-moving context signal.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data_fetcher import fetch_price_history


_CACHE_FILE = "market_regime.json"
_CACHE_MAX_AGE_HOURS = 24


def _cache_path(data_dir: str) -> Path:
    return Path(data_dir) / _CACHE_FILE


def _is_stale(path: Path, max_age_hours: int = _CACHE_MAX_AGE_HOURS) -> bool:
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(hours=max_age_hours)


def _latest_close(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _spy_vs_200ma(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return (current_price, 200-day MA). Either may be None."""
    if df is None or df.empty or "Close" not in df.columns:
        return None, None
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(s) < 200:
        # Not enough history for a 200DMA; return current price only
        return (float(s.iloc[-1]) if not s.empty else None), None
    ma200 = float(s.tail(200).mean())
    return float(s.iloc[-1]), ma200


def _spy_20d_return(df: pd.DataFrame) -> float | None:
    """Return 20-trading-day return as a decimal (0.05 = +5%)."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(s) < 21:
        return None
    return float(s.iloc[-1] / s.iloc[-21] - 1.0)


def get_market_regime(data_dir: str = "data", *, force_refresh: bool = False) -> dict:
    """
    Classify current market regime.

    Returns
    -------
    {
        'regime':      'RISK_ON' | 'NEUTRAL' | 'RISK_OFF' | 'UNKNOWN',
        'vix':         float | None,
        'spy':         float | None,
        'spy_200ma':   float | None,
        'spy_vs_200ma': float | None,   # (spy - 200ma) / 200ma as decimal
        'spy_20d_ret': float | None,    # 20-day trailing return as decimal
        'reasons':     list[str],       # human-readable rule triggers
        'asof':        str,             # ISO timestamp the regime was computed
    }

    Falls back to 'UNKNOWN' only if both VIX and SPY fetches fail.
    """
    cache = _cache_path(data_dir)
    if not force_refresh and not _is_stale(cache):
        try:
            with open(cache, "r") as f:
                return json.load(f)
        except Exception:
            pass

    result: dict = {
        "regime": "UNKNOWN",
        "vix": None,
        "spy": None,
        "spy_200ma": None,
        "spy_vs_200ma": None,
        "spy_20d_ret": None,
        "reasons": [],
        "asof": datetime.now().isoformat(timespec="seconds"),
    }

    # Fetch VIX (1-year history is plenty)
    try:
        vix_df = fetch_price_history("^VIX", years=1, data_dir=data_dir)
        vix_now = _latest_close(vix_df)
        if vix_now is not None:
            result["vix"] = round(vix_now, 2)
    except Exception as e:
        print(f"[market_regime] ^VIX fetch failed: {e}")

    # Fetch SPY (2 years so 200DMA is always available)
    try:
        spy_df = fetch_price_history("SPY", years=2, data_dir=data_dir)
        spy_now, spy_200ma = _spy_vs_200ma(spy_df)
        spy_20d = _spy_20d_return(spy_df)
        if spy_now is not None:
            result["spy"] = round(spy_now, 2)
        if spy_200ma is not None:
            result["spy_200ma"] = round(spy_200ma, 2)
            if spy_now is not None and spy_200ma > 0:
                result["spy_vs_200ma"] = round((spy_now - spy_200ma) / spy_200ma, 4)
        if spy_20d is not None:
            result["spy_20d_ret"] = round(spy_20d, 4)
    except Exception as e:
        print(f"[market_regime] SPY fetch failed: {e}")

    # --- Classify ---
    vix = result["vix"]
    spy_delta = result["spy_vs_200ma"]   # None if 200DMA unavailable
    spy_20d = result["spy_20d_ret"]
    reasons: list[str] = []

    # Collect votes
    risk_off_votes = 0
    risk_on_votes = 0

    if vix is not None:
        if vix > 30:
            risk_off_votes += 1
            reasons.append(f"VIX={vix:.1f} > 30（恐慌）")
        elif vix < 20:
            risk_on_votes += 1
            reasons.append(f"VIX={vix:.1f} < 20（平靜）")
        else:
            reasons.append(f"VIX={vix:.1f}（中性區間）")

    if spy_delta is not None:
        pct = spy_delta * 100
        if spy_delta < -0.03:
            risk_off_votes += 1
            reasons.append(f"SPY 較 200MA 低 {abs(pct):.1f}%")
            if spy_20d is not None and spy_20d < -0.05:
                risk_off_votes += 1  # acute sell-off confirmation
                reasons.append(f"SPY 近 20 日跌 {abs(spy_20d*100):.1f}%（急跌）")
        elif spy_delta > 0.03:
            risk_on_votes += 1
            reasons.append(f"SPY 較 200MA 高 {pct:.1f}%")
        else:
            reasons.append(f"SPY 與 200MA 相差 {pct:+.1f}%（盤整）")

    # Decide regime: RISK_OFF if any strong risk-off signal;
    #                RISK_ON only if both vix<20 and spy>200MA+3%;
    #                otherwise NEUTRAL.
    if vix is None and spy_delta is None:
        result["regime"] = "UNKNOWN"
        result["reasons"] = ["無法取得 VIX 或 SPY 資料"]
    elif risk_off_votes >= 1:
        result["regime"] = "RISK_OFF"
    elif risk_on_votes >= 2:
        result["regime"] = "RISK_ON"
    else:
        result["regime"] = "NEUTRAL"

    result["reasons"] = reasons

    # Write-through cache
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_REGIME_DISPLAY: dict[str, str] = {
    "RISK_ON":  "🟢 RISK ON（風險偏好）",
    "NEUTRAL":  "⚪ NEUTRAL（中性）",
    "RISK_OFF": "🔴 RISK OFF（避險）",
    "UNKNOWN":  "❔ UNKNOWN（資料不足）",
}

_REGIME_COLOR: dict[str, str] = {
    "RISK_ON":  "#d4edda",
    "NEUTRAL":  "#f0f0f0",
    "RISK_OFF": "#f8d7da",
    "UNKNOWN":  "#e9ecef",
}


def regime_display(regime_key: str) -> str:
    return _REGIME_DISPLAY.get(regime_key, regime_key)


def regime_color(regime_key: str) -> str:
    return _REGIME_COLOR.get(regime_key, "")
