"""
Composite signal: combines the PE-based signal with news sentiment and
supplementary valuation indicators (P/CF, PEG, Forward P/E, Strategy D).

Design principles:
  1. Valuation (PE/PB percentile) is the primary signal.
  2. News sentiment shifts the PE signal by at most one level (matrix lookup).
  3. Supplementary indicators (P/CF, PEG, Forward P/E, Strategy D) add a
     second-pass adjustment of at most ±1 level so they cannot dominate.
  4. Missing / unavailable indicators are ignored (no forced penalty).

Matrix (PE signal × news sentiment → base composite):

                positive    neutral     negative
  BUY           STRONG_BUY  BUY         CAUTIOUS_BUY
  WATCH         BUY         WATCH       NEUTRAL
  NEUTRAL       WATCH       NEUTRAL     CAUTION
  CAUTION       NEUTRAL     CAUTION     SELL
  SELL          CAUTIOUS_SELL SELL      STRONG_SELL
"""

# ---------------------------------------------------------------------------
# Composite lookup matrix
# ---------------------------------------------------------------------------

_MATRIX: dict[tuple[str, str], tuple[str, str]] = {
    ("BUY",     "positive"): ("STRONG_BUY",    "🌟 強力買進"),
    ("BUY",     "neutral"):  ("BUY",            "🟢 買進"),
    ("BUY",     "negative"): ("CAUTIOUS_BUY",   "🟢⚠️ 謹慎買進"),
    ("WATCH",   "positive"): ("BUY",            "🟢 買進"),
    ("WATCH",   "neutral"):  ("WATCH",          "🔵 觀察"),
    ("WATCH",   "negative"): ("NEUTRAL",        "⚪ 中性"),
    ("NEUTRAL", "positive"): ("WATCH",          "🔵 觀察"),
    ("NEUTRAL", "neutral"):  ("NEUTRAL",        "⚪ 中性"),
    ("NEUTRAL", "negative"): ("CAUTION",        "🟡 謹慎"),
    ("CAUTION", "positive"): ("NEUTRAL",        "⚪ 中性"),
    ("CAUTION", "neutral"):  ("CAUTION",        "🟡 謹慎"),
    ("CAUTION", "negative"): ("SELL",           "🔴 賣出"),
    ("SELL",    "positive"): ("CAUTIOUS_SELL",  "🔴⚠️ 謹慎賣出"),
    ("SELL",    "neutral"):  ("SELL",           "🔴 賣出"),
    ("SELL",    "negative"): ("STRONG_SELL",    "🚨 強力賣出"),
}

# Colour mapping for app.py's color_signal()
COMPOSITE_COLORS: dict[str, str] = {
    "STRONG_BUY":    "background-color: #a8e6a3",  # darker green
    "BUY":           "background-color: #d4edda",
    "CAUTIOUS_BUY":  "background-color: #d4edda",
    "WATCH":         "background-color: #cce5ff",
    "NEUTRAL":       "",
    "CAUTION":       "background-color: #fff3cd",
    "SELL":          "background-color: #f8d7da",
    "CAUTIOUS_SELL": "background-color: #f8d7da",
    "STRONG_SELL":   "background-color: #f5a0a8",  # darker red
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Signal ordering (weakest → strongest) used for level-shifting
# ---------------------------------------------------------------------------

_SIGNAL_ORDER: list[str] = [
    "STRONG_SELL",
    "SELL",
    "CAUTIOUS_SELL",
    "CAUTION",
    "NEUTRAL",
    "WATCH",
    "CAUTIOUS_BUY",
    "BUY",
    "STRONG_BUY",
]

_SIGNAL_DISPLAY: dict[str, str] = {
    "STRONG_BUY":    "🌟 強力買進",
    "BUY":           "🟢 買進",
    "CAUTIOUS_BUY":  "🟢⚠️ 謹慎買進",
    "WATCH":         "🔵 觀察",
    "NEUTRAL":       "⚪ 中性",
    "CAUTION":       "🟡 謹慎",
    "CAUTIOUS_SELL": "🔴⚠️ 謹慎賣出",
    "SELL":          "🔴 賣出",
    "STRONG_SELL":   "🚨 強力賣出",
}


def compute_composite(pe_signal: str, sentiment_label: str) -> tuple[str, str]:
    """
    Map (pe_signal, sentiment_label) → (composite_key, display_string).

    Parameters
    ----------
    pe_signal       : one of BUY / WATCH / NEUTRAL / CAUTION / SELL
    sentiment_label : one of "positive" / "neutral" / "negative"

    Returns
    -------
    (composite_key, display_string)

    Falls back to the PE signal unchanged if either argument is unrecognised.
    """
    key = (pe_signal, sentiment_label)
    if key in _MATRIX:
        return _MATRIX[key]

    # Graceful fallback — preserve the original PE signal display
    _PE_DISPLAY = {
        "BUY":     "🟢 BUY ZONE",
        "WATCH":   "🔵 WATCH ZONE",
        "NEUTRAL": "⚪ NEUTRAL",
        "CAUTION": "🟡 CAUTION",
        "SELL":    "🔴 SELL ZONE",
    }
    return pe_signal, _PE_DISPLAY.get(pe_signal, pe_signal)


def composite_color(composite_key: str) -> str:
    """Return the CSS background-color style for a composite signal key."""
    return COMPOSITE_COLORS.get(composite_key, "")


# ---------------------------------------------------------------------------
# Multi-factor composite (PE × sentiment + supplementary indicators)
# ---------------------------------------------------------------------------

def compute_multi_factor_composite(
    pe_signal: str,
    sentiment_label: str,
    *,
    pcf: float | None = None,
    peg: float | None = None,
    forward_pe: float | None = None,
    trailing_pe: float | None = None,
    strategy_d: bool | None = None,
) -> tuple[str, str, dict]:
    """
    Multi-factor composite signal.

    Step 1 — base composite: PE signal × news sentiment (existing matrix).
    Step 2 — supplementary adjustment: count bullish (+1) / bearish (-1) votes
              from available indicators; cap net at ±1; shift signal one level.

    Indicator voting rules
    ----------------------
    P/CF       : < 10 → +1 (cheap)       | > 20 → -1 (expensive)
    PEG        : < 1.0 → +1 (undervalued)| > 2.0 → -1 (overvalued)
    Forward P/E: forward < 0.90 × trailing → +1 (earnings growth expected)
                 forward > 1.10 × trailing → -1 (earnings decline expected)
    Strategy D : active (True) → +1 (technical momentum turning)

    Parameters
    ----------
    pe_signal      : BUY / WATCH / NEUTRAL / CAUTION / SELL
    sentiment_label: positive / neutral / negative
    pcf            : P/CF ratio (or None)
    peg            : PEG ratio (or None)
    forward_pe     : forward 12-month P/E (or None)
    trailing_pe    : current trailing P/E used only as reference for forward_pe
    strategy_d     : True = signal active, False = no signal, None = disabled

    Returns
    -------
    (composite_key, display_string, factor_details)
    factor_details: dict[factor_name → int]  (+1 / 0 / -1 per factor)
    """
    # Step 1 — base
    base_key, base_display = compute_composite(pe_signal, sentiment_label)

    # Step 2 — supplementary votes
    factors: dict[str, int] = {}
    net = 0

    if pcf is not None:
        if pcf < 10:
            factors["P/CF"] = 1
            net += 1
        elif pcf > 20:
            factors["P/CF"] = -1
            net -= 1
        else:
            factors["P/CF"] = 0

    if peg is not None:
        if peg < 1.0:
            factors["PEG"] = 1
            net += 1
        elif peg > 2.0:
            factors["PEG"] = -1
            net -= 1
        else:
            factors["PEG"] = 0

    if forward_pe is not None and trailing_pe is not None and trailing_pe > 0:
        ratio = forward_pe / trailing_pe
        if ratio < 0.90:
            factors["Forward P/E"] = 1
            net += 1
        elif ratio > 1.10:
            factors["Forward P/E"] = -1
            net -= 1
        else:
            factors["Forward P/E"] = 0

    if strategy_d is True:
        factors["Strategy D"] = 1
        net += 1
    elif strategy_d is False:
        factors["Strategy D"] = 0

    # No supplementary data available → return base unchanged
    if not factors:
        return base_key, base_display, factors

    # Cap net adjustment to ±1 (supplementary cannot override primary signal)
    net = max(-1, min(1, net))

    if net == 0:
        return base_key, base_display, factors

    # Shift signal one level up or down
    try:
        idx = _SIGNAL_ORDER.index(base_key)
        new_idx = max(0, min(len(_SIGNAL_ORDER) - 1, idx + net))
        new_key = _SIGNAL_ORDER[new_idx]
        new_display = _SIGNAL_DISPLAY.get(new_key, new_key)
        return new_key, new_display, factors
    except ValueError:
        return base_key, base_display, factors
