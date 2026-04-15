"""
Composite signal: combines the PE-based signal with news sentiment.

Design principle: news sentiment shifts the PE signal by at most one level.
Valuation (PE/PB percentile) is the primary signal; news is the modifier.

Matrix (PE signal × news sentiment → composite):

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
