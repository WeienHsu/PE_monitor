"""
Composite signal: combines PE-based signal with news sentiment and
supplementary valuation indicators (P/CF, PEG, Forward P/E, EV/EBITDA,
Strategy D).

Design principles:
  1. Valuation (PE/PB percentile) is the primary signal.
  2. News sentiment shifts the PE signal by at most one level (matrix lookup).
  3. Supplementary indicators add a second-pass type-adaptive adjustment
     (Plan B): which factors participate and by how much depends on
     stock_type.  Cap is ±1 for unknown/cyclical/etf, ±2 for growth/stable.
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


# ---------------------------------------------------------------------------
# Plan B: Type-adaptive factor configuration
# ---------------------------------------------------------------------------

_TYPE_CONFIG: dict[str, dict] = {
    # growth — PEG + Forward P/E are the relevant lenses; P/CF irrelevant
    #          (growth companies reinvest heavily, depressing OCF)
    "growth": {
        "factors": {"peg", "forward_pe", "strategy_d"},
        "weights": {"peg": 2, "forward_pe": 2, "strategy_d": 1},
        "cap": 2,
    },
    # stable — cash flow quality matters most; all factors active
    "stable": {
        "factors": {"pcf", "peg", "forward_pe", "strategy_d"},
        "weights": {"pcf": 2, "peg": 1, "forward_pe": 1, "strategy_d": 1},
        "cap": 2,
    },
    # cyclical — PEG/Forward P/E are unreliable at cycle troughs/peaks;
    #            EV/EBITDA is more cycle-neutral; conservative ±1 cap
    "cyclical": {
        "factors": {"pcf", "ev_ebitda", "strategy_d"},
        "weights": {"pcf": 1, "ev_ebitda": 1, "strategy_d": 1},
        "cap": 1,
    },
    # etf — only forward P/E reflects the blended consensus valuation
    "etf": {
        "factors": {"forward_pe"},
        "weights": {"forward_pe": 1},
        "cap": 1,
    },
    # unknown — legacy behaviour: all factors, equal weight, ±1 cap
    "unknown": {
        "factors": {"pcf", "peg", "forward_pe", "strategy_d"},
        "weights": {"pcf": 1, "peg": 1, "forward_pe": 1, "strategy_d": 1},
        "cap": 1,
    },
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
    stock_type: str = "unknown",
    pcf: float | None = None,
    peg: float | None = None,
    forward_pe: float | None = None,
    trailing_pe: float | None = None,
    strategy_d: bool | None = None,
    ev_ebitda: float | None = None,
) -> tuple[str, str, dict]:
    """
    DEPRECATED — superseded by `src.factors.qvm_composite.compute_qvm`.

    Kept for reference & ad-hoc tooling; not called in the main scan pipeline
    after the QVM refactor. The QVM composite encodes the same intent (base
    signal + factor adjustments + sentiment overlay) with a more robust
    multi-factor valuation input (V) and an explicit quality / trend gate.

    Multi-factor composite signal (Plan B: type-adaptive).

    Step 1 — base composite: PE signal × news sentiment (existing matrix).
    Step 2 — type-adaptive supplementary adjustment:
              which factors participate and their weights depend on stock_type.
              Net vote is capped per type (±1 for unknown/cyclical/etf,
              ±2 for growth/stable) then converted to a level shift.

    Voting rules (direction, before weighting)
    -------------------------------------------
    P/CF       : < 10 → +1  |  10–20 → 0  |  > 20 → -1
    PEG        : < 1.0 → +1 |  1–2 → 0    |  > 2.0 → -1
    Forward P/E: fpe/trailing < 0.90 → +1  |  > 1.10 → -1  |  else 0
    EV/EBITDA  : < 8 → +1   |  8–15 → 0   |  > 15 → -1  (cyclical only)
    Strategy D : True → +1  |  False/None → 0

    Parameters
    ----------
    pe_signal      : BUY / WATCH / NEUTRAL / CAUTION / SELL
    sentiment_label: positive / neutral / negative
    stock_type     : stable / growth / cyclical / etf / unknown
    pcf            : P/CF ratio (or None)
    peg            : PEG ratio (or None)
    forward_pe     : forward 12-month P/E (or None)
    trailing_pe    : trailing P/E used as reference for forward_pe comparison
    strategy_d     : True = signal active, False = no signal, None = disabled
    ev_ebitda      : EV/EBITDA ratio (or None; relevant for cyclical)

    Returns
    -------
    (composite_key, display_string, factor_details)
    factor_details: dict[factor_name → int]  raw vote +1/0/-1 per factor
                    (factors not in the type config are omitted entirely)
    """
    # Step 1 — base
    base_key, base_display = compute_composite(pe_signal, sentiment_label)

    # Step 2 — look up type config; fall back to "unknown" for unrecognised types
    cfg = _TYPE_CONFIG.get(stock_type, _TYPE_CONFIG["unknown"])
    allowed = cfg["factors"]
    weights = cfg["weights"]
    cap = cfg["cap"]

    factors: dict[str, int] = {}
    net = 0

    # P/CF
    if "pcf" in allowed and pcf is not None:
        v = 1 if pcf < 10 else (-1 if pcf > 20 else 0)
        factors["P/CF"] = v
        net += v * weights.get("pcf", 1)

    # PEG
    if "peg" in allowed and peg is not None:
        v = 1 if peg < 1.0 else (-1 if peg > 2.0 else 0)
        factors["PEG"] = v
        net += v * weights.get("peg", 1)

    # Forward P/E
    if "forward_pe" in allowed and forward_pe is not None and trailing_pe is not None and trailing_pe > 0:
        ratio = forward_pe / trailing_pe
        v = 1 if ratio < 0.90 else (-1 if ratio > 1.10 else 0)
        factors["Forward P/E"] = v
        net += v * weights.get("forward_pe", 1)

    # EV/EBITDA
    if "ev_ebitda" in allowed and ev_ebitda is not None:
        v = 1 if ev_ebitda < 8 else (-1 if ev_ebitda > 15 else 0)
        factors["EV/EBITDA"] = v
        net += v * weights.get("ev_ebitda", 1)

    # Strategy D
    if "strategy_d" in allowed:
        if strategy_d is True:
            factors["Strategy D"] = 1
            net += 1 * weights.get("strategy_d", 1)
        elif strategy_d is False:
            factors["Strategy D"] = 0

    # No supplementary data available → return base unchanged
    if not factors:
        return base_key, base_display, factors

    # Cap weighted net, then convert to level shift
    net = max(-cap, min(cap, net))
    if net == 0:
        return base_key, base_display, factors

    # Determine level shift: ±1 always; ±2 if cap allows and |net| ≥ 2
    if net >= 2:
        level_shift = 2
    elif net >= 1:
        level_shift = 1
    elif net <= -2:
        level_shift = -2
    else:
        level_shift = -1

    try:
        idx = _SIGNAL_ORDER.index(base_key)
        new_idx = max(0, min(len(_SIGNAL_ORDER) - 1, idx + level_shift))
        new_key = _SIGNAL_ORDER[new_idx]
        new_display = _SIGNAL_DISPLAY.get(new_key, new_key)
        return new_key, new_display, factors
    except ValueError:
        return base_key, base_display, factors
