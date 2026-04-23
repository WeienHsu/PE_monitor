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
    # growth — P/S replaces P/CF (growth companies reinvest → OCF depressed)
    #          Revenue growth added as growth-thesis confirmatory factor
    "growth": {
        "factors": {"ps", "peg", "forward_pe", "revenue_growth", "strategy_d",
                    "volume", "position_52w"},
        "weights": {"ps": 1, "peg": 2, "forward_pe": 2, "revenue_growth": 1,
                    "strategy_d": 1, "volume": 1, "position_52w": 1},
        "cap": 2,
    },
    # stable — dividend yield added as cash-return quality signal
    "stable": {
        "factors": {"pcf", "peg", "forward_pe", "dividend_yield", "strategy_d",
                    "volume", "position_52w"},
        "weights": {"pcf": 2, "peg": 1, "forward_pe": 1, "dividend_yield": 1,
                    "strategy_d": 1, "volume": 1, "position_52w": 1},
        "cap": 2,
    },
    # cyclical — PEG/Forward P/E are unreliable at cycle troughs/peaks;
    #            EV/EBITDA is more cycle-neutral; conservative ±1 cap
    "cyclical": {
        "factors": {"pcf", "ev_ebitda", "strategy_d", "volume", "position_52w"},
        "weights": {"pcf": 1, "ev_ebitda": 1, "strategy_d": 1,
                    "volume": 1, "position_52w": 1},
        "cap": 1,
    },
    # etf — only forward P/E reflects the blended consensus valuation
    "etf": {
        "factors": {"forward_pe", "volume", "position_52w"},
        "weights": {"forward_pe": 1, "volume": 1, "position_52w": 1},
        "cap": 1,
    },
    # unknown — legacy behaviour: all factors, equal weight, ±1 cap
    "unknown": {
        "factors": {"pcf", "peg", "forward_pe", "strategy_d",
                    "volume", "position_52w"},
        "weights": {"pcf": 1, "peg": 1, "forward_pe": 1, "strategy_d": 1,
                    "volume": 1, "position_52w": 1},
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
    ps: float | None = None,
    revenue_growth: float | None = None,
    dividend_yield: float | None = None,
    market_regime: str | None = None,
    value_trap_severity: int = 0,
    volume_vote: int | None = None,
    position_52w_vote: int | None = None,
    sentiment_score: float | None = None,
) -> tuple[str, str, dict]:
    """
    Multi-factor composite signal (Plan B: type-adaptive).

    Step 1 — base composite: PE signal × news sentiment (existing matrix).
    Step 2 — type-adaptive supplementary adjustment:
              which factors participate and their weights depend on stock_type.
              Net vote is capped per type (±1 for unknown/cyclical/etf,
              ±2 for growth/stable) then converted to a level shift.

    Voting rules (direction, before weighting)
    -------------------------------------------
    P/CF          : < 10 → +1  |  10–20 → 0  |  > 20 → -1
    PEG           : < 1.0 → +1 |  1–2 → 0    |  > 2.0 → -1
    Forward P/E   : fpe/trailing < 0.90 → +1  |  > 1.10 → -1  |  else 0
    EV/EBITDA     : < 8 → +1   |  8–15 → 0   |  > 15 → -1  (cyclical only)
    Strategy D    : True → +1  |  False/None → 0
    P/S           : < 4 → +1   |  4–20 → 0   |  > 20 → -1  (growth only)
    Revenue growth: > 15% → +1 |  5–15% → 0  |  < 5% → -1  (growth only)
    Dividend yield: ≥ 2% → +1  |  0.5–2% → 0 |  < 0.5% → -1  (stable only)
    Volume (5d/20d): > 1.5 → +1 | 0.5–1.5 → 0 | < 0.5 → -1 (P2-11 confirm)
    52w position    : ≥ 0.95 → +1 | 0.7–0.95 → 0 | < 0.7 → -1 (P2-11 trend)

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
    ps             : P/S ratio (or None; relevant for growth)
    revenue_growth : YoY revenue growth rate as decimal (or None; growth only)
    dividend_yield : trailing dividend yield as decimal (or None; stable only)
    market_regime  : 'RISK_ON' | 'NEUTRAL' | 'RISK_OFF' | 'UNKNOWN' | None
                     When 'RISK_OFF', demote BUY-side composites by 1 level
                     (STRONG_BUY→BUY, BUY→CAUTIOUS_BUY, CAUTIOUS_BUY→WATCH,
                      WATCH→NEUTRAL). Asymmetric on purpose — we never promote
                     in RISK_ON to avoid FOMO at tops.
    value_trap_severity : 0..4 count of value-trap flags from
                          value_trap_filter.check_value_trap(). When >= 2 and
                          the resulting signal is BUY-class, force the signal
                          down to WATCH (overrides everything else). Designed
                          to stop the system from chasing falling-knife cheap
                          stocks with deteriorating fundamentals.
    sentiment_score    : raw aggregated news-sentiment score in [-1, 1]
                          (from analyze_sentiment).  When < -0.6 AND the
                          resulting signal is in CAUTION / NEUTRAL, demote
                          an extra level (P3-16). Deliberately asymmetric —
                          we never promote on extreme positive sentiment to
                          avoid amplifying FOMO at tops.

    Returns
    -------
    (composite_key, display_string, factor_details)
    factor_details: dict[factor_name → int]  raw vote +1/0/-1 per factor
                    (factors not in the type config are omitted entirely)
                    When regime demotes the signal, an entry
                    "Market Regime" = -1 is recorded.
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

    # P/S ratio (growth stocks; P/S norms vary by sector — conservative thresholds)
    if "ps" in allowed and ps is not None:
        v = 1 if ps < 4 else (-1 if ps > 20 else 0)
        factors["P/S"] = v
        net += v * weights.get("ps", 1)

    # Revenue growth confirmatory vote (growth stocks)
    if "revenue_growth" in allowed and revenue_growth is not None:
        v = 1 if revenue_growth > 0.15 else (-1 if revenue_growth < 0.05 else 0)
        factors["Revenue Growth"] = v
        net += v * weights.get("revenue_growth", 1)

    # Dividend yield (stable stocks)
    if "dividend_yield" in allowed and dividend_yield is not None:
        v = 1 if dividend_yield >= 0.02 else (-1 if dividend_yield < 0.005 else 0)
        factors["Div Yield"] = v
        net += v * weights.get("dividend_yield", 1)

    # Strategy D
    if "strategy_d" in allowed:
        if strategy_d is True:
            factors["Strategy D"] = 1
            net += 1 * weights.get("strategy_d", 1)
        elif strategy_d is False:
            factors["Strategy D"] = 0

    # P2-11: Volume factor (-1 / 0 / +1 from momentum_factors.get_volume_factor)
    if "volume" in allowed and volume_vote is not None:
        v = max(-1, min(1, int(volume_vote)))
        factors["Volume"] = v
        net += v * weights.get("volume", 1)

    # P2-11: 52-week position factor
    if "position_52w" in allowed and position_52w_vote is not None:
        v = max(-1, min(1, int(position_52w_vote)))
        factors["52w Position"] = v
        net += v * weights.get("position_52w", 1)

    def _finalize(signal_key: str, display: str, f: dict) -> tuple[str, str, dict]:
        """Apply regime → value-trap → extreme-sentiment filters in order."""
        k, d, ff = _apply_regime_filter(signal_key, display, f, market_regime)
        k, d, ff = _apply_value_trap_filter(k, d, ff, value_trap_severity)
        k, d, ff = _apply_extreme_sentiment_filter(k, d, ff, sentiment_score)
        return k, d, ff

    # No supplementary data available → start from base
    if not factors:
        return _finalize(base_key, base_display, factors)

    # Cap weighted net, then convert to level shift
    net = max(-cap, min(cap, net))
    if net == 0:
        return _finalize(base_key, base_display, factors)

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
        return _finalize(new_key, new_display, factors)
    except ValueError:
        return _finalize(base_key, base_display, factors)


# ---------------------------------------------------------------------------
# Market-regime post-filter
# ---------------------------------------------------------------------------

# BUY-side keys that get demoted one level in RISK_OFF.
# CAUTIOUS_BUY and WATCH are included — they still imply leaning long.
_RISK_OFF_DEMOTION: dict[str, str] = {
    "STRONG_BUY":   "BUY",
    "BUY":          "CAUTIOUS_BUY",
    "CAUTIOUS_BUY": "WATCH",
    "WATCH":        "NEUTRAL",
}


def _apply_regime_filter(
    signal_key: str,
    display: str,
    factors: dict,
    market_regime: str | None,
) -> tuple[str, str, dict]:
    """
    Post-processing: demote BUY-side signals by 1 level when market_regime == 'RISK_OFF'.

    Intentionally asymmetric — we do not promote in RISK_ON to avoid FOMO at tops.
    Records a "Market Regime" = -1 vote in factors when a demotion occurs.
    """
    if market_regime != "RISK_OFF":
        return signal_key, display, factors
    new_key = _RISK_OFF_DEMOTION.get(signal_key)
    if new_key is None:
        return signal_key, display, factors
    # Record the demotion in factors for transparency
    new_factors = dict(factors)
    new_factors["Market Regime"] = -1
    return new_key, _SIGNAL_DISPLAY.get(new_key, new_key), new_factors


# BUY-class keys that get force-capped to WATCH when value-trap severity >= 2.
_BUY_CLASS = {"STRONG_BUY", "BUY", "CAUTIOUS_BUY"}


def _apply_value_trap_filter(
    signal_key: str,
    display: str,
    factors: dict,
    severity: int,
) -> tuple[str, str, dict]:
    """
    Post-processing: when value-trap severity >= 2, force BUY-class signals
    down to WATCH so the system stops chasing cheap stocks with deteriorating
    fundamentals (falling knives).

    Rationale for capping at WATCH rather than NEUTRAL: the valuation signal
    is still saying "cheap", and we want the ticker to remain on the radar
    for when the fundamentals stabilise — we just refuse to commit capital
    while the thesis is deteriorating.
    """
    if severity < 2:
        return signal_key, display, factors
    if signal_key not in _BUY_CLASS:
        return signal_key, display, factors
    new_factors = dict(factors)
    new_factors["Value Trap"] = -1
    return "WATCH", _SIGNAL_DISPLAY["WATCH"], new_factors


# ---------------------------------------------------------------------------
# P3-16: Extreme negative sentiment — 2-level jump (asymmetric)
# ---------------------------------------------------------------------------

# Threshold for "extreme" negative sentiment.  analyze_sentiment returns
# a time-weighted score in [-1, 1]; -0.6 is strongly negative after decay.
_EXTREME_NEGATIVE_THRESHOLD = -0.6

# Signals where the extra demotion applies.  Intentionally NOT applied to
# BUY-class — those are already aggressively covered by value-trap filter
# and regime filter.  Applied to the middle zone where a neutral valuation
# signal might mask a genuinely bad fundamental situation.
_EXTREME_SENT_APPLIES_TO = {"CAUTION", "NEUTRAL", "WATCH"}


def _apply_extreme_sentiment_filter(
    signal_key: str,
    display: str,
    factors: dict,
    sentiment_score: float | None,
) -> tuple[str, str, dict]:
    """
    Post-processing: when sentiment_score is extremely negative (< -0.6),
    demote CAUTION / NEUTRAL / WATCH by one extra level — the base sentiment
    matrix already applied -1, so the net effect is a 2-level drop.

    Asymmetric by design — we never amplify positive sentiment to avoid
    creating FOMO signals at market tops.
    """
    if sentiment_score is None or sentiment_score >= _EXTREME_NEGATIVE_THRESHOLD:
        return signal_key, display, factors
    if signal_key not in _EXTREME_SENT_APPLIES_TO:
        return signal_key, display, factors
    try:
        idx = _SIGNAL_ORDER.index(signal_key)
    except ValueError:
        return signal_key, display, factors
    new_idx = max(0, idx - 1)
    new_key = _SIGNAL_ORDER[new_idx]
    if new_key == signal_key:
        return signal_key, display, factors
    new_factors = dict(factors)
    new_factors["Extreme Sentiment"] = -1
    return new_key, _SIGNAL_DISPLAY.get(new_key, new_key), new_factors
