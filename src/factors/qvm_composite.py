"""
QVM composite: combine V/Q/M scores into a 5-level signal, apply quality gate
and trend filter, then overlay news sentiment (via composite_signal matrix).

Pipeline
--------
  QVM_raw = w_V · V + w_Q · Q + w_M · M          # weights per stock_type

  Quality Gate (for non-ETF equities only):
      if OCF ≤ 0 or TTM EPS ≤ 0 → cap base signal at WATCH (no BUY/STRONG_BUY)

  QVM_raw → base signal:
      > 75      BUY
      65–75     WATCH
      35–65     NEUTRAL
      25–35     CAUTION
      < 25      SELL

  Trend Filter:
      if price < SMA200 × 0.85 and base in {BUY, STRONG_BUY}
          → downgrade to WATCH

  Final composite:
      base × news sentiment  (reuse composite_signal._MATRIX)
"""

from typing import Optional

from src.composite_signal import compute_composite


# ---------------------------------------------------------------------------
# Type-weight table (V, Q, M) — must sum to 1.0
# ---------------------------------------------------------------------------

QVM_WEIGHTS: dict[str, dict[str, float]] = {
    "stable":         {"V": 0.40, "Q": 0.35, "M": 0.25},
    "growth":         {"V": 0.30, "Q": 0.35, "M": 0.35},
    "cyclical":       {"V": 0.30, "Q": 0.20, "M": 0.50},
    "etf_broad":      {"V": 0.50, "Q": 0.15, "M": 0.35},
    "etf_sector":     {"V": 0.45, "Q": 0.15, "M": 0.40},
    "etf_dividend":   {"V": 0.50, "Q": 0.10, "M": 0.40},
    "etf_commodity":  {"V": 0.25, "Q": 0.00, "M": 0.75},
    "etf_bond":       {"V": 0.35, "Q": 0.00, "M": 0.65},
    # Legacy ETF type without subtype → fall through to etf_broad
    "etf":            {"V": 0.50, "Q": 0.15, "M": 0.35},
    "unknown":        {"V": 0.35, "Q": 0.30, "M": 0.35},
}


def _get_weights(stock_type: str, etf_subtype: Optional[str]) -> dict[str, float]:
    if etf_subtype and f"etf_{etf_subtype}" in QVM_WEIGHTS:
        return QVM_WEIGHTS[f"etf_{etf_subtype}"]
    return QVM_WEIGHTS.get(stock_type, QVM_WEIGHTS["unknown"])


# ---------------------------------------------------------------------------
# QVM → base signal mapping
# ---------------------------------------------------------------------------

# Thresholds on QVM_raw (will be retuned in Phase 4 backtest)
BUY_CUT: float = 75.0
WATCH_CUT: float = 65.0
CAUTION_CUT: float = 35.0
SELL_CUT: float = 25.0


def _qvm_to_signal(qvm: float) -> str:
    if qvm > BUY_CUT:
        return "BUY"
    if qvm > WATCH_CUT:
        return "WATCH"
    if qvm >= CAUTION_CUT:
        return "NEUTRAL"
    if qvm >= SELL_CUT:
        return "CAUTION"
    return "SELL"


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def _apply_quality_gate(
    base: str,
    operating_cashflow: Optional[float],
    ttm_eps: Optional[float],
    is_etf: bool,
) -> tuple[str, bool]:
    """Cap base signal at WATCH when quality fails. ETFs skip the gate.

    Returns (possibly-downgraded signal, gate_triggered_bool).
    """
    if is_etf:
        return base, False
    fails_ocf = operating_cashflow is not None and operating_cashflow <= 0
    fails_eps = ttm_eps is not None and ttm_eps <= 0
    if fails_ocf or fails_eps:
        if base == "BUY":
            return "WATCH", True
        # Already at/below WATCH — no change
        return base, True
    return base, False


# ---------------------------------------------------------------------------
# Trend filter
# ---------------------------------------------------------------------------

def _apply_trend_filter(
    base: str,
    price: Optional[float],
    sma200: Optional[float],
    drop_ratio: float = 0.85,
) -> tuple[str, bool]:
    """If price < SMA200 × drop_ratio and base is BUY, downgrade to WATCH.

    Returns (possibly-downgraded signal, trend_triggered_bool).
    """
    if price is None or sma200 is None or sma200 <= 0:
        return base, False
    if base == "BUY" and price < sma200 * drop_ratio:
        return "WATCH", True
    return base, False


# ---------------------------------------------------------------------------
# Position sizing suggestion
# ---------------------------------------------------------------------------

_POSITION_MAP: list[tuple[float, str]] = [
    (BUY_CUT,    "可加碼 10%"),
    (WATCH_CUT,  "可加碼 5%"),
    (CAUTION_CUT, "維持現有倉位"),
    (SELL_CUT,   "建議減倉 5%"),
    (-1.0,       "建議減倉 10%"),
]


def compute_position_suggestion(qvm_raw: Optional[float]) -> str:
    """Return a plain-text position-sizing suggestion based on QVM composite score."""
    if qvm_raw is None:
        return "N/A"
    for threshold, label in _POSITION_MAP:
        if qvm_raw > threshold:
            return label
    return "建議減倉 10%"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_qvm(
    *,
    v_score: Optional[float],
    q_score: Optional[float],
    m_score: Optional[float],
    stock_type: str = "unknown",
    etf_subtype: Optional[str] = None,
    sentiment_label: str = "neutral",
    operating_cashflow: Optional[float] = None,
    ttm_eps: Optional[float] = None,
    price: Optional[float] = None,
    sma200: Optional[float] = None,
    is_etf: bool = False,
) -> dict:
    """Run the full QVM pipeline and return a result dict.

    Returned dict:
        qvm_raw            : float 0-100 (None if all factor scores are None)
        base_signal        : BUY / WATCH / NEUTRAL / CAUTION / SELL  (after gates)
        composite_signal   : key after sentiment overlay (e.g. STRONG_BUY)
        composite_display  : emoji + label string
        weights            : {V, Q, M} used
        gates              : {quality: bool, trend: bool}
        component_signal   : base signal BEFORE gates (for transparency)
    """
    weights = _get_weights(stock_type, etf_subtype)

    # Weighted sum over AVAILABLE scores, renormalising weights on the fly
    # so that missing factors don't get implicit 0.
    pairs = [
        ("V", v_score, weights["V"]),
        ("Q", q_score, weights["Q"]),
        ("M", m_score, weights["M"]),
    ]
    usable = [(s, w) for _, s, w in pairs if s is not None and w > 0]
    if not usable:
        return {
            "qvm_raw": None,
            "base_signal": "N/A",
            "composite_signal": "N/A",
            "composite_display": "N/A",
            "weights": weights,
            "gates": {"quality": False, "trend": False},
            "component_signal": "N/A",
        }

    total_w = sum(w for _, w in usable)
    qvm_raw = sum(s * w for s, w in usable) / total_w
    qvm_raw = round(qvm_raw, 2)

    pre_gate_signal = _qvm_to_signal(qvm_raw)

    # Quality gate
    post_quality, quality_triggered = _apply_quality_gate(
        pre_gate_signal, operating_cashflow, ttm_eps, is_etf
    )

    # Trend filter
    post_trend, trend_triggered = _apply_trend_filter(post_quality, price, sma200)

    base_signal = post_trend

    # News sentiment overlay
    composite_key, composite_display = compute_composite(base_signal, sentiment_label)

    return {
        "qvm_raw": qvm_raw,
        "base_signal": base_signal,
        "composite_signal": composite_key,
        "composite_display": composite_display,
        "weights": weights,
        "gates": {
            "quality": quality_triggered,
            "trend": trend_triggered,
        },
        "component_signal": pre_gate_signal,
        "position_suggestion": compute_position_suggestion(qvm_raw),
    }
