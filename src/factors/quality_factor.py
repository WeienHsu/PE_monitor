"""
Quality factor (Q): profitability & balance-sheet quality score on 0-100.

Absolute thresholds (literature-informed; Phase 4 will retune per sector):

  Gross Margin    : ≥ 40% → 100 ; ≤ 10% → 0   (Novy-Marx 2013)
  ROE             : ≥ 20% → 100 ; ≤  5% → 0
  Operating Margin: ≥ 20% → 100 ; ≤  0% → 0
  EPS Stability   : std(YoY growth) ≤ 30% → 100 ; ≥ 60% → 0  (inverse)
  Debt/Equity     : ≤ 50  → 100 ; ≥ 200 → 0   (inverse; yfinance scale 0-∞)

Missing fields are skipped (not penalised). Q_score = mean of available.
"""

from dataclasses import dataclass
from typing import Optional


def _linear(value: Optional[float], good: float, bad: float) -> Optional[float]:
    """Linear interpolation from bad → good, clamped to [0, 100].

    If good > bad: higher value is better (score increases with value).
    If good < bad: lower value is better (score decreases with value).
    """
    if value is None:
        return None
    if good > bad:
        if value >= good:
            return 100.0
        if value <= bad:
            return 0.0
        return float(100.0 * (value - bad) / (good - bad))
    # good < bad: reversed
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return float(100.0 * (bad - value) / (bad - good))


@dataclass
class QualityInputs:
    gross_margin: Optional[float] = None        # 0.0-1.0 (yfinance fraction)
    return_on_equity: Optional[float] = None    # 0.0-1.0
    operating_margin: Optional[float] = None    # 0.0-1.0
    eps_growth_std_pct: Optional[float] = None  # % (e.g. 45.0 means 45%)
    debt_to_equity: Optional[float] = None      # yfinance scale (50 = 50%)


def compute_q_score(inputs: QualityInputs) -> tuple[Optional[float], dict]:
    """Compute the Q score (0-100, higher = better quality).

    Returns (q_score, details). q_score is None if no inputs are usable.
    """
    components: dict[str, dict] = {}

    # Gross margin — Novy-Marx gross profitability anchor
    s = _linear(inputs.gross_margin, good=0.40, bad=0.10)
    if s is not None:
        components["Gross Margin"] = {
            "raw": inputs.gross_margin,
            "score": s,
        }

    # ROE
    s = _linear(inputs.return_on_equity, good=0.20, bad=0.05)
    if s is not None:
        components["ROE"] = {
            "raw": inputs.return_on_equity,
            "score": s,
        }

    # Operating margin
    s = _linear(inputs.operating_margin, good=0.20, bad=0.0)
    if s is not None:
        components["Operating Margin"] = {
            "raw": inputs.operating_margin,
            "score": s,
        }

    # EPS stability — lower std is better (good=30, bad=60)
    s = _linear(inputs.eps_growth_std_pct, good=30.0, bad=60.0)
    if s is not None:
        components["EPS Stability"] = {
            "raw": inputs.eps_growth_std_pct,
            "score": s,
        }

    # Debt/Equity — lower is better
    s = _linear(inputs.debt_to_equity, good=50.0, bad=200.0)
    if s is not None:
        components["Debt/Equity"] = {
            "raw": inputs.debt_to_equity,
            "score": s,
        }

    if not components:
        return None, {"components": {}, "score": None}

    scores = [c["score"] for c in components.values()]
    q_score = sum(scores) / len(scores)
    return round(q_score, 2), {
        "components": components,
        "score": round(q_score, 2),
    }
