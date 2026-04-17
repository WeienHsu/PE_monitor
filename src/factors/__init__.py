"""QVM three-factor model (Quality / Value / Momentum).

Public entry points:
    compute_v_score  — multi-ratio valuation score (0-100, higher = cheaper)
    compute_q_score  — profitability & balance-sheet quality (0-100)
    compute_m_score  — 12-1 momentum + Strategy D bonus (0-100)
    compute_qvm      — combine via type-weighted sum, apply gates, map to signal
"""

from src.factors.value_factor import compute_v_score
from src.factors.quality_factor import compute_q_score
from src.factors.momentum_factor import compute_m_score
from src.factors.qvm_composite import compute_qvm

__all__ = [
    "compute_v_score",
    "compute_q_score",
    "compute_m_score",
    "compute_qvm",
]
