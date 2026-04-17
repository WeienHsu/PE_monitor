"""Unit tests for the Q-factor scoring logic (no network)."""

import pytest

from src.factors.quality_factor import QualityInputs, _linear, compute_q_score


def test_linear_positive_direction():
    # good=20 (high is better), bad=5
    assert _linear(25, good=20, bad=5) == 100.0
    assert _linear(20, good=20, bad=5) == 100.0
    assert _linear(5, good=20, bad=5) == 0.0
    assert _linear(0, good=20, bad=5) == 0.0
    assert _linear(12.5, good=20, bad=5) == pytest.approx(50.0)


def test_linear_reversed_direction():
    # good=50, bad=200 (lower is better, e.g. Debt/Equity)
    assert _linear(30, good=50, bad=200) == 100.0
    assert _linear(50, good=50, bad=200) == 100.0
    assert _linear(200, good=50, bad=200) == 0.0
    assert _linear(300, good=50, bad=200) == 0.0
    assert _linear(125, good=50, bad=200) == pytest.approx(50.0)


def test_linear_none():
    assert _linear(None, 20, 5) is None


def test_compute_q_score_empty():
    q, details = compute_q_score(QualityInputs())
    assert q is None
    assert details["score"] is None


def test_compute_q_score_all_inputs():
    q, details = compute_q_score(
        QualityInputs(
            gross_margin=0.40,            # ≥ 0.40 → 100
            return_on_equity=0.20,        # ≥ 0.20 → 100
            operating_margin=0.20,        # ≥ 0.20 → 100
            eps_growth_std_pct=30.0,      # ≤ 30 → 100
            debt_to_equity=50.0,          # ≤ 50 → 100
        )
    )
    assert q == 100.0
    assert set(details["components"]) == {
        "Gross Margin", "ROE", "Operating Margin", "EPS Stability", "Debt/Equity"
    }


def test_compute_q_score_mixed_quality():
    q, details = compute_q_score(
        QualityInputs(
            gross_margin=0.25,        # halfway → 50
            return_on_equity=0.05,    # at bad → 0
            operating_margin=0.10,    # halfway → 50
        )
    )
    assert q == pytest.approx((50 + 0 + 50) / 3, rel=1e-2)


def test_compute_q_score_debt_heavy_penalised():
    q, details = compute_q_score(QualityInputs(debt_to_equity=300))
    assert q == 0.0
