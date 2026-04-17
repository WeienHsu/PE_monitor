"""Unit tests for QVM composite: weight routing, gates, signal mapping."""

import pytest

from src.factors.qvm_composite import (
    BUY_CUT,
    WATCH_CUT,
    CAUTION_CUT,
    SELL_CUT,
    QVM_WEIGHTS,
    _apply_quality_gate,
    _apply_trend_filter,
    _get_weights,
    _qvm_to_signal,
    compute_qvm,
)


# ---------------------------------------------------------------------------
# Weight routing
# ---------------------------------------------------------------------------

def test_weights_sum_to_one():
    for key, w in QVM_WEIGHTS.items():
        total = w["V"] + w["Q"] + w["M"]
        assert total == pytest.approx(1.0, abs=1e-6), f"{key} sums to {total}"


def test_get_weights_routes_stock_type():
    assert _get_weights("growth", None) == QVM_WEIGHTS["growth"]
    assert _get_weights("cyclical", None) == QVM_WEIGHTS["cyclical"]


def test_get_weights_prefers_etf_subtype():
    assert _get_weights("etf", "broad") == QVM_WEIGHTS["etf_broad"]
    assert _get_weights("etf", "commodity") == QVM_WEIGHTS["etf_commodity"]
    # Unknown subtype falls through to stock_type lookup
    assert _get_weights("etf", "unknownsub") == QVM_WEIGHTS["etf"]


def test_get_weights_unknown_falls_back():
    assert _get_weights("garbage", None) == QVM_WEIGHTS["unknown"]


# ---------------------------------------------------------------------------
# Signal mapping
# ---------------------------------------------------------------------------

def test_signal_thresholds():
    assert _qvm_to_signal(90) == "BUY"
    assert _qvm_to_signal(BUY_CUT + 0.01) == "BUY"
    assert _qvm_to_signal(BUY_CUT) == "WATCH"       # strict >
    assert _qvm_to_signal(70) == "WATCH"
    assert _qvm_to_signal(WATCH_CUT) == "NEUTRAL"    # strict >
    assert _qvm_to_signal(50) == "NEUTRAL"
    assert _qvm_to_signal(CAUTION_CUT) == "NEUTRAL"  # ≥ CAUTION_CUT
    assert _qvm_to_signal(CAUTION_CUT - 0.01) == "CAUTION"
    assert _qvm_to_signal(SELL_CUT) == "CAUTION"     # ≥ SELL_CUT
    assert _qvm_to_signal(SELL_CUT - 0.01) == "SELL"
    assert _qvm_to_signal(10) == "SELL"


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def test_quality_gate_passes_healthy_stock():
    sig, triggered = _apply_quality_gate(
        "BUY", operating_cashflow=1e9, ttm_eps=5.0, is_etf=False
    )
    assert sig == "BUY"
    assert triggered is False


def test_quality_gate_caps_buy_when_ocf_negative():
    sig, triggered = _apply_quality_gate(
        "BUY", operating_cashflow=-100, ttm_eps=5.0, is_etf=False
    )
    assert sig == "WATCH"
    assert triggered is True


def test_quality_gate_caps_buy_when_eps_negative():
    sig, triggered = _apply_quality_gate(
        "BUY", operating_cashflow=1e9, ttm_eps=-1.0, is_etf=False
    )
    assert sig == "WATCH"
    assert triggered is True


def test_quality_gate_doesnt_downgrade_below_watch():
    sig, triggered = _apply_quality_gate(
        "NEUTRAL", operating_cashflow=-1, ttm_eps=-1, is_etf=False
    )
    # Already below BUY — still triggers the flag but no change
    assert sig == "NEUTRAL"
    assert triggered is True


def test_quality_gate_skips_etf():
    sig, triggered = _apply_quality_gate(
        "BUY", operating_cashflow=None, ttm_eps=None, is_etf=True
    )
    assert sig == "BUY"
    assert triggered is False


# ---------------------------------------------------------------------------
# Trend filter
# ---------------------------------------------------------------------------

def test_trend_filter_passes_when_price_above_threshold():
    sig, triggered = _apply_trend_filter("BUY", price=100, sma200=100)
    assert sig == "BUY"
    assert triggered is False


def test_trend_filter_downgrades_buy_when_price_crashes():
    # 80 < 100 * 0.85 = 85 → triggered
    sig, triggered = _apply_trend_filter("BUY", price=80, sma200=100)
    assert sig == "WATCH"
    assert triggered is True


def test_trend_filter_ignores_non_buy():
    sig, triggered = _apply_trend_filter("NEUTRAL", price=50, sma200=100)
    assert sig == "NEUTRAL"
    assert triggered is False


def test_trend_filter_missing_sma_is_passthrough():
    sig, triggered = _apply_trend_filter("BUY", price=50, sma200=None)
    assert sig == "BUY"
    assert triggered is False


# ---------------------------------------------------------------------------
# End-to-end compute_qvm
# ---------------------------------------------------------------------------

def test_compute_qvm_all_none_returns_n_a():
    result = compute_qvm(v_score=None, q_score=None, m_score=None)
    assert result["qvm_raw"] is None
    assert result["base_signal"] == "N/A"
    assert result["composite_signal"] == "N/A"


def test_compute_qvm_renormalises_missing_factors():
    # Only V provided. Weights for 'stable' are V:0.40, Q:0.35, M:0.25.
    # With only V usable, the weight renormalises to 1.0 → qvm == v.
    result = compute_qvm(
        v_score=70, q_score=None, m_score=None, stock_type="stable"
    )
    assert result["qvm_raw"] == 70.0
    assert result["base_signal"] == "WATCH"    # 70 is within (65, 75]


def test_compute_qvm_high_qvm_yields_buy_for_stable():
    result = compute_qvm(
        v_score=90, q_score=90, m_score=90, stock_type="stable"
    )
    assert result["qvm_raw"] == 90.0
    assert result["base_signal"] == "BUY"


def test_compute_qvm_quality_gate_caps_buy():
    result = compute_qvm(
        v_score=90, q_score=90, m_score=90,
        stock_type="stable",
        operating_cashflow=-100,
        ttm_eps=5.0,
    )
    assert result["component_signal"] == "BUY"
    assert result["base_signal"] == "WATCH"
    assert result["gates"]["quality"] is True


def test_compute_qvm_trend_filter_downgrades():
    result = compute_qvm(
        v_score=90, q_score=90, m_score=90,
        stock_type="stable",
        price=80,
        sma200=100,
    )
    assert result["component_signal"] == "BUY"
    assert result["base_signal"] == "WATCH"
    assert result["gates"]["trend"] is True


def test_compute_qvm_etf_bypasses_quality_gate():
    result = compute_qvm(
        v_score=90, q_score=50, m_score=90,
        stock_type="etf", etf_subtype="broad",
        is_etf=True,
        operating_cashflow=None, ttm_eps=None,
    )
    # ETF, so quality gate is skipped → signal stays BUY
    assert result["base_signal"] == "BUY"
    assert result["gates"]["quality"] is False
