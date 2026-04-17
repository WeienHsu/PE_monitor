"""Unit tests for the V-factor scoring logic (no network)."""

import pytest

from src.factors.value_factor import (
    ValueInputs,
    _linear_band,
    _percentile_to_cheapness,
    compute_v_score,
)


def test_percentile_to_cheapness_inverts():
    assert _percentile_to_cheapness(0) == 100
    assert _percentile_to_cheapness(50) == 50
    assert _percentile_to_cheapness(100) == 0
    assert _percentile_to_cheapness(None) is None


def test_linear_band_low_is_cheap():
    # cheap_end=10, expensive_end=30 → below 10 is 100, above 30 is 0
    assert _linear_band(5, 10, 30) == 100.0
    assert _linear_band(10, 10, 30) == 100.0
    assert _linear_band(30, 10, 30) == 0.0
    assert _linear_band(50, 10, 30) == 0.0
    assert _linear_band(20, 10, 30) == 50.0


def test_linear_band_reversed_high_is_cheap():
    # cheap_end=5%, expensive_end=1% (dividend-yield-style: high yield = cheap)
    assert _linear_band(6, 5, 1) == 100.0
    assert _linear_band(5, 5, 1) == 100.0
    assert _linear_band(1, 5, 1) == 0.0
    assert _linear_band(3, 5, 1) == 50.0


def test_compute_v_score_empty_returns_none():
    v, details = compute_v_score(ValueInputs())
    assert v is None
    assert details["score"] is None
    assert details["components"] == {}


def test_compute_v_score_single_input():
    v, details = compute_v_score(ValueInputs(ttm_pe_percentile=20))
    # rank 20 → cheapness 80
    assert v == 80.0
    assert details["components"]["TTM P/E"]["score"] == 80.0
    assert details["components"]["TTM P/E"]["kind"] == "percentile"


def test_compute_v_score_averages_components():
    v, details = compute_v_score(
        ValueInputs(
            ttm_pe_percentile=20,          # → 80
            pb_percentile=50,               # → 50
            forward_pe=30,                  # at expensive_end → 0
        )
    )
    # mean(80, 50, 0) = 43.33
    assert v == pytest.approx(43.33, rel=1e-2)
    assert set(details["components"]) == {"TTM P/E", "P/B", "Forward P/E"}


def test_cape_percentile_takes_precedence_over_absolute():
    v, details = compute_v_score(
        ValueInputs(cape_percentile=10, cape_absolute=45)
    )
    # percentile path wins → cheapness = 90
    assert v == 90.0
    assert details["components"]["CAPE-like P/E"]["kind"] == "percentile"


def test_cape_absolute_used_when_percentile_missing():
    v, details = compute_v_score(ValueInputs(cape_absolute=15))
    # cape_absolute=15 maps to 100 (cheap_end)
    assert v == 100.0
    assert details["components"]["CAPE-like P/E"]["kind"] == "absolute"


def test_industry_pe_ratio_input():
    v, details = compute_v_score(
        ValueInputs(ttm_pe=20, industry_pe=40, industry_name="Foo")
    )
    # ratio 0.5 ≤ 0.7 → 100
    assert v == 100.0
    comp_key = next(k for k in details["components"] if "Foo" in k)
    assert details["components"][comp_key]["score"] == 100.0


def test_industry_pe_ignored_when_missing():
    # No industry_pe → no "PE vs Industry" component
    v, details = compute_v_score(
        ValueInputs(ttm_pe_percentile=25, ttm_pe=20, industry_pe=None)
    )
    assert v == 75.0
    assert all("Industry" not in k and "PE vs" not in k for k in details["components"])
