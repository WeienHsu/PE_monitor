"""Smoke tests for ETF V/Q score plumbing.

These tests only exercise the pure-Python glue: classification and score
composition. They monkey-patch external fetches so the suite stays offline.
"""

import pandas as pd
import pytest

from src import etf_signal


@pytest.fixture(autouse=True)
def stub_external_data(monkeypatch):
    """Default stubs — individual tests override as needed."""

    def fake_info(ticker, data_dir="data"):
        return {
            "trailingPE": 20.0,
            "totalAssets": 2_000_000_000,
            "annualReportExpenseRatio": 0.0009,  # 0.09%
        }

    def fake_price_history(ticker, years=5, data_dir="data"):
        idx = pd.date_range("2020-01-01", periods=1000, freq="B")
        # Prices ramp from 100 → 200, so current price 200 is at 100th pct.
        return pd.DataFrame({"Close": list(range(100, 1100))}, index=idx)

    def fake_shiller(value=None, data_dir="data", years=None):
        return (30.0, 70.0)

    def fake_industry_pe(industry, data_dir="data"):
        return 40.0 if industry else None

    monkeypatch.setattr(etf_signal, "fetch_info", fake_info)
    monkeypatch.setattr(etf_signal, "fetch_price_history", fake_price_history)
    monkeypatch.setattr(etf_signal, "get_shiller_cape_percentile", fake_shiller)
    monkeypatch.setattr(etf_signal, "get_industry_trailing_pe", fake_industry_pe)


def test_etf_q_score_combines_aum_and_expense():
    q, details = etf_signal.compute_etf_q_score("SPY")
    assert q is not None
    assert "AUM (stability)" in details["components"]
    assert "Expense ratio" in details["components"]
    # Large AUM and low ER → near 100
    assert q > 90


def test_etf_q_score_returns_none_when_info_empty(monkeypatch):
    monkeypatch.setattr(etf_signal, "fetch_info", lambda *a, **kw: {})
    q, details = etf_signal.compute_etf_q_score("XYZ")
    assert q is None
    assert details["components"] == {}


def test_broad_uses_shiller_percentile():
    v, details = etf_signal.compute_etf_v_score("SPY", "broad", 200.0)
    assert v is not None
    assert any("Shiller CAPE" in k for k in details["components"])


def test_broad_falls_back_when_shiller_fails(monkeypatch):
    monkeypatch.setattr(
        etf_signal, "get_shiller_cape_percentile",
        lambda *a, **kw: (None, None),
    )
    v, details = etf_signal.compute_etf_v_score("VTI", "broad", 500.0)
    assert v is not None
    # Should fall back to own-price percentile
    assert any("Price percentile" in k for k in details["components"])


def test_sector_uses_industry_pe_ratio(monkeypatch):
    monkeypatch.setattr(
        etf_signal, "get_industry_for_etf",
        lambda ticker: "Software",
    )
    v, details = etf_signal.compute_etf_v_score("XLK", "sector", 150.0)
    assert v is not None
    assert any("ETF PE /" in k for k in details["components"])


def test_commodity_uses_price_percentile():
    v, details = etf_signal.compute_etf_v_score("GLD", "commodity", 400.0)
    assert v is not None
    assert "Price percentile (5Y)" in details["components"]


def test_bond_uses_treasury_yield(monkeypatch):
    monkeypatch.setattr(
        etf_signal, "_treasury_10y_percentile",
        lambda years, data_dir: (4.5, 80.0),
    )
    v, details = etf_signal.compute_etf_v_score("TLT", "bond", 90.0)
    assert v is not None
    assert any("Treasury" in k for k in details["components"])


def test_unknown_subtype_uses_price_percentile():
    # Truly unknown subtype string → else branch uses price percentile
    v, details = etf_signal.compute_etf_v_score("ZZZ", "weird_new_kind", 200.0)
    assert v is not None
    assert "Price percentile (5Y)" in details["components"]


def test_none_subtype_defaults_to_broad():
    # Defensive default: None subtype → broad branch (Shiller-driven)
    v, details = etf_signal.compute_etf_v_score("ZZZ", None, 200.0)
    assert v is not None
    assert any("Shiller CAPE" in k or "Price percentile" in k for k in details["components"])
