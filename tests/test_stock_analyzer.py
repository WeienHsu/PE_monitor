"""Unit tests for stock_analyzer.py"""

import pandas as pd
import pytest
from unittest.mock import patch

from src.stock_analyzer import analyze_suitability


def _make_quarterly_df(eps_by_year: dict[int, float]) -> pd.DataFrame:
    """Build a mock quarterly income statement DataFrame from annual EPS values."""
    rows = {}
    for year, annual_net in eps_by_year.items():
        for q in range(1, 5):
            col = pd.Timestamp(f"{year}-{q*3:02d}-01")
            rows[col] = annual_net / 4
    s = pd.Series(rows, name="Net Income")
    return pd.DataFrame(s).T.rename(index={0: "Net Income"})


def _make_annual_df(eps_by_year: dict[int, float]) -> pd.DataFrame:
    """Build a mock annual income statement (one column per fiscal year end)."""
    cols = {}
    for year, net in eps_by_year.items():
        cols[pd.Timestamp(f"{year}-12-31")] = net
    s = pd.Series(cols, name="Net Income")
    return pd.DataFrame(s).T.rename(index={0: "Net Income"})


STABLE_QUARTERLY = _make_quarterly_df({2021: 10e9, 2022: 10.5e9, 2023: 11e9, 2024: 11.5e9})
GROWTH_QUARTERLY = _make_quarterly_df({2021: 5e9, 2022: 8e9, 2023: 13e9, 2024: 20e9})
CYCLICAL_QUARTERLY = _make_quarterly_df({2021: 10e9, 2022: -2e9, 2023: 8e9, 2024: 12e9})

STABLE_ANNUAL = _make_annual_df({2021: 10e9, 2022: 10.5e9, 2023: 11e9, 2024: 11.5e9})
GROWTH_ANNUAL = _make_annual_df({2021: 5e9, 2022: 8e9, 2023: 13e9, 2024: 20e9})
CYCLICAL_ANNUAL = _make_annual_df({2021: 10e9, 2022: -2e9, 2023: 8e9, 2024: 12e9})


class TestAnalyzeSuitabilityETF:
    def test_etf_returns_early(self):
        with patch("src.stock_analyzer.is_etf", return_value=True), \
             patch("src.stock_analyzer.fetch_info", return_value={"longName": "Test ETF"}):
            result = analyze_suitability("SPY")
        assert result["type"] == "etf"
        assert result["recommended_metric"] == "PE"


class TestAnalyzeSuitabilityInsufficientData:
    def test_single_year_returns_unknown(self):
        single_year_df = _make_quarterly_df({2024: 5e9})
        single_year_annual = _make_annual_df({2024: 5e9})
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value={"longName": "Test Corp"}), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=single_year_annual), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=single_year_df), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value={}):
            result = analyze_suitability("TEST")
        assert result["type"] == "unknown"
        assert result["suitability_score"] == 2

    def test_two_year_data_marks_insufficient_volatility(self):
        """
        P0-2 regression: 2 years → 1 growth rate → len(growth_rates) < 2 → std_pct=None.
        Previously np.std on 1 element returned 0, falsely triggering "stable +2".
        Now EPS volatility dimension should skip voting and reason should say
        "EPS資料不足" instead of "EPS波動率0%".
        """
        two_year_df = _make_quarterly_df({2023: 10e9, 2024: 12e9})
        two_year_annual = _make_annual_df({2023: 10e9, 2024: 12e9})
        # Growth-signal extras (high rev growth + high beta) should dominate now
        # that the bogus EPS-stable +2 is gone.
        extras = {"beta": 1.35, "revenue_growth": 0.25, "dividend_yield": None, "ps_ratio": 10.0}
        info = {"longName": "Short History Corp", "quoteType": "EQUITY"}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=two_year_annual), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=two_year_df), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("TEST")
        assert "EPS資料不足" in result["reason"]
        assert "EPS波動率0%" not in result["reason"]
        # With growth extras and no fake stable vote, should classify as growth
        assert result["type"] == "growth"


class TestAnalyzeSuitabilityStable:
    def test_stable_all_factors(self, mock_stable_info):
        extras = {"beta": 0.75, "revenue_growth": 0.04, "dividend_yield": 0.028, "ps_ratio": 5.2}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=mock_stable_info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=STABLE_ANNUAL), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=STABLE_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("AAPL")
        assert result["type"] == "stable"
        assert result["recommended_metric"] == "PE"
        assert result["suitability_score"] == 5
        assert "穩定型" in result["reason"]


class TestAnalyzeSuitabilityGrowth:
    def test_growth_high_revenue_and_beta(self, mock_growth_info):
        extras = {"beta": 1.35, "revenue_growth": 0.25, "dividend_yield": None, "ps_ratio": 10.5}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=mock_growth_info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=GROWTH_ANNUAL), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=GROWTH_ANNUAL), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=GROWTH_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("NVDA")
        assert result["type"] == "growth"
        assert result["recommended_metric"] == "PE"
        assert "成長型" in result["reason"]


class TestAnalyzeSuitabilityCyclical:
    def test_negative_eps_yields_cyclical(self):
        extras = {"beta": 1.6, "revenue_growth": -0.05, "dividend_yield": 0.01, "ps_ratio": 2.0}
        info = {"longName": "Cyclical Corp", "quoteType": "EQUITY"}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=CYCLICAL_ANNUAL), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=CYCLICAL_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("MU")
        assert result["type"] == "cyclical"
        assert result["recommended_metric"] == "PB"
        assert result["suitability_score"] == 2


class TestAnalyzeSuitabilityTiebreak:
    def test_tiebreak_stable_wins_over_growth(self):
        # Low EPS volatility (stable +2) + no other signals → stable wins tiebreak
        extras = {"beta": None, "revenue_growth": None, "dividend_yield": None, "ps_ratio": None}
        info = {"longName": "Tied Corp", "quoteType": "EQUITY"}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=STABLE_ANNUAL), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=STABLE_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("TEST")
        assert result["type"] == "stable"


class TestAnnualStatementPath:
    """
    P0-2 follow-up: the analyzer should prefer the annual income statement
    (5 years of real annual Net Income) over quarterly aggregation, because
    yfinance quarterly_income_stmt only returns 4-6 quarters (1-2 annual
    points), which is insufficient for EPS volatility.
    """

    def test_annual_data_used_when_available(self):
        """Annual df has 5 years → classifier gets 4 growth rates → std_pct is a real number."""
        # Only 2 quarters of quarterly data (too few), but 5 years of annual
        tiny_q = _make_quarterly_df({2024: 10e9})
        full_annual = _make_annual_df({2020: 8e9, 2021: 9e9, 2022: 10e9, 2023: 11e9, 2024: 12e9})
        extras = {"beta": 0.9, "revenue_growth": 0.10, "dividend_yield": 0.02, "ps_ratio": 5.0}
        info = {"longName": "Test Corp", "quoteType": "EQUITY"}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=full_annual), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=tiny_q), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("TEST")
        # Reason should contain a real volatility number, not "EPS資料不足"
        assert "EPS資料不足" not in result["reason"]
        assert "EPS波動率" in result["reason"]

    def test_quarterly_fallback_drops_partial_years(self):
        """
        When annual statement is empty, fall back to quarterly grouping — but
        discard years that don't have 4 quarters, to avoid partial-year artifacts.
        """
        # Build a quarterly df with 4 full quarters for 2023, but only 1 for 2024
        rows = {}
        for q in range(1, 5):
            rows[pd.Timestamp(f"2023-{q*3:02d}-01")] = 2.5e9
        rows[pd.Timestamp("2024-03-01")] = 1e9  # only Q1 2024
        s = pd.Series(rows, name="Net Income")
        quarterly_partial = pd.DataFrame(s).T.rename(index={0: "Net Income"})

        extras = {"beta": None, "revenue_growth": None, "dividend_yield": None, "ps_ratio": None}
        info = {"longName": "Partial Corp", "quoteType": "EQUITY"}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=pd.DataFrame()), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=quarterly_partial), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("TEST")
        # Only 1 full year survives → std_pct=None → 資料不足 path
        assert "EPS資料不足" in result["reason"]


class TestHardClassify:
    """P2-12: industry / sector hard-classification overrides multi-factor scoring."""

    def test_semiconductor_industry_forced_cyclical(self, mock_growth_info):
        """
        NVDA-like fundamentals (growth signals everywhere) — but industry
        'Semiconductors' forces cyclical type regardless.
        """
        info = {
            **mock_growth_info,
            "industry": "Semiconductors",
            "sector": "Technology",
            "marketCap": 3_000_000_000_000,  # $3T
        }
        extras = {"beta": 1.4, "revenue_growth": 0.30, "dividend_yield": None, "ps_ratio": 20}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_annual_financials", return_value=GROWTH_ANNUAL), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=GROWTH_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("NVDA")
        assert result["type"] == "cyclical"
        assert result["recommended_metric"] == "PB"
        assert result["type_source"] == "hard_rule"
        assert "Semiconductors" in result["reason"]
        assert result["small_cap"] is False

    def test_regulated_utility_forced_stable(self):
        info = {
            "longName": "Great Plains Electric",
            "industry": "Utilities - Regulated Electric",
            "sector": "Utilities",
            "marketCap": 15_000_000_000,
        }
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=CYCLICAL_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value={}):
            result = analyze_suitability("SO")
        assert result["type"] == "stable"
        assert result["recommended_metric"] == "PE"
        assert result["type_source"] == "hard_rule"

    def test_sector_fallback_when_industry_unknown(self):
        """Industry not in table, but sector='Energy' → cyclical."""
        info = {
            "longName": "Unknown Oil Corp",
            "industry": "Something Obscure",
            "sector": "Energy",
            "marketCap": 5_000_000_000,
        }
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=STABLE_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value={}):
            result = analyze_suitability("XOM")
        assert result["type"] == "cyclical"

    def test_no_rule_falls_back_to_multifactor(self, mock_stable_info):
        """Generic software industry is not in the table → multi-factor runs."""
        info = {
            **mock_stable_info,
            "industry": "Software - Application",
            "sector": "Technology",
            "marketCap": 100_000_000_000,
        }
        extras = {"beta": 0.75, "revenue_growth": 0.04, "dividend_yield": 0.028, "ps_ratio": 5.2}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=STABLE_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("TEST")
        assert result["type"] == "stable"
        assert result["type_source"] == "auto"  # multi-factor path

    def test_small_cap_flag(self, mock_growth_info):
        info = {
            **mock_growth_info,
            "industry": "Software - Application",
            "sector": "Technology",
            "marketCap": 500_000_000,  # $500M → small cap
        }
        extras = {"beta": 1.3, "revenue_growth": 0.25, "dividend_yield": None, "ps_ratio": 10}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=info), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=GROWTH_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("TEST")
        assert result["small_cap"] is True


class TestReasonString:
    def test_reason_includes_factor_summary(self, mock_stable_info):
        extras = {"beta": 0.75, "revenue_growth": 0.04, "dividend_yield": 0.028, "ps_ratio": 5.2}
        with patch("src.stock_analyzer.is_etf", return_value=False), \
             patch("src.stock_analyzer.fetch_info", return_value=mock_stable_info), \
             patch("src.stock_analyzer.fetch_quarterly_financials", return_value=STABLE_QUARTERLY), \
             patch("src.stock_analyzer.fetch_fundamental_extras", return_value=extras):
            result = analyze_suitability("TEST")
        assert "EPS波動率" in result["reason"]
        assert "Beta=" in result["reason"]
        assert "營收成長" in result["reason"]
        assert "殖利率" in result["reason"]
