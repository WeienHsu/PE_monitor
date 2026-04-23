"""Unit tests for data_fetcher.py — focused on P0 bug fixes."""

from unittest.mock import patch

import pytest

from src.data_fetcher import fetch_fundamental_extras


class TestDividendYieldUnitDetection:
    """
    yfinance changed dividendYield unit from decimal (0.005) to percent
    (0.5 = 0.5%) in late 2024, which caused AAPL to display as 39%, MSFT 87%.

    Fix: prefer `trailingAnnualDividendYield` (unit unchanged); fall back to
    `dividendYield` assuming the new percent format.
    """

    def _mock_info(self, **extras) -> dict:
        base = {
            "beta": 1.0, "revenueGrowth": 0.10,
            "priceToSalesTrailing12Months": 5.0, "marketCap": 1e11,
        }
        base.update(extras)
        return base

    def test_trailing_annual_preferred(self):
        """When both fields exist, prefer the stable-unit trailingAnnualDividendYield."""
        with patch("src.data_fetcher.fetch_info",
                   return_value=self._mock_info(
                       trailingAnnualDividendYield=0.005,  # 0.5% decimal
                       dividendYield=0.5,                  # would also become 0.005 via fallback
                   )):
            result = fetch_fundamental_extras("TEST")
        assert result["dividend_yield"] == 0.005

    def test_dividend_yield_percent_format_fallback(self):
        """Only dividendYield present — assume percent, divide by 100."""
        # AAPL-like: yfinance returns 0.39 meaning 0.39%
        with patch("src.data_fetcher.fetch_info",
                   return_value=self._mock_info(dividendYield=0.39)):
            result = fetch_fundamental_extras("TEST")
        assert result["dividend_yield"] == pytest.approx(0.0039)

    def test_msft_like_percent_fallback(self):
        """MSFT-like: yfinance returns 0.87 meaning 0.87%."""
        with patch("src.data_fetcher.fetch_info",
                   return_value=self._mock_info(dividendYield=0.87)):
            result = fetch_fundamental_extras("TEST")
        assert result["dividend_yield"] == pytest.approx(0.0087)

    def test_absurd_trailing_value_rejected(self):
        """trailingAnnualDividendYield > 50% is rejected as dirty."""
        with patch("src.data_fetcher.fetch_info",
                   return_value=self._mock_info(trailingAnnualDividendYield=5.0)):
            result = fetch_fundamental_extras("TEST")
        assert result["dividend_yield"] is None

    def test_absurd_fallback_value_rejected(self):
        """dividendYield that stays absurd after /100 (e.g. 9700) is rejected."""
        with patch("src.data_fetcher.fetch_info",
                   return_value=self._mock_info(dividendYield=9700.0)):
            result = fetch_fundamental_extras("TEST")
        assert result["dividend_yield"] is None

    def test_zero_dividend_kept(self):
        """0% is a valid value (non-dividend-paying stock)."""
        with patch("src.data_fetcher.fetch_info",
                   return_value=self._mock_info(trailingAnnualDividendYield=0.0)):
            result = fetch_fundamental_extras("TEST")
        assert result["dividend_yield"] == 0.0

    def test_missing_both_fields_returns_none(self):
        """No dividend fields at all → None."""
        info = {"beta": 1.0}
        with patch("src.data_fetcher.fetch_info", return_value=info):
            result = fetch_fundamental_extras("TEST")
        assert result["dividend_yield"] is None
