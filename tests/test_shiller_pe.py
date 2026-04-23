"""Unit tests for P3-14 Shiller / Normalized P/E calculation."""

from unittest.mock import patch

import pandas as pd

from src.pe_calculator import calc_shiller_pe


def _annual_income_df(eps_list: list[float], shares: float) -> pd.DataFrame:
    """
    Build a fake annual income-statement DataFrame with the supplied EPS
    values (most recent first → oldest last). ``shares`` is the divisor used
    to back out Net Income for each year.
    """
    # Annual dates, newest first
    years = len(eps_list)
    dates = pd.to_datetime(
        [f"{2025 - i}-12-31" for i in range(years)]
    )
    net_income = [eps * shares for eps in eps_list]
    # Rows = metric names, cols = dates
    return pd.DataFrame({d: {"Net Income": ni} for d, ni in zip(dates, net_income)}).T.T


class TestShillerPE:
    def test_basic_10_year_average(self, tmp_path):
        """With 10 flat years of $5 EPS and price=100, Shiller PE should
        converge to price / normalized_eps. Inflation adjustment makes the
        normalized EPS slightly higher than the flat $5 (older years get
        inflated to today's dollars), so Shiller PE < 20."""
        eps_list = [5.0] * 10
        shares = 1_000_000
        df = _annual_income_df(eps_list, shares)

        with patch("src.pe_calculator.fetch_annual_financials", return_value=df), \
             patch("src.pe_calculator.fetch_shares_outstanding", return_value=shares), \
             patch("src.pe_calculator.fetch_info", return_value={"currentPrice": 100.0}):
            r = calc_shiller_pe("FAKE", price=100.0, data_dir=str(tmp_path))

        assert r["available"] is True
        assert r["years_used"] == 10
        # normalized_pe (no inflation) exactly 100/5 = 20
        assert r["normalized_pe"] == 20.0
        # shiller_pe slightly less than 20 because inflation lifts denominator
        assert r["shiller_pe"] < 20.0
        assert r["shiller_pe"] > 15.0  # sanity bound

    def test_returns_unavailable_when_no_annual_data(self, tmp_path):
        with patch("src.pe_calculator.fetch_annual_financials", return_value=pd.DataFrame()), \
             patch("src.pe_calculator.fetch_shares_outstanding", return_value=1_000_000), \
             patch("src.pe_calculator.fetch_info", return_value={"currentPrice": 100.0}):
            r = calc_shiller_pe("FAKE", price=100.0, data_dir=str(tmp_path))

        assert r["available"] is False
        assert "無年度淨利" in r["reason"]

    def test_returns_unavailable_on_negative_mean_eps(self, tmp_path):
        """If the 10-year mean EPS is negative, we return unavailable and
        tag the reason as 'cyclical loss'."""
        eps_list = [-2.0, -3.0, -1.5, -4.0, -2.5]
        shares = 1_000_000
        df = _annual_income_df(eps_list, shares)
        with patch("src.pe_calculator.fetch_annual_financials", return_value=df), \
             patch("src.pe_calculator.fetch_shares_outstanding", return_value=shares), \
             patch("src.pe_calculator.fetch_info", return_value={"currentPrice": 20.0}):
            r = calc_shiller_pe("FAKE", price=20.0, data_dir=str(tmp_path))
        assert r["available"] is False
        assert "週期性虧損" in r["reason"]

    def test_partial_window_still_usable(self, tmp_path):
        """Only 4 years available → still computes, flags years_used=4."""
        eps_list = [3.0, 4.0, 5.0, 6.0]
        shares = 1_000_000
        df = _annual_income_df(eps_list, shares)
        with patch("src.pe_calculator.fetch_annual_financials", return_value=df), \
             patch("src.pe_calculator.fetch_shares_outstanding", return_value=shares), \
             patch("src.pe_calculator.fetch_info", return_value={"currentPrice": 60.0}):
            r = calc_shiller_pe("FAKE", price=60.0, data_dir=str(tmp_path))
        assert r["available"] is True
        assert r["years_used"] == 4
        # Mean EPS = 4.5 (unadjusted), normalized_pe = 60/4.5 ≈ 13.33
        assert 13.0 < r["normalized_pe"] < 14.0

    def test_uses_current_price_when_not_provided(self, tmp_path):
        eps_list = [2.0] * 10
        shares = 1_000_000
        df = _annual_income_df(eps_list, shares)
        with patch("src.pe_calculator.fetch_annual_financials", return_value=df), \
             patch("src.pe_calculator.fetch_shares_outstanding", return_value=shares), \
             patch("src.pe_calculator.fetch_info", return_value={"currentPrice": 40.0}):
            # price=None → function should call fetch_info
            r = calc_shiller_pe("FAKE", price=None, data_dir=str(tmp_path))
        assert r["available"] is True
        assert r["normalized_pe"] == 20.0
