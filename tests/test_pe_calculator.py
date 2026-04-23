"""Unit tests for pe_calculator.py"""

import pandas as pd
import pytest
from unittest.mock import patch

from src.pe_calculator import (
    build_historical_pe_series,
    classify_signal,
    current_percentile_rank,
    get_percentiles,
    get_ps_ratio,
)


class TestClassifySignal:
    def test_buy_zone(self):
        assert classify_signal(0.0) == "BUY"
        assert classify_signal(24.9) == "BUY"

    def test_watch_zone(self):
        assert classify_signal(25.0) == "WATCH"
        assert classify_signal(34.9) == "WATCH"

    def test_neutral_zone(self):
        assert classify_signal(35.0) == "NEUTRAL"
        assert classify_signal(64.9) == "NEUTRAL"

    def test_caution_zone(self):
        assert classify_signal(65.0) == "CAUTION"
        assert classify_signal(74.9) == "CAUTION"

    def test_sell_zone(self):
        assert classify_signal(75.0) == "SELL"
        assert classify_signal(100.0) == "SELL"

    def test_custom_entry_exit(self):
        assert classify_signal(20.0, entry=30, exit_=70) == "BUY"
        assert classify_signal(30.0, entry=30, exit_=70) == "WATCH"
        assert classify_signal(60.0, entry=30, exit_=70) == "CAUTION"
        assert classify_signal(70.0, entry=30, exit_=70) == "SELL"


class TestTypeAdaptiveThresholds:
    """P1-9: different stock types get different entry/exit percentile bands."""

    def test_growth_aggressive_entry_at_15pct(self):
        """A growth stock at 20th percentile is NEUTRAL (past 15% entry), not BUY."""
        # growth thresholds: (15, 60)
        assert classify_signal(10.0, stock_type="growth") == "BUY"
        assert classify_signal(14.0, stock_type="growth") == "BUY"
        # 15 is the boundary → WATCH
        assert classify_signal(15.0, stock_type="growth") == "WATCH"
        # mid range → NEUTRAL
        assert classify_signal(30.0, stock_type="growth") == "NEUTRAL"
        # past exit-10 (60-buffer) → CAUTION
        assert classify_signal(58.0, stock_type="growth") == "CAUTION"
        assert classify_signal(70.0, stock_type="growth") == "SELL"

    def test_cyclical_wider_bands_35_85(self):
        """A cyclical at 30th percentile is still NEUTRAL (past 35% but below SELL zone)."""
        # cyclical thresholds: (35, 85)
        assert classify_signal(30.0, stock_type="cyclical") == "BUY"
        assert classify_signal(34.9, stock_type="cyclical") == "BUY"
        # 35 → WATCH
        assert classify_signal(35.0, stock_type="cyclical") == "WATCH"
        # 84 is near exit, below 85 → CAUTION
        assert classify_signal(84.0, stock_type="cyclical") == "CAUTION"
        assert classify_signal(90.0, stock_type="cyclical") == "SELL"

    def test_stable_preserves_25_75(self):
        """Stable type matches the legacy 25/75 defaults."""
        assert classify_signal(24.9, stock_type="stable") == "BUY"
        assert classify_signal(25.0, stock_type="stable") == "WATCH"
        assert classify_signal(75.0, stock_type="stable") == "SELL"

    def test_etf_and_unknown_default_25_75(self):
        assert classify_signal(24.9, stock_type="etf") == "BUY"
        assert classify_signal(25.0, stock_type="unknown") == "WATCH"
        assert classify_signal(25.0, stock_type=None) == "WATCH"

    def test_explicit_entry_overridden_by_stock_type(self):
        """
        When stock_type is given, it takes precedence over entry/exit args.
        With stock_type='growth' → 15/60 bands, 20 falls in WATCH (15-25).
        With the legacy stable 25/75 bands, 20 would be BUY — the override
        is what we're verifying here.
        """
        result = classify_signal(20.0, entry=25, exit_=75, stock_type="growth")
        assert result == "WATCH"
        # Sanity check: without the override, same value is BUY
        assert classify_signal(20.0, entry=25, exit_=75) == "BUY"


class TestGetPercentiles:
    def test_empty_series_returns_empty_dict(self):
        result = get_percentiles(pd.Series(dtype=float))
        assert result == {}

    def test_known_series(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = get_percentiles(s)
        assert 25 in result
        assert 50 in result
        assert result[50] == pytest.approx(30.0)

    def test_returns_all_percentile_keys(self):
        s = pd.Series(range(100), dtype=float)
        result = get_percentiles(s)
        assert set(result.keys()) == {10, 25, 35, 50, 65, 75, 90}


class TestCurrentPercentileRank:
    def test_median_value(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        rank = current_percentile_rank(3.0, s)
        assert rank == pytest.approx(40.0)

    def test_lowest_value(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0])
        rank = current_percentile_rank(5.0, s)
        assert rank == 0.0

    def test_highest_value(self):
        s = pd.Series([10.0, 20.0, 30.0])
        rank = current_percentile_rank(40.0, s)
        assert rank == 100.0

    def test_empty_series_returns_50(self):
        rank = current_percentile_rank(10.0, pd.Series(dtype=float))
        assert rank == 50.0


class TestHistoricalPEWithHistoricalShares:
    """
    P1-6: per-quarter historical shares fix. The old code used current shares
    for every historical quarter, which under-priced historical P/E for
    aggressive buyback names (AAPL, GOOGL). The new code pulls per-quarter
    Ordinary Shares Number from the balance sheet.
    """

    def _make_quarterly_ni(self) -> pd.DataFrame:
        """4 quarters of constant net income = 1e9 each → TTM NI = 4e9."""
        dates = [pd.Timestamp(f"2025-{m:02d}-01") for m in (3, 6, 9, 12)]
        s = pd.Series([1e9] * 4, index=dates, name="Net Income")
        return pd.DataFrame(s).T.rename(index={0: "Net Income"})

    def _make_shrinking_shares_bs(self) -> pd.DataFrame:
        """Shares drop from 2B → 1B via buybacks. Current shares = 1B."""
        # yfinance orders balance-sheet columns most-recent-first
        dates = [pd.Timestamp(f"2025-{m:02d}-01") for m in (12, 9, 6, 3)]
        shares = [1e9, 1.3e9, 1.6e9, 2e9]
        return pd.DataFrame({d: {"Ordinary Shares Number": v} for d, v in zip(dates, shares)})

    def _price_df(self, price: float) -> pd.DataFrame:
        dates = pd.date_range(end="2026-01-01", periods=300, freq="D")
        return pd.DataFrame({"Close": [price] * len(dates)}, index=dates)

    def test_uses_per_quarter_shares_not_current(self, tmp_path):
        """
        With NI=1e9/quarter and shrinking shares 2B→1B, per-quarter EPS
        should reflect the actual per-quarter share counts, not the current 1B.

        TTM EPS (at end-of-period) = sum(1e9/q_shares for 4 quarters)
          = 1/2 + 1/1.6 + 1/1.3 + 1/1 = 0.5 + 0.625 + 0.769 + 1.0 ≈ 2.894
        Legacy (all current shares = 1B): = 4.0

        At price=100, legacy P/E = 100/4 = 25; correct P/E = 100/2.894 ≈ 34.5
        """
        price_df = self._price_df(100.0)
        ni_df = self._make_quarterly_ni()
        bs_df = self._make_shrinking_shares_bs()

        with patch("src.pe_calculator.fetch_price_history", return_value=price_df), \
             patch("src.pe_calculator.fetch_quarterly_financials", return_value=ni_df), \
             patch("src.pe_calculator.fetch_quarterly_balance_sheet", return_value=bs_df), \
             patch("src.pe_calculator.fetch_shares_outstanding", return_value=1e9):
            series = build_historical_pe_series("TEST", years=1, data_dir=str(tmp_path))

        assert not series.empty
        # Latest P/E should be price / (sum of per-quarter EPS)
        expected_ttm_eps = (1e9 / 2e9) + (1e9 / 1.6e9) + (1e9 / 1.3e9) + (1e9 / 1e9)
        expected_pe = 100.0 / expected_ttm_eps
        assert series.iloc[-1] == pytest.approx(expected_pe, rel=1e-3)
        # Confirm it's meaningfully different from the legacy value of 25.0
        assert series.iloc[-1] > 30.0

    def test_falls_back_to_current_when_bs_empty(self, tmp_path):
        """No balance-sheet data → use current shares for every quarter (legacy)."""
        price_df = self._price_df(100.0)
        ni_df = self._make_quarterly_ni()

        with patch("src.pe_calculator.fetch_price_history", return_value=price_df), \
             patch("src.pe_calculator.fetch_quarterly_financials", return_value=ni_df), \
             patch("src.pe_calculator.fetch_quarterly_balance_sheet", return_value=pd.DataFrame()), \
             patch("src.pe_calculator.fetch_shares_outstanding", return_value=1e9):
            series = build_historical_pe_series("TEST", years=1, data_dir=str(tmp_path))

        assert not series.empty
        # Legacy math: TTM EPS = 4e9 / 1e9 = 4; P/E = 100/4 = 25
        assert series.iloc[-1] == pytest.approx(25.0, rel=1e-3)


class TestGetPsRatio:
    def test_returns_ps_from_info(self, mock_growth_info):
        with patch("src.pe_calculator.fetch_fundamental_extras") as mock_fe:
            mock_fe.return_value = {"ps_ratio": 10.5}
            result = get_ps_ratio("TEST")
        assert result == pytest.approx(10.5)

    def test_returns_none_when_missing(self):
        with patch("src.pe_calculator.fetch_fundamental_extras") as mock_fe:
            mock_fe.return_value = {"ps_ratio": None}
            result = get_ps_ratio("TEST")
        assert result is None

    def test_returns_none_when_zero(self):
        with patch("src.pe_calculator.fetch_fundamental_extras") as mock_fe:
            mock_fe.return_value = {"ps_ratio": 0.0}
            result = get_ps_ratio("TEST")
        assert result is None
