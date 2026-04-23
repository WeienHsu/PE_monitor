"""Unit tests for value_trap_filter.py and the composite override."""

from unittest.mock import patch

import pandas as pd

from src.composite_signal import compute_multi_factor_composite
from src.value_trap_filter import check_value_trap


def _quarterly_df_from_rows(rows: dict, n_quarters: int = 8) -> pd.DataFrame:
    """
    Build a quarterly financial DataFrame (rows=metrics, cols=dates descending).
    yfinance orders columns most-recent-first.

    rows: dict[metric_name -> list of values, index 0 = most recent quarter]
    """
    dates = pd.date_range(end="2025-12-31", periods=n_quarters, freq="QE")[::-1]
    data = {d: {} for d in dates}
    for metric, values in rows.items():
        for i, d in enumerate(dates):
            if i < len(values):
                data[d][metric] = values[i]
    return pd.DataFrame(data)


class TestRevenueDeclineFlag:
    def test_four_quarters_declining_triggers(self, tmp_path):
        """Revenue YoY < 0 for 4 consecutive quarters → trap flag."""
        # 8 quarters: older 4 are 100, newer 4 are 80 (each -20% YoY)
        rev_old = [100.0] * 4
        rev_new = [80.0] * 4
        revenue = rev_new + rev_old  # most-recent-first
        fin_df = _quarterly_df_from_rows({"Total Revenue": revenue}, n_quarters=8)

        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=fin_df), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert "營收連續 4 季 YoY 下滑" in result["flags"]

    def test_revenue_stable_no_flag(self, tmp_path):
        revenue = [100.0] * 8
        fin_df = _quarterly_df_from_rows({"Total Revenue": revenue}, n_quarters=8)
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=fin_df), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert "營收連續 4 季 YoY 下滑" not in result["flags"]

    def test_only_3_declining_quarters_no_flag(self, tmp_path):
        # Latest is flat YoY, earlier 3 are declining
        revenue = [100.0, 80.0, 80.0, 80.0, 100.0, 100.0, 100.0, 100.0]
        fin_df = _quarterly_df_from_rows({"Total Revenue": revenue}, n_quarters=8)
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=fin_df), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert "營收連續 4 季 YoY 下滑" not in result["flags"]


class TestNegativeFCFFlag:
    def test_negative_fcf_triggers(self, tmp_path):
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=pd.DataFrame()), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": -1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert "自由現金流為負" in result["flags"]

    def test_positive_fcf_no_flag(self, tmp_path):
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=pd.DataFrame()), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 5e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert "自由現金流為負" not in result["flags"]


class TestGrossMarginDrop:
    def test_margin_drop_over_300bps_triggers(self, tmp_path):
        # GM latest = 40/100 = 40%; prev = 45/100 = 45% → drop 500bps
        fin_df = _quarterly_df_from_rows(
            {
                "Total Revenue": [100.0, 100.0, 100.0, 100.0],
                "Gross Profit": [40.0, 45.0, 45.0, 45.0],
            },
            n_quarters=4,
        )
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=fin_df), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert any("毛利率" in f for f in result["flags"])

    def test_margin_drop_under_300bps_no_flag(self, tmp_path):
        fin_df = _quarterly_df_from_rows(
            {
                "Total Revenue": [100.0, 100.0, 100.0, 100.0],
                "Gross Profit": [44.0, 45.0, 45.0, 45.0],  # -100bps
            },
            n_quarters=4,
        )
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=fin_df), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert not any("毛利率" in f for f in result["flags"])


class TestOCFNegativeConsecutive:
    def test_two_negative_quarters_trigger(self, tmp_path):
        # OCF: [-1e8, -2e8, 5e8, 6e8] — two most-recent are negative
        dates = pd.date_range(end="2025-12-31", periods=4, freq="QE")[::-1]
        df = pd.DataFrame(
            {d: {"Operating Cash Flow": v} for d, v in zip(dates, [-1e8, -2e8, 5e8, 6e8])}
        )
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=pd.DataFrame()), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=df):
            result = check_value_trap("TEST", str(tmp_path))
        assert "營業現金流連 2 季為負" in result["flags"]

    def test_one_negative_quarter_no_flag(self, tmp_path):
        dates = pd.date_range(end="2025-12-31", periods=4, freq="QE")[::-1]
        df = pd.DataFrame(
            {d: {"Operating Cash Flow": v} for d, v in zip(dates, [-1e8, 5e8, 5e8, 6e8])}
        )
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=pd.DataFrame()), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": 1e9}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=df):
            result = check_value_trap("TEST", str(tmp_path))
        assert "營業現金流連 2 季為負" not in result["flags"]


class TestSeverityAndIsTrapFlag:
    def test_severity_2_triggers_is_trap(self, tmp_path):
        # Negative FCF + OCF 2 consecutive negatives
        dates = pd.date_range(end="2025-12-31", periods=4, freq="QE")[::-1]
        ocf_df = pd.DataFrame(
            {d: {"Operating Cash Flow": v} for d, v in zip(dates, [-1e8, -2e8, 5e8, 6e8])}
        )
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=pd.DataFrame()), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": -5e8}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=ocf_df):
            result = check_value_trap("TEST", str(tmp_path))
        assert result["severity"] == 2
        assert result["is_trap"] is True

    def test_severity_1_not_trap(self, tmp_path):
        with patch("src.value_trap_filter.fetch_quarterly_financials", return_value=pd.DataFrame()), \
             patch("src.value_trap_filter.fetch_cashflow", return_value={"free_cashflow": -5e8}), \
             patch("src.value_trap_filter.fetch_quarterly_cashflow", return_value=pd.DataFrame()):
            result = check_value_trap("TEST", str(tmp_path))
        assert result["severity"] == 1
        assert result["is_trap"] is False


class TestCompositeValueTrapOverride:
    def test_buy_class_capped_to_watch(self):
        """Any BUY-class signal with severity>=2 becomes WATCH."""
        key, _, factors = compute_multi_factor_composite(
            "BUY", "positive",  # would be STRONG_BUY
            stock_type="stable",
            value_trap_severity=2,
        )
        assert key == "WATCH"
        assert factors.get("Value Trap") == -1

    def test_non_buy_unaffected(self):
        """CAUTION/NEUTRAL/SELL are not affected by the cap."""
        key, _, factors = compute_multi_factor_composite(
            "CAUTION", "neutral",  # CAUTION base
            stock_type="stable",
            value_trap_severity=4,
        )
        assert key == "CAUTION"
        assert "Value Trap" not in factors

    def test_severity_1_does_not_cap(self):
        """Severity < 2 does not trigger override."""
        key, _, _ = compute_multi_factor_composite(
            "BUY", "positive",
            stock_type="stable",
            value_trap_severity=1,
        )
        assert key == "STRONG_BUY"

    def test_regime_and_value_trap_compose(self):
        """RISK_OFF first demotes BUY→CAUTIOUS_BUY, then value_trap caps BUY-class to WATCH."""
        key, _, factors = compute_multi_factor_composite(
            "BUY", "neutral",  # base = BUY
            stock_type="stable",
            market_regime="RISK_OFF",
            value_trap_severity=2,
        )
        # BUY → CAUTIOUS_BUY (RISK_OFF) → WATCH (value trap cap)
        assert key == "WATCH"
        assert factors.get("Market Regime") == -1
        assert factors.get("Value Trap") == -1
