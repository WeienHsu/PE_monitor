"""Unit tests for the P3-15 backtest framework."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.backtest.backtest_metrics import (
    metrics_by_signal,
    overall_summary,
)
from tests.backtest.compute_forward_returns import forward_return
from tests.backtest.load_historical_signals import load_historical_signals


class TestLoadHistoricalSignals:
    def test_returns_empty_for_missing_dir(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        df = load_historical_signals(report_dir=str(missing))
        assert df.empty

    def test_loads_single_report(self, tmp_path):
        # Build a minimal daily_YYYYMMDD.csv
        p = tmp_path / "daily_20260101.csv"
        pd.DataFrame(
            [
                {"ticker": "AAPL", "signal": "BUY", "composite_signal": "BUY",
                 "metric_value": 20.0, "stock_type": "growth",
                 "percentile_rank": 15.0, "price": 150.0,
                 "metric_label": "PE", "name": "Apple",
                 "signal_display": "🟢 BUY", "composite_display": "🟢 BUY",
                 "news_label": "neutral"},
                {"ticker": "MSFT", "signal": "WATCH", "composite_signal": "WATCH",
                 "metric_value": 30.0, "stock_type": "stable",
                 "percentile_rank": 30.0, "price": 380.0,
                 "metric_label": "PE", "name": "Microsoft",
                 "signal_display": "🔵 WATCH", "composite_display": "🔵 WATCH",
                 "news_label": "neutral"},
            ]
        ).to_csv(p, index=False, encoding="utf-8-sig")

        df = load_historical_signals(report_dir=str(tmp_path))
        assert len(df) == 2
        assert set(df["ticker"]) == {"AAPL", "MSFT"}
        assert df["date"].iloc[0] == pd.Timestamp("2026-01-01")

    def test_date_range_filtering(self, tmp_path):
        for d in ["20260101", "20260115", "20260201"]:
            pd.DataFrame([{"ticker": "X", "signal": "BUY",
                           "composite_signal": "BUY"}]).to_csv(
                tmp_path / f"daily_{d}.csv", index=False, encoding="utf-8-sig")

        df = load_historical_signals(
            report_dir=str(tmp_path),
            start_date="2026-01-10",
            end_date="2026-01-31",
        )
        # Only 2026-01-15 should pass
        assert len(df) == 1
        assert df["date"].iloc[0] == pd.Timestamp("2026-01-15")


class TestForwardReturn:
    def test_simple_60_day_return(self, tmp_path, monkeypatch):
        """Construct a linear price series and verify horizon math."""
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        prices = pd.DataFrame({"Close": np.arange(100.0, 100.0 + len(dates))},
                              index=dates)
        monkeypatch.setattr(
            "tests.backtest.compute_forward_returns.fetch_price_history",
            lambda ticker, years=5, data_dir="data": prices,
        )

        # Day 0 = 100; day 60 trading days later = 160 → return = 0.60
        r = forward_return("FAKE", "2024-01-01", horizon_days=60, data_dir=str(tmp_path))
        assert r is not None
        assert abs(r - 0.60) < 1e-9

    def test_returns_none_when_not_enough_future_data(self, tmp_path, monkeypatch):
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        prices = pd.DataFrame({"Close": [100.0] * len(dates)}, index=dates)
        monkeypatch.setattr(
            "tests.backtest.compute_forward_returns.fetch_price_history",
            lambda ticker, years=5, data_dir="data": prices,
        )
        r = forward_return("FAKE", "2024-01-01", horizon_days=60, data_dir=str(tmp_path))
        assert r is None


class TestMetricsBySignal:
    def test_empty_input_returns_empty_frame(self):
        df = pd.DataFrame()
        assert metrics_by_signal(df).empty

    def test_aggregates_per_signal(self):
        # Two BUYs (one up, one down) and one SELL (down)
        df = pd.DataFrame(
            [
                {"composite_signal": "BUY", "forward_return": 0.10, "alpha": 0.05},
                {"composite_signal": "BUY", "forward_return": -0.05, "alpha": -0.02},
                {"composite_signal": "SELL", "forward_return": -0.08, "alpha": -0.03},
                {"composite_signal": "SELL", "forward_return": -0.12, "alpha": -0.05},
            ]
        )
        m = metrics_by_signal(df, horizon_days=60)
        assert set(m["signal"]) == {"BUY", "SELL"}
        buy = m[m["signal"] == "BUY"].iloc[0]
        assert buy["count"] == 2
        assert abs(buy["mean_return"] - 0.025) < 1e-9  # (0.10 - 0.05) / 2
        assert buy["win_rate"] == 0.5
        sell = m[m["signal"] == "SELL"].iloc[0]
        assert sell["win_rate"] == 0.0  # neither SELL had positive return

    def test_overall_summary(self):
        df = pd.DataFrame(
            [
                {"composite_signal": "BUY", "forward_return": 0.10, "alpha": 0.05},
                {"composite_signal": "SELL", "forward_return": -0.05, "alpha": -0.02},
            ]
        )
        s = overall_summary(df)
        assert s["count"] == 2
        assert abs(s["mean_return"] - 0.025) < 1e-9
        assert s["win_rate"] == 0.5


class TestCliSmoke:
    def test_cli_exits_nonzero_with_empty_dir(self, tmp_path, capsys):
        from tests.backtest.run_backtest import main
        code = main([
            "--report-dir", str(tmp_path),
            "--data-dir", str(tmp_path),
            "--horizon", "60",
            "--years", "1",
        ])
        assert code == 1  # no reports → warn and exit 1
