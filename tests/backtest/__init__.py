"""
P3-15: Backtest framework.

Purpose
-------
Validate that our composite signals (BUY, STRONG_BUY, CAUTIOUS_BUY, ...) are
actually predictive of forward returns.  Too much of the system today is
"rules that feel right" — Phase 3 ended without ever measuring how BUY-class
signals perform over the next 20/60/120 days.

Module layout
-------------
- ``load_historical_signals.py``
      Walk ``reports/daily_*.csv`` and return a time-series of
      (date, ticker, signal, metric_value, ...) records.
- ``compute_forward_returns.py``
      Given a (date, ticker) pair, compute N-day forward return using cached
      price data.  Returns None when insufficient future data is available.
- ``backtest_metrics.py``
      Aggregate: win-rate, mean return, Sharpe (annualised using √252),
      maximum drawdown, alpha vs SPY over the same window.
- ``run_backtest.py``
      CLI entry: ``python -m tests.backtest.run_backtest --years 3
      --signal BUY --horizon 60``.

Initial-data caveat
-------------------
On a freshly-cloned repo with only one or two daily reports in ``reports/``
there simply isn't enough signal data to draw conclusions. The framework is
built to degrade gracefully — ``run_backtest.py`` prints a warning and
exits with meaningful output even when the sample is tiny. Meaningful
backtests require ~6 months of daily reports or a one-off historical-signal
replay (not in scope here).
"""
