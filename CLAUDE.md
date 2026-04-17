# CLAUDE.md

Guidance for Claude Code when working in this repo. Kept short on purpose — the
code is the source of truth; this file explains architecture and pitfalls that
aren't obvious from reading files in isolation.

---

## 1. What this project does

PE_monitor is a personal-watchlist signal scanner. For each ticker it produces a
5-level buy/sell recommendation (BUY / WATCH / NEUTRAL / CAUTION / SELL) plus an
optional sentiment-adjusted composite label. Outputs go to:

- **CLI** (`python main.py`) — daily CSV to `reports/daily_YYYYMMDD.csv`, email
  alerts if SMTP is configured.
- **Streamlit** (`streamlit run app.py`) — interactive per-ticker breakdown with
  V/Q/M factor bars, news, supplementary valuation metrics.

---

## 2. Signal pipeline — the new core (QVM)

The primary signal is a **three-factor composite** (V·Value + Q·Quality +
M·Momentum), not a single P/E percentile. Each factor is scored 0-100 (higher
is better), weighted by stock type, then mapped to a 5-level signal. A
quality gate and trend filter can downgrade BUY to WATCH. Finally news
sentiment overlays a ±1-step adjustment.

```
 V, Q, M  ∈ [0, 100]
     │
     ▼
  QVM_raw = w_V·V + w_Q·Q + w_M·M     (weights in QVM_WEIGHTS[stock_type])
     │
     ▼
  map: > 75 → BUY, 65-75 → WATCH, 35-65 → NEUTRAL, 25-35 → CAUTION, < 25 → SELL
     │
     ▼
  Quality gate (non-ETF only):
      if OCF ≤ 0 or TTM EPS ≤ 0 → cap at WATCH
     │
     ▼
  Trend filter:
      if price < SMA200 × 0.85 and base is BUY → downgrade to WATCH
     │
     ▼
  News sentiment overlay → STRONG_BUY / BUY / ... / STRONG_SELL
```

All this lives in [src/factors/qvm_composite.py](src/factors/qvm_composite.py).
Weight table per stock_type is `QVM_WEIGHTS` at the top of that file.

### Per-factor files

| Factor | File | What it reads |
|--------|------|---------------|
| V | [src/factors/value_factor.py](src/factors/value_factor.py) | TTM/Forward P/E, P/B percentile, CAPE-like, P/FCF, EV/EBITDA, stock-vs-industry ratio |
| Q | [src/factors/quality_factor.py](src/factors/quality_factor.py) | Gross margin, ROE, operating margin, EPS stability, Debt/Equity |
| M | [src/factors/momentum_factor.py](src/factors/momentum_factor.py) | 12-1 momentum percentile + Strategy D bonus |

All three **skip missing inputs instead of penalising** them — a factor with
three missing inputs out of five averages only the two it has. In
`compute_qvm`, missing *factors* (entire V, Q, or M) cause weight
renormalisation to the available ones.

### ETF path

ETFs do not have consolidated fundamentals, so V and Q are replaced by
subtype-specific proxies in [src/etf_signal.py](src/etf_signal.py):

| subtype | V input | Q input |
|---------|---------|---------|
| `broad` | Shiller CAPE percentile → price-pct fallback | AUM + expense ratio |
| `sector` | ETF P/E vs Damodaran industry P/E | ditto |
| `dividend` | 5Y dividend-yield percentile (reversed: high yield = cheap) | ditto |
| `commodity` | 5Y price percentile (reversed: low price = cheap) | AUM only |
| `bond` | 10Y Treasury yield percentile | AUM only |

The quality gate is **bypassed** for ETFs (no OCF/EPS concept).

---

## 3. Code layout

```
src/
  data_fetcher.py       yfinance wrapper + CSV/JSON cache
  stock_analyzer.py     classify_etf_subtype, stock_type heuristic
  pe_calculator.py      TTM EPS, PE/PB series, percentile, 5-level mapping
  technical_signals.py  Strategy D (KD + MACD convergence)
  sentiment_analyzer.py VADER news sentiment (English only — known limitation)
  news_fetcher.py       Yahoo Finance RSS + fallback scrape
  composite_signal.py   news sentiment × base signal matrix
  factors/
    value_factor.py
    quality_factor.py
    momentum_factor.py
    qvm_composite.py    weight table, gates, compute_qvm
  etf_signal.py         per-subtype V and Q for ETFs
  external_data.py      Shiller CAPE scrape, Damodaran industry PE scrape
  etf_industry_map.py   ETF→industry and yfinance→Damodaran lookup tables
  report_generator.py   scan_ticker: the one entry point used by CLI/UI
  backtest.py           walk-forward QVM replay
  notifier.py           SMTP + macOS local notification
  utils.py              config load/save

tests/                  pytest suite (50 tests, all offline via monkeypatch)
PLAN.md                 phased roadmap; check the progress table before starting
```

`scan_ticker(ticker, config)` in [src/report_generator.py](src/report_generator.py)
is the single function the CLI (`main.py`) and UI (`app.py`) call per ticker.
If you need to change what a signal considers, it's almost always this function
plus one of the `factors/*.py` files.

---

## 4. Caching rules

[src/data_fetcher.py](src/data_fetcher.py) stores everything under `data/`:

| File | TTL | Notes |
|------|-----|-------|
| `{ticker}_info.json` | 6 h | yfinance .info dict |
| `{ticker}_price_history.csv` | 6 h | **Years parameter is ignored after cache hit** — if you need more history, delete the file first |
| `{ticker}_quarterly_financials.csv` | 12 h | |
| `{ticker}_cashflow.json` | 12 h | |
| `{ticker}_news.json` | 1 h | |
| `_shiller_cape.csv` | 30 d | scraped from multpl.com |
| `_damodaran_pe.csv` | 365 d | scraped from NYU pedata.html |

Cache is **not aware of earnings announcements** — a stale cache can hold
pre-earnings EPS for hours after a release. Delete the relevant file to force a
refetch when this matters.

---

## 5. Tests and backtest

```bash
# Run the full suite (offline, no network)
.venv/bin/python -m pytest tests/ -q

# Walk-forward QVM replay on a single ticker
.venv/bin/python -m src.backtest --ticker AAPL --start 2022-01-01
```

All test modules monkeypatch network-touching functions so CI / new clones can
run without credentials. The backtest holds V and Q **constant at current
values** — true point-in-time fundamentals aren't in yfinance. Signal
frequencies and momentum dynamics are reliable, but treat absolute return
numbers as indicative, not tradeable.

---

## 6. Common pitfalls

1. **yfinance returns DataFrame, not Series** — some endpoints changed format
   between minor versions. When a calculation yields NaN unexpectedly, check
   with `.squeeze()` / `.iloc[:, 0]` whether the input is a 1-col DataFrame.

2. **Timezone-aware vs tz-naive indexes** — yfinance dividends and prices
   occasionally come back with `UTC` or ticker-local timezones while a
   derived series is tz-naive. Align via
   `idx.tz_convert("UTC").tz_localize(None)` (see [`_to_tz_naive`](src/factors/momentum_factor.py:23))
   before comparing or reindexing.

3. **Dividend yield units** — yfinance returns decimal (0.03) for most US
   tickers but percent (3.0) for some Taiwan ETFs. Normalise in
   `_dividend_yield_percentile` by checking `if yield > 1: yield /= 100`.

4. **MultiIndex columns from `yf.download`** — newer yfinance versions return
   `MultiIndex([('Close', 'TICKER'), ...])`. The cache flattens via
   `_flatten_columns` but old CSVs written before that fix will read back as
   two-row-header disasters. `_is_valid_price_df` detects this and deletes;
   if you see KeyError on "Close", remove the `_price_history.csv` file.

5. **Config merging** — `load_config` fills in missing keys from
   `_SETTINGS_DEFAULTS`. If you add a new setting, put its default there so old
   user configs pick it up on next run without needing a manual edit.

6. **lxml is NOT a dependency** — external_data.py uses BeautifulSoup
   directly for Shiller and Damodaran because lxml wasn't available in the
   reference environment. Don't introduce `pd.read_html` without adding lxml
   to requirements.

7. **Watchlist entries drive stock_type** — if a ticker isn't in `config.json`
   watchlist, it falls back to `stock_type="unknown"` with equal-weight
   QVM_WEIGHTS. The backtest and CLI both try to load the real config first.

---

## 7. When adding a factor or input

1. Extend the factor's `Inputs` dataclass (V: `ValueInputs`, Q: `QualityInputs`).
2. In `compute_*_score`, score it into components with a clear label and
   `"kind"` of `"percentile" | "absolute"`.
3. In [src/report_generator.py](src/report_generator.py) `scan_ticker`,
   populate the input before calling `compute_v_score` / `compute_q_score`.
4. Add a unit test under `tests/` mirroring the existing patterns (monkeypatch
   any network call).
5. Run the backtest on a known ticker to sanity-check no regression in
   signal counts.

---

## 8. Related documents

- [PLAN.md](PLAN.md) — phased roadmap with machine-readable progress checkboxes.
  Check it first to see which phase is current.
- [README.md](README.md) — user-facing setup and usage guide.
