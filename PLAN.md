# PE_monitor 專案體檢 + QVM 三因子核心重構 + ETF 訊號方案

> **跨電腦續做指南**：這份計畫是專案重構的主文件。在新電腦上只要 clone 下來、建好 venv、裝 `requirements.txt`，然後打開此檔對照 **"進度快照"** 區的勾選狀態，找出下一個未完成的 Phase 繼續做即可。所有檔案路徑都是相對於 repo 根目錄，可以直接點開。

---

## 進度快照（每完成一項就勾起來）

- [x] **Phase 1** — QVM 因子基礎設施（個股已可產出 QVM 訊號）
  - [x] `src/factors/value_factor.py` / `quality_factor.py` / `momentum_factor.py` / `qvm_composite.py`
  - [x] `src/pe_calculator.py::calc_cape_style_pe()`
  - [x] `src/report_generator.py::scan_ticker()` 改走 12 階段 QVM pipeline
  - [x] `src/composite_signal.py` 多因子投票降為 display-only
  - [x] `app.py` UI 顯示 V/Q/M 三進度條 + 閘門狀態
  - [x] Smoke test：`WATCHLIST="AAPL:stable,JNJ:stable,NVDA:growth,XOM:cyclical" python main.py` 四檔皆非 N/A
- [x] **Phase 2** — ETF 整合進 QVM
  - [x] `src/stock_analyzer.py::classify_etf_subtype()`
  - [x] `src/etf_signal.py` (compute_etf_v_score / compute_etf_q_score)
  - [x] `src/report_generator.py` 在 ETF 分支切換到 ETF V/Q
  - [x] 驗證：SPY / XLK / 0056 / GLD / TLT 皆非 N/A
- [x] **Phase 3** — 外部資料源（Shiller CAPE、Damodaran 產業 PE）
  - [x] `src/external_data.py`
  - [x] `src/etf_industry_map.py`
  - [x] ETF broad_equity 吃 CAPE；sector_equity 吃產業 PE
- [x] **Phase 4** — 測試 + 回測 + 文件
  - [x] `tests/test_*.py`（value/quality/momentum/qvm_composite/etf_signal）— 50 tests
  - [x] `src/backtest.py` — `python -m src.backtest --ticker AAPL --start 2022-01-01`
  - [x] `CLAUDE.md`
- [ ] **Phase 5** — 長線補強（擇期）
  - [ ] 中文新聞情緒、yfinance 備援、快取感知財報日、倉位建議

---

## Context

現狀的核心訊號是「TTM EPS → 歷史 PE 百分位」，經使用者與研究後確認這個做法**在規模／景氣劇變／商業模式轉型時會系統性失真**（2008–09、2020 期間 S&P 500 TTM PE 分別飆到三位數與 46.7，就是反例）。同時 ETF 因為沒有合併 EPS，整條主訊號鏈被跳過，最終 composite_signal 都是 N/A。

使用者已確認三項方向：
1. **核心訊號**改為 Quality + Value + Momentum 三因子模型（QVM，AQR/Asness 做法）
2. **品質閘門**輕量：只擋 OCF > 0 且 TTM EPS > 0 的負值
3. **趨勢過濾**：現價 < SMA200 × 0.85 時，BUY 級別降為 WATCH（避免接刀子）

此計畫交付：(A) 專案體檢、(B) QVM 核心重構方案、(C) ETF 納入 QVM 的設計、(D) 參考依據、(E) 分期實作步驟。

---

## Part A｜整體架構體檢

### 強項（保留）
- 模組分層乾淨；data_fetcher / pe_calculator / composite_signal / news / technical 各司其職
- Type-adaptive 設計是很好的抽象——QVM 會直接擴充這張表
- 補充指標不阻斷主訊號
- 快取策略合理（價格 6h / 財報 12h / 新聞 1h）

### 弱項（按影響排序）

| # | 問題 | 位置 | 本計畫處理？ |
|---|------|------|--------------|
| 1 | **單一 TTM PE 百分位做主訊號脆弱** | `pe_calculator.classify_signal` 被當作全系統起點 | ✅ 改 QVM 核心 |
| 2 | **ETF 全無訊號** | `report_generator.py` | ✅ ETF 也走 QVM |
| 3 | **沒有任何測試** | 無 `tests/` | ✅ Phase 4 補 |
| 4 | **沒有回測** | 無 | ✅ Phase 4 補簡易 backtest |
| 5 | VADER 只支英文，台股新聞情緒失效 | `sentiment_analyzer.py` | 延後（Phase 5） |
| 6 | yfinance 無備援，來源風險 | `data_fetcher.py` | 延後（Phase 5） |
| 7 | Plan B 中 Forward PE 與 TTM PE 有雙重計入疑慮 | `composite_signal.py` | ✅ QVM 會重整此邏輯 |
| 8 | 無倉位／風險管理 | — | 範圍外 |
| 9 | 快取無法感知財報日 | `data_fetcher.py` | 延後 |
| 10 | 無 CLAUDE.md | 根目錄 | ✅ Phase 4 補 |

---

## Part B｜QVM 三因子核心重構（專案的新心臟）

### 為什麼要重構
- **PE 百分位失效情境**：①景氣劇變分母崩潰（2008、2020）②成長股被歷史區間低估 ③商業模式改變（Meta 2022 案例）④市場結構長期漂移
- **學術證據**：O'Shaughnessy 多比率 composite 勝單一比率 82%；Piotroski F-score 高者年化 34.1% vs 低者 7.8%；Novy-Marx 顯示 Value + Quality 互補；AQR 多因子橫跨 8 個市場一致勝出單因子

### QVM 三因子定義

每個因子輸出 **0–100 分數**（100 最佳），彙總得 QVM composite 再對應 5 級訊號。

#### V — Value Factor（估值便宜度）
多估值比率百分位平均，每個比率各自算該股**過去 5 年的百分位**，再把「便宜」方向調成高分：

| 輸入比率 | 來源 | 分數轉換 |
|---------|------|---------|
| TTM P/E | 既有 `calc_ttm_eps` + history | 低百分位 → 高 V 分 |
| Forward P/E | 既有 `get_forward_pe` | 低百分位 → 高 V 分 |
| P/B | 既有 `get_pb_ratio` | 低百分位 → 高 V 分 |
| P/FCF (或 P/OCF) | 既有 `get_pcf_ratio` | 低百分位 → 高 V 分 |
| EV/EBITDA | 既有 `get_ev_ebitda` | 低百分位 → 高 V 分 |
| CAPE-like（5Y 平均 EPS 的 PE） | 新增：`calc_cape_style_pe()` | 低百分位 → 高 V 分 |

V_score = 可得到值的平均。缺值不罰分、不強補（與既有「缺值不阻塞」原則一致）。

#### Q — Quality Factor（公司品質）
絕對門檻對映分數（不做橫向排名，因為自選股只有少數幾檔）：

| 指標 | 來源 | 映射（示意，Phase 4 回測校準） |
|------|------|-------------------------------|
| Gross Margin (Novy-Marx) | `info.grossMargins` | > 40% → 100；< 10% → 0；線性插值 |
| ROE | `info.returnOnEquity` | > 20% → 100；< 5% → 0 |
| Operating Margin | `info.operatingMargins` | > 20% → 100；< 0% → 0 |
| EPS 穩定度（σ of YoY growth） | 既有 `analyze_suitability` 已算 std_pct | std_pct < 30% → 100；> 60% → 0 |
| Debt/Equity | `info.debtToEquity` | < 50 → 100；> 200 → 0（反向） |

Q_score = 可得分數的平均。

#### M — Momentum Factor（價格動能）
- **12-1 Momentum**（AQR 經典做法）：過去 12 個月報酬減掉最近 1 個月報酬，`(P_{t-21} / P_{t-252}) - 1`
- 百分位：跟這支股票自己過去 5 年的 12-1 momentum 分布比
- 加上 Strategy D 訊號作為 +10 分 bonus（若啟用且觸發）

M_score = 12-1 momentum 百分位 + Strategy D bonus，截斷 0–100。

### 最終訊號產生流程

```
  V_score, Q_score, M_score  ∈ [0, 100]

  QVM_raw = w_V × V + w_Q × Q + w_M × M        # 權重依 stock_type

  ─── Quality Gate（輕量，使用者指定）───
  if OCF ≤ 0 or TTM EPS ≤ 0:
      base_signal 封頂為 WATCH（不得 BUY/STRONG_BUY）

  ─── 5 級訊號對應 ───
  QVM_raw > 75  → BUY
  QVM_raw 65-75 → WATCH_BUY   （對齊現有 WATCH）
  QVM_raw 35-65 → NEUTRAL
  QVM_raw 25-35 → CAUTION
  QVM_raw < 25  → SELL

  ─── Trend Filter（使用者指定）───
  if price < SMA200 × 0.85 and base_signal in (BUY, STRONG_BUY):
      降級為 WATCH

  ─── News Sentiment 疊加（沿用既有 matrix）───
  final = base_signal × sentiment  →  STRONG_BUY / BUY / ... / STRONG_SELL
```

### 各類型預設權重

| 類型 | V | Q | M | 說明 |
|------|---|---|---|------|
| stable | 0.40 | 0.35 | 0.25 | 穩定型重基本面 |
| growth | 0.30 | 0.35 | 0.35 | 成長型 V 與 M 持平，Q 看獲利品質 |
| cyclical | 0.30 | 0.20 | 0.50 | 景氣循環看動能／趨勢 |
| etf_broad | 0.50 | 0.15 | 0.35 | ETF 品質弱，CAPE 主導 |
| etf_sector | 0.45 | 0.15 | 0.40 | 產業 PE 主導 + 動能 |
| etf_dividend | 0.50 | 0.10 | 0.40 | 殖利率百分位主導 |
| etf_commodity | 0.25 | 0.00 | 0.75 | 無估值，純技術／動能 |
| etf_bond | 0.35 | 0.00 | 0.65 | 利率曲線 + 動能 |
| unknown | 0.35 | 0.30 | 0.35 | 等權 fallback |

---

## Part C｜ETF 在 QVM 框架裡的處理

### ETF 子類型分類（新增 `etf_subtype` 欄位）
在 `src/stock_analyzer.py` ETF 分支內細分：

| subtype | 判斷 | 範例 |
|---------|------|------|
| `broad` | `info.category` 含 "Blend"/"S&P"/"Total Market"/"世界" | SPY, VOO, VTI, 0050 |
| `sector` | `info.category` 含 "Sector"/"Industry" | XLK, SOXX, XLF |
| `dividend` | 殖利率 > 3% 的股票 ETF | 0056, 00878, SCHD |
| `commodity` | `category` 含 "Commodity" | GLD, SLV |
| `bond` | `category` 含 "Bond"/"Treasury" | TLT, IEF, AGG |

### ETF 的 V 因子特殊處理

個股 V 的六個輸入對 ETF 大多不可得，改走子類型專屬的估值替代物：

| subtype | V 輸入 | 資料來源 |
|---------|--------|---------|
| broad | Shiller CAPE 百分位（對 SPY/VOO/VTI）；0050 用 TWSE 本益比歷史 | multpl.com 每月；TWSE opendata |
| sector | Damodaran 產業 PE / ETF trailing PE 比率 | NYU `pedata.xls` 年更 |
| dividend | 殖利率 5 年歷史百分位（**反向**：高殖利率 = 便宜） | yfinance `dividends` + 價格序列自算 |
| commodity | 價格 5 年百分位（反向：低價 = 便宜） | yfinance price |
| bond | 10Y Treasury yield 百分位（高殖利率 = 便宜） | yfinance `^TNX` |

### ETF 的 Q 因子簡化
ETF 無法取得合併 Gross Margin 等品質指標。改用兩個簡單代理：
- `totalAssets` ≥ 500M（規模穩定性）→ 100 分；≤ 50M → 0；線性
- `annualReportExpenseRatio` 倒數縮放（費率越低越好）→ ≤0.10% → 100；≥0.75% → 0

### ETF 的 M 因子與個股一致
12-1 momentum + Strategy D bonus。ETF 技術面原本就跟個股相容。

### Quality Gate 對 ETF 的處理
不套用「OCF > 0」（ETF 沒有 OCF 概念），Quality Gate 對 ETF 直接略過（等同 pass）。

---

## Part D｜參考依據彙整

### 為何多比率 composite 勝單一 PE
- O'Shaughnessy《What Works on Wall Street》(2012, 4th ed.)：Value Composite 勝單一比率 82%
- **Piotroski (2000) "Value Investing: Financial Statement Information to Separate Winners from Losers"**：F-score 高組年化 34.1% vs 低組 7.8%
- **Novy-Marx (2013) "The Other Side of Value: The Gross Profitability Premium"**：毛利率／總資產是最穩健的品質指標

### 為何 QVM 三因子
- **Asness, Moskowitz & Pedersen (2013) "Value and Momentum Everywhere"**（Journal of Finance）——Value + Momentum 跨 8 個市場一致勝出
- **Asness, Frazzini & Pedersen (2019) "Quality Minus Junk"**（Review of Accounting Studies）——Quality 獨立貢獻；與 Value 組合時互相補償
- **AQR "Understanding Factor Investing"** 白皮書——V/Q/M 組合是主流多因子做法
- **12-1 Momentum**：Jegadeesh & Titman (1993)

### 為何 TTM PE 單一百分位脆弱
- **Shiller, Irrational Exuberance**（2000, 2015）——CAPE 的提出根據；單期 EPS 受景氣循環干擾嚴重
- **Advisor Perspectives P/E10 系列**——TTM PE 在 2008/2020 的失真數據
- **Siegel "The Shiller CAPE Ratio: A New Look"**（2016）——CAPE 也有缺陷，但比 TTM 穩

### 趨勢過濾／SMA
- **Faber, Mebane (2007) "A Quantitative Approach to Tactical Asset Allocation"**——SMA200 作為股票與 ETF 配置進出場的經典規則

---

## Part E｜分期實作步驟

### Phase 1｜QVM 因子基礎設施 ✅
**目標**：個股（非 ETF）能以 QVM 產生訊號。

新檔案：
- `src/factors/__init__.py`
- `src/factors/value_factor.py` — `compute_v_score(inputs) → (v_score, details)`
- `src/factors/quality_factor.py` — `compute_q_score(inputs) → (q_score, details)`
- `src/factors/momentum_factor.py` — `compute_m_score(ticker, data_dir, sd_signal) → (m_score, details)`
- `src/factors/qvm_composite.py` — `compute_qvm(v, q, m, stock_type, ...) → {qvm_raw, base_signal, composite_signal, ...}`

修改檔案：
- `src/pe_calculator.py` — 新增 `calc_cape_style_pe()`
- `src/report_generator.py` — `scan_ticker()` 改走 12 階段 QVM 流程
- `src/composite_signal.py` — Plan B 投票改為 display-only
- `app.py` — V/Q/M 進度條 + 閘門徽章

### Phase 2｜ETF 整合進 QVM
- `src/stock_analyzer.py::classify_etf_subtype(ticker, data_dir) → "broad"|"sector"|"dividend"|"commodity"|"bond"`
- 在 `ensure_watchlist_analyzed()` 或新 helper 中把 `etf_subtype` 寫進 watchlist entry
- `src/etf_signal.py`（新）：
  - `compute_etf_v_score(ticker, subtype, price, data_dir, years)` — 依 subtype 走不同估值替代物
  - `compute_etf_q_score(ticker, data_dir)` — AUM + expense ratio
- `src/report_generator.py::scan_ticker()` — is_etf 分支改呼叫 ETF V/Q；M 仍用 `compute_m_score`
- 驗證：SPY / XLK / 0056 / GLD / TLT 各跑一次，全部非 N/A；CSV 出現 `etf_subtype` 欄位

### Phase 3｜外部資料源接入
- `src/external_data.py`（新）
  - `fetch_shiller_cape_series(data_dir)` — multpl.com monthly table scrape；快取 `data/_shiller_cape.csv` TTL 30 天
  - `fetch_damodaran_industry_pe(data_dir)` — NYU `pedata.xls`；快取 `data/_damodaran_pe.csv` TTL 365 天
- `src/etf_industry_map.py`（新）— 硬編碼約 30 檔常見 ETF → Damodaran industry
- ETF V 因子升級：broad_equity 吃 CAPE 百分位、sector_equity 吃 `ETF PE / industry PE` 比值
- 個股延伸：個股 V 亦加一個 "TTM PE vs sector median" 輸入（額外穩健性）

### Phase 4｜測試、回測、文件
- `tests/test_value_factor.py` / `test_quality_factor.py` / `test_momentum_factor.py`
- `tests/test_qvm_composite.py` — 涵蓋所有 stock_type 的權重切換與閘門行為
- `tests/test_etf_signal.py` — 五種 subtype
- `src/backtest.py`（新）— 讀 5 年歷史，replay 每日 QVM 訊號，輸出勝率／平均報酬／最大回撤
- `CLAUDE.md`（新）— 專案結構、訊號語意、偵錯常見問題

### Phase 5｜長線補強（擇期）
- 中文新聞情緒（FinBERT-ZH 或 LLM 替代 VADER）
- yfinance 備援（FMP / Alpha Vantage 抽象化）
- 快取感知財報日
- 倉位建議（依 QVM 分數建議加碼/減碼百分比）

---

## 關鍵檔案對照

| 目的 | 路徑 |
|------|------|
| ETF 偵測 | `src/data_fetcher.py::is_etf()` |
| ETF 分類（擴 subtype） | `src/stock_analyzer.py` |
| 主訊號流程 | `src/report_generator.py::scan_ticker` |
| QVM 三因子 | `src/factors/` |
| Plan B 投票（display-only） | `src/composite_signal.py` |
| 5 級訊號對應 | `src/pe_calculator.py::classify_signal` |
| 技術訊號（M bonus） | `src/technical_signals.py::compute_strategy_d` |
| UI factor breakdown | `app.py`（搜 "composite_factors"） |

---

## 驗證方式

### Phase 1 完成後（個股 QVM 就位）
1. 自選股加入 AAPL、JNJ（stable）、NVDA（growth）、XOM（cyclical）。
2. `python main.py`：每檔顯示 `signal`、`composite_signal`、`v_score`、`q_score`、`m_score`；負 EPS 公司訊號封頂 WATCH。
3. `streamlit run app.py`：細節區出現 V/Q/M 三進度條，每因子可展開看貢獻。
4. CSV `daily_YYYYMMDD.csv` 新增 V/Q/M 三欄。

### Phase 2 完成後（ETF 不再 N/A）
1. 加入 SPY、XLK、0056、GLD、TLT。
2. 所有 ETF 的 `composite_signal` 非 N/A，有 QVM_raw。
3. ETF 細節顯示 `etf_subtype` 徽章。

### Phase 3 完成後（外部資料接上）
1. SPY 的 V_score 主導因子顯示為 CAPE（而非 price_percentile）。
2. XLK V_score 顯示 "ETF trailing PE vs Damodaran Technology industry median"。
3. Damodaran 資料每年 1/5 自動更新；Shiller 每月更新。

### Phase 4 完成後（測試 + 回測）
1. `pytest tests/` 全綠。
2. `python -m src.backtest --ticker AAPL --start 2020-01-01`：輸出訊號時序、各訊號的後續 1M/3M/6M 平均報酬、勝率、最大回撤。

## 風險與取捨

- **回測樣本限制**：yfinance 歷史只有 5 年完整 fundamentals，金融海嘯／網路泡沫無法納入驗證；回測結果僅反映近期市場條件。
- **絕對 Q 門檻需微調**：20% ROE 算「100 分」的門檻，對不同產業（科技 vs 公用事業）尺度不同；Phase 4 回測後可能拆成產業敏感門檻。
- **M 動能的反轉風險**：12-1 momentum 在市場轉折點（2022 熊市初期）會給錯誤訊號；Trend Filter 是關鍵防線但不是萬靈丹。
- **ETF Q 很弱**：subtype 權重已把 Q 降到 0–0.15，但這是「知道自己不知道」的處理，不是解決。
- **與既有 Plan B 的關係**：Plan B 的 cyclical/growth/stable 權重表會併入 QVM 的類型權重表；Plan B 投票（factor breakdown）仍顯示在 UI 供使用者看內部怎麼推出結論，只是不再獨立做升降級。
