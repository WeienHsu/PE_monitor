# PE Monitor

個人自選股訊號掃描工具，以 **QVM 三因子模型**（Value + Quality + Momentum）為核心，每日輸出 5 級買賣訊號，支援個股與 ETF。

---

## 功能亮點

- **QVM 三因子複合訊號**：V（估值便宜度）、Q（公司品質）、M（價格動能）加權合成 0-100 分，映射至 BUY／WATCH／NEUTRAL／CAUTION／SELL
- **品質閘門**：OCF ≤ 0 或 TTM EPS ≤ 0 時，BUY 封頂為 WATCH
- **趨勢過濾**：價格 < SMA200 × 0.85 時，BUY 降為 WATCH
- **新聞情緒疊加**：VADER（英文）+ SnowNLP（中文）雙語情緒分析，訊號 ±1 步調整
- **ETF 完整支援**：broad／sector／dividend／commodity／bond 五種子類型各有專屬估值替代物（Shiller CAPE、Damodaran 產業 PE、殖利率百分位等）
- **倉位建議**：依 QVM 分數給出「可加碼 10%」至「建議減倉 10%」五段建議
- **快取感知財報日**：財報發佈日 ±1 天自動縮短快取 TTL 至 1h，避免使用過期 EPS
- **yfinance 容錯機制**：網路失敗自動 retry（2 次），仍失敗時退回 stale cache
- **持倉管理**：追蹤成本價、股數、即時損益 %
- **CLI + Streamlit**：命令列每日掃描輸出 CSV；互動式 Web UI 含 V/Q/M 進度條

---

## 快速開始

### 1. 安裝

```bash
git clone https://github.com/WeienHsu/PE_monitor.git
cd PE_monitor

python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
python -m nltk.downloader vader_lexicon   # 下載 VADER 情緒模型（首次需要）
```

### 2. 初始設定（選用）

```bash
cp .env.example .env
```

編輯 `.env`：

```dotenv
# 自選股（逗號分隔；可附類型：AAPL:stable）
WATCHLIST=GOOGL,AAPL:stable,NVDA:growth,XOM:cyclical,SPY,0056

# 持倉（ticker:成本:股數，可留空）
HOLDINGS=AAPL:195.0:5,NVDA:850.0:2

# Finnhub API Key（免費：https://finnhub.io/register）
FINNHUB_API_KEY=your_key_here

# 其他（有預設值可不填）
PE_HISTORY_YEARS=5
ENTRY_PERCENTILE=25
EXIT_PERCENTILE=75
```

> `.env` 只在第一次啟動時初始化 `config.json`，之後請透過 Streamlit 介面操作。

### 3. 啟動 Streamlit

```bash
.venv/bin/streamlit run app.py
```

開啟瀏覽器 → `http://localhost:8501`

---

## 命令列每日掃描

```bash
python main.py
```

輸出至 `reports/daily_YYYYMMDD.csv`，欄位包含：ticker、price、signal、composite_signal、v_score、q_score、m_score、qvm_raw、position_suggestion 等。

---

## 訊號說明

### QVM 複合訊號（5 級）

| 訊號 | QVM 分數 | 說明 |
|------|---------|------|
| 🟢 BUY | > 75 | 三因子綜合低估，考慮進場 |
| 🔵 WATCH | 65–75 | 接近進場區，觀察 |
| ⚪ NEUTRAL | 35–65 | 合理估值 |
| 🟡 CAUTION | 25–35 | 接近出場區，謹慎 |
| 🔴 SELL | < 25 | 綜合高估，考慮減倉 |

品質閘門（OCF ≤ 0 或 EPS ≤ 0）和趨勢過濾（價格 < SMA200 × 0.85）可將 BUY 降為 WATCH。

### 新聞情緒（雙語）

| 語言 | 模型 |
|------|------|
| 英文 | VADER（離線，NLTK） |
| 中文 | SnowNLP（離線，無需 GPU） |

情緒最多讓訊號偏移一級，不會整個翻轉估值訊號。

### 倉位建議

| QVM 分數 | 建議 |
|---------|------|
| > 75 | 可加碼 10% |
| 65–75 | 可加碼 5% |
| 35–65 | 維持現有倉位 |
| 25–35 | 建議減倉 5% |
| < 25 | 建議減倉 10% |

> 僅供參考，非投資建議。

---

## ETF 支援

| 子類型 | 範例 | V 因子替代物 |
|-------|------|-------------|
| broad | SPY, VOO, 0050 | Shiller CAPE 百分位 |
| sector | XLK, SOXX, XLF | ETF P/E vs Damodaran 產業 PE |
| dividend | 0056, SCHD | 5 年殖利率百分位（反向） |
| commodity | GLD, SLV | 5 年價格百分位（反向） |
| bond | TLT, IEF | 10Y 美債殖利率百分位 |

ETF 不套用品質閘門（無 OCF／EPS 概念）。

---

## V / Q / M 因子詳細

### V — 估值便宜度（Value）

| 輸入 | 方向 |
|------|------|
| TTM P/E 5 年歷史百分位 | 低 = 高分 |
| Forward P/E | 低 = 高分 |
| P/B 5 年歷史百分位 | 低 = 高分 |
| P/FCF | 低 = 高分 |
| EV/EBITDA | 低 = 高分 |
| CAPE-like（5 年平均 EPS 的 P/E） | 低 = 高分 |
| TTM P/E vs 產業中位數 | 低 = 高分 |

### Q — 公司品質（Quality）

| 指標 | 門檻 |
|------|------|
| 毛利率 | > 40% → 100 分；< 10% → 0 分 |
| ROE | > 20% → 100 分；< 5% → 0 分 |
| 營業利益率 | > 20% → 100 分；< 0% → 0 分 |
| EPS 穩定度（YoY std%） | < 30% → 100 分；> 60% → 0 分 |
| 負債權益比 | < 50 → 100 分；> 200 → 0 分 |

### M — 價格動能（Momentum）

- **12-1 Momentum**：過去 12 個月報酬減最近 1 個月報酬，對自身 5 年歷史取百分位
- **Strategy D**：KD + MACD 收斂訊號觸發時給予 +10 分 bonus（需在設定中啟用）

---

## 各類型 QVM 預設權重

| 類型 | V | Q | M |
|------|---|---|---|
| stable | 0.40 | 0.35 | 0.25 |
| growth | 0.30 | 0.35 | 0.35 |
| cyclical | 0.30 | 0.20 | 0.50 |
| etf_broad | 0.50 | 0.15 | 0.35 |
| etf_sector | 0.45 | 0.15 | 0.40 |
| etf_dividend | 0.50 | 0.10 | 0.40 |
| etf_commodity | 0.25 | 0.00 | 0.75 |
| etf_bond | 0.35 | 0.00 | 0.65 |
| unknown | 0.35 | 0.30 | 0.35 |

---

## 快取策略

| 檔案 | TTL | 備註 |
|------|-----|------|
| `{ticker}_info.json` | 6 h | yfinance .info |
| `{ticker}_price_history.csv` | 6 h | 網路失敗退回 stale cache |
| `{ticker}_quarterly_financials.csv` | 12 h* | *財報日 ±1 天縮至 1 h |
| `{ticker}_cashflow.json` | 12 h* | *同上 |
| `{ticker}_earnings_date.json` | 24 h | 財報日快取 |
| `{ticker}_news.json` | 1 h | |
| `_shiller_cape.csv` | 30 d | multpl.com |
| `_damodaran_pe.csv` | 365 d | NYU pedata.html |

---

## 專案結構

```
PE_monitor/
├── main.py                      # CLI 每日掃描入口
├── app.py                       # Streamlit Web UI
├── config.json                  # 自選股、持倉、設定（自動產生，不上傳 git）
├── .env                         # 個人設定（不上傳 git）
├── .env.example
├── requirements.txt
├── data/                        # 本地快取
├── reports/                     # 每日 CSV 報告
├── tests/                       # pytest（50 tests，全離線）
└── src/
    ├── data_fetcher.py          # yfinance wrapper + 快取 + 容錯
    ├── pe_calculator.py         # TTM EPS、PE/PB 序列、百分位
    ├── stock_analyzer.py        # 股票類型分類、ETF 子類型判斷
    ├── technical_signals.py     # Strategy D（KD + MACD 收斂）
    ├── sentiment_analyzer.py    # VADER（英文）+ SnowNLP（中文）情緒分析
    ├── news_fetcher.py          # Yahoo Finance RSS + Finnhub 新聞抓取
    ├── composite_signal.py      # 估值 × 情緒複合訊號矩陣
    ├── report_generator.py      # scan_ticker：完整 QVM pipeline
    ├── backtest.py              # walk-forward QVM 回測
    ├── notifier.py              # SMTP + macOS 本地通知
    ├── etf_signal.py            # ETF V/Q 計算（依子類型）
    ├── external_data.py         # Shiller CAPE、Damodaran PE 抓取
    ├── etf_industry_map.py      # ETF → Damodaran 產業對照表
    ├── utils.py                 # config 載入／儲存
    └── factors/
        ├── value_factor.py      # V 因子
        ├── quality_factor.py    # Q 因子
        ├── momentum_factor.py   # M 因子
        └── qvm_composite.py     # QVM 加權、閘門、倉位建議
```

---

## 測試與回測

```bash
# 全套測試（離線，不需網路）
.venv/bin/python -m pytest tests/ -q

# walk-forward QVM 回測
.venv/bin/python -m src.backtest --ticker AAPL --start 2022-01-01
```

回測保持 V、Q 在當前值固定，僅 M（動能）逐日重算，適合觀察訊號頻率與動能動態，絕對報酬數字僅供參考。

---

## 設定說明

### Sidebar 設定項目（Streamlit）

| 設定 | 預設值 | 說明 |
|------|--------|------|
| PE 歷史年數 | 5 年 | 計算百分位的歷史範圍 |
| 買進百分位 | 25 | 低於此值 → BUY 訊號 |
| 賣出百分位 | 75 | 高於此值 → SELL 訊號 |
| 新聞回顧天數 | 14 天 | 情緒分析時間窗口 |
| Strategy D | 停用 | KD + MACD 技術訊號 bonus |

### `.env` 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `WATCHLIST` | 初始自選股（`ticker` 或 `ticker:type`） | `GOOGL,AAPL,MSFT` |
| `HOLDINGS` | 初始持倉（`ticker:成本:股數`） | 空白 |
| `FINNHUB_API_KEY` | Finnhub API Key（新聞主力來源） | 空白（fallback RSS） |
| `PE_HISTORY_YEARS` | 歷史年數 | `5` |
| `ENTRY_PERCENTILE` | 進場百分位 | `25` |
| `EXIT_PERCENTILE` | 出場百分位 | `75` |

---

## 注意事項

- 資料來源為 [yfinance](https://github.com/ranaroussi/yfinance)，僅供個人研究，非投資建議
- VADER 為英文情緒模型；中文標題使用 SnowNLP
- Finnhub 免費方案限 60 次/分鐘；10 支股票以內通常不觸發限制
- 快取不感知財報日以外的突發事件；有重要消息時請手動刪除 `data/{ticker}_info.json`
