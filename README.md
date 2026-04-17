# PE Monitor 📊

每日監測自選美股的估值位置，整合新聞情緒分析、多因子複合訊號與技術動能偵測，幫助判斷進出場時機。

系統根據股票類型（穩定 / 成長 / 景氣循環 / ETF）自動選擇最適合的估值指標，結合 5 年歷史百分位、近期新聞情緒、補充估值因子（P/CF、PEG、Forward P/E、EV/EBITDA），以及 Strategy D 技術訊號（KD + MACD），產生最終複合建議。

---

## 功能亮點

| 功能 | 說明 |
|------|------|
| **估值訊號（5 級）** | P/E 或 P/B 歷史 5 年百分位，BUY → SELL |
| **新聞情緒分析** | Finnhub / Yahoo RSS，VADER 離線評分，時間加權 |
| **複合訊號矩陣** | 估值 × 情緒 = 15 格矩陣（🌟 強力買進 → 🚨 強力賣出） |
| **多因子調整（Plan B）** | P/CF、PEG、Forward P/E、EV/EBITDA 依股票類型加權投票 |
| **Strategy D 技術訊號** | KD 低區黃金交叉 + MACD 柱狀收斂，偵測動能反轉 |
| **股票類型自動識別** | 依 EPS 波動度 / 成長率分類，決定適用指標與因子權重 |
| **P/E Band 走勢圖** | 歷史 P/E 曲線 + 百分位色帶 + 雙軸股價疊圖 |
| **持倉管理** | 成本價 / 股數 / 即時損益 % |
| **每日報告 + Email 通知** | CSV 報告 + 訊號變化觸發 Gmail 通知 |
| **本地快取** | 股價 / 財報 / 新聞分別快取，避免重複 API 呼叫 |
| **Streamlit Web 介面** | 三頁 + 設定，互動式操作 |
| **CLI 掃描** | `python main.py` 一鍵掃描 + 自動儲存報告 |

---

## 快速開始

### 1. 安裝

```bash
git clone https://github.com/WeienHsu/PE_monitor.git
cd PE_monitor

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
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
# 自選股（可附加類型標記）
WATCHLIST=GOOGL:growth,AAPL:growth,MSFT:stable,XOM:cyclical

# 持倉（ticker:成本:股數）
HOLDINGS=GOOGL:175.0:10,AAPL:195.0:5

# 歷史年數與進出場門檻
PE_HISTORY_YEARS=5
ENTRY_PERCENTILE=25
EXIT_PERCENTILE=75

# Finnhub API Key（免費申請：https://finnhub.io/register）
FINNHUB_API_KEY=pk_free...

# Email 通知（可選，需 Gmail App Password）
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=<app_password>
NOTIFICATION_EMAIL=recipient@example.com
SMTP_ENABLED=true
```

> `.env` 只在**第一次啟動**時讀取並初始化 `config.json`。之後請透過 Streamlit 介面或直接編輯 `config.json` 來調整設定。

### 3. 啟動 Streamlit

```bash
streamlit run app.py
```

開啟瀏覽器 → `http://localhost:8501`

### 4. 命令列每日掃描

```bash
python main.py
```

---

## 指標說明

### 主要估值指標

#### P/E 本益比（Price-to-Earnings）

```
TTM EPS  = 最近 4 季淨利 ÷ 稀釋股數
日 P/E   = 當日收盤價 ÷ TTM EPS
```

- EPS 在每次財報公布後更新，兩次財報之間維持固定值
- 距上次財報超過 100 天時顯示 ⚠️ 過期警告
- EPS 為負時，自動切換為 P/B 作為主要指標
- 適用股票類型：**穩定型**（主要）、**成長型**（主要）

#### P/B 帳面價值比（Price-to-Book）

```
P/B = 當日收盤價 ÷ 每股帳面價值
```

- EPS 為負或缺乏獲利記錄時自動啟用
- 適用股票類型：**景氣循環型**、**金融股**（主要）

---

### 補充估值指標（多因子調整用）

以下四個指標作為 Plan B 多因子投票的依據，依股票類型決定是否納入及權重。

#### P/CF 股價現金流比（Price-to-Cash-Flow）

```
P/CF = 當日股價 ÷ 每股 TTM 營業現金流
```

| P/CF 值 | 解讀 |
|---------|------|
| < 10 | 低估（投票 +1） |
| 10 – 20 | 合理（投票 0） |
| > 20 | 高估（投票 −1） |

- 比 P/E 更不易被會計手段操縱
- 適用：**穩定型**（×2 加權）、**景氣循環型**（×1）
- 排除：**成長型**（高成長期現金流常為負，不具代表性）

#### PEG 成長調整本益比

```
PEG = 分析師共識 P/E ÷ 預期年 EPS 成長率（%）
```

| PEG 值 | 解讀 |
|--------|------|
| < 1.0 | 低估（投票 +1） |
| 1.0 – 2.0 | 合理（投票 0） |
| > 2.0 | 高估（投票 −1） |

- 修正 P/E 不考慮成長速度的缺點
- 適用：**穩定型**（×1）、**成長型**（×2 加權）
- 排除：**景氣循環型**（EPS 波動劇烈，分析師成長率預測不可靠）

#### Forward P/E 預期本益比

```
Forward P/E 比率 = 預期 P/E ÷ 當前 TTM P/E
```

| 比率 | 解讀 |
|------|------|
| < 0.90 | 市場預期獲利加速（投票 +1） |
| 0.90 – 1.10 | 獲利預期平穩（投票 0） |
| > 1.10 | 市場預期獲利趨緩（投票 −1） |

- 反映市場對未來 12 個月獲利的預期
- 適用：**穩定型**（×1）、**成長型**（×2 加權）、**ETF**（×1）

#### EV/EBITDA 企業倍數

```
EV/EBITDA = 企業價值 ÷ 稅息折舊攤銷前盈餘
```

| EV/EBITDA 值 | 解讀 |
|-------------|------|
| < 8 | 低估（投票 +1） |
| 8 – 15 | 合理（投票 0） |
| > 15 | 高估（投票 −1） |

- 排除資本結構與稅務差異，適合跨公司比較
- 適用：**景氣循環型**（×1）——景氣循環股 EPS 波動大，此指標更穩定
- 排除：**穩定型**、**成長型**（使用 PEG / Forward P/E 更精準）

---

## 訊號說明

### 估值訊號（5 級）

根據當前 P/E 或 P/B 在 5 年歷史中的百分位位置輸出：

| 訊號 | 條件 | Emoji | 說明 |
|------|------|-------|------|
| BUY | 百分位 < 進場門檻（預設 25） | 🟢 | 歷史低估區間，考慮進場 |
| WATCH | 進場門檻 ~ 進場門檻+10 | 🔵 | 接近進場區，觀察 |
| NEUTRAL | 中間區段 | ⚪ | 合理估值，無明顯訊號 |
| CAUTION | 出場門檻-10 ~ 出場門檻（預設 75） | 🟡 | 接近出場區，謹慎 |
| SELL | 百分位 > 出場門檻 | 🔴 | 歷史高估區間，考慮減倉 |

> 百分位門檻可在設定頁調整。

### 新聞情緒（3 級）

| 標籤 | Emoji | 說明 |
|------|-------|------|
| 正面 | 📰🟢 | 近期新聞以利多為主 |
| 中性 | 📰⚪ | 無明顯情緒偏向 |
| 負面 | 📰🔴 | 近期新聞以利空為主 |

**評分方式：**
- 每篇文章：標題佔 70%、摘要佔 30%，用 VADER 計算 compound score
- 時間加權：今天權重 1.0，7 天前約 0.1（指數衰減），越新的消息影響越大
- 情緒閾值：≥ 0.05 = 正面，≤ −0.05 = 負面，其餘 = 中性

**資料來源（依序降級）：**

| 優先 | 來源 | 說明 |
|------|------|------|
| 1 | Finnhub API | 需免費 API Key，60 次/分鐘，1 年歷史 |
| 2 | Yahoo Finance RSS | 無需 Key，穩定性較低，自動啟用 |
| 3 | 無資料 | 複合訊號直接採用估值訊號，不中斷 |

### 複合訊號矩陣（估值 × 情緒）

|  | 正面新聞 📰🟢 | 中性新聞 📰⚪ | 負面新聞 📰🔴 |
|--|------------|------------|------------|
| **🟢 BUY** | 🌟 強力買進 | 🟢 買進 | 🟢⚠️ 謹慎買進 |
| **🔵 WATCH** | 🟢 買進 | 🔵 觀察 | ⚪ 中性 |
| **⚪ NEUTRAL** | 🔵 觀察 | ⚪ 中性 | 🟡 謹慎 |
| **🟡 CAUTION** | ⚪ 中性 | 🟡 謹慎 | 🔴 賣出 |
| **🔴 SELL** | 🔴⚠️ 謹慎賣出 | 🔴 賣出 | 🚨 強力賣出 |

> **設計原則**：新聞情緒最多讓訊號偏移一級，不會整個翻轉。估值是主訊號，新聞是修正因子。

---

## 多因子調整（Plan B）

在估值 × 情緒複合訊號的基礎上，再根據補充指標（P/CF、PEG、Forward P/E、EV/EBITDA）與 Strategy D 技術訊號進行投票，依股票類型加權後微調最終訊號。

### 各類型股票的因子權重

| 股票類型 | P/CF | PEG | Forward P/E | EV/EBITDA | Strategy D | 調整上限 |
|---------|------|-----|------------|-----------|------------|---------|
| 穩定型 | ×2 | ×1 | ×1 | — | ×1 | ±2 級 |
| 成長型 | — | ×2 | ×2 | — | ×1 | ±2 級 |
| 景氣循環型 | ×1 | — | — | ×1 | ×1 | ±1 級 |
| ETF | — | — | ×1 | — | — | ±1 級 |
| 未知型 | ×1 | ×1 | ×1 | ×1 | ×1 | ±1 級 |

### 投票規則

每個因子依數值輸出 +1 / 0 / −1，乘以權重後加總，超過上限則截斷：

| 因子 | +1（偏多） | 0（中性） | −1（偏空） |
|------|---------|---------|---------|
| P/CF | < 10 | 10 – 20 | > 20 |
| PEG | < 1.0 | 1.0 – 2.0 | > 2.0 |
| Forward P/E | 比率 < 0.90 | 0.90 – 1.10 | 比率 > 1.10 |
| EV/EBITDA | < 8 | 8 – 15 | > 15 |
| Strategy D | 觸發 = True | — | False = 0 |

加權總分 > 0 → 訊號調升；< 0 → 訊號調降；= 0 → 維持不變。

---

## Strategy D：技術動能訊號

偵測 **KD 低區黃金交叉** + **MACD 柱狀圖收斂**同時成立的反轉先行訊號。

### 觸發條件（全部成立才輸出 True）

1. **KD 黃金交叉**：過去 N 根 K 線內（預設 10 根），K 線由下穿越 D 線，且交叉時 K 值 < 20（超賣區）
2. **MACD 負值連續**：最近 N 根 K 線（預設 3 根）的 MACD 柱狀圖均為負值
3. **柱狀圖收斂**：最新柱狀圖已從低谷回升至少 70%（動能正在建立）

### 可調整參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `kd_window` | 10 | KD 黃金交叉有效期（根數） |
| `kd_k_threshold` | 20 | 黃金交叉時 K 值必須低於此值 |
| `n_bars` | 3 | MACD 柱狀圖負值連續根數 |
| `recovery_pct` | 0.70 | 柱狀圖從低谷回升比例門檻 |

> Strategy D 預設**關閉**，可在設定頁開啟並調整參數。觸發時在多因子投票中加 +1。

---

## 股票類型分類

系統在新增股票時自動分析並分類，使用者可手動覆蓋。

### 分類邏輯

| 類型 | 自動識別條件 | 主要估值指標 | 說明 |
|------|------------|------------|------|
| **穩定型** | EPS 年波動 < 30%，成長率 ±5–20% | P/E | 藍籌、公用事業、金融 |
| **成長型** | EPS 波動 30–60%，成長率 > 20% | P/E（搭配 PEG） | 科技、生技高成長股 |
| **景氣循環型** | EPS 波動 > 60% 或週期性震盪 | P/B + EV/EBITDA | 原物料、能源、航運 |
| **ETF** | 無 EPS 記錄 | Forward P/E | 指數型基金 |
| **未知型** | 資料不足 | 全因子等權 | 預設 fallback |

### 類型對指標適用性的影響

- **穩定型**：P/CF 可靠，PEG 有意義（成長可預測）→ 加重 P/CF
- **成長型**：PEG 最關鍵（成長溢價合理性），Forward P/E 反映市場預期 → 排除 P/CF（高成長期現金流為負）
- **景氣循環型**：EPS 高波動使 PEG 失準，改用 P/CF + EV/EBITDA → 排除 PEG / Forward P/E
- **ETF**：只用 Forward P/E 反映指數整體預期

---

## Streamlit 介面

### 頁面一：自選股管理

- 輸入 Ticker 新增股票，系統自動識別類型與適合指標
- 設定持倉（成本價 / 股數）
- 刪除股票（同步清除持倉）

### 頁面二：每日監測儀表板

點「🔍 掃描全部」後顯示彙總表格：

| 欄位 | 說明 |
|------|------|
| 股票 / 名稱 | Ticker 與公司名 |
| 類型 | 股票分類（穩定 / 成長 / 循環 / ETF） |
| 現價 | 最新收盤價（USD） |
| 指標值 | 當前 P/E 或 P/B |
| 百分位 | 在 5 年歷史中的排名（0 = 最便宜，100 = 最貴） |
| 估值訊號 | 5 級估值訊號 |
| 新聞情緒 | 近 7 天 VADER 情緒評分 |
| **綜合訊號** | 估值 × 情緒 × 多因子 = 最終建議 |

個股展開後顯示：
- 補充指標（P/CF、PEG、Forward P/E、EV/EBITDA）
- Strategy D 技術訊號狀態
- 多因子投票明細
- 最新 3 篇新聞（含情緒標籤 + 可點擊連結）
- P/E Band 走勢圖（雙軸：左 P/E，右股價）
- 持倉資訊（成本 / 損益 %）

### 頁面三：歷史報告

查看 / 下載過去任一天的掃描 CSV（存放於 `reports/`）

### 頁面四：策略設置（⚙️）

| 設定項目 | 預設值 | 說明 |
|---------|--------|------|
| PE 歷史年數 | 5 年 | 計算百分位的歷史範圍 |
| 買進百分位 | 25 | 低於此值 → BUY 訊號 |
| 賣出百分位 | 75 | 高於此值 → SELL 訊號 |
| 新聞回顧天數 | 7 天 | 情緒分析的時間窗口 |
| Strategy D 開關 | 關閉 | 是否納入技術訊號投票 |
| Strategy D 參數 | 見上表 | KD / MACD 觸發條件 |

---

## Email 通知設定

每日掃描偵測到訊號變化時，自動發送 HTML 格式的 Email 通知。

### 設定步驟

1. Gmail 帳戶啟用兩步驟驗證
2. 前往 [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) 產生「應用程式密碼」
3. 在 `.env` 設定：

```dotenv
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=<16 位應用程式密碼>
NOTIFICATION_EMAIL=recipient@example.com
SMTP_ENABLED=true
```

4. 或在 `config.json` 直接設定 `"smtp_enabled": true`

---

## 命令列掃描

```bash
python main.py
```

輸出範例：

```
=================================================================
  PE Monitor — Daily Scan  |  2025-04-17 08:30:00
=================================================================

Ticker  名稱                  現價    TTM EPS  指標   百分位  估值訊號      新聞情緒       綜合訊號
-----------------------------------------------------------------------------------------------------
GOOGL   Alphabet Inc.        185.50    7.23   25.7   42%   ⚪ NEUTRAL   📰🟢 正面    🔵 觀察
MU      Micron Technology     98.30    2.10   46.8   18%   🟢 BUY       📰⚪ 中性    🟢 買進
AAPL    Apple Inc.           195.00    6.57   29.7   68%   🟡 CAUTION   📰🔴 負面   🔴 賣出

報告已儲存：reports/daily_20250417.csv
```

---

## 本地快取策略

所有外部資料均快取到 `data/` 目錄，降低 API 呼叫頻率：

| 資料類型 | 快取檔案 | TTL | 說明 |
|---------|---------|-----|------|
| 股價歷史 | `{ticker}_price_history.csv` | 6 小時 | 自動過期更新 |
| 季度財報 | `{ticker}_quarterly_financials.csv` | 12 小時 | 自動過期更新 |
| 公司基本資料 | `{ticker}_info.json` | 6 小時 | 自動過期更新 |
| 現金流指標 | `{ticker}_cashflow.json` | 12 小時 | 自動過期更新 |
| 新聞文章 | `{ticker}_news.json` | 1 小時 | 自動過期更新 |
| P/E 歷史序列 | `{ticker}_pe_series.csv` | 永久 | 遺失時自動重建 |
| P/B 歷史序列 | `{ticker}_pb_series.csv` | 永久 | 遺失時自動重建 |

---

## 專案結構

```
PE_monitor/
├── main.py                      # CLI 每日掃描入口
├── app.py                       # Streamlit Web 介面入口
├── config.json                  # 自選股、持倉、設定（自動產生，不上傳 git）
├── .env                         # 個人設定（不上傳 git）
├── .env.example                 # 環境變數範本
├── requirements.txt             # Python 相依套件
├── README.md
├── .gitignore
├── data/                        # 本地快取（股價 CSV、新聞 JSON 等）
├── reports/                     # 每日掃描 CSV 報告
└── src/
    ├── data_fetcher.py          # yfinance 股價與財報資料抓取 + 快取
    ├── pe_calculator.py         # TTM P/E、P/B、歷史百分位計算
    ├── stock_analyzer.py        # 股票類型分類（穩定/成長/循環/ETF）
    ├── news_fetcher.py          # Finnhub / Yahoo RSS 新聞抓取 + 快取
    ├── sentiment_analyzer.py    # VADER 情緒評分（時間加權）
    ├── technical_signals.py     # Strategy D：KD + MACD 動能反轉偵測
    ├── composite_signal.py      # 估值 × 情緒矩陣 + 多因子調整（Plan B）
    ├── report_generator.py      # 每日掃描流程 + CSV 報告輸出
    ├── notifier.py              # Gmail SMTP Email 通知
    └── utils.py                 # config 載入/儲存、.env 初始化
```

---

## 相依套件

```
yfinance>=0.2.36          # 股價與財報資料
streamlit>=1.32.0         # Web UI
plotly>=5.20.0            # 互動式圖表
pandas>=2.0.0             # 資料處理
numpy>=1.26.0             # 數值運算
python-dotenv>=1.0.0      # .env 解析
finnhub-python>=2.4.19    # 新聞 API
nltk>=3.8.1               # VADER 情緒分析
feedparser>=6.0.0         # RSS 解析
pandas-ta>=0.3.14b        # 技術指標（Strategy D 用）
```

---

## 資料流程

```
.env / CLI 參數
    ↓
load_config() → config.json
    ↓
main() 或 app.py
    ↓
[每支股票]
    ├─ fetch_price_history()          → yfinance → data/{ticker}_price_history.csv
    ├─ calc_ttm_eps()                 → 季度財報 + 股數
    ├─ build_historical_pe_series()   → data/{ticker}_pe_series.csv
    ├─ classify_signal()              → 5 級估值訊號
    ├─ fetch_news()                   → Finnhub / RSS → data/{ticker}_news.json
    ├─ analyze_sentiment()            → VADER 時間加權
    ├─ base_composite_signal()        → 15 格矩陣
    ├─ compute_strategy_d()           → KD + MACD（若啟用）
    ├─ compute_supplementary_metrics()→ P/CF, PEG, Forward P/E, EV/EBITDA
    └─ compute_multi_factor_composite()→ Plan B 多因子投票 → 最終訊號
    ↓
[輸出]
├─ main.py → 印出表格 + save_daily_report() → reports/daily_YYYYMMDD.csv
│            + compare_signals() → send_signal_change_email()（若啟用）
└─ app.py  → 渲染儀表板 + 圖表 + 個股詳情
```

---

## 注意事項

- 資料來源為 [yfinance](https://github.com/ranaroussi/yfinance) 及 [Finnhub](https://finnhub.io/)，僅供個人研究，**非投資建議**
- VADER 為英文情緒模型，中文標題評分僅供參考
- Finnhub 免費方案限 60 次/分鐘；10 支股票以內通常不觸發限制
- 更換 Finnhub Key 後需清除舊快取：`rm data/*_news.json`
- ETF（如 GLD、SOXX）無 EPS，估值訊號顯示 N/A，但仍會抓取新聞情緒
- `config.json` 與 `data/`、`reports/` 目錄均已列入 `.gitignore`，不會上傳個人資料
