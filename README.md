# PE Monitor 📊

每日監測自選美股的本益比（P/E）位置，整合即時新聞情緒分析，幫助判斷進出場時機。

系統會自動分析每檔股票是否適合使用 P/E 或 P/B 區間法，根據歷史百分位輸出估值訊號，並結合最新新聞情緒產生複合訊號。

---

## 功能亮點

- **估值訊號**：根據 5 年歷史 P/E 百分位輸出 5 級訊號（BUY → SELL）
- **新聞情緒分析**：抓取 Finnhub / Yahoo Finance RSS 新聞，用 VADER 離線評分
- **複合訊號**：估值 × 情緒 = 15 格矩陣（🌟 強力買進 → 🚨 強力賣出）
- **P/E Band 走勢圖**：歷史每日 P/E 曲線 + 百分位色帶 + 雙軸股價疊圖
- **股票適合度自動分析**：穩定型 ✅ / 成長型 🟡 / 景氣循環型 ❌ / ETF 📦
- **持倉管理**：追蹤成本、即時損益 %
- **本地快取**：股價 / 財報 / 新聞分別快取，避免重複 API 呼叫
- **Streamlit 介面**：互動式 Web UI，三個頁面

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
# 自選股（逗號分隔）
WATCHLIST=GOOGL,AAPL,MU,NVDA

# 持倉（ticker:成本:股數，沒有可留空）
HOLDINGS=GOOGL:175.0:10,AAPL:195.0:5

# Finnhub API Key（免費申請：https://finnhub.io/register）
FINNHUB_API_KEY=your_key_here

# 其他設定（有預設值，可不填）
PE_HISTORY_YEARS=5
ENTRY_PERCENTILE=25
EXIT_PERCENTILE=75
```

> `.env` 只在**第一次啟動**時用來初始化 `config.json`，之後請透過 Streamlit 介面操作。

### 3. 啟動 Streamlit

```bash
.venv/bin/streamlit run app.py
```

開啟瀏覽器 → `http://localhost:8501`

---

## Streamlit 介面

### 頁面一：自選股管理

- 輸入 Ticker 新增股票，系統自動分析適合用 P/E 還是 P/B
- 設定持倉（成本價 / 股數）
- 刪除股票（同步清除持倉）

### 頁面二：每日監測儀表板

點「🔍 掃描全部」後顯示：

| 欄位 | 說明 |
|------|------|
| 股票 / 名稱 | Ticker 與公司名 |
| 現價 | 最新收盤價（USD） |
| 指標值 | 當前 P/E 或 P/B |
| 百分位 | 在 5 年歷史中的排名（0 = 最便宜，100 = 最貴） |
| 估值訊號 | 5 級訊號（見下方說明） |
| 新聞情緒 | 近 7 天新聞的 VADER 情緒評分 |
| **綜合訊號** | 估值 × 情緒的最終建議（見矩陣說明） |

個股展開後顯示：
- 第二排指標：PE 訊號 / 新聞情緒 / 綜合訊號
- 最新 3 篇相關新聞標題（含情緒標籤、可點擊連結）
- P/E Band 走勢圖（雙軸：左 P/E，右股價）
- 持倉資訊（成本 / 損益 %）

### 頁面三：歷史報告

查看 / 下載過去任一天的掃描 CSV（存放於 `reports/`）

---

## 訊號說明

### 估值訊號（5 級）

| 訊號 | 條件 | 說明 |
|------|------|------|
| 🟢 BUY | P/E < 25th 百分位 | 歷史低估區間，考慮進場 |
| 🔵 WATCH | 25th–35th 百分位 | 接近進場區，觀察 |
| ⚪ NEUTRAL | 35th–65th 百分位 | 合理估值 |
| 🟡 CAUTION | 65th–75th 百分位 | 接近出場區，謹慎 |
| 🔴 SELL | P/E > 75th 百分位 | 歷史高估區間，考慮減倉 |

> 百分位門檻可在 Sidebar「設定」中調整。

### 新聞情緒（3 級）

| 標籤 | 說明 |
|------|------|
| 📰🟢 正面 | 近期新聞以利多為主 |
| 📰⚪ 中性 | 無明顯情緒偏向 |
| 📰🔴 負面 | 近期新聞以利空為主 |

VADER 對每篇文章的標題（70%）+ 摘要（30%）評分，並對近期文章給予更高時間權重（指數衰減）。

### 複合訊號矩陣（估值 × 情緒）

|  | 正面新聞 | 中性新聞 | 負面新聞 |
|--|---------|---------|---------|
| **BUY** | 🌟 強力買進 | 🟢 買進 | 🟢⚠️ 謹慎買進 |
| **WATCH** | 🟢 買進 | 🔵 觀察 | ⚪ 中性 |
| **NEUTRAL** | 🔵 觀察 | ⚪ 中性 | 🟡 謹慎 |
| **CAUTION** | ⚪ 中性 | 🟡 謹慎 | 🔴 賣出 |
| **SELL** | 🔴⚠️ 謹慎賣出 | 🔴 賣出 | 🚨 強力賣出 |

> **設計原則**：新聞情緒最多讓訊號偏移一級，不會整個翻轉。估值是主訊號，新聞是修正因子。

---

## 新聞情緒設定

### 設定 Finnhub API Key（建議）

**方法一：直接編輯 `config.json`**

```json
"settings": {
    "finnhub_api_key": "你的key填這裡",
    ...
}
```

**方法二：`.env` 檔案（Key 不進 git）**

```bash
echo 'FINNHUB_API_KEY=你的key填這裡' > .env
# 刪除 config.json 後重啟 Streamlit（系統會自動重新生成）
```

免費申請：[finnhub.io/register](https://finnhub.io/register)（60 次/分鐘，1 年歷史）

### Sidebar 狀態指示

| 狀態 | 顯示 |
|------|------|
| Finnhub 正常 | `✅ Finnhub 正常運作` |
| 未設定 Key，使用 RSS | `⚠️ 使用 RSS 備用（建議設定 Finnhub Key）` |
| 達速率上限，切換 RSS | `⚠️ Finnhub 達速率上限，已切換 RSS` |
| API Key 無效 | `🔴 Finnhub API Key 無效，請重新設定` |
| 所有來源失敗 | `🔴 無法取得新聞資料` |

### 降級策略

1. **Finnhub**（主力）：需免費 API Key，60 req/min，1 年歷史
2. **Yahoo Finance RSS**（備用）：無需 Key，穩定性較低，自動啟用
3. **無資料**：`available=False`，複合訊號直接沿用估值訊號，不報錯

---

## 每日執行（命令列）

```bash
python main.py
```

輸出範例：

```
=================================================================
  PE Monitor — Daily Scan  |  2025-01-15 08:30:00
=================================================================

Ticker  名稱                  現價    TTM EPS  指標   百分位  估值訊號        新聞情緒       綜合訊號
--------------------------------------------------------------------------------------------------------
GOOGL   Alphabet Inc.        185.50    7.23   25.7   42%   ⚪ NEUTRAL     📰🟢 正面     🔵 觀察
MU      Micron Technology     98.30    2.10   46.8   18%   🟢 BUY         📰⚪ 中性     🟢 買進
AAPL    Apple Inc.           195.00    6.57   29.7   68%   🟡 CAUTION     📰🔴 負面     🔴 賣出

報告已儲存：reports/daily_20250115.csv
```

---

## 設定說明

### Sidebar 設定項目

| 設定 | 預設值 | 說明 |
|------|--------|------|
| PE 歷史年數 | 5 年 | 計算百分位的歷史範圍 |
| 買進百分位 | 25 | 低於此值 → BUY 訊號 |
| 賣出百分位 | 75 | 高於此值 → SELL 訊號 |
| 新聞回顧天數 | 7 天 | 情緒分析的時間窗口 |

### `.env` 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `WATCHLIST` | 初始自選股（逗號分隔） | `GOOGL,AAPL,MSFT` |
| `HOLDINGS` | 初始持倉（`ticker:成本:股數`） | 空白 |
| `FINNHUB_API_KEY` | Finnhub API Key | 空白 |
| `PE_HISTORY_YEARS` | 歷史年數 | `5` |
| `ENTRY_PERCENTILE` | 進場百分位 | `25` |
| `EXIT_PERCENTILE` | 出場百分位 | `75` |

---

## 專案結構

```
PE_monitor/
├── main.py                      # CLI 每日執行入口
├── app.py                       # Streamlit 介面入口
├── config.json                  # 自選股、持倉、設定（自動產生，不上傳 git）
├── .env                         # 個人設定（不上傳 git）
├── .env.example                 # 環境變數範本
├── requirements.txt
├── README.md
├── .gitignore
├── data/                        # 本地快取（股價 CSV、新聞 JSON）
├── reports/                     # 每日掃描 CSV 報告
└── src/
    ├── data_fetcher.py          # 股價與財報資料（yfinance）
    ├── pe_calculator.py         # TTM P/E、P/B、歷史百分位計算
    ├── stock_analyzer.py        # 股票類型判斷（穩定/成長/循環/ETF）
    ├── news_fetcher.py          # Finnhub / Yahoo RSS 新聞抓取 + 快取
    ├── sentiment_analyzer.py    # VADER 情緒評分（時間加權）
    ├── composite_signal.py      # 估值 × 情緒複合訊號矩陣
    ├── report_generator.py      # 每日掃描邏輯 + CSV 報告輸出
    └── utils.py                 # config 載入/儲存、.env 初始化
```

---

## P/E 計算原理

```
TTM EPS = 最近 4 季 淨利 ÷ 稀釋股數（四季加總）
日 P/E  = 當日收盤價 ÷ TTM EPS
```

- EPS 在每次財報公布後更新，兩次財報之間為固定值
- 距上次財報超過 100 天時顯示 ⚠️ 過期警告
- EPS 為負時自動切換為 P/B 顯示
- ETF（如 GLD、SOXX）無 EPS，估值訊號顯示 N/A，但仍會抓取並顯示新聞情緒

---

## 注意事項

- 資料來源為 [yfinance](https://github.com/ranaroussi/yfinance)，僅供個人研究，非投資建議
- 新聞快取 TTL 1 小時（`data/{TICKER}_news.json`）；股價快取同天有效
- 更換 Finnhub Key 後需清除舊快取：`rm data/*_news.json`
- VADER 為英文情緒模型，中文標題評分僅供參考
- Finnhub 免費方案限 60 次/分鐘；10 支股票以內通常不會觸發限制
