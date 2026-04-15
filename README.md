# PE Monitor

每日監測自選美股的本益比（P/E）位置，幫助判斷進出場時機。

系統會自動分析每檔股票是否適合使用 P/E 區間法，並根據歷史百分位輸出 BUY / WATCH / NEUTRAL / CAUTION / SELL 訊號。

---

## 功能亮點

- **股票適合度自動判斷**：穩定型 ✅ / 成長型 🟡 / 景氣循環型 ❌ / ETF 📦
- **歷史 P/E Band**：計算過去 5 年每日 P/E，標示 10/25/50/75/90 百分位
- **每日訊號**：🟢 BUY / 🔵 WATCH / ⚪ NEUTRAL / 🟡 CAUTION / 🔴 SELL
- **持倉管理**：追蹤成本、損益%，在出場區間時提醒
- **Streamlit 介面**：互動式 Web 介面，支援新增/刪除股票與查看歷史報告
- **本地快取**：資料存到 `data/`，避免重複呼叫 API

---

## 安裝步驟

```bash
# 1. Clone 或下載專案
git clone <your-repo-url>
cd pe-monitor

# 2. 建立虛擬環境（建議）
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. 安裝相依套件
pip install -r requirements.txt
```

---

## 初始設定

```bash
# 複製範本並填入自己的股票清單
cp .env.example .env
```

編輯 `.env`：

```dotenv
# 自選股（逗號分隔）
WATCHLIST=GOOGL,MU,SOXX,AAPL

# 持倉（ticker:成本:股數，沒有可留空）
HOLDINGS=GOOGL:175.0:10

# 其他設定（有預設值，可不填）
PE_HISTORY_YEARS=5
ENTRY_PERCENTILE=25
EXIT_PERCENTILE=75
```

> **注意**：`.env` 只在**第一次啟動**時用來初始化 `config.json`。  
> 之後新增/刪除股票請透過 Streamlit 介面操作，直接更新 `config.json`。

---

## 執行 Streamlit 介面

```bash
streamlit run app.py
```

瀏覽器會自動開啟 `http://localhost:8501`，包含三個頁面：

| 頁面 | 功能 |
|------|------|
| 自選股管理 | 新增股票（含適合度分析）、設定持倉 |
| 每日監測儀表板 | 即時 P/E 訊號、Band 走勢圖、持倉損益 |
| 歷史報告 | 查看/下載過去任一天的掃描結果 |

---

## 每日執行（命令列）

```bash
python main.py
```

輸出範例：
```
=================================================================
  PE Monitor — Daily Scan
  2025-01-15 08:30:00
=================================================================

掃描 4 檔股票...

Ticker  Name                  Price  TTM EPS  Metric  %ile  Signal
----------------------------------------------------------------
GOOGL   Alphabet Inc.        185.50    7.23   25.7   42%   ⚪ NEUTRAL
MU      Micron Technology     98.30    2.10   46.8   18%   🟢 BUY ZONE
AAPL    Apple Inc.           195.00    6.57   29.7   68%   🟡 CAUTION

報告已儲存：reports/daily_20250115.csv

🟢 BUY ZONE：1 檔
```

---

## 推送到 GitHub

```bash
# 初始化（已完成，僅供參考）
git init
git add .
git commit -m "init: pe-monitor initial setup"

# 新增遠端並推送
git remote add origin https://github.com/<your-username>/pe-monitor.git
git branch -M main
git push -u origin main
```

---

## .env 說明

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `WATCHLIST` | 初始自選股，逗號分隔 | `GOOGL,AAPL,MSFT` |
| `HOLDINGS` | 初始持倉，格式 `ticker:成本:股數` | 空白 |
| `PE_HISTORY_YEARS` | 歷史區間年數 | `5` |
| `ENTRY_PERCENTILE` | 進場百分位門檻 | `25` |
| `EXIT_PERCENTILE` | 出場百分位門檻 | `75` |

> **`.env` 只作為首次初始化來源。**  
> 一旦 `config.json` 建立後，日常操作請使用 Streamlit 介面。  
> `.env` 不會上傳到 Git（已加入 `.gitignore`）。

---

## 訊號說明

| 訊號 | 條件 | 說明 |
|------|------|------|
| 🟢 BUY ZONE | P/E < 25th 百分位 | 歷史低估區間，考慮進場 |
| 🔵 WATCH ZONE | 25th ~ 35th 百分位 | 接近進場區，觀察 |
| ⚪ NEUTRAL | 35th ~ 65th 百分位 | 合理估值區間 |
| 🟡 CAUTION | 65th ~ 75th 百分位 | 接近出場區，謹慎 |
| 🔴 SELL ZONE | P/E > 75th 百分位 | 歷史高估區間，考慮減倉 |

---

## 專案結構

```
pe-monitor/
├── main.py                  # 每日執行入口
├── app.py                   # Streamlit 介面入口
├── config.json              # 自選股與持倉設定（自動產生）
├── .env                     # 個人設定（不上傳 Git）
├── .env.example             # 環境變數範本
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── data_fetcher.py      # 抓股價與財報資料（yfinance）
    ├── pe_calculator.py     # 計算 TTM P/E、P/B、歷史百分位
    ├── stock_analyzer.py    # 判斷股票適合度 + 類型
    ├── report_generator.py  # 產生每日報告
    └── utils.py             # config 載入/儲存、.env 初始化
```

---

## 注意事項

- 資料來源為 [yfinance](https://github.com/ranaroussi/yfinance)，僅供個人研究用途
- EPS 資料距今超過 100 天時會顯示 ⚠️ 過期警告
- EPS 為負值時自動切換為 P/B 顯示
- ETF（如 SOXX）的 P/E 為成分股加權平均，僅供參考
- 所有快取資料儲存在 `data/`，可定期清除以強制重抓
