"""
技術訊號模組 — Strategy D (KD 低檔金叉 + MACD 柱狀圖收斂)

核心邏輯移植自：
  stock-screener/indicators/calculator.py（2026-04-16）

欄位命名規範（傳入 DataFrame 需符合）：
  OHLCV → close, high, low, open, volume（小寫）
  MACD  → macd_line, signal_line, histogram
  Stoch → K, D

注意：PE Monitor 的 fetch_price_history() 回傳大寫欄位（Close/High/Low），
      請使用本模組的 compute_strategy_d() 適配器，它會自動做欄位重命名。
"""

import pandas as pd
import pandas_ta as ta


# ---------------------------------------------------------------------------
# 指標計算
# ---------------------------------------------------------------------------

def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    計算標準 MACD 並附加至 DataFrame。

    新增欄位：
        macd_line   - MACD 線（DIF）
        signal_line - 訊號線（DEA）
        histogram   - 柱狀圖
    """
    df = df.copy()
    raw = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if raw is None or raw.empty:
        raise ValueError(f"MACD 計算失敗，資料筆數不足（需至少 {slow + signal} 筆）。")

    df["macd_line"]   = raw[f"MACD_{fast}_{slow}_{signal}"]
    df["signal_line"] = raw[f"MACDs_{fast}_{slow}_{signal}"]
    df["histogram"]   = raw[f"MACDh_{fast}_{slow}_{signal}"]
    return df


def add_kd(
    df: pd.DataFrame,
    k: int = 9,
    d: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    計算 KD 隨機指標並附加至 DataFrame。

    新增欄位：
        K - 平滑後的 %K 線
        D - %D 線
    """
    df = df.copy()
    raw = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth_k)
    if raw is None or raw.empty:
        raise ValueError(f"KD 計算失敗，資料筆數不足（需至少 {k} 筆）。")

    df["K"] = raw[f"STOCHk_{k}_{d}_{smooth_k}"]
    df["D"] = raw[f"STOCHd_{k}_{d}_{smooth_k}"]
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 MACD 與 KD 並附加至 DataFrame。"""
    df = add_macd(df)
    df = add_kd(df)
    return df


# ---------------------------------------------------------------------------
# 訊號偵測（最新一根）
# ---------------------------------------------------------------------------

def detect_macd_hist_converging(df: pd.DataFrame, n_bars: int = 3, recovery_pct: float = 0.7) -> bool:
    """
    偵測最新一根 K 線是否出現 MACD 柱狀圖收斂訊號（提前卡位型）。

    條件（全部成立）：
        1. 最近 n_bars 根 histogram 都是負數
        2. 每根都比前一根更接近零（連續收斂）
        3. 今日 histogram 已從近期谷底回彈 recovery_pct 以上
    """
    if "histogram" not in df.columns:
        raise ValueError("DataFrame 缺少 histogram 欄位，請先呼叫 add_macd()。")
    if len(df) < n_bars + 1:
        return False

    hist = df["histogram"].iloc[-(n_bars + 1):]
    if hist.isna().any():
        return False

    recent = hist.iloc[-n_bars:]

    if (recent >= 0).any():
        return False

    for i in range(1, len(recent)):
        if recent.iloc[i] <= recent.iloc[i - 1]:
            return False

    lookback = df["histogram"].iloc[-20:] if len(df) >= 20 else df["histogram"]
    neg_vals = lookback[lookback < 0]
    if neg_vals.empty:
        return False
    trough = neg_vals.min()
    threshold = abs(trough) * (1 - recovery_pct)
    return abs(recent.iloc[-1]) < threshold


def _build_kd_prefilter_mask(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    建立 KD 低檔金叉前置過濾遮罩。

    對每一列，若該列本身或往前 window 根內曾出現 KD 低檔金叉（K<20 且 K 由下穿上 D），
    則該位置為 True。
    """
    k = df["K"]
    d = df["D"]
    kd_signal = (k.shift(1) < d.shift(1)) & (k > d) & (k < 20)

    mask = pd.Series(False, index=df.index)
    for offset in range(0, window + 1):
        mask = mask | kd_signal.shift(offset).fillna(False)
    return mask


def detect_macd_converging_kd_prefilter(
    df: pd.DataFrame,
    kd_window: int = 10,
    n_bars: int = 3,
    recovery_pct: float = 0.7,
) -> bool:
    """
    Strategy D — 偵測「KD 前置過濾 + MACD 柱狀圖收斂」訊號。

    條件（兩者皆須成立）：
        1. 今日出現 MACD 柱狀圖收斂（連續 n_bars 日負數收斂，回彈達 recovery_pct）
        2. 今日或往前 kd_window 根內曾出現 KD 低檔黃金交叉（K<20）
    """
    required = {"K", "D", "histogram"}
    if not required.issubset(df.columns):
        raise ValueError("DataFrame 缺少必要欄位，請先呼叫 add_all_indicators()。")

    if not detect_macd_hist_converging(df, n_bars=n_bars, recovery_pct=recovery_pct):
        return False

    window_df = df.iloc[-(kd_window + 1):]
    k = window_df["K"]
    d = window_df["D"]
    kd_fired = ((k.shift(1) < d.shift(1)) & (k > d) & (k < 20)).any()
    return bool(kd_fired)


# ---------------------------------------------------------------------------
# 批次掃描：找出歷史中所有訊號日期
# ---------------------------------------------------------------------------

def scan_macd_converging_kd_prefilter(
    df: pd.DataFrame,
    kd_window: int = 10,
    n_bars: int = 3,
    recovery_pct: float = 0.7,
) -> pd.DataFrame:
    """
    Strategy D — 掃描 MACD 柱狀圖收斂訊號，只保留 KD 低檔金叉後 kd_window 根內的訊號。

    回傳含訊號日期（date 欄位）的 DataFrame。
    需先呼叫 add_all_indicators()。
    """
    required = {"K", "D", "histogram"}
    if not required.issubset(df.columns):
        raise ValueError("DataFrame 缺少必要欄位，請先呼叫 add_all_indicators()。")

    hist = df["histogram"]

    converging_idxs = set()
    for i in range(n_bars, len(df)):
        recent = hist.iloc[i - n_bars + 1: i + 1]
        if recent.isna().any():
            continue
        if (recent >= 0).any():
            continue
        converging = all(recent.iloc[j] > recent.iloc[j - 1] for j in range(1, len(recent)))
        if not converging:
            continue
        lb_start = max(0, i - 19)
        lookback = hist.iloc[lb_start: i + 1]
        neg_vals = lookback[lookback < 0]
        if neg_vals.empty:
            continue
        trough = neg_vals.min()
        threshold = abs(trough) * (1 - recovery_pct)
        if abs(recent.iloc[-1]) < threshold:
            converging_idxs.add(i)

    converging_mask = pd.Series(False, index=df.index)
    if converging_idxs:
        converging_mask.iloc[list(converging_idxs)] = True

    kd_mask = _build_kd_prefilter_mask(df, window=kd_window)
    signal = converging_mask & kd_mask

    cols = [c for c in ["date", "close", "K", "D", "macd_line", "signal_line", "histogram"]
            if c in df.columns]
    return df[signal][cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 適配器：與 PE Monitor 整合的入口
# ---------------------------------------------------------------------------

def compute_strategy_d(
    ticker: str,
    data_dir: str = "data",
    kd_window: int = 10,
    n_bars: int = 3,
    recovery_pct: float = 0.7,
    **kwargs,
) -> dict:
    """
    為單一股票計算 Strategy D 訊號，使用已快取的價格資料。

    參數：
        ticker       - 股票代號（與 PE Monitor watchlist 相同格式）
        data_dir     - 快取目錄（對應 config["settings"]["data_dir"]）
        kd_window    - KD 金叉後等待 MACD 訊號的最大根數，預設 10
        n_bars       - MACD 柱狀圖連續收斂根數，預設 3
        recovery_pct - 從谷底回彈比例門檻，預設 0.7
        **kwargs     - 吸收額外參數（例如 config 中的 enabled 欄位）

    回傳：
        {
            "signal":       bool,       # True = 今日觸發訊號
            "signal_dates": list[str],  # 歷史訊號日期（ISO 格式 YYYY-MM-DD）
            "error":        str | None,
        }
    """
    try:
        from src.data_fetcher import fetch_price_history

        # 取得 1 年價格資料（命中 fetch_price_history 的快取，不額外 API 請求）
        df = fetch_price_history(ticker, years=1, data_dir=data_dir)
        if df.empty:
            return {"signal": False, "signal_dates": [], "error": "無法取得價格資料"}

        # PE Monitor 回傳大寫欄位，calculator.py 需小寫
        df = df.rename(columns={
            "Close":  "close",
            "High":   "high",
            "Low":    "low",
            "Open":   "open",
            "Volume": "volume",
        })

        # 將 DatetimeIndex 轉為 date 欄位（scan 函式用來回傳日期）
        df = df.copy()
        df["date"] = df.index
        df = df.reset_index(drop=True)

        # 計算技術指標
        df = add_all_indicators(df)

        # 今日訊號（最新一根）
        signal = detect_macd_converging_kd_prefilter(
            df, kd_window=kd_window, n_bars=n_bars, recovery_pct=recovery_pct
        )

        # 歷史訊號日期（供圖表標記）
        signal_df = scan_macd_converging_kd_prefilter(
            df, kd_window=kd_window, n_bars=n_bars, recovery_pct=recovery_pct
        )
        signal_dates: list[str] = []
        if not signal_df.empty and "date" in signal_df.columns:
            signal_dates = [str(d)[:10] for d in signal_df["date"]]

        return {"signal": signal, "signal_dates": signal_dates, "error": None}

    except Exception as exc:
        return {"signal": False, "signal_dates": [], "error": str(exc)}
