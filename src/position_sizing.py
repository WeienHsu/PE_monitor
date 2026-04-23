"""
Position-sizing recommendations — rule-based advice for each signal.

The goal is to make the output actionable: instead of just "BUY", tell the
user *how much* and *when*. We intentionally avoid Kelly-style calculations
because we don't have a reliable edge estimate (see P3-15 backtest).

Rules
-----
No holding (holding_shares falsy) + signal:
    STRONG_BUY   → INITIAL  33%   (leave room for adding on dip)
    BUY          → INITIAL  25%
    CAUTIOUS_BUY → INITIAL  15%   (toe-in)
    WATCH        → HOLD     0%    (on the radar, not yet)
    NEUTRAL/above→ HOLD     0%

Existing holding + signal:
    STRONG_BUY   → ADD 25% (if percentile_rank below initial 15th → deep discount)
                 → ADD 15% otherwise
    BUY          → ADD 15%
    CAUTIOUS_BUY → HOLD (don't ladder into weak buys)
    WATCH/NEUTRAL/CAUTION → HOLD
    CAUTIOUS_SELL → TRIM 25%
    SELL         → TRIM 50%
    STRONG_SELL  → EXIT 100%

size_pct in the result is "% of total portfolio to commit for this action"
(not "% of current position"). The UI should translate based on portfolio
size the user enters.
"""

from __future__ import annotations


_BUY_SIGNALS = {"STRONG_BUY", "BUY", "CAUTIOUS_BUY"}
_SELL_SIGNALS = {"CAUTIOUS_SELL", "SELL", "STRONG_SELL"}


def suggest_position(
    signal: str,
    percentile_rank: float | None = None,
    holding_shares: int | float | None = None,
) -> dict:
    """
    Return a position-sizing recommendation.

    Parameters
    ----------
    signal         : composite signal key (STRONG_BUY … STRONG_SELL)
    percentile_rank: current valuation percentile (0-100); deep discounts
                     (< 15) bump the ADD size. None = unknown.
    holding_shares : current holding quantity (falsy → no position)

    Returns
    -------
    {
        'action':   'INITIAL' | 'ADD' | 'HOLD' | 'TRIM' | 'EXIT',
        'size_pct': float,    # suggested % of portfolio to commit
        'advice':   str,      # human-readable 中文 advice
    }
    """
    has_position = bool(holding_shares)

    # Default — hold / wait
    result = {"action": "HOLD", "size_pct": 0.0, "advice": "維持現狀，無建議動作"}

    # --- SELL side ---
    if signal == "STRONG_SELL":
        if has_position:
            return {
                "action": "EXIT",
                "size_pct": 100.0,
                "advice": "強烈賣出訊號：建議全數出清部位，停損或獲利了結",
            }
        return {"action": "HOLD", "size_pct": 0.0, "advice": "強烈賣出區，暫不建倉"}

    if signal == "SELL":
        if has_position:
            return {
                "action": "TRIM",
                "size_pct": 50.0,
                "advice": "賣出訊號：建議減碼 50%，保留部分倉位觀察",
            }
        return {"action": "HOLD", "size_pct": 0.0, "advice": "賣出區，暫不建倉"}

    if signal == "CAUTIOUS_SELL":
        if has_position:
            return {
                "action": "TRIM",
                "size_pct": 25.0,
                "advice": "謹慎賣出：建議減碼 25%",
            }
        return {"action": "HOLD", "size_pct": 0.0, "advice": "謹慎賣出區，暫不建倉"}

    # --- BUY side ---
    deep_discount = percentile_rank is not None and percentile_rank < 15

    if signal == "STRONG_BUY":
        if has_position:
            size = 25.0 if deep_discount else 15.0
            note = "（處於歷史低位，加碼幅度放大）" if deep_discount else ""
            return {
                "action": "ADD",
                "size_pct": size,
                "advice": f"強力買進：建議加碼 {size:.0f}% {note}".strip(),
            }
        return {
            "action": "INITIAL",
            "size_pct": 33.0,
            "advice": "強力買進訊號：建議首次建倉 33%，保留 2/3 資金日後逢低加碼",
        }

    if signal == "BUY":
        if has_position:
            return {
                "action": "ADD",
                "size_pct": 15.0,
                "advice": "買進訊號：建議加碼 15%",
            }
        return {
            "action": "INITIAL",
            "size_pct": 25.0,
            "advice": "買進訊號：建議首次建倉 25%（分批進場留餘裕）",
        }

    if signal == "CAUTIOUS_BUY":
        if has_position:
            return {
                "action": "HOLD",
                "size_pct": 0.0,
                "advice": "謹慎買進：維持現有部位，不急著加碼",
            }
        return {
            "action": "INITIAL",
            "size_pct": 15.0,
            "advice": "謹慎買進：建議小倉試水溫 15%，確認訊號再加碼",
        }

    # WATCH / NEUTRAL / CAUTION — no action either way
    if signal == "WATCH":
        result["advice"] = "觀察區：納入名單等待更佳時機" if not has_position else "觀察區：續抱現有部位"
    elif signal == "NEUTRAL":
        result["advice"] = "中性區：無明確訊號"
    elif signal == "CAUTION":
        result["advice"] = "謹慎區：高估邊緣，暫不加碼"
    return result
