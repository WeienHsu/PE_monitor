"""Unit tests for utils.py — including type/reason mismatch detection."""

from src.utils import detect_reason_type_mismatch


class TestDetectReasonTypeMismatch:
    def test_consistent_returns_none(self):
        """type matches reason — no mismatch."""
        entry = {
            "type": "stable",
            "reason": "EPS波動率25%，多因子評分（成長1分／穩定3分／循環0分）判定為穩定型，適合 P/E 區間",
        }
        assert detect_reason_type_mismatch(entry) is None

    def test_growth_override_to_stable_detected(self):
        """reason says stable but type is growth (manual override)."""
        entry = {
            "type": "growth",
            "reason": "EPS波動率0%，Beta=1.1，多因子評分判定為穩定型，適合 P/E 區間",
        }
        assert detect_reason_type_mismatch(entry) == "stable"

    def test_stable_override_to_cyclical_detected(self):
        entry = {
            "type": "stable",
            "reason": "EPS有負值，判定為景氣循環型，建議改用 P/B",
        }
        assert detect_reason_type_mismatch(entry) == "cyclical"

    def test_etf_reason_matches_etf_type(self):
        entry = {
            "type": "etf",
            "reason": "ETF：成分股混合，P/E 為加權平均，僅供參考",
        }
        assert detect_reason_type_mismatch(entry) is None

    def test_empty_reason_returns_none(self):
        entry = {"type": "growth", "reason": ""}
        assert detect_reason_type_mismatch(entry) is None

    def test_empty_type_returns_none(self):
        entry = {"type": "", "reason": "判定為成長型"}
        assert detect_reason_type_mismatch(entry) is None

    def test_reason_without_classification_phrase(self):
        """reason contains no 判定為 pattern → cannot detect."""
        entry = {
            "type": "growth",
            "reason": "EPS 資料不足，無法自動判斷類型",
        }
        assert detect_reason_type_mismatch(entry) is None

    def test_p2_12_hard_rule_reason_still_detected(self):
        """P2-13: industry hard-rule reason format ('產業...依規則判定為景氣循環型')
        should still be caught when stored type is different."""
        entry = {
            "type": "growth",
            "reason": "產業「Semiconductors」依規則判定為景氣循環型（跳過多因子評分）",
        }
        assert detect_reason_type_mismatch(entry) == "cyclical"

    def test_p2_12_hard_rule_consistent_no_mismatch(self):
        entry = {
            "type": "cyclical",
            "reason": "產業「Semiconductors」依規則判定為景氣循環型（跳過多因子評分）",
        }
        assert detect_reason_type_mismatch(entry) is None
