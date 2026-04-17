"""
Value factor (V): blended valuation score on a 0-100 scale.

Combines up to six valuation inputs, each converted into a "cheapness score"
where higher = cheaper. V = mean of available inputs (missing values are
skipped, never penalised).

Inputs with historical time-series (percentile-based)
-----------------------------------------------------
  - TTM P/E percentile   (lower percentile → cheaper → higher score)
  - P/B percentile
  - CAPE-like P/E percentile  (price / 5-year-average quarterly EPS)

Inputs without usable history (absolute-threshold-based)
--------------------------------------------------------
  - Forward P/E absolute bands
  - P/FCF  absolute bands
  - EV/EBITDA absolute bands

The thresholds are literature-informed defaults; Phase 4 backtest will retune.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_to_cheapness(rank: Optional[float]) -> Optional[float]:
    """Invert a 0-100 percentile rank into a 0-100 cheapness score.

    A rank of 0 (lowest-ever valuation in the window) → score 100.
    A rank of 100 (highest-ever)                      → score 0.
    """
    if rank is None:
        return None
    return float(max(0.0, min(100.0, 100.0 - rank)))


def _linear_band(
    value: Optional[float],
    cheap_end: float,
    expensive_end: float,
) -> Optional[float]:
    """Map an absolute ratio to a 0-100 cheapness score via linear interpolation.

    value ≤ cheap_end      → 100
    value ≥ expensive_end  → 0
    between                → linear

    If cheap_end < expensive_end, the mapping is *inverted* (low = cheap);
    if reversed, the caller signals that "higher is cheaper" (e.g. dividend yield).
    """
    if value is None:
        return None
    if cheap_end < expensive_end:
        if value <= cheap_end:
            return 100.0
        if value >= expensive_end:
            return 0.0
        frac = (value - cheap_end) / (expensive_end - cheap_end)
        return float(100.0 * (1.0 - frac))
    # Reversed band: higher is cheaper
    if value >= cheap_end:
        return 100.0
    if value <= expensive_end:
        return 0.0
    frac = (cheap_end - value) / (cheap_end - expensive_end)
    return float(100.0 * (1.0 - frac))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ValueInputs:
    """All raw inputs to the V factor. Caller supplies what's available."""
    ttm_pe_percentile: Optional[float] = None       # 0-100 (from historical PE series)
    pb_percentile: Optional[float] = None           # 0-100 (from historical PB series)
    cape_percentile: Optional[float] = None         # 0-100 (filled in Phase 3 once Shiller series is available)
    cape_absolute: Optional[float] = None           # current CAPE-like P/E (price / 5Y avg TTM EPS)
    forward_pe: Optional[float] = None              # absolute
    pcf_ratio: Optional[float] = None               # absolute (P/OCF per share)
    ev_ebitda: Optional[float] = None               # absolute
    # Stock's TTM P/E vs Damodaran industry-median trailing P/E.
    # Populated in Phase 3; None if industry is unmapped or data unavailable.
    ttm_pe: Optional[float] = None
    industry_pe: Optional[float] = None
    industry_name: Optional[str] = None


def compute_v_score(inputs: ValueInputs) -> tuple[Optional[float], dict]:
    """Compute the V score (0-100, higher = cheaper) and a per-component breakdown.

    Returns (v_score, details). v_score is None if no inputs are usable.
    details keys are human-readable component names mapped to a sub-dict:
        {"raw": <raw value>, "score": <0-100>, "kind": "percentile" | "absolute"}
    """
    components: dict[str, dict] = {}

    # Percentile-based inputs
    s = _percentile_to_cheapness(inputs.ttm_pe_percentile)
    if s is not None:
        components["TTM P/E"] = {
            "raw": inputs.ttm_pe_percentile,
            "score": s,
            "kind": "percentile",
        }

    s = _percentile_to_cheapness(inputs.pb_percentile)
    if s is not None:
        components["P/B"] = {
            "raw": inputs.pb_percentile,
            "score": s,
            "kind": "percentile",
        }

    s = _percentile_to_cheapness(inputs.cape_percentile)
    if s is not None:
        components["CAPE-like P/E"] = {
            "raw": inputs.cape_percentile,
            "score": s,
            "kind": "percentile",
        }
    else:
        # Fallback: use current CAPE value against absolute bands
        # (≤15 cheap, ≥30 expensive — Shiller long-run means are ~15-17)
        s = _linear_band(inputs.cape_absolute, cheap_end=15.0, expensive_end=30.0)
        if s is not None:
            components["CAPE-like P/E"] = {
                "raw": inputs.cape_absolute,
                "score": s,
                "kind": "absolute",
            }

    # Absolute-threshold inputs (literature defaults)
    # Forward P/E: ≤10 cheap, ≥30 expensive
    s = _linear_band(inputs.forward_pe, cheap_end=10.0, expensive_end=30.0)
    if s is not None:
        components["Forward P/E"] = {
            "raw": inputs.forward_pe,
            "score": s,
            "kind": "absolute",
        }

    # P/FCF: ≤10 cheap, ≥25 expensive
    s = _linear_band(inputs.pcf_ratio, cheap_end=10.0, expensive_end=25.0)
    if s is not None:
        components["P/FCF"] = {
            "raw": inputs.pcf_ratio,
            "score": s,
            "kind": "absolute",
        }

    # EV/EBITDA: ≤8 cheap, ≥20 expensive
    s = _linear_band(inputs.ev_ebitda, cheap_end=8.0, expensive_end=20.0)
    if s is not None:
        components["EV/EBITDA"] = {
            "raw": inputs.ev_ebitda,
            "score": s,
            "kind": "absolute",
        }

    # Stock PE vs industry PE (Damodaran). Ratio ≤0.7 → 100, ≥1.3 → 0.
    if inputs.ttm_pe and inputs.ttm_pe > 0 and inputs.industry_pe and inputs.industry_pe > 0:
        ratio = float(inputs.ttm_pe) / float(inputs.industry_pe)
        if ratio <= 0.7:
            s = 100.0
        elif ratio >= 1.3:
            s = 0.0
        else:
            s = 100.0 * (1.0 - (ratio - 0.7) / (1.3 - 0.7))
        label = f"PE vs {inputs.industry_name}" if inputs.industry_name else "PE vs Industry"
        components[label] = {
            "raw": f"{inputs.ttm_pe:.1f} / {inputs.industry_pe:.1f} = {ratio:.2f}",
            "score": s,
            "kind": "absolute",
        }

    if not components:
        return None, {"components": {}, "score": None}

    scores = [c["score"] for c in components.values()]
    v_score = sum(scores) / len(scores)
    return round(v_score, 2), {
        "components": components,
        "score": round(v_score, 2),
    }
