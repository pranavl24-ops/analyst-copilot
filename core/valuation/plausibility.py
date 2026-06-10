"""Plausibility checks that flag results the generic models can't be trusted on.

These checks never alter a computed value; they only APPEND human-readable
warnings when a result carries the fingerprints of a model-fit failure — a
valuation gap too wide to be a real mispricing, an unstable FCFF base typical of
heavy-capex names, or a bank book-equity jump that distorts the ROE seed. The
caller surfaces the warnings alongside the value and folds them into the memo's
Risks section; the value itself is left untouched.
"""

from __future__ import annotations

import statistics
from typing import Optional, Sequence

EXTREME_GAP_THRESHOLD = 0.35
FCFF_CV_THRESHOLD = 0.5
EQUITY_JUMP_THRESHOLD = 0.40


def extreme_gap_warnings(value_per_share: float, current_price: float) -> list[str]:
    """Flag a value-vs-price gap too wide to read as a simple mispricing."""
    if current_price <= 0:
        return []
    gap = value_per_share / current_price - 1.0
    if abs(gap) <= EXTREME_GAP_THRESHOLD:
        return []
    return [
        f"Extreme valuation gap: model value/share is {gap:+.0%} versus the market "
        f"price (beyond ±35%). A gap this wide usually means the generic model "
        f"does not fit the company — a heavy capex/investment phase, a conglomerate "
        f"needing sum-of-the-parts, cyclical earnings, or M&A distortion — rather than "
        f"a real mispricing. Investigate before trusting the number; a genuine "
        f"mispricing is also possible."
    ]


def unstable_fcff_warnings(historical_fcff: Sequence[float]) -> list[str]:
    """Flag an FCFF base too unstable to grow off its average (FCFF path)."""
    if len(historical_fcff) < 2:
        return []
    reasons: list[str] = []
    if any(value < 0 for value in historical_fcff):
        reasons.append("at least one historical FCFF year is negative")
    cv = _coefficient_of_variation(historical_fcff)
    if cv is not None and cv > FCFF_CV_THRESHOLD:
        reasons.append(f"the coefficient of variation of historical FCFF is {cv:.0%} (>50%)")
    if not reasons:
        return []
    return [
        "Unreliable FCFF base: " + " and ".join(reasons) + ". The grow-off-average "
        "forecast is unstable and likely unreliable — typical of heavy-capex names — "
        "so the DCF output should be treated with caution."
    ]


def book_equity_distortion_warnings(total_equity_series: Sequence[float]) -> list[str]:
    """Flag a book-equity jump that distorts the ROE seed (bank path)."""
    warnings: list[str] = []
    for earlier, later in zip(total_equity_series, total_equity_series[1:]):
        if earlier <= 0:
            continue
        jump = later / earlier - 1.0
        if jump > EQUITY_JUMP_THRESHOLD:
            warnings.append(
                f"Bank book-equity distortion: total equity jumped {jump:+.0%} "
                f"year-over-year. A capital raise or M&A likely distorted the ROE seed, "
                f"so the result needs a normalized-ROE override before it can be trusted."
            )
    return warnings


def _coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.pstdev(values) / abs(mean)
