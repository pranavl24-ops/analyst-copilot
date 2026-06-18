"""Sanity checks for company financial statements.

Separates broken data, which raises, from suspicious-but-usable data, which is
returned as human-readable flags. The caller decides how to surface flags; a
raise means the statements cannot be trusted downstream at all.
"""

from __future__ import annotations

import math

from core.models import Statements

MARGIN_SWING_LIMIT = 0.25
SCALE_GAP_DECADES = 1.0
PE_BAND = (3.0, 100.0)
PB_BAND = (0.2, 30.0)


class BrokenDataError(ValueError):
    """Raised when statement data is internally broken and cannot be used."""


def sanity_checks(s: Statements, market_cap: float | None = None) -> list[str]:
    """Return flags for suspicious-but-usable statement data.

    Raises :class:`BrokenDataError` on unusable data (non-finite numbers, a
    non-positive market cap). Merely-suspicious findings are returned as strings:
    non-positive revenue, negative EBIT, EBIT-margin swings beyond
    :data:`MARGIN_SWING_LIMIT`, and a revenue/market-cap scale gap that betrays
    the known yfinance year-misalignment off-by-one.
    """
    _require_finite(s, market_cap)

    flags: list[str] = []
    flags += _revenue_flags(s)
    flags += _ebit_flags(s)
    flags += _margin_swing_flags(s)
    flags += _scale_flags(s, market_cap)
    return flags


def _series(s: Statements) -> dict[str, list[float]]:
    return {
        "revenue": s.revenue,
        "ebit": s.ebit,
        "depreciation": s.depreciation,
        "capex": s.capex,
        "nwc": s.nwc,
    }


def _require_finite(s: Statements, market_cap: float | None) -> None:
    for name, series in _series(s).items():
        for year, value in enumerate(series):
            if not math.isfinite(value):
                raise BrokenDataError(f"{name}[{year}] is not a finite number: {value!r}")
    if not math.isfinite(s.tax_rate):
        raise BrokenDataError(f"tax_rate is not a finite number: {s.tax_rate!r}")
    if market_cap is not None and (not math.isfinite(market_cap) or market_cap <= 0):
        raise BrokenDataError(f"market_cap must be a positive finite number: {market_cap!r}")


def _revenue_flags(s: Statements) -> list[str]:
    return [
        f"revenue in year {year} is non-positive ({value:,.0f})"
        for year, value in enumerate(s.revenue)
        if value <= 0
    ]


def _ebit_flags(s: Statements) -> list[str]:
    return [
        f"EBIT in year {year} is negative ({value:,.0f})"
        for year, value in enumerate(s.ebit)
        if value < 0
    ]


def _margin_swing_flags(s: Statements) -> list[str]:
    margins = [
        ebit / revenue if revenue > 0 else None
        for revenue, ebit in zip(s.revenue, s.ebit)
    ]
    flags: list[str] = []
    for year in range(1, len(margins)):
        prev, curr = margins[year - 1], margins[year]
        if prev is None or curr is None:
            continue
        swing = abs(curr - prev)
        if swing > MARGIN_SWING_LIMIT:
            flags.append(
                f"EBIT margin swung {swing * 100:.0f}pp between year {year - 1} "
                f"({prev * 100:.0f}%) and year {year} ({curr * 100:.0f}%)"
            )
    return flags


def price_scale_flags(
    market_cap: float,
    pretax_income: float,
    tax_expense: float,
    total_equity: float,
) -> list[str]:
    """Flag when the pulled price and statements look like different scales.

    Computes the implied P/E (``market_cap`` over net income, where net income is
    ``pretax_income - tax_expense``) and the implied P/B (``market_cap`` over
    ``total_equity``) and flags either ratio that falls outside a sane band. A
    ratio far outside its band usually means the price and the statements are
    quoted on inconsistent bases — for instance a split applied to the price but
    not to shares or earnings — which would silently corrupt the upside/downside
    call. Sub-checks with a non-positive denominator carry no scale signal and are
    skipped. This never raises; it only flags.
    """
    flags: list[str] = []
    net_income = pretax_income - tax_expense
    if net_income > 0:
        implied_pe = market_cap / net_income
        if not PE_BAND[0] <= implied_pe <= PE_BAND[1]:
            flags.append(_inconsistent_basis_warning("P/E", implied_pe, PE_BAND))
    if total_equity > 0:
        implied_pb = market_cap / total_equity
        if not PB_BAND[0] <= implied_pb <= PB_BAND[1]:
            flags.append(_inconsistent_basis_warning("P/B", implied_pb, PB_BAND))
    return flags


def _inconsistent_basis_warning(label: str, ratio: float, band: tuple[float, float]) -> str:
    return (
        f"implied {label} of {ratio:,.1f}x is outside the sane band "
        f"[{band[0]:g}, {band[1]:g}] — price and statements may be on inconsistent "
        f"bases (e.g. a split applied to price but not to shares or earnings); "
        f"verify against a trusted source"
    )


def _scale_flags(s: Statements, market_cap: float | None) -> list[str]:
    if market_cap is None:
        return []
    latest_revenue = s.revenue[-1]
    if latest_revenue <= 0:
        return []
    decades = abs(math.log10(market_cap) - math.log10(latest_revenue))
    if decades >= SCALE_GAP_DECADES:
        return [
            f"latest revenue ({latest_revenue:,.0f}) and market cap ({market_cap:,.0f}) "
            f"differ by {decades:.1f} orders of magnitude — possible yfinance "
            f"year-misalignment"
        ]
    return []
