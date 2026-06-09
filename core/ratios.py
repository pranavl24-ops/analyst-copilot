"""Financial ratios derived from the data layer's merged payload.

Each function returns a single named number computed from raw figures; no
commentary, no interpretation. Missing inputs raise rather than coerce to zero.

Inputs follow the shape returned by :func:`data.fetch.merged`: market fields at
the top level (``total_debt``, ``cash``, ``market_cap``) and a ``statements``
list of per-period dicts ordered oldest to newest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

Period = Mapping[str, Any]


class RatioError(ValueError):
    """Raised when a ratio cannot be computed from the available data."""


@dataclass(frozen=True)
class Ratios:
    """The named ratios for one company.

    ``ebit_margin_trend`` is the average annual change in EBIT margin across the
    statement window, expressed as a fraction per year. ``roce`` divides latest
    EBIT by book capital employed: total debt plus latest book shareholders'
    equity. It is a book measure and never uses market capitalisation.
    """

    revenue_cagr: float
    ebit_margin_latest: float
    ebit_margin_trend: float
    roce: float
    net_debt_to_ebitda: float


def compute_ratios(data: Mapping[str, Any]) -> Ratios:
    statements = data.get("statements")
    if not statements:
        raise RatioError("no statements available")

    return Ratios(
        revenue_cagr=revenue_cagr(statements),
        ebit_margin_latest=ebit_margin_latest(statements),
        ebit_margin_trend=ebit_margin_trend(statements),
        roce=roce(statements, _require(data, "total_debt")),
        net_debt_to_ebitda=net_debt_to_ebitda(
            statements, _require(data, "total_debt"), _require(data, "cash")
        ),
    )


def revenue_cagr(statements: Sequence[Period]) -> float:
    revenue = _series(statements, "revenue")
    first, last = revenue[0], revenue[-1]
    if first <= 0:
        raise RatioError("first-period revenue must be positive for a CAGR")
    years = len(revenue) - 1
    return (last / first) ** (1 / years) - 1


def ebit_margin_latest(statements: Sequence[Period]) -> float:
    return _ebit_margins(statements)[-1]


def ebit_margin_trend(statements: Sequence[Period]) -> float:
    margins = _ebit_margins(statements)
    return (margins[-1] - margins[0]) / (len(margins) - 1)


def roce(statements: Sequence[Period], total_debt: float) -> float:
    ebit = _latest(statements, "ebit")
    book_equity = _latest(statements, "total_equity")
    capital_employed = float(total_debt) + book_equity
    if capital_employed <= 0:
        raise RatioError("capital employed (debt + book equity) must be positive for ROCE")
    return ebit / capital_employed


def net_debt_to_ebitda(statements: Sequence[Period], total_debt: float, cash: float) -> float:
    ebitda = _latest(statements, "ebit") + _latest(statements, "depreciation")
    if ebitda <= 0:
        raise RatioError("EBITDA must be positive for net-debt/EBITDA")
    return (float(total_debt) - float(cash)) / ebitda


def _ebit_margins(statements: Sequence[Period]) -> list[float]:
    revenue = _series(statements, "revenue")
    ebit = _series(statements, "ebit")
    margins = []
    for period, rev, ebit_value in zip(statements, revenue, ebit):
        if rev <= 0:
            raise RatioError(f"revenue for FY{period.get('year')} must be positive for a margin")
        margins.append(ebit_value / rev)
    return margins


def _series(statements: Sequence[Period], field: str) -> list[float]:
    if len(statements) < 2:
        raise RatioError(f"{field} needs at least two periods, got {len(statements)}")
    values = []
    for period in statements:
        value = period.get(field)
        if value is None:
            raise RatioError(f"{field} missing for FY{period.get('year')}")
        values.append(float(value))
    return values


def _latest(statements: Sequence[Period], field: str) -> float:
    value = statements[-1].get(field)
    if value is None:
        raise RatioError(f"{field} missing for the latest period")
    return float(value)


def _require(data: Mapping[str, Any], field: str) -> float:
    value = data.get(field)
    if value is None:
        raise RatioError(f"{field} is missing")
    return float(value)
