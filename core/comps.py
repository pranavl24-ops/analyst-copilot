"""Comparable-company analysis.

Pulls EV/EBITDA and P/E for a set of peer tickers through the data layer and
turns them into a peer median and an implied value-per-share range for a target.

Peers that cannot be fetched, or whose figures yield no usable multiple, are
recorded in ``CompsResult.missing`` rather than aborting the run. The target,
by contrast, must be valuable: a target missing the figures a chosen multiple
needs raises :class:`CompsError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Optional

from data.fetch import DataError, merged


class CompsError(ValueError):
    """Raised when comps cannot be produced for the target."""


@dataclass(frozen=True)
class PeerMultiples:
    ticker: str
    ev_ebitda: Optional[float]
    pe: Optional[float]


@dataclass(frozen=True)
class CompsResult:
    peers: list[PeerMultiples]
    missing: list[str]
    median_ev_ebitda: Optional[float]
    median_pe: Optional[float]
    implied_from_ev_ebitda: Optional[float]
    implied_from_pe: Optional[float]
    value_per_share_low: float
    value_per_share_high: float


def run_comps(target: str, peers: list[str]) -> CompsResult:
    valid: list[PeerMultiples] = []
    missing: list[str] = []
    for peer in peers:
        try:
            data = merged(peer)
        except DataError:
            missing.append(peer)
            continue
        ev_ebitda = _safe(ev_ebitda_multiple, data)
        pe = _safe(pe_multiple, data)
        if ev_ebitda is None and pe is None:
            missing.append(peer)
            continue
        valid.append(PeerMultiples(peer, ev_ebitda, pe))

    if not valid:
        raise CompsError("no peer produced a usable multiple")

    median_ev_ebitda = _median([p.ev_ebitda for p in valid])
    median_pe = _median([p.pe for p in valid])

    target_data = merged(target)
    implied_from_ev_ebitda = (
        _imply_from_ev_ebitda(target_data, median_ev_ebitda)
        if median_ev_ebitda is not None
        else None
    )
    implied_from_pe = (
        _imply_from_pe(target_data, median_pe) if median_pe is not None else None
    )

    estimates = [v for v in (implied_from_ev_ebitda, implied_from_pe) if v is not None]
    if not estimates:
        raise CompsError("no peer median available to value the target")

    return CompsResult(
        peers=valid,
        missing=missing,
        median_ev_ebitda=median_ev_ebitda,
        median_pe=median_pe,
        implied_from_ev_ebitda=implied_from_ev_ebitda,
        implied_from_pe=implied_from_pe,
        value_per_share_low=min(estimates),
        value_per_share_high=max(estimates),
    )


def ev_ebitda_multiple(data: Mapping[str, Any]) -> float:
    ebitda = _ebitda(data)
    if ebitda <= 0:
        raise CompsError("EBITDA must be positive for EV/EBITDA")
    return _enterprise_value(data) / ebitda


def pe_multiple(data: Mapping[str, Any]) -> float:
    net_income = _net_income(data)
    if net_income <= 0:
        raise CompsError("net income must be positive for P/E")
    return _require(data, "market_cap") / net_income


def _imply_from_ev_ebitda(data: Mapping[str, Any], multiple: float) -> float:
    enterprise_value = multiple * _ebitda(data)
    equity_value = enterprise_value - _net_debt(data)
    return equity_value / _require(data, "shares")


def _imply_from_pe(data: Mapping[str, Any], multiple: float) -> float:
    equity_value = multiple * _net_income(data)
    return equity_value / _require(data, "shares")


def _enterprise_value(data: Mapping[str, Any]) -> float:
    return _require(data, "market_cap") + _net_debt(data)


def _net_debt(data: Mapping[str, Any]) -> float:
    return _require(data, "total_debt") - _require(data, "cash")


def _ebitda(data: Mapping[str, Any]) -> float:
    period = _latest_period(data)
    return _field(period, "ebit") + _field(period, "depreciation")


def _net_income(data: Mapping[str, Any]) -> float:
    period = _latest_period(data)
    return _field(period, "pretax_income") - _field(period, "tax_expense")


def _latest_period(data: Mapping[str, Any]) -> Mapping[str, Any]:
    statements = data.get("statements")
    if not statements:
        raise CompsError("no statements available")
    return statements[-1]


def _field(period: Mapping[str, Any], field: str) -> float:
    value = period.get(field)
    if value is None:
        raise CompsError(f"{field} missing for FY{period.get('year')}")
    return float(value)


def _require(data: Mapping[str, Any], field: str) -> float:
    value = data.get(field)
    if value is None:
        raise CompsError(f"{field} is missing")
    return float(value)


def _safe(fn: Any, data: Mapping[str, Any]) -> Optional[float]:
    try:
        return fn(data)
    except CompsError:
        return None


def _median(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return median(present) if present else None
