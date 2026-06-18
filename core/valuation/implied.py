"""Reverse expectations: what the market price implies, never a calibration.

This module never adjusts a model toward the price — doing so would make the
tool indefensible. Instead, holding every base-case input fixed, it inverts the
valuation for the single assumption that makes value-per-share equal the current
price, so the gap to the base case can be stated as an explicit, falsifiable
market expectation ("to justify the price, the market implies a sustained ROE of
~Y% versus a base case of Z%"). Each solved assumption is monotonic in price, so
a bracketed bisection recovers it exactly; a price unreachable within sane bounds
is reported as such rather than forced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from core.valuation import assumptions as macro
from core.valuation.fcff import (
    FCFF_GROWTH_CAP,
    DCFInputs,
    base_near_term_growth,
    forecast_from_drivers,
    run_dcf,
)
from core.valuation.financials import BankInputs, run_bank_excess_return

TERMINAL_GROWTH_CEILING = 0.07
TERMINAL_ROE_CEILING = 0.20
BISECTION_ITERATIONS = 200
_EPS = 1e-6


@dataclass
class ImpliedAssumption:
    """One assumption solved back from the price, contrasted with its base case."""

    label: str
    base: float
    implied: Optional[float]
    reachable: bool
    implausible: bool


@dataclass
class ImpliedExpectations:
    base_value_per_share: float
    current_price: float
    assumptions: list[ImpliedAssumption]


def implied_expectations(model: Any, inputs: Any, current_price: float) -> ImpliedExpectations:
    """Solve, per valuation model, for the assumptions the market price implies.

    ``model`` supplies the base value through its own engine; the solver is chosen
    from the concrete input type so the contract holds for any model the registry
    can dispatch.
    """
    base_value = model.value(inputs).value_per_share
    if isinstance(inputs, DCFInputs):
        assumptions = _fcff_assumptions(inputs, current_price)
    elif isinstance(inputs, BankInputs):
        assumptions = _bank_assumptions(inputs, current_price)
    else:
        raise TypeError(
            f"implied_expectations has no solver for inputs of type {type(inputs).__name__}"
        )
    return ImpliedExpectations(base_value, current_price, assumptions)


def _fcff_assumptions(inputs: DCFInputs, price: float) -> list[ImpliedAssumption]:
    assumptions = [_fcff_terminal_growth(inputs, price)]
    drivers = inputs.forecast_drivers
    if drivers is not None:
        assumptions.append(_fcff_ebit_margin(inputs, drivers, price))
        assumptions.append(_fcff_revenue_growth(inputs, drivers, price))
    return assumptions


def _fcff_terminal_growth(inputs: DCFInputs, price: float) -> ImpliedAssumption:
    def value(g: float) -> float:
        return run_dcf(replace(inputs, terminal_growth=g)).value_per_share

    implied, reachable = _solve(value, -0.50, inputs.wacc - _EPS, price)
    return _build(
        "terminal growth", inputs.terminal_growth, implied, reachable,
        lambda x: x > TERMINAL_GROWTH_CEILING,
    )


def _fcff_ebit_margin(inputs: DCFInputs, drivers: Any, price: float) -> ImpliedAssumption:
    def value(margin: float) -> float:
        forecast = forecast_from_drivers(drivers, inputs.terminal_growth, ebit_margin=margin)
        return run_dcf(replace(inputs, fcff_forecast=forecast)).value_per_share

    implied, reachable = _solve(value, 0.0, 1.0, price)
    return _build(
        "target EBIT margin", drivers.target_ebit_margin, implied, reachable,
        lambda x: x > drivers.max_ebit_margin,
    )


def _fcff_revenue_growth(inputs: DCFInputs, drivers: Any, price: float) -> ImpliedAssumption:
    def value(growth: float) -> float:
        forecast = forecast_from_drivers(drivers, inputs.terminal_growth, near_term_growth=growth)
        return run_dcf(replace(inputs, fcff_forecast=forecast)).value_per_share

    implied, reachable = _solve(value, -0.50, 0.60, price)
    return _build(
        "near-term revenue growth", base_near_term_growth(drivers), implied, reachable,
        lambda x: x > FCFF_GROWTH_CAP,
    )


def _bank_assumptions(inputs: BankInputs, price: float) -> list[ImpliedAssumption]:
    return [
        _bank_terminal_roe(inputs, price),
        _bank_terminal_growth(inputs, price),
        _bank_cost_of_equity(inputs, price),
    ]


def _bank_terminal_roe(inputs: BankInputs, price: float) -> ImpliedAssumption:
    def value(roe: float) -> float:
        return run_bank_excess_return(replace(inputs, terminal_roe=roe)).value_per_share

    implied, reachable = _solve(value, 0.0, 0.60, price)
    return _build(
        "terminal ROE", inputs.terminal_roe, implied, reachable,
        lambda x: x > TERMINAL_ROE_CEILING,
    )


def _bank_terminal_growth(inputs: BankInputs, price: float) -> ImpliedAssumption:
    def value(g: float) -> float:
        return run_bank_excess_return(replace(inputs, terminal_growth=g)).value_per_share

    implied, reachable = _solve(value, -0.20, inputs.cost_of_equity - _EPS, price)
    return _build(
        "terminal growth", inputs.terminal_growth, implied, reachable,
        lambda x: x > TERMINAL_GROWTH_CEILING,
    )


def _bank_cost_of_equity(inputs: BankInputs, price: float) -> ImpliedAssumption:
    def value(ke: float) -> float:
        return run_bank_excess_return(replace(inputs, cost_of_equity=ke)).value_per_share

    implied, reachable = _solve(value, inputs.terminal_growth + _EPS, 0.40, price)
    return _build(
        "cost of equity", inputs.cost_of_equity, implied, reachable,
        lambda x: x < macro.RF_INDIA,
    )


def _build(label: str, base: float, implied: Optional[float], reachable: bool,
           implausible_if: Callable[[float], bool]) -> ImpliedAssumption:
    implausible = reachable and implied is not None and implausible_if(implied)
    return ImpliedAssumption(label, base, implied, reachable, implausible)


def _solve(evaluate: Callable[[float], float], lo: float, hi: float,
           target: float) -> tuple[Optional[float], bool]:
    """Invert a monotonic ``evaluate`` over ``[lo, hi]`` for ``evaluate(x) == target``.

    The direction is read from the endpoints, so it works whether value rises or
    falls with the assumption. Returns ``(None, False)`` when the target lies
    outside ``[evaluate(lo), evaluate(hi)]`` — the price is unreachable in band.
    """
    f_lo, f_hi = evaluate(lo), evaluate(hi)
    increasing = f_hi >= f_lo
    low_bound, high_bound = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    if not low_bound <= target <= high_bound:
        return None, False

    a, b = lo, hi
    for _ in range(BISECTION_ITERATIONS):
        mid = (a + b) / 2.0
        if (evaluate(mid) < target) == increasing:
            a = mid
        else:
            b = mid
    return (a + b) / 2.0, True


def describe(expectations: ImpliedExpectations) -> list[str]:
    """Factual, non-editorial lines contrasting each implied assumption with its base."""
    lines: list[str] = []
    price = expectations.current_price
    for item in expectations.assumptions:
        if not item.reachable or item.implied is None:
            lines.append(
                f"Implied {item.label}: the market price of {price:,.2f} implies a "
                f"{item.label} beyond the plausible range (base case {_pct(item.base)}); "
                f"the price is unreachable within sane bounds."
            )
            continue
        flag = " — outside the plausible band" if item.implausible else ""
        lines.append(
            f"Implied {item.label}: to justify the price of {price:,.2f}, the market "
            f"implies a {item.label} of ~{_pct(item.implied)} (base case {_pct(item.base)})"
            f"{flag}."
        )
    return lines


def to_payload(expectations: ImpliedExpectations) -> dict[str, Any]:
    """JSON-serialisable view for the memo payload and the saved payload.json."""
    return {
        "base_value_per_share": round(expectations.base_value_per_share, 2),
        "current_price": round(expectations.current_price, 2),
        "assumptions": [
            {
                "label": item.label,
                "base_pct": round(item.base * 100, 2),
                "implied_pct": round(item.implied * 100, 2) if item.implied is not None else None,
                "reachable": item.reachable,
                "implausible": item.implausible,
            }
            for item in expectations.assumptions
        ],
        "lines": describe(expectations),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"
