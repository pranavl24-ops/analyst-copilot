"""Free-cash-flow-to-firm (FCFF) discounted cash flow valuation.

Projects unlevered free cash flows, discounts them at the WACC, and derives an
intrinsic equity value per share. Suited to non-financial companies. ``run_dcf``
is the verified, standalone engine; ``StandardFCFF`` wraps it behind the model
protocol.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from core.valuation import assumptions as macro
from core.valuation.base import ValuationModel

FCFF_GROWTH_CAP = 0.15


@dataclass
class DCFInputs:
    fcff_forecast: list[float]
    wacc: float
    terminal_growth: float
    net_debt: float
    shares_outstanding: float
    minority_interest: float = 0.0
    preferred_equity: float = 0.0
    investments_associates: float = 0.0
    terminal_ebitda: Optional[float] = None
    exit_multiple: Optional[float] = None


@dataclass
class DCFResult:
    pv_explicit: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    tv_pct_of_ev: float
    implied_exit_multiple: Optional[float]
    warnings: list[str] = field(default_factory=list)


def _validate(inp: DCFInputs) -> list[str]:
    w = []
    if inp.terminal_growth >= inp.wacc:
        raise ValueError(
            f"terminal_growth ({inp.terminal_growth:.2%}) >= WACC ({inp.wacc:.2%}); "
            "Gordon Growth is undefined. Lower g or raise WACC."
        )
    if inp.terminal_growth > 0.07:
        w.append(f"terminal_growth {inp.terminal_growth:.2%} exceeds a sane long-run ceiling (~6-7%).")
    if not 0.05 <= inp.wacc <= 0.20:
        w.append(f"WACC {inp.wacc:.2%} outside the normal 5-20% band; recheck CAPM inputs.")
    if inp.shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive.")
    if len(inp.fcff_forecast) < 3:
        w.append("Fewer than 3 explicit forecast years; standard is 5-10.")
    return w


def run_dcf(inp: DCFInputs) -> DCFResult:
    warnings = _validate(inp)
    n = len(inp.fcff_forecast)

    pv_explicit = sum(
        fcff / (1 + inp.wacc) ** t
        for t, fcff in enumerate(inp.fcff_forecast, start=1)
    )

    fcff_terminal_year = inp.fcff_forecast[-1]
    tv = fcff_terminal_year * (1 + inp.terminal_growth) / (inp.wacc - inp.terminal_growth)
    pv_terminal = tv / (1 + inp.wacc) ** n

    ev = pv_explicit + pv_terminal
    equity_value = (
        ev - inp.net_debt - inp.minority_interest
        - inp.preferred_equity + inp.investments_associates
    )
    value_per_share = equity_value / inp.shares_outstanding

    tv_pct = pv_terminal / ev if ev else 0.0
    if tv_pct > 0.75:
        warnings.append(
            f"Terminal value is {tv_pct:.0%} of EV (>75%). The answer leans heavily on the "
            "terminal assumption; extend the explicit window or sanity-check g."
        )

    implied_exit = None
    if inp.terminal_ebitda:
        implied_exit = tv / inp.terminal_ebitda
        if inp.exit_multiple is not None:
            gap = abs(implied_exit - inp.exit_multiple) / inp.exit_multiple
            if gap > 0.25:
                warnings.append(
                    f"Gordon-implied exit multiple {implied_exit:.1f}x vs assumed "
                    f"{inp.exit_multiple:.1f}x differ by {gap:.0%}; the TV methods disagree."
                )

    return DCFResult(
        pv_explicit, tv, pv_terminal, ev, equity_value,
        value_per_share, tv_pct, implied_exit, warnings,
    )


def sensitivity(inp: DCFInputs,
                wacc_steps=(-0.01, -0.005, 0, 0.005, 0.01),
                g_steps=(-0.01, -0.005, 0, 0.005, 0.01)) -> dict:
    grid = {}
    for dw in wacc_steps:
        for dg in g_steps:
            trial = DCFInputs(**{**inp.__dict__})
            trial.wacc = inp.wacc + dw
            trial.terminal_growth = inp.terminal_growth + dg
            try:
                grid[(round(trial.wacc, 4), round(trial.terminal_growth, 4))] = \
                    round(run_dcf(trial).value_per_share, 1)
            except ValueError:
                grid[(round(trial.wacc, 4), round(trial.terminal_growth, 4))] = None
    return grid


def _nwc(period: dict[str, Any]) -> Optional[float]:
    receivables = period.get("receivables")
    inventory = period.get("inventory")
    payables = period.get("payables")
    if receivables is None or inventory is None or payables is None:
        return None
    return float(receivables) + float(inventory) - float(payables)


def _delta_nwc(period: dict[str, Any], prev: Optional[dict[str, Any]]) -> Optional[float]:
    """Increase in net working capital (a use of cash), positive when NWC grows.

    The cash-flow ``change_in_working_capital`` line carries the opposite sign (it is
    the cash impact), so it is negated to match the balance-sheet derivation.
    """
    reported = period.get("change_in_working_capital")
    if reported is not None:
        return -float(reported)
    if prev is None:
        return None
    current, previous = _nwc(period), _nwc(prev)
    if current is None or previous is None:
        return None
    return current - previous


def _historical_fcff(statements: list[dict[str, Any]], tax_rate: float) -> list[float]:
    fcff: list[float] = []
    prev: Optional[dict[str, Any]] = None
    for period in statements:
        ebit = period.get("ebit")
        depreciation = period.get("depreciation")
        capex = period.get("capex")
        delta_nwc = _delta_nwc(period, prev)
        prev = period
        if ebit is None or depreciation is None or capex is None or delta_nwc is None:
            continue
        fcff.append(
            float(ebit) * (1 - tax_rate)
            + float(depreciation)
            - abs(float(capex))
            - delta_nwc
        )
    return fcff


def _revenue_cagr(statements: list[dict[str, Any]]) -> float:
    revenues = [
        float(p["revenue"])
        for p in statements
        if p.get("revenue") is not None and float(p["revenue"]) > 0
    ]
    if len(revenues) < 2:
        macro.warn("Fewer than two positive revenue years; forecasting FCFF flat.")
        return 0.0
    return (revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1


def _forecast_fcff(history: list[float], revenue_cagr: float) -> list[float]:
    recent = history[-3:]
    base = sum(recent) / len(recent)
    growth = min(revenue_cagr, FCFF_GROWTH_CAP)
    return [base * (1 + growth) ** t for t in range(1, macro.FORECAST_YEARS + 1)]


def _wacc(market_data: dict[str, Any], tax_rate: float,
          profile: dict[str, Any], beta_value: float) -> float:
    equity = float(market_data["market_cap"])
    debt = market_data.get("total_debt")
    debt = float(debt) if debt is not None else 0.0
    total = equity + debt
    after_tax_kd = macro.cost_of_debt(profile) * (1 - tax_rate)
    return (equity / total) * macro.cost_of_equity(beta_value, profile) + (debt / total) * after_tax_kd


def build_dcf_inputs(statements: list[dict[str, Any]],
                     market_data: dict[str, Any],
                     profile: dict[str, Any]) -> DCFInputs:
    if market_data.get("market_cap") is None:
        raise ValueError("market_cap is required to weight the WACC and bridge to equity; none was available.")
    if market_data.get("shares") is None:
        raise ValueError("shares_outstanding is required for a per-share value; none was available.")

    tax_rate = macro.effective_tax_rate(statements, profile)
    history = _historical_fcff(statements, tax_rate)
    if len(history) < 2:
        raise ValueError(
            "Need at least two years of computable FCFF; each needs EBIT, depreciation, "
            "capex, and a working-capital change (change_in_working_capital, or "
            "receivables/inventory/payables for a year-over-year delta)."
        )

    forecast = _forecast_fcff(history, _revenue_cagr(statements))
    beta_value = macro.beta_for_ticker(market_data.get("ticker"), market_data, profile)

    debt = market_data.get("total_debt")
    cash = market_data.get("cash")
    if debt is None:
        macro.warn("total_debt unavailable; treating as 0 in net debt.")
    if cash is None:
        macro.warn("cash unavailable; treating as 0 in net debt.")
    net_debt = (float(debt) if debt is not None else 0.0) - (float(cash) if cash is not None else 0.0)

    minority = float(macro.profile_assumptions(profile).get("minority_interest", 0.0))

    return DCFInputs(
        fcff_forecast=forecast,
        wacc=_wacc(market_data, tax_rate, profile, beta_value),
        terminal_growth=macro.terminal_growth(profile),
        net_debt=net_debt,
        shares_outstanding=float(market_data["shares"]),
        minority_interest=minority,
    )


class StandardFCFF(ValuationModel[DCFInputs]):
    """FCFF DCF for industrials, consumer, pharma, and IT names."""

    name = "fcff"

    def value(self, inputs: DCFInputs) -> DCFResult:
        return run_dcf(inputs)

    def from_statements(self, statements: list[dict[str, Any]],
                        market_data: dict[str, Any], profile: dict[str, Any]) -> DCFInputs:
        return build_dcf_inputs(statements, market_data, profile)
