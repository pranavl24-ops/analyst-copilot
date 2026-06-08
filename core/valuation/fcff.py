"""Free-cash-flow-to-firm (FCFF) discounted cash flow valuation.

Projects unlevered free cash flows, discounts them at the WACC, and derives an
intrinsic equity value per share. Suited to non-financial companies. ``run_dcf``
is the verified, standalone engine; ``StandardFCFF`` wraps it behind the model
protocol.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.models import MarketData, Statements
from core.valuation.base import ValuationModel


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


class StandardFCFF(ValuationModel[DCFInputs]):
    """FCFF DCF for industrials, consumer, pharma, and IT names."""

    name = "fcff"

    def value(self, inputs: DCFInputs) -> DCFResult:
        return run_dcf(inputs)

    def from_statements(self, statements: Statements,
                        market_data: MarketData, profile: dict) -> DCFInputs:
        raise NotImplementedError("Company/MarketData -> DCFInputs wiring is a later step.")
