"""Excess-return (residual-income) valuation for banks and NBFCs.

FCFF is wrong for financials: leverage is the business, not a financing choice.
This values the bank as book equity plus the present value of returns earned
above the cost of equity, discounted at the cost of equity (CAPM), never WACC.
``run_bank_excess_return`` is the verified, standalone engine; ``BankExcessReturn``
wraps it behind the model protocol.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from core.valuation import assumptions as macro
from core.valuation.base import ValuationModel

TERMINAL_ROE_DEFAULT = 0.15
RETENTION_RATIO_DEFAULT = 0.55


@dataclass
class BankInputs:
    book_equity_0: float
    roe_forecast: list[float]
    bv_growth_forecast: list[float]
    cost_of_equity: float
    terminal_roe: float
    terminal_growth: float
    shares_outstanding: float
    tier1_ratio: Optional[float] = None
    tier1_min: float = 0.115


@dataclass
class BankResult:
    book_equity_0: float
    pv_excess_explicit: float
    pv_excess_terminal: float
    equity_value: float
    value_per_share: float
    implied_pb: float
    warnings: list[str] = field(default_factory=list)


def run_bank_excess_return(inp: BankInputs) -> BankResult:
    if inp.terminal_growth >= inp.cost_of_equity:
        raise ValueError(
            f"terminal_growth ({inp.terminal_growth:.2%}) >= Ke ({inp.cost_of_equity:.2%}); "
            "terminal excess-return is undefined."
        )
    if len(inp.roe_forecast) != len(inp.bv_growth_forecast):
        raise ValueError("roe_forecast and bv_growth_forecast must be the same length.")

    warnings = []
    ke = inp.cost_of_equity
    n = len(inp.roe_forecast)

    bv_prev = inp.book_equity_0
    pv_excess = 0.0
    for t in range(1, n + 1):
        roe_t = inp.roe_forecast[t - 1]
        excess_return_t = (roe_t - ke) * bv_prev
        pv_excess += excess_return_t / (1 + ke) ** t
        bv_prev = bv_prev * (1 + inp.bv_growth_forecast[t - 1])

    bv_n = bv_prev
    terminal_excess = (inp.terminal_roe - ke) * bv_n * (1 + inp.terminal_growth) \
        / (ke - inp.terminal_growth)
    pv_terminal = terminal_excess / (1 + ke) ** n

    equity_value = inp.book_equity_0 + pv_excess + pv_terminal
    value_per_share = equity_value / inp.shares_outstanding
    implied_pb = equity_value / inp.book_equity_0

    if inp.terminal_roe > 0.20:
        warnings.append(f"Terminal ROE {inp.terminal_roe:.0%} > 20%; few banks sustain this forever.")
    if inp.tier1_ratio is not None and inp.tier1_ratio < inp.tier1_min:
        warnings.append(f"Tier-1 {inp.tier1_ratio:.1%} below {inp.tier1_min:.1%} floor; growth may be capital-constrained.")
    if implied_pb > 4:
        warnings.append(f"Implied P/B {implied_pb:.1f}x is very high; check the ROE-Ke spread.")

    return BankResult(inp.book_equity_0, pv_excess, pv_terminal,
                      equity_value, value_per_share, implied_pb, warnings)


def _net_income(period: dict[str, Any], tax_rate: float) -> tuple[Optional[float], bool]:
    pretax = period.get("pretax_income")
    tax = period.get("tax_expense")
    if pretax is not None and tax is not None:
        return float(pretax) - float(tax), False
    ebit = period.get("ebit")
    if ebit is not None:
        return float(ebit) * (1 - tax_rate), True
    return None, False


def _historical_roe(statements: list[dict[str, Any]], tax_rate: float) -> tuple[list[float], bool]:
    roes: list[float] = []
    approximated = False
    for index in range(1, len(statements)):
        opening = statements[index - 1].get("total_equity")
        if opening is None or float(opening) == 0:
            continue
        net_income, approx = _net_income(statements[index], tax_rate)
        if net_income is None:
            continue
        roes.append(net_income / float(opening))
        approximated = approximated or approx
    return roes, approximated


def build_bank_inputs(statements: list[dict[str, Any]],
                      market_data: dict[str, Any],
                      profile: dict[str, Any]) -> BankInputs:
    if not statements:
        raise ValueError("No statements available to value the bank.")
    book_equity_0 = statements[-1].get("total_equity")
    if book_equity_0 is None or float(book_equity_0) == 0:
        raise ValueError("book_equity_0 (most recent total_equity) is missing or zero; cannot value the bank.")
    book_equity_0 = float(book_equity_0)
    if market_data.get("shares") is None:
        raise ValueError("shares_outstanding is required for a per-share value; none was available.")

    assumptions = macro.profile_assumptions(profile)
    tax_rate = macro.effective_tax_rate(statements, profile)
    terminal_roe = float(assumptions.get("terminal_roe", TERMINAL_ROE_DEFAULT))
    retention = float(assumptions.get("retention_ratio", RETENTION_RATIO_DEFAULT))

    roes, approximated = _historical_roe(statements, tax_rate)
    if approximated:
        macro.warn("net_income approximated as EBIT*(1-tax_rate) for at least one year; ROE is indicative.")
    if not roes:
        macro.warn("Could not compute a historical ROE; fading from the terminal ROE instead.")
    start_roe = roes[-1] if roes else terminal_roe

    years = macro.FORECAST_YEARS
    roe_forecast = [start_roe + (terminal_roe - start_roe) * (t / years) for t in range(1, years + 1)]
    bv_growth_forecast = [roe * retention for roe in roe_forecast]

    beta_value = macro.beta_for_ticker(market_data.get("ticker"), market_data, profile)
    return BankInputs(
        book_equity_0=book_equity_0,
        roe_forecast=roe_forecast,
        bv_growth_forecast=bv_growth_forecast,
        cost_of_equity=macro.cost_of_equity(beta_value, profile),
        terminal_roe=terminal_roe,
        terminal_growth=macro.terminal_growth(profile),
        shares_outstanding=float(market_data["shares"]),
    )


class BankExcessReturn(ValuationModel[BankInputs]):
    """Excess-return valuation for banks and NBFCs."""

    name = "bank"

    def value(self, inputs: BankInputs) -> BankResult:
        return run_bank_excess_return(inputs)

    def from_statements(self, statements: list[dict[str, Any]],
                        market_data: dict[str, Any], profile: dict[str, Any]) -> BankInputs:
        return build_bank_inputs(statements, market_data, profile)
