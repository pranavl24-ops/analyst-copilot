"""Excess-return (residual-income) valuation for banks and NBFCs.

FCFF is wrong for financials: leverage is the business, not a financing choice.
This values the bank as book equity plus the present value of returns earned
above the cost of equity, discounted at the cost of equity (CAPM), never WACC.
``run_bank_excess_return`` is the verified, standalone engine; ``BankExcessReturn``
wraps it behind the model protocol.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.models import MarketData, Statements
from core.valuation.base import ValuationModel


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


class BankExcessReturn(ValuationModel[BankInputs]):
    """Excess-return valuation for banks and NBFCs."""

    name = "bank"

    def value(self, inputs: BankInputs) -> BankResult:
        return run_bank_excess_return(inputs)

    def from_statements(self, statements: Statements,
                        market_data: MarketData, profile: dict) -> BankInputs:
        raise NotImplementedError("Company/MarketData -> BankInputs wiring is a later step.")
