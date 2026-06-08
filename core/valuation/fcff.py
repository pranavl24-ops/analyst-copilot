"""Free-cash-flow-to-firm (FCFF) discounted cash flow valuation.

Projects unlevered free cash flows, discounts them at the WACC, and derives
an intrinsic equity value per share. Suited to non-financial companies.
"""

from __future__ import annotations

from core.models import Company
from core.valuation.base import ValuationModel, ValuationResult


class FCFFModel(ValuationModel):
    """Two-stage FCFF DCF valuation model."""

    name: str = "fcff"

    def project_fcff(self, company: Company) -> list[float]:
        """Project unlevered free cash flows over the forecast horizon."""
        ...

    def wacc(self, company: Company) -> float:
        """Estimate the weighted average cost of capital."""
        ...

    def terminal_value(self, company: Company) -> float:
        """Compute the terminal value at the end of the forecast horizon."""
        ...

    def value(self, company: Company) -> ValuationResult:
        """Run the FCFF DCF and return the valuation result."""
        ...
