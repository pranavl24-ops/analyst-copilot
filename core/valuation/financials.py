"""Valuation model for financial institutions (banks, insurers).

Uses equity-side approaches such as the dividend discount or excess-returns
(residual income) model, since FCFF is ill-suited to financials.
"""

from __future__ import annotations

from core.models import Company
from core.valuation.base import ValuationModel, ValuationResult


class FinancialsModel(ValuationModel):
    """Excess-returns / dividend-discount valuation for financial firms."""

    name: str = "financials"

    def cost_of_equity(self, company: Company) -> float:
        """Estimate the cost of equity (e.g. via CAPM)."""
        ...

    def excess_returns(self, company: Company) -> list[float]:
        """Project excess returns (ROE above cost of equity) on book equity."""
        ...

    def value(self, company: Company) -> ValuationResult:
        """Run the financials valuation and return the result."""
        ...
