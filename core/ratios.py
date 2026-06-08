"""Financial ratio calculations.

Computes profitability, leverage, liquidity, and efficiency ratios from a
company's statements.
"""

from __future__ import annotations

from core.models import Company


def profitability(company: Company) -> dict[str, float]:
    """Compute margin and return ratios (e.g. gross margin, ROE, ROIC)."""
    ...


def leverage(company: Company) -> dict[str, float]:
    """Compute leverage ratios (e.g. debt/equity, interest coverage)."""
    ...


def liquidity(company: Company) -> dict[str, float]:
    """Compute liquidity ratios (e.g. current ratio, quick ratio)."""
    ...


def all_ratios(company: Company) -> dict[str, float]:
    """Compute the full set of ratios for a company."""
    ...
