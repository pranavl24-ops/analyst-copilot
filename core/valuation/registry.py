"""Registry mapping model names and company profiles to valuation models.

Lets the pipeline select the appropriate valuation model (e.g. FCFF vs.
financials) based on a company's industry profile.
"""

from __future__ import annotations

from core.valuation.base import ValuationModel


def register(model: ValuationModel) -> None:
    """Register a valuation model by its name."""
    ...


def get(name: str) -> ValuationModel:
    """Retrieve a registered valuation model by name."""
    ...


def for_profile(profile: str) -> ValuationModel:
    """Select the appropriate valuation model for a company profile."""
    ...
