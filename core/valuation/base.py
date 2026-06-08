"""Base interface for valuation models.

Defines the common protocol every valuation model implements so they can be
registered and dispatched uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import Company


class ValuationResult:
    """The output of a valuation: equity value, per-share value, and assumptions."""


class ValuationModel(ABC):
    """Abstract base class for all valuation models."""

    name: str

    @abstractmethod
    def value(self, company: Company) -> ValuationResult:
        """Estimate the value of a company."""
        ...
