"""Base interface for valuation models.

Defines the common protocol every valuation model implements so they can be
selected by the registry and dispatched uniformly. Each model declares its own
input type via ``Generic[InputT]`` and returns a concrete result that satisfies
the ``ValuationResult`` protocol structurally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT")


@runtime_checkable
class ValuationResult(Protocol):
    """The common surface every valuation result exposes."""

    value_per_share: float
    warnings: list[str]


class ValuationModel(ABC, Generic[InputT]):
    """Abstract base class for all valuation models."""

    name: str

    @abstractmethod
    def value(self, inputs: InputT) -> ValuationResult:
        """Estimate the value of a company from its model-specific inputs."""
        ...
