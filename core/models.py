"""Pydantic data models for financial statements and company snapshots.

Defines the typed structures that flow through validation, ratio analysis,
comps, and valuation.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

MAX_TAX_RATE = 0.45
MIN_PERIODS = 2


class Statements(BaseModel):
    """Aligned annual time series feeding the valuation models.

    Each series is ordered oldest to newest and the series share a common length:
    index ``i`` refers to the same fiscal year across every field. ``tax_rate`` is
    the effective rate expressed as a fraction.
    """

    revenue: list[float]
    ebit: list[float]
    depreciation: list[float]
    capex: list[float]
    nwc: list[float]
    tax_rate: float

    @field_validator("revenue", "ebit", "depreciation", "capex", "nwc")
    @classmethod
    def _at_least_two_years(cls, series: list[float], info: ValidationInfo) -> list[float]:
        if len(series) < MIN_PERIODS:
            raise ValueError(
                f"{info.field_name} needs at least {MIN_PERIODS} years, got {len(series)}"
            )
        return series

    @field_validator("tax_rate")
    @classmethod
    def _tax_rate_in_range(cls, rate: float) -> float:
        if not 0.0 <= rate <= MAX_TAX_RATE:
            raise ValueError(f"tax_rate must be within [0, {MAX_TAX_RATE}], got {rate}")
        return rate

    @model_validator(mode="after")
    def _series_aligned(self) -> Statements:
        lengths = {
            len(self.revenue),
            len(self.ebit),
            len(self.depreciation),
            len(self.capex),
            len(self.nwc),
        }
        if len(lengths) != 1:
            raise ValueError("all series must cover the same number of years")
        return self


class IncomeStatement(BaseModel):
    """A single-period income statement."""


class BalanceSheet(BaseModel):
    """A single-period balance sheet."""


class CashFlow(BaseModel):
    """A single-period cash flow statement."""


class MarketData(BaseModel):
    """Market-derived inputs such as price, shares, and beta."""


class Company(BaseModel):
    """A company snapshot bundling statements, market data, and metadata."""
