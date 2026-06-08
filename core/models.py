"""Pydantic data models for financial statements and company snapshots.

Defines the typed structures that flow through validation, ratio analysis,
comps, and valuation.
"""

from __future__ import annotations

from pydantic import BaseModel


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
