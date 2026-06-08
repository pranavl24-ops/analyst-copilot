"""Fetch raw financial data for a ticker and apply local overrides.

Pulls statements and market data from external providers (e.g. yfinance),
then merges any analyst-supplied overrides from ``data/overrides/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def fetch_statements(ticker: str) -> dict[str, Any]:
    """Fetch income statement, balance sheet, and cash flow for a ticker."""
    ...


def fetch_market_data(ticker: str) -> dict[str, Any]:
    """Fetch market data such as price, shares outstanding, and beta."""
    ...


def load_overrides(ticker: str, overrides_dir: Path | None = None) -> dict[str, Any]:
    """Load analyst override values for a ticker, if any exist."""
    ...


def fetch(ticker: str, overrides_dir: Path | None = None) -> dict[str, Any]:
    """Fetch all data for a ticker with overrides applied."""
    ...
