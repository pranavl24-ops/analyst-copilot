"""Validation and sanity checks for company financial data.

Verifies internal consistency (e.g. balance sheet balances, statement
linkages) and flags missing or implausible values before downstream use.
"""

from __future__ import annotations

from core.models import Company


class ValidationIssue:
    """A single validation finding with severity and message."""


def validate(company: Company) -> list[ValidationIssue]:
    """Run all validation checks and return any issues found."""
    ...


def balance_sheet_balances(company: Company) -> bool:
    """Check that assets equal liabilities plus equity within tolerance."""
    ...
