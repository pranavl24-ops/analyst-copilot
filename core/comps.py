"""Comparable-company analysis.

Builds a peer set and computes relative valuation multiples (e.g. EV/EBITDA,
P/E, P/B) for benchmarking the subject company.
"""

from __future__ import annotations

from core.models import Company


def build_peer_set(company: Company, candidates: list[Company]) -> list[Company]:
    """Select comparable peers for the subject company."""
    ...


def multiples(company: Company) -> dict[str, float]:
    """Compute valuation multiples for a single company."""
    ...


def comps_table(company: Company, peers: list[Company]) -> dict[str, dict[str, float]]:
    """Build a comps table of multiples across the subject and its peers."""
    ...
