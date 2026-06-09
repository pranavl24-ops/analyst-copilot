"""Tests for comparable-company analysis."""

from __future__ import annotations

import math

import pytest

from core import comps
from data.fetch import DataError


def _company(market_cap, total_debt, cash, shares, ebit, depreciation, pretax, tax) -> dict:
    return {
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "shares": shares,
        "statements": [
            {
                "year": 2023,
                "revenue": 100.0,
                "ebit": ebit,
                "depreciation": depreciation,
                "pretax_income": pretax,
                "tax_expense": tax,
            }
        ],
    }


UNIVERSE = {
    "TGT": _company(200, 50, 10, 10, 30, 9, 25, 5),
    "PEER1": _company(300, 100, 20, 15, 40, 10, 40, 10),
    "PEER2": _company(500, 50, 50, 20, 60, 20, 60, 15),
}


@pytest.fixture
def patched_merged(monkeypatch):
    def fake(ticker: str) -> dict:
        if ticker == "BAD":
            raise DataError("no data")
        return UNIVERSE[ticker]

    monkeypatch.setattr(comps, "merged", fake)


def test_medians_and_implied_range(patched_merged):
    result = comps.run_comps("TGT", ["PEER1", "PEER2"])

    assert math.isclose(result.median_ev_ebitda, (7.6 + 6.25) / 2)
    assert math.isclose(result.median_pe, (10.0 + 500 / 45) / 2)

    median_ev_ebitda = (7.6 + 6.25) / 2
    expected_ev_implied = (median_ev_ebitda * 39 - 40) / 10
    median_pe = (10.0 + 500 / 45) / 2
    expected_pe_implied = (median_pe * 20) / 10

    assert math.isclose(result.implied_from_ev_ebitda, expected_ev_implied)
    assert math.isclose(result.implied_from_pe, expected_pe_implied)
    assert math.isclose(result.value_per_share_low, min(expected_ev_implied, expected_pe_implied))
    assert math.isclose(result.value_per_share_high, max(expected_ev_implied, expected_pe_implied))


def test_missing_peer_is_recorded_not_fatal(patched_merged):
    result = comps.run_comps("TGT", ["PEER1", "BAD", "PEER2"])
    assert result.missing == ["BAD"]
    assert {p.ticker for p in result.peers} == {"PEER1", "PEER2"}


def test_all_peers_missing_raises(patched_merged):
    with pytest.raises(comps.CompsError):
        comps.run_comps("TGT", ["BAD"])
