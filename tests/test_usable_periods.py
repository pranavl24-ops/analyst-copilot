"""Tests for the single source of truth that filters statements to usable periods.

A pulled period that is not fully empty but is missing fields the model needs (a
real case: APOLLOHOSP.NS / BHARTIARTL.NS have an FY where capex and most lines are
``None`` while receivables survive) must be dropped, not blindly floated into a
crash. The valuation then runs off the remaining complete periods.
"""

from __future__ import annotations

from copilot import usable_periods
from core.valuation.fcff import StandardFCFF, run_dcf


def _complete_period(year: int, revenue: float) -> dict[str, float]:
    return {
        "year": year,
        "revenue": revenue,
        "ebit": revenue * 0.18,
        "depreciation": revenue * 0.05,
        "capex": -revenue * 0.06,
        "receivables": revenue * 0.12,
        "inventory": revenue * 0.15,
        "payables": revenue * 0.10,
        "change_in_working_capital": -revenue * 0.01,
        "tax_expense": revenue * 0.18 * 0.25,
        "pretax_income": revenue * 0.18,
        "total_equity": revenue * 1.2,
    }


def _partial_period(year: int) -> dict[str, object]:
    period = {field: None for field in _complete_period(year, 1.0)}
    period["year"] = year
    period["receivables"] = 3100.0
    return period


def _statements_with_one_partial() -> list[dict[str, object]]:
    return [
        _complete_period(2021, 19160.0),
        _partial_period(2022),
        _complete_period(2023, 22753.0),
        _complete_period(2024, 25774.0),
        _complete_period(2025, 27548.0),
    ]


def _market() -> dict[str, float]:
    return {"ticker": "APOLLOHOSP.NS", "price": 1500.0, "market_cap": 121050.0,
            "shares": 80.7, "beta": 0.7, "total_debt": 1500.0, "cash": 9000.0}


def test_partial_fcff_period_is_dropped_with_reason():
    usable, dropped = usable_periods(_statements_with_one_partial(), "fcff")
    assert [p["year"] for p in usable] == [2021, 2023, 2024, 2025]
    assert len(dropped) == 1
    year, reason = dropped[0]
    assert year == 2022
    assert "capex" in reason


def test_values_off_the_remaining_complete_periods():
    usable, _ = usable_periods(_statements_with_one_partial(), "fcff")
    model = StandardFCFF()
    inputs = model.from_statements(usable, _market(), {})
    result = run_dcf(inputs)
    assert len(inputs.fcff_forecast) == 10
    assert result.value_per_share > 0


def test_partial_period_does_not_crash_the_payload_floats():
    usable, _ = usable_periods(_statements_with_one_partial(), "fcff")
    assert all(p.get("capex") is not None for p in usable)
    assert all(p.get("depreciation") is not None for p in usable)


def test_fcff_needs_either_cashflow_line_or_balance_components():
    period = _complete_period(2024, 25774.0)
    period["change_in_working_capital"] = None
    period["payables"] = None
    usable, dropped = usable_periods([period, _complete_period(2025, 27548.0)], "fcff")
    assert [p["year"] for p in usable] == [2025]
    assert "working-capital inputs" in dropped[0][1]


def test_bank_criteria_drops_periods_missing_equity_or_income():
    good = {"year": 2024, "total_equity": 20000.0, "pretax_income": 3000.0, "tax_expense": 700.0}
    bad = {"year": 2023, "total_equity": None, "pretax_income": 3000.0, "tax_expense": 700.0}
    usable, dropped = usable_periods([bad, good], "bank")
    assert [p["year"] for p in usable] == [2024]
    assert "total_equity" in dropped[0][1]


def test_too_few_usable_periods_is_detectable():
    statements = [_complete_period(2024, 25774.0), _partial_period(2025)]
    usable, _ = usable_periods(statements, "fcff")
    assert len(usable) < 2
