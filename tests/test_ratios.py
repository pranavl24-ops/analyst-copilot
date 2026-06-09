"""Tests for the financial ratio calculations."""

from __future__ import annotations

import math

import pytest

from core.ratios import RatioError, compute_ratios


def _data() -> dict:
    return {
        "ticker": "TGT",
        "market_cap": 200.0,
        "total_debt": 50.0,
        "cash": 10.0,
        "statements": [
            {"year": 2021, "revenue": 100.0, "ebit": 18.0, "depreciation": 5.0, "total_equity": 80.0},
            {"year": 2022, "revenue": 120.0, "ebit": 24.0, "depreciation": 6.0, "total_equity": 90.0},
            {"year": 2023, "revenue": 150.0, "ebit": 30.0, "depreciation": 9.0, "total_equity": 100.0},
        ],
    }


def test_named_numbers_match_manual_calc():
    r = compute_ratios(_data())
    assert math.isclose(r.revenue_cagr, (150 / 100) ** 0.5 - 1)
    assert math.isclose(r.ebit_margin_latest, 30 / 150)
    assert math.isclose(r.ebit_margin_trend, (30 / 150 - 18 / 100) / 2)
    assert math.isclose(r.roce, 30 / (50 + 100))
    assert math.isclose(r.net_debt_to_ebitda, (50 - 10) / (30 + 9))


def test_roce_is_a_book_measure_not_market():
    data = _data()
    market_roce = data["statements"][-1]["ebit"] / (data["total_debt"] + data["market_cap"])
    book_roce = compute_ratios(data).roce
    assert math.isclose(book_roce, 30 / (50 + 100))
    assert book_roce > market_roce


def test_missing_book_equity_raises_without_market_fallback():
    data = _data()
    data["statements"][-1]["total_equity"] = None
    with pytest.raises(RatioError):
        compute_ratios(data)


def _cipla() -> dict:
    return {
        "ticker": "CIPLA.NS",
        "market_cap": 100000.0,
        "total_debt": 300.0,
        "cash": 7000.0,
        "statements": [
            {"year": 2022, "revenue": 21763.0, "ebit": 3819.0, "depreciation": 900.0, "total_equity": 19831.0},
            {"year": 2023, "revenue": 22753.0, "ebit": 4128.0, "depreciation": 950.0, "total_equity": 22431.0},
            {"year": 2024, "revenue": 25774.0, "ebit": 4800.0, "depreciation": 1000.0, "total_equity": 24500.0},
        ],
    }


def test_cipla_shaped_roce_is_book_realistic():
    roce = compute_ratios(_cipla()).roce
    assert 0.18 <= roce <= 0.20


def test_missing_field_raises():
    data = _data()
    data["statements"][-1]["ebit"] = None
    with pytest.raises(RatioError):
        compute_ratios(data)


def test_no_statements_raises():
    with pytest.raises(RatioError):
        compute_ratios({"market_cap": 1.0, "total_debt": 0.0, "cash": 0.0, "statements": []})
