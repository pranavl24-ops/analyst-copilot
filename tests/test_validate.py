"""Tests for statement validation and sanity checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.models import Statements
from core.validate import BrokenDataError, sanity_checks


def _clean() -> Statements:
    return Statements(
        revenue=[100.0, 110.0, 125.0],
        ebit=[20.0, 22.0, 26.0],
        depreciation=[5.0, 6.0, 7.0],
        capex=[-8.0, -9.0, -10.0],
        nwc=[3.0, 4.0, 5.0],
        tax_rate=0.25,
    )


def test_clean_statements_have_no_flags() -> None:
    assert sanity_checks(_clean(), market_cap=900.0) == []


def test_tax_rate_above_cap_rejected() -> None:
    with pytest.raises(ValidationError):
        Statements(
            revenue=[100.0, 110.0],
            ebit=[20.0, 22.0],
            depreciation=[5.0, 6.0],
            capex=[-8.0, -9.0],
            nwc=[3.0, 4.0],
            tax_rate=0.60,
        )


def test_single_year_series_rejected() -> None:
    with pytest.raises(ValidationError):
        Statements(
            revenue=[100.0],
            ebit=[20.0],
            depreciation=[5.0],
            capex=[-8.0],
            nwc=[3.0],
            tax_rate=0.25,
        )


def test_misaligned_series_rejected() -> None:
    with pytest.raises(ValidationError):
        Statements(
            revenue=[100.0, 110.0, 125.0],
            ebit=[20.0, 22.0],
            depreciation=[5.0, 6.0],
            capex=[-8.0, -9.0],
            nwc=[3.0, 4.0],
            tax_rate=0.25,
        )


def test_non_positive_revenue_flagged() -> None:
    s = _clean().model_copy(update={"revenue": [100.0, 0.0, 125.0]})
    flags = sanity_checks(s)
    assert any("revenue in year 1 is non-positive" in f for f in flags)


def test_negative_ebit_flagged() -> None:
    s = _clean().model_copy(update={"ebit": [20.0, -5.0, 26.0]})
    flags = sanity_checks(s)
    assert any("EBIT in year 1 is negative" in f for f in flags)


def test_margin_swing_flagged() -> None:
    s = _clean().model_copy(update={"ebit": [20.0, 22.0, 80.0]})
    flags = sanity_checks(s)
    assert any("EBIT margin swung" in f for f in flags)


def test_scale_mismatch_flagged() -> None:
    flags = sanity_checks(_clean(), market_cap=5_000.0)
    assert any("orders of magnitude" in f for f in flags)


def test_non_finite_value_raises() -> None:
    s = _clean().model_copy(update={"revenue": [100.0, float("nan"), 125.0]})
    with pytest.raises(BrokenDataError):
        sanity_checks(s)


def test_non_positive_market_cap_raises() -> None:
    with pytest.raises(BrokenDataError):
        sanity_checks(_clean(), market_cap=0.0)
