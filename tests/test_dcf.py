from core.valuation.fcff import DCFInputs, run_dcf
import pytest


def base():
    return DCFInputs(
        fcff_forecast=[4300, 4800, 5400, 5900, 6542],
        wacc=0.1136, terminal_growth=0.06,
        net_debt=-3000, shares_outstanding=80.78, minority_interest=96,
    )


def test_lands_in_corrected_range():
    r = run_dcf(base())
    assert 1100 <= r.value_per_share <= 1400


def test_terminal_uses_year_n_not_year_1():
    correct = run_dcf(base()).value_per_share
    wrong = base(); wrong.fcff_forecast = wrong.fcff_forecast[:-1] + [3978]
    assert run_dcf(wrong).value_per_share < correct


def test_g_above_wacc_raises():
    bad = base(); bad.terminal_growth = 0.15
    with pytest.raises(ValueError):
        run_dcf(bad)
