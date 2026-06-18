from dataclasses import replace

import pytest

from core.valuation.fcff import StandardFCFF, forecast_from_drivers, run_dcf
from core.valuation.financials import BankExcessReturn, BankInputs, run_bank_excess_return
from core.valuation.implied import implied_expectations


def _statements():
    return [
        {"year": 2021, "revenue": 19160.0, "ebit": 3200.0, "depreciation": 900.0,
         "capex": -900.0, "receivables": 3100.0, "inventory": 4200.0, "payables": 2600.0,
         "change_in_working_capital": -450.0, "tax_expense": 850.0,
         "pretax_income": 3050.0, "total_equity": 18500.0},
        {"year": 2022, "revenue": 21764.0, "ebit": 3450.0, "depreciation": 980.0,
         "capex": -1000.0, "receivables": 3300.0, "inventory": 4600.0, "payables": 2750.0,
         "change_in_working_capital": -380.0, "tax_expense": 900.0,
         "pretax_income": 3300.0, "total_equity": 20800.0},
        {"year": 2023, "revenue": 22753.0, "ebit": 3850.0, "depreciation": 1050.0,
         "capex": -1100.0, "receivables": 3500.0, "inventory": 4900.0, "payables": 2900.0,
         "change_in_working_capital": -320.0, "tax_expense": 1000.0,
         "pretax_income": 3700.0, "total_equity": 22900.0},
        {"year": 2024, "revenue": 25774.0, "ebit": 5050.0, "depreciation": 1150.0,
         "capex": -1300.0, "receivables": 3850.0, "inventory": 5300.0, "payables": 3100.0,
         "change_in_working_capital": -300.0, "tax_expense": 1300.0,
         "pretax_income": 4900.0, "total_equity": 26500.0},
        {"year": 2025, "revenue": 27548.0, "ebit": 5900.0, "depreciation": 1250.0,
         "capex": -1400.0, "receivables": 4100.0, "inventory": 5650.0, "payables": 3300.0,
         "change_in_working_capital": -300.0, "tax_expense": 1500.0,
         "pretax_income": 5750.0, "total_equity": 30200.0},
    ]


def _market():
    return {"price": 1500.0, "market_cap": 121050.0, "shares": 80.7,
            "beta": 0.7, "total_debt": 1500.0, "cash": 9000.0}


def _fcff_inputs():
    return StandardFCFF().from_statements(_statements(), _market(), {})


def _bank_inputs():
    return BankInputs(
        book_equity_0=1000.0,
        roe_forecast=[0.15] * 10,
        bv_growth_forecast=[0.08] * 10,
        cost_of_equity=0.135,
        terminal_roe=0.15,
        terminal_growth=0.05,
        shares_outstanding=100.0,
    )


def _by_label(expectations, label):
    for assumption in expectations.assumptions:
        if assumption.label == label:
            return assumption
    raise AssertionError(f"no implied assumption labelled {label!r}")


def test_fcff_implied_terminal_growth_round_trips():
    inputs = _fcff_inputs()
    base = run_dcf(inputs).value_per_share
    price = base * 1.3

    implied = _by_label(implied_expectations(StandardFCFF(), inputs, price), "terminal growth")
    assert implied.reachable

    replayed = run_dcf(replace(inputs, terminal_growth=implied.implied)).value_per_share
    assert replayed == pytest.approx(price, rel=1e-4)


def test_fcff_implied_ebit_margin_round_trips():
    inputs = _fcff_inputs()
    base = run_dcf(inputs).value_per_share
    price = base * 1.2

    implied = _by_label(implied_expectations(StandardFCFF(), inputs, price), "target EBIT margin")
    assert implied.reachable

    forecast = forecast_from_drivers(
        inputs.forecast_drivers, inputs.terminal_growth, ebit_margin=implied.implied
    )
    replayed = run_dcf(replace(inputs, fcff_forecast=forecast)).value_per_share
    assert replayed == pytest.approx(price, rel=1e-4)


def test_fcff_solving_at_base_value_recovers_every_base_assumption():
    inputs = _fcff_inputs()
    drivers = inputs.forecast_drivers
    base = run_dcf(inputs).value_per_share

    expectations = implied_expectations(StandardFCFF(), inputs, base)
    assert _by_label(expectations, "terminal growth").implied == pytest.approx(
        inputs.terminal_growth, abs=1e-4
    )
    assert _by_label(expectations, "target EBIT margin").implied == pytest.approx(
        drivers.target_ebit_margin, abs=1e-4
    )
    assert _by_label(expectations, "near-term revenue growth").implied == pytest.approx(
        min(drivers.revenue_cagr, 0.15), abs=1e-4
    )


def test_bank_implied_terminal_roe_round_trips():
    inputs = _bank_inputs()
    base = run_bank_excess_return(inputs).value_per_share
    price = base * 1.4

    implied = _by_label(implied_expectations(BankExcessReturn(), inputs, price), "terminal ROE")
    assert implied.reachable

    replayed = run_bank_excess_return(replace(inputs, terminal_roe=implied.implied)).value_per_share
    assert replayed == pytest.approx(price, rel=1e-4)


def test_bank_implied_cost_of_equity_round_trips():
    inputs = _bank_inputs()
    base = run_bank_excess_return(inputs).value_per_share
    price = base * 0.8

    implied = _by_label(implied_expectations(BankExcessReturn(), inputs, price), "cost of equity")
    assert implied.reachable

    replayed = run_bank_excess_return(replace(inputs, cost_of_equity=implied.implied)).value_per_share
    assert replayed == pytest.approx(price, rel=1e-4)


def test_unreachable_price_is_reported_not_forced():
    inputs = _bank_inputs()
    implied = _by_label(implied_expectations(BankExcessReturn(), inputs, 1e9), "terminal ROE")
    assert not implied.reachable
    assert implied.implied is None


def test_implied_above_band_is_flagged_implausible():
    inputs = _bank_inputs()
    base = run_bank_excess_return(inputs).value_per_share
    expectations = implied_expectations(BankExcessReturn(), inputs, base * 2.5)
    roe = _by_label(expectations, "terminal ROE")
    if roe.reachable:
        assert roe.implausible == (roe.implied > 0.20)
