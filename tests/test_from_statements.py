import pytest

from core.valuation import assumptions as macro
from core.valuation.fcff import StandardFCFF, build_dcf_inputs, run_dcf
from core.valuation.financials import BankExcessReturn


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


def test_builds_a_valuable_dcf():
    inputs = StandardFCFF().from_statements(_statements(), _market(), {})
    assert len(inputs.fcff_forecast) == 5
    assert all(f > 0 for f in inputs.fcff_forecast)
    assert 0.05 <= inputs.wacc <= 0.20
    assert inputs.terminal_growth < inputs.wacc
    assert inputs.net_debt == pytest.approx(1500.0 - 9000.0)
    assert inputs.shares_outstanding == 80.7
    result = run_dcf(inputs)
    assert result.value_per_share > 0


def test_tax_rate_is_clamped_and_averaged():
    inputs = build_dcf_inputs(_statements(), _market(), {})
    forecast = inputs.fcff_forecast
    assert forecast[-1] > forecast[0]


def test_fcff_growth_capped_at_15_percent():
    statements = _statements()
    for i, period in enumerate(statements):
        period["revenue"] = 1000.0 * (3.0 ** i)
    inputs = build_dcf_inputs(statements, _market(), {})
    ratio = inputs.fcff_forecast[1] / inputs.fcff_forecast[0]
    assert ratio == pytest.approx(1.15, abs=1e-9)


def test_delta_nwc_derived_when_cashflow_line_absent():
    with_line = _statements()
    without_line = _statements()
    for period in without_line:
        period.pop("change_in_working_capital")
    a = build_dcf_inputs(with_line, _market(), {})
    b = build_dcf_inputs(without_line, _market(), {})
    assert a.fcff_forecast[0] > 0
    assert b.fcff_forecast[0] > 0


def test_raises_without_two_fcff_years():
    statements = _statements()[:1]
    statements[0].pop("change_in_working_capital")
    with pytest.raises(ValueError, match="two years"):
        build_dcf_inputs(statements, _market(), {})


def test_raises_when_market_cap_missing():
    market = _market()
    market["market_cap"] = None
    with pytest.raises(ValueError, match="market_cap"):
        build_dcf_inputs(_statements(), market, {})


def test_raises_when_shares_missing():
    market = _market()
    market["shares"] = None
    with pytest.raises(ValueError, match="shares"):
        build_dcf_inputs(_statements(), market, {})


def test_missing_debt_and_cash_treated_as_zero():
    market = _market()
    market["total_debt"] = None
    market["cash"] = None
    inputs = build_dcf_inputs(_statements(), market, {})
    assert inputs.net_debt == 0.0


def test_profile_overrides_terminal_growth_and_beta():
    profile = {"assumptions": {"terminal_growth": 0.04, "default_beta": 1.2, "rf": 0.07, "erp": 0.055}}
    market = _market()
    market["beta"] = None
    inputs = build_dcf_inputs(_statements(), market, profile)
    assert inputs.terminal_growth == 0.04


def test_bank_from_statements_builds_inputs():
    market = {"price": 1000.0, "market_cap": 700000.0, "shares": 700.0, "beta": 1.1,
              "total_debt": None, "cash": None}
    inputs = BankExcessReturn().from_statements(_statements(), market, {})
    assert inputs.book_equity_0 == 30200.0
    assert len(inputs.roe_forecast) == 5
    assert len(inputs.bv_growth_forecast) == 5
    assert inputs.roe_forecast[-1] == pytest.approx(0.15)
    assert inputs.terminal_growth < inputs.cost_of_equity


def test_regression_beta_recovers_known_slope():
    index_returns = [0.02, -0.01, 0.03, 0.00, -0.02, 0.015]
    stock_returns = [2.0 * r for r in index_returns]
    assert macro.regression_beta(stock_returns, index_returns) == pytest.approx(2.0)


def test_regression_beta_matches_manual_cov_over_var():
    stock_returns = [0.05, -0.02, 0.01, 0.04, -0.03]
    index_returns = [0.03, -0.01, 0.02, 0.01, -0.02]
    n = len(stock_returns)
    mean_s = sum(stock_returns) / n
    mean_i = sum(index_returns) / n
    cov = sum((s - mean_s) * (i - mean_i) for s, i in zip(stock_returns, index_returns)) / (n - 1)
    var = sum((i - mean_i) ** 2 for i in index_returns) / (n - 1)
    assert macro.regression_beta(stock_returns, index_returns) == pytest.approx(cov / var)


def test_regression_beta_rejects_misaligned_series():
    with pytest.raises(ValueError):
        macro.regression_beta([0.01, 0.02], [0.01])


def test_regression_beta_rejects_zero_variance():
    with pytest.raises(ValueError):
        macro.regression_beta([0.01, 0.02, 0.03], [0.01, 0.01, 0.01])


def test_profile_beta_override_wins_without_network():
    inputs = build_dcf_inputs(_statements(), _market(), {"beta": 0.9})
    expected_ke = macro.cost_of_equity(0.9, {"beta": 0.9})
    assert macro.cost_of_equity(macro.beta_for_ticker("CIPLA.NS", _market(), {"beta": 0.9}), {}) == pytest.approx(expected_ke)


def test_non_indian_ticker_uses_field_beta():
    market = _market()
    market["beta"] = 1.3
    assert macro.beta_for_ticker("AAPL", market, {}) == 1.3


def test_bank_raises_on_zero_book_equity():
    statements = _statements()
    statements[-1]["total_equity"] = 0.0
    market = {"shares": 700.0, "market_cap": 1.0, "beta": 1.0}
    with pytest.raises(ValueError, match="book_equity_0"):
        BankExcessReturn().from_statements(statements, market, {})
