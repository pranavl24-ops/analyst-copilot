from core.comps import CompsResult, PeerMultiples
from core.ratios import Ratios
from core.valuation.fcff import DCFInputs, run_dcf, sensitivity
from core.valuation.financials import BankInputs, run_bank_excess_return
from memo.prompt import build_payload, generate_memo, render_offline_memo

SECTION_HEADERS = [
    "## 1. Recommendation & Target",
    "## 2. Business",
    "## 3. Valuation",
    "## 4. Assumptions",
    "## 5. Sensitivity",
    "## 6. Risks",
    "## 7. Bottom Line",
]


def _ratios() -> Ratios:
    return Ratios(
        revenue_cagr=0.095,
        ebit_margin_latest=0.214,
        ebit_margin_trend=0.0118,
        roce=0.186,
        net_debt_to_ebitda=-1.05,
    )


def _fcff_payload() -> dict:
    inputs = DCFInputs(
        fcff_forecast=[4270.0, 4650.0, 5050.0, 5480.0, 5920.0, 6330.0, 6710.0],
        wacc=0.107,
        terminal_growth=0.05,
        net_debt=-7500.0,
        shares_outstanding=80.7,
    )
    result = run_dcf(inputs)
    comps = CompsResult(
        peers=[PeerMultiples("SUNPHARMA.NS", 18.0, 32.0)],
        missing=["LUPIN.NS"],
        median_ev_ebitda=16.0,
        median_pe=28.0,
        implied_from_ev_ebitda=1510.53,
        implied_from_pe=1474.60,
        value_per_share_low=1474.60,
        value_per_share_high=1510.53,
    )
    return build_payload(
        company="Cipla Ltd",
        ticker="CIPLA.NS",
        currency="INR crore",
        current_price=1500.0,
        method="FCFF DCF",
        valuation=result,
        assumptions={"wacc_pct": 10.7, "terminal_growth_pct": 5.0, "forecast_years": 7},
        ratios=_ratios(),
        comps=comps,
        sensitivity=sensitivity(inputs),
    )


def _bank_payload() -> dict:
    inputs = BankInputs(
        book_equity_0=50000.0,
        roe_forecast=[0.16, 0.16, 0.17, 0.17, 0.18],
        bv_growth_forecast=[0.14, 0.14, 0.13, 0.13, 0.12],
        cost_of_equity=0.135,
        terminal_roe=0.16,
        terminal_growth=0.06,
        shares_outstanding=700.0,
    )
    result = run_bank_excess_return(inputs)
    return build_payload(
        company="Some Bank Ltd",
        ticker="SOMEBANK.NS",
        currency="INR crore",
        current_price=1600.0,
        method="Bank excess-return",
        valuation=result,
        assumptions={"cost_of_equity_pct": 13.5, "terminal_roe_pct": 16.0},
        ratios=_ratios(),
    )


def test_offline_memo_has_all_seven_sections():
    memo = render_offline_memo(_fcff_payload())
    for header in SECTION_HEADERS:
        assert header in memo


def test_offline_memo_marks_itself_offline():
    memo = render_offline_memo(_fcff_payload())
    assert "OFFLINE" in memo
    assert "NOT written by the Claude API" in memo


def test_offline_memo_includes_value_per_share_and_upside():
    payload = _fcff_payload()
    memo = render_offline_memo(payload)
    assert f"{payload['value_per_share']:,.2f}" in memo
    assert f"{payload['upside_pct']:+.1f}%" in memo


def test_offline_memo_uses_fcff_terminology_not_bank():
    memo = render_offline_memo(_fcff_payload())
    assert "WACC" in memo
    assert "free cash flow" in memo.lower()
    assert "cost of equity" not in memo.lower()


def test_offline_memo_uses_bank_terminology_not_dcf():
    memo = render_offline_memo(_bank_payload())
    assert "cost of equity" in memo.lower()
    assert "price-to-book" in memo.lower()
    assert "WACC" not in memo
    assert "DCF" not in memo


def test_offline_memo_lists_every_warning_in_risks():
    payload = _fcff_payload()
    payload["warnings"] = ["First flagged issue", "Second flagged issue"]
    memo = render_offline_memo(payload)
    risks = memo.split("## 6. Risks", 1)[1]
    assert "First flagged issue" in risks
    assert "Second flagged issue" in risks


def test_negative_upside_yields_sell():
    memo = render_offline_memo(_fcff_payload())
    assert "**SELL.**" in memo


def test_generate_memo_defaults_to_offline_without_network(monkeypatch):
    monkeypatch.delenv("MEMO_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    memo = generate_memo(_fcff_payload())
    assert "OFFLINE" in memo
