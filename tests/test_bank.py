from core.valuation.financials import BankInputs, run_bank_excess_return


def test_reduces_to_closed_form_pb():
    roe, ke, g = 0.17, 0.135, 0.055
    N = 40
    inp = BankInputs(
        book_equity_0=100.0,
        roe_forecast=[roe] * N,
        bv_growth_forecast=[g] * N,
        cost_of_equity=ke,
        terminal_roe=roe,
        terminal_growth=g,
        shares_outstanding=1.0,
    )
    r = run_bank_excess_return(inp)
    closed_form = (roe - g) / (ke - g)
    assert abs(r.implied_pb - closed_form) / closed_form < 0.005
