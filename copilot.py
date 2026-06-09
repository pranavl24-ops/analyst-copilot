"""Command-line entry point for the analyst copilot.

Wires the pipeline end to end: load the trust-anchor override, validate the
statements, pick the sector-appropriate valuation model, value the company, run
ratios and a comps cross-check, draft the memo, and render a one-page PDF with a
football-field chart — all into ``out/<TICKER>/``.

    python copilot.py CIPLA.NS
    python copilot.py ICICIBANK.NS --model bank

Valuation inputs come from ``data/overrides/<TICKER>.yaml`` because the
statements->inputs projection (``ValuationModel.from_statements``) is deferred;
a ticker without an override is refused loudly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

from core.comps import CompsResult, PeerMultiples
from core.models import Statements
from core.ratios import compute_ratios
from core.validate import sanity_checks
from core.valuation.fcff import DCFInputs, run_dcf, sensitivity
from core.valuation.financials import BankInputs, run_bank_excess_return
from core.valuation.registry import NotSupported, pick_model
from data.fetch import load_override
from memo.prompt import build_payload, generate_memo
from memo.render import football_field, render_pdf

OUT_ROOT = Path("out")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="copilot",
        description="Ticker in, one-page IC memo (and PDF) out.",
    )
    parser.add_argument("ticker", help="Exchange-suffixed ticker, e.g. CIPLA.NS")
    parser.add_argument(
        "--model",
        default=None,
        help="Force a valuation model (e.g. 'bank' or 'fcff'); overrides the sector map.",
    )
    return parser


def run(ticker: str, model: Optional[str] = None) -> int:
    """Run the full analysis pipeline for a ticker and return an exit code."""
    data = load_override(ticker)
    if data is None:
        print(
            f"No override at data/overrides/{ticker}.yaml. Valuation-from-statements "
            f"wiring is deferred, so a trust-anchor override is required to value {ticker}."
        )
        return 1
    if "valuation" not in data:
        print(f"Override for {ticker} has no 'valuation' block; cannot value it.")
        return 1

    sector = str(data.get("sector", ""))
    try:
        model_cls = pick_model(sector, model)
    except NotSupported as error:
        print(f"Refusing to value {ticker}: {error}")
        return 2

    model_name = model_cls.name
    vblock = data["valuation"]
    declared = str(vblock.get("model", model_name))
    if declared != model_name:
        print(
            f"Model mismatch for {ticker}: registry picked {model_name!r} but the override's "
            f"valuation block is {declared!r}. Align --model/sector with the override."
        )
        return 1

    print(f"[{ticker}] model={model_name} sector={sector or '—'}")
    if model_name == "bank":
        payload, chart_methods = _value_bank(ticker, data)
    else:
        payload, chart_methods = _value_fcff(ticker, data)

    memo_markdown = generate_memo(payload)

    outdir = OUT_ROOT / ticker
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "payload.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (outdir / "memo.md").write_text(memo_markdown)

    chart = football_field(chart_methods, payload["current_price"], outdir / "football_field.png")
    pdf = render_pdf(memo_markdown, chart, outdir / "memo.pdf")

    print(
        f"[{ticker}] value/share {payload['value_per_share']:,.2f} vs price "
        f"{payload['current_price']:,.2f} ({payload['upside_pct']:+.1f}%)"
    )
    print(f"[{ticker}] wrote {outdir}/memo.pdf, memo.md, payload.json, football_field.png")
    return 0


def _value_fcff(ticker: str, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    flags = _validate_fcff(data)
    shares = float(data["shares"])
    net_debt = float(data["total_debt"]) - float(data["cash"])
    vblock = data["valuation"]

    inputs = DCFInputs(
        fcff_forecast=[float(x) for x in vblock["fcff_forecast"]],
        wacc=float(vblock["wacc"]),
        terminal_growth=float(vblock["terminal_growth"]),
        net_debt=net_debt,
        shares_outstanding=shares,
    )
    result = run_dcf(inputs)
    grid = sensitivity(inputs)
    ratios = compute_ratios(data)
    comps = _fcff_comps(data, net_debt, shares)

    cblock = data["comps"]
    assumptions = {
        "forecast_years": len(inputs.fcff_forecast),
        "wacc_pct": round(inputs.wacc * 100, 2),
        "terminal_growth_pct": round(inputs.terminal_growth * 100, 2),
        "net_debt": net_debt,
        "shares_outstanding": shares,
        "comps_median_ev_ebitda": cblock["median_ev_ebitda"],
        "comps_median_pe": cblock["median_pe"],
    }

    payload = build_payload(
        company=str(data["company"]),
        ticker=ticker,
        currency=str(data.get("currency", "")),
        current_price=float(data["price"]),
        method="FCFF DCF",
        valuation=result,
        assumptions=assumptions,
        ratios=ratios,
        comps=comps,
        sensitivity=grid,
        extra_warnings=flags,
    )
    methods = {
        "DCF (FCFF)": _range_from_grid(grid),
        "Comps": (comps.value_per_share_low, comps.value_per_share_high),
        "52-week": _range_52w(data),
    }
    return payload, methods


def _value_bank(ticker: str, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    flags = _validate_bank(data)
    shares = float(data["shares"])
    vblock = data["valuation"]

    inputs = BankInputs(
        book_equity_0=float(vblock["book_equity_0"]),
        roe_forecast=[float(x) for x in vblock["roe_forecast"]],
        bv_growth_forecast=[float(x) for x in vblock["bv_growth_forecast"]],
        cost_of_equity=float(vblock["cost_of_equity"]),
        terminal_roe=float(vblock["terminal_roe"]),
        terminal_growth=float(vblock["terminal_growth"]),
        shares_outstanding=shares,
        tier1_ratio=vblock.get("tier1_ratio"),
    )
    result = run_bank_excess_return(inputs)
    grid = _bank_sensitivity(inputs)
    ratios = _bank_ratios(data)
    comps = _bank_comps(data, shares)

    cblock = data["comps"]
    assumptions = {
        "cost_of_equity_pct": round(inputs.cost_of_equity * 100, 2),
        "terminal_roe_pct": round(inputs.terminal_roe * 100, 2),
        "terminal_growth_pct": round(inputs.terminal_growth * 100, 2),
        "book_equity_0": inputs.book_equity_0,
        "shares_outstanding": shares,
        "comps_median_pb": cblock["median_pb"],
        "comps_median_pe": cblock["median_pe"],
    }
    if inputs.tier1_ratio is not None:
        assumptions["tier1_ratio_pct"] = round(inputs.tier1_ratio * 100, 1)

    payload = build_payload(
        company=str(data["company"]),
        ticker=ticker,
        currency=str(data.get("currency", "")),
        current_price=float(data["price"]),
        method="Bank excess-return",
        valuation=result,
        assumptions=assumptions,
        ratios=ratios,
        comps=comps,
        sensitivity=grid,
        extra_warnings=flags,
    )
    methods = {
        "Excess return": _range_from_grid(grid),
        "Comps": (comps.value_per_share_low, comps.value_per_share_high),
        "52-week": _range_52w(data),
    }
    return payload, methods


def _validate_fcff(data: dict[str, Any]) -> list[str]:
    periods = data["statements"]
    statements = Statements(
        revenue=[float(p["revenue"]) for p in periods],
        ebit=[float(p["ebit"]) for p in periods],
        depreciation=[float(p["depreciation"]) for p in periods],
        capex=[abs(float(p["capex"])) for p in periods],
        nwc=[float(p["receivables"]) + float(p["inventory"]) - float(p["payables"]) for p in periods],
        tax_rate=_latest_tax_rate(periods),
    )
    return sanity_checks(statements, market_cap=float(data["market_cap"]))


def _validate_bank(data: dict[str, Any]) -> list[str]:
    book_equity = float(data["valuation"]["book_equity_0"])
    if book_equity <= 0:
        from core.validate import BrokenDataError

        raise BrokenDataError(f"book_equity_0 must be positive, got {book_equity}")
    equity_series = [float(p["total_equity"]) for p in data["statements"]]
    flags: list[str] = []
    if any(later < earlier for earlier, later in zip(equity_series, equity_series[1:])):
        flags.append("Book equity declined in at least one year of the statement window.")
    return flags


def _latest_tax_rate(periods: list[dict[str, Any]]) -> float:
    latest = periods[-1]
    pretax = float(latest["pretax_income"])
    if pretax <= 0:
        return 0.0
    return min(max(float(latest["tax_expense"]) / pretax, 0.0), 0.45)


def _fcff_comps(data: dict[str, Any], net_debt: float, shares: float) -> CompsResult:
    cblock = data["comps"]
    latest = data["statements"][-1]
    ebitda = float(latest["ebit"]) + float(latest["depreciation"])
    net_income = float(latest["pretax_income"]) - float(latest["tax_expense"])

    implied_ev_ebitda = (float(cblock["median_ev_ebitda"]) * ebitda - net_debt) / shares
    implied_pe = (float(cblock["median_pe"]) * net_income) / shares
    estimates = [implied_ev_ebitda, implied_pe]
    return CompsResult(
        peers=[PeerMultiples(p, None, None) for p in cblock["peers"]],
        missing=[],
        median_ev_ebitda=float(cblock["median_ev_ebitda"]),
        median_pe=float(cblock["median_pe"]),
        implied_from_ev_ebitda=implied_ev_ebitda,
        implied_from_pe=implied_pe,
        value_per_share_low=min(estimates),
        value_per_share_high=max(estimates),
    )


def _bank_comps(data: dict[str, Any], shares: float) -> CompsResult:
    cblock = data["comps"]
    book_equity = float(data["valuation"]["book_equity_0"])
    net_income = float(cblock["net_income_latest"])

    implied_pb = float(cblock["median_pb"]) * book_equity / shares
    implied_pe = float(cblock["median_pe"]) * net_income / shares
    estimates = [implied_pb, implied_pe]
    return CompsResult(
        peers=[PeerMultiples(p, None, None) for p in cblock["peers"]],
        missing=[],
        median_ev_ebitda=None,
        median_pe=float(cblock["median_pe"]),
        implied_from_ev_ebitda=None,
        implied_from_pe=implied_pe,
        value_per_share_low=min(estimates),
        value_per_share_high=max(estimates),
    )


def _bank_ratios(data: dict[str, Any]) -> dict[str, float]:
    periods = data["statements"]
    equity = [float(p["total_equity"]) for p in periods]
    bv_cagr = (equity[-1] / equity[0]) ** (1 / (len(equity) - 1)) - 1
    pat_latest = float(periods[-1]["pretax_income"]) - float(periods[-1]["tax_expense"])
    roe_latest = pat_latest / equity[-1]
    ke = float(data["valuation"]["cost_of_equity"])
    return {
        "roe_latest_pct": round(roe_latest * 100, 1),
        "book_value_cagr_pct": round(bv_cagr * 100, 1),
        "cost_of_equity_pct": round(ke * 100, 1),
        "roe_minus_ke_spread_pp": round((roe_latest - ke) * 100, 1),
    }


def _bank_sensitivity(
    inp: BankInputs,
    ke_steps: tuple[float, ...] = (-0.01, -0.005, 0.0, 0.005, 0.01),
    g_steps: tuple[float, ...] = (-0.01, -0.005, 0.0, 0.005, 0.01),
) -> dict[tuple[float, float], Optional[float]]:
    grid: dict[tuple[float, float], Optional[float]] = {}
    for dke in ke_steps:
        for dg in g_steps:
            ke = inp.cost_of_equity + dke
            g = inp.terminal_growth + dg
            trial = BankInputs(
                book_equity_0=inp.book_equity_0,
                roe_forecast=inp.roe_forecast,
                bv_growth_forecast=inp.bv_growth_forecast,
                cost_of_equity=ke,
                terminal_roe=inp.terminal_roe,
                terminal_growth=g,
                shares_outstanding=inp.shares_outstanding,
                tier1_ratio=inp.tier1_ratio,
            )
            try:
                grid[(round(ke, 4), round(g, 4))] = round(run_bank_excess_return(trial).value_per_share, 1)
            except ValueError:
                grid[(round(ke, 4), round(g, 4))] = None
    return grid


def _range_from_grid(grid: dict[tuple[float, float], Optional[float]]) -> tuple[float, float]:
    values = [v for v in grid.values() if v is not None]
    if not values:
        raise ValueError("sensitivity grid produced no valid scenarios")
    return min(values), max(values)


def _range_52w(data: dict[str, Any]) -> tuple[float, float]:
    low, high = data["range_52w"]
    return float(low), float(high)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to :func:`run`."""
    args = build_parser().parse_args(argv)
    return run(args.ticker, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
