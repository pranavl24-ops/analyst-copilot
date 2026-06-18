"""Command-line entry point for the analyst copilot.

Wires the pipeline end to end: resolve the valuation inputs once, value the
company, run ratios and (when available) a comps cross-check, collect the
plausibility warnings, draft the memo, and render a one-page PDF with a
football-field chart — all into ``out/<TICKER>/``.

    python copilot.py CIPLA.NS
    python copilot.py ICICIBANK.NS --model bank

There is a SINGLE pipeline with two input sources, not two pipelines: if a
``data/overrides/<TICKER>.yaml`` carries a ``valuation`` block it is the trust
anchor and the model inputs are built from it; otherwise the inputs are projected
from pulled statements via ``ValuationModel.from_statements``. Either way the same
downstream runs — memo, PDF, payload, and chart — so no ticker needs a hand-made
override to get a memo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from core.comps import CompsResult, PeerMultiples
from core.models import Statements
from core.ratios import RatioError, compute_ratios
from core.validate import price_scale_flags, sanity_checks
from core.valuation import assumptions as macro
from core.valuation.fcff import (
    DCFInputs,
    StandardFCFF,
    _historical_fcff,
    _revenue_cagr,
    run_dcf,
    sensitivity,
)
from core.valuation.financials import BankExcessReturn, BankInputs, run_bank_excess_return
from core.valuation.implied import implied_expectations
from core.valuation.implied import to_payload as implied_to_payload
from core.valuation.plausibility import (
    book_equity_distortion_warnings,
    extreme_gap_warnings,
    sum_of_the_parts_warnings,
    unstable_fcff_warnings,
)
from core.valuation.registry import NotSupported, pick_model
from data.fetch import MARKET_FIELDS, STATEMENT_FIELDS, DataError, load_override, merged
from memo.prompt import build_payload, generate_memo
from memo.render import football_field, render_pdf

OUT_ROOT = Path("out")
PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

FCFF_REQUIRED_FIELDS = ("ebit", "depreciation", "capex", "tax_expense", "pretax_income")
BANK_REQUIRED_FIELDS = ("total_equity", "pretax_income", "tax_expense")


def _has_working_capital(period: dict[str, Any]) -> bool:
    if period.get("change_in_working_capital") is not None:
        return True
    return all(period.get(field) is not None for field in ("receivables", "inventory", "payables"))


def _missing_fields(period: dict[str, Any], model_name: str) -> list[str]:
    if model_name == "bank":
        return [field for field in BANK_REQUIRED_FIELDS if period.get(field) is None]
    missing = [field for field in FCFF_REQUIRED_FIELDS if period.get(field) is None]
    if not _has_working_capital(period):
        missing.append("working-capital inputs")
    return missing


def usable_periods(
    statements: list[dict[str, Any]], model_name: str
) -> tuple[list[dict[str, Any]], list[tuple[Any, str]]]:
    """Split pulled statements into periods complete enough for the chosen model.

    Returns the usable periods (order preserved) and, for each dropped period, a
    ``(year, reason)`` pair naming the absent fields. This is the single source of
    truth for which periods the valuation, payload, and console report all consume:
    nothing downstream re-reads the raw unfiltered list and blindly floats fields
    that may be ``None``.
    """
    usable: list[dict[str, Any]] = []
    dropped: list[tuple[Any, str]] = []
    for period in statements:
        missing = _missing_fields(period, model_name)
        if missing:
            dropped.append((period.get("year", "?"), f"missing {', '.join(missing)}"))
        else:
            usable.append(period)
    return usable, dropped


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
    """Run the full analysis pipeline for a ticker and return an exit code.

    Resolves the model inputs once — from the override's ``valuation`` block when
    present, otherwise projected from pulled statements — then runs the one shared
    downstream (memo, PDF, payload, chart) for whichever source produced them.
    """
    data = load_override(ticker)
    sector = str((data or {}).get("sector", ""))
    try:
        model_cls = pick_model(sector, model)
    except NotSupported as error:
        print(f"Refusing to value {ticker}: {error}")
        return 2

    if data is not None and "valuation" in data:
        resolved = _resolve_from_override(ticker, data, model_cls, sector)
    else:
        if data is None:
            print(f"[{ticker}] no override found; valuing from pulled statements.")
        else:
            print(f"[{ticker}] override has no 'valuation' block; valuing from pulled statements.")
        resolved = _resolve_from_statements(ticker, model_cls, sector)

    if isinstance(resolved, int):
        return resolved
    payload, chart_methods = resolved
    return _emit(ticker, payload, chart_methods)


def _emit(ticker: str, payload: dict[str, Any],
          chart_methods: dict[str, tuple[float, float]]) -> int:
    """Run the shared downstream: memo, chart, PDF, files, and console summary."""
    memo_markdown = generate_memo(payload)

    outdir = OUT_ROOT / ticker
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "payload.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (outdir / "memo.md").write_text(memo_markdown)

    chart = football_field(chart_methods, payload["current_price"], outdir / "football_field.png")
    render_pdf(memo_markdown, payload, chart, outdir / "memo.pdf")

    print(
        f"[{ticker}] value/share {payload['value_per_share']:,.2f} vs price "
        f"{payload['current_price']:,.2f} ({payload['upside_pct']:+.1f}%)"
    )
    for warning in payload["warnings"]:
        print(f"[{ticker}] warning: {warning}")
    market_implied = payload.get("market_implied")
    if market_implied:
        print(f"[{ticker}] what the market is implying:")
        for line in market_implied["lines"]:
            print(f"[{ticker}]   {line}")
    print(f"[{ticker}] wrote {outdir}/memo.pdf, memo.md, payload.json, football_field.png")
    return 0


def _market_implied(
    model: Any, inputs: Any, price: float, gap_flags: list[str]
) -> Optional[dict[str, Any]]:
    """Solve what the price implies, but only when the extreme-gap warning fired."""
    if not gap_flags:
        return None
    return implied_to_payload(implied_expectations(model, inputs, price))


def _resolve_from_override(
    ticker: str, data: dict[str, Any], model_cls: type, sector: str
) -> int | tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    model_name = model_cls.name
    declared = str(data["valuation"].get("model", model_name))
    if declared != model_name:
        print(
            f"Model mismatch for {ticker}: registry picked {model_name!r} but the override's "
            f"valuation block is {declared!r}. Align --model/sector with the override."
        )
        return 1

    print(f"[{ticker}] model={model_name} sector={sector or '—'} (override valuation block)")
    if model_name == "bank":
        return _value_bank(ticker, data)
    return _value_fcff(ticker, data)


def _load_profile(sector: str) -> dict[str, Any]:
    name = _profile_name_for_sector(sector)
    path = PROFILES_DIR / f"{name}.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _profile_name_for_sector(sector: str) -> str:
    normalized = sector.strip().lower()
    if any(key in normalized for key in ("bank", "nbfc", "financial")):
        return "financials"
    if "pharma" in normalized:
        return "pharma"
    return "industrial"


def _resolve_from_statements(
    ticker: str, model_cls: type, sector: str
) -> int | tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    try:
        pulled = merged(ticker)
    except DataError as error:
        print(f"Could not pull data for {ticker}: {error}")
        return 1

    statements = pulled.get("statements")
    if not statements:
        print(f"No statements could be pulled for {ticker}; cannot value from statements.")
        return 1

    market = {field: pulled.get(field) for field in MARKET_FIELDS}
    market["ticker"] = ticker
    profile = _load_profile(sector)
    model = model_cls()

    print(f"[{ticker}] model={model.name} sector={sector or '—'} (from-statements projection)")
    _print_pulled(pulled, market)

    if market.get("price") is None:
        print(f"Cannot build a memo for {ticker}: no market price was available to bridge to.")
        return 1

    usable, dropped = usable_periods(statements, model.name)
    for year, reason in dropped:
        print(f"[{ticker}] dropped FY{year}: {reason}")
    if len(usable) < 2:
        print(
            f"Cannot value {ticker}: not enough complete statement periods to value "
            f"{ticker} from pulled data; supply an override."
        )
        return 1

    try:
        inputs = model.from_statements(usable, market, profile)
        result = model.value(inputs)
    except (ValueError, KeyError) as error:
        print(f"Cannot value {ticker} from statements: {error}")
        return 1

    if model.name == "bank":
        return _payload_bank_from_statements(ticker, model, usable, market, profile, inputs, result)
    return _payload_fcff_from_statements(ticker, model, pulled, usable, market, profile, inputs, result)


def _payload_fcff_from_statements(
    ticker: str, model: Any, pulled: dict[str, Any], statements: list[dict[str, Any]],
    market: dict[str, Any], profile: dict[str, Any], inputs: DCFInputs, result: Any,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    tax_rate = macro.effective_tax_rate(statements, profile)
    history = _historical_fcff(statements, tax_rate)
    beta_value = macro.beta_for_ticker(market.get("ticker"), market, profile)
    ke = macro.cost_of_equity(beta_value, profile)

    print("  derived FCFF inputs:")
    print(f"    effective tax rate        {tax_rate:.2%}")
    print(f"    revenue CAGR              {_revenue_cagr(statements):.2%} (FCFF growth capped at 15%)")
    print(f"    historical FCFF           [{', '.join(f'{x:,.0f}' for x in history)}]")
    print(f"    FCFF forecast ({len(inputs.fcff_forecast)}y)       [{', '.join(f'{x:,.0f}' for x in inputs.fcff_forecast)}]")
    print(f"    cost of equity (CAPM)     {ke:.2%} (beta {beta_value:.2f} vs ^NSEI)")
    print(f"    WACC                      {inputs.wacc:.2%}")
    print(f"    terminal growth           {inputs.terminal_growth:.2%}")
    print(f"    net debt                  {inputs.net_debt:,.0f}")
    print(f"    diluted shares            {inputs.shares_outstanding:,.0f}")
    print("  valuation:")
    print(f"    enterprise value          {result.enterprise_value:,.0f}")
    print(f"    equity value              {result.equity_value:,.0f}")
    print(f"    terminal value % of EV    {result.tv_pct_of_ev:.0%}")

    price = float(market["price"])
    grid = sensitivity(inputs)
    flags = unstable_fcff_warnings(history)
    sotp_periods = [p for p in statements if p.get("revenue") is not None]
    flags += sum_of_the_parts_warnings(
        history,
        [float(p["capex"]) for p in sotp_periods],
        [float(p["depreciation"]) for p in sotp_periods],
        [float(p["revenue"]) for p in sotp_periods],
    )
    gap_flags = extreme_gap_warnings(result.value_per_share, price)
    flags += gap_flags
    market_implied = _market_implied(model, inputs, price, gap_flags)
    latest = statements[-1]
    if market.get("market_cap") is not None and latest.get("total_equity") is not None:
        flags += price_scale_flags(
            market_cap=float(market["market_cap"]),
            pretax_income=float(latest["pretax_income"]),
            tax_expense=float(latest["tax_expense"]),
            total_equity=float(latest["total_equity"]),
        )
    ratios = _ratios_or_warning(pulled, flags)

    assumptions = {
        "forecast_years": len(inputs.fcff_forecast),
        "wacc_pct": round(inputs.wacc * 100, 2),
        "terminal_growth_pct": round(inputs.terminal_growth * 100, 2),
        "cost_of_equity_pct": round(ke * 100, 2),
        "effective_tax_rate_pct": round(tax_rate * 100, 2),
        "net_debt": inputs.net_debt,
        "shares_outstanding": inputs.shares_outstanding,
    }
    payload = build_payload(
        company=ticker,
        ticker=ticker,
        currency="",
        current_price=price,
        method="FCFF DCF",
        valuation=result,
        assumptions=assumptions,
        ratios=ratios,
        comps=None,
        sensitivity=grid,
        extra_warnings=flags,
        market_implied=market_implied,
    )
    methods = {"DCF (FCFF)": _range_from_grid(grid)}
    return payload, methods


def _payload_bank_from_statements(
    ticker: str, model: Any, statements: list[dict[str, Any]], market: dict[str, Any],
    profile: dict[str, Any], inputs: BankInputs, result: Any,
) -> tuple[dict[str, Any], dict[str, tuple[float, float]]]:
    beta_value = macro.beta_for_ticker(market.get("ticker"), market, profile)

    print("  derived bank inputs:")
    print(f"    book equity (opening)     {inputs.book_equity_0:,.0f}")
    print(f"    ROE forecast ({len(inputs.roe_forecast)}y)        [{', '.join(f'{x:.1%}' for x in inputs.roe_forecast)}]")
    print(f"    BV growth forecast ({len(inputs.bv_growth_forecast)}y)  [{', '.join(f'{x:.1%}' for x in inputs.bv_growth_forecast)}]")
    print(f"    cost of equity (CAPM)     {inputs.cost_of_equity:.2%} (beta {beta_value:.2f} vs ^NSEI)")
    print(f"    terminal ROE              {inputs.terminal_roe:.2%}")
    print(f"    terminal growth           {inputs.terminal_growth:.2%}")
    print(f"    diluted shares            {inputs.shares_outstanding:,.0f}")
    print("  valuation:")
    print(f"    equity value              {result.equity_value:,.0f}")
    print(f"    implied P/B               {result.implied_pb:.2f}x")

    price = float(market["price"])
    grid = _bank_sensitivity(inputs)
    equity_series = [float(p["total_equity"]) for p in statements if p.get("total_equity") is not None]
    flags = book_equity_distortion_warnings(equity_series)
    gap_flags = extreme_gap_warnings(result.value_per_share, price)
    flags += gap_flags
    market_implied = _market_implied(model, inputs, price, gap_flags)
    ratios = _bank_ratios_or_warning(statements, inputs.cost_of_equity, flags)

    assumptions = {
        "cost_of_equity_pct": round(inputs.cost_of_equity * 100, 2),
        "terminal_roe_pct": round(inputs.terminal_roe * 100, 2),
        "terminal_growth_pct": round(inputs.terminal_growth * 100, 2),
        "book_equity_0": inputs.book_equity_0,
        "shares_outstanding": inputs.shares_outstanding,
    }
    if inputs.tier1_ratio is not None:
        assumptions["tier1_ratio_pct"] = round(inputs.tier1_ratio * 100, 1)

    payload = build_payload(
        company=ticker,
        ticker=ticker,
        currency="",
        current_price=price,
        method="Bank excess-return",
        valuation=result,
        assumptions=assumptions,
        ratios=ratios,
        comps=None,
        sensitivity=grid,
        extra_warnings=flags,
        market_implied=market_implied,
    )
    methods = {"Excess return": _range_from_grid(grid)}
    return payload, methods


def _ratios_or_warning(data: dict[str, Any], warnings: list[str]) -> Any:
    try:
        return compute_ratios(data)
    except RatioError as error:
        warnings.append(f"Operating ratios could not be computed from the pulled data: {error}.")
        return {}


def _bank_ratios_or_warning(
    statements: list[dict[str, Any]], cost_of_equity: float, warnings: list[str]
) -> dict[str, float]:
    try:
        return _bank_ratios(statements, cost_of_equity)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        warnings.append(f"Bank ratios could not be computed from the pulled data: {error}.")
        return {}


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

    periods = data["statements"]
    history = _historical_fcff(periods, _latest_tax_rate(periods))
    flags += unstable_fcff_warnings(history)
    flags += sum_of_the_parts_warnings(
        history,
        [float(p["capex"]) for p in periods],
        [float(p["depreciation"]) for p in periods],
        [float(p["revenue"]) for p in periods],
    )
    gap_flags = extreme_gap_warnings(result.value_per_share, float(data["price"]))
    flags += gap_flags
    market_implied = _market_implied(StandardFCFF(), inputs, float(data["price"]), gap_flags)

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
        market_implied=market_implied,
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
    ratios = _bank_ratios(data["statements"], float(vblock["cost_of_equity"]))
    comps = _bank_comps(data, shares)

    equity_series = [float(p["total_equity"]) for p in data["statements"]]
    flags += book_equity_distortion_warnings(equity_series)
    gap_flags = extreme_gap_warnings(result.value_per_share, float(data["price"]))
    flags += gap_flags
    market_implied = _market_implied(BankExcessReturn(), inputs, float(data["price"]), gap_flags)

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
        market_implied=market_implied,
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
    market_cap = float(data["market_cap"])
    flags = sanity_checks(statements, market_cap=market_cap)
    latest = periods[-1]
    flags += price_scale_flags(
        market_cap=market_cap,
        pretax_income=float(latest["pretax_income"]),
        tax_expense=float(latest["tax_expense"]),
        total_equity=float(latest["total_equity"]),
    )
    return flags


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


def _bank_ratios(periods: list[dict[str, Any]], cost_of_equity: float) -> dict[str, float]:
    equity = [float(p["total_equity"]) for p in periods]
    bv_cagr = (equity[-1] / equity[0]) ** (1 / (len(equity) - 1)) - 1
    pat_latest = float(periods[-1]["pretax_income"]) - float(periods[-1]["tax_expense"])
    roe_latest = pat_latest / equity[-1]
    return {
        "roe_latest_pct": round(roe_latest * 100, 1),
        "book_value_cagr_pct": round(bv_cagr * 100, 1),
        "cost_of_equity_pct": round(cost_of_equity * 100, 1),
        "roe_minus_ke_spread_pp": round((roe_latest - cost_of_equity) * 100, 1),
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


def _fmt(value: Any) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def _print_pulled(pulled: dict[str, Any], market: dict[str, Any]) -> None:
    sources = pulled.get("_sources", {})
    print("  pulled market data:")
    for field in MARKET_FIELDS:
        print(f"    {field:<12} {_fmt(market[field]):>20}   [{sources.get(field, '—')}]")
    print(f"  pulled statements [{sources.get('statements', '—')}]:")
    for period in pulled["statements"]:
        print(f"    --- FY{period.get('year', '?')} ---")
        for field in STATEMENT_FIELDS:
            print(f"      {field:<26} {_fmt(period.get(field))}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to :func:`run`."""
    args = build_parser().parse_args(argv)
    return run(args.ticker, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
