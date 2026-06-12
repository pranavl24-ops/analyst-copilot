"""Assemble computed outputs into a payload and draft the IC memo via Claude.

``build_payload`` gathers every figure Python has already computed — valuation,
comps, ratios, the EV->equity bridge — into one JSON-serialisable dict.
``write_memo`` hands that dict to the Anthropic API, which writes prose only:
the model is forbidden from inventing, estimating, or recomputing any number.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional, Union

import anthropic
from dotenv import load_dotenv

from core.comps import CompsResult
from core.ratios import Ratios
from core.valuation.base import ValuationResult

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500


def build_payload(
    *,
    company: str,
    ticker: str,
    currency: str,
    current_price: float,
    method: str,
    valuation: ValuationResult,
    assumptions: dict[str, Any],
    ratios: Union[Ratios, Mapping[str, Any]],
    comps: Optional[CompsResult] = None,
    sensitivity: Optional[dict[tuple[float, float], Optional[float]]] = None,
    extra_warnings: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Collect computed outputs into one dict for the memo prompt.

    ``valuation`` is any object satisfying :class:`ValuationResult`; method-specific
    detail (the EV->equity bridge for a DCF, the book-value/PB bridge for a bank) is
    pulled in when present. ``upside_pct`` is the only figure derived here, from the
    valuation's value-per-share against ``current_price``.
    """
    value_per_share = valuation.value_per_share
    upside_pct = value_per_share / current_price - 1.0

    warnings = list(valuation.warnings)
    if comps is not None and comps.missing:
        warnings.append(
            f"Comps: {len(comps.missing)} peer(s) dropped for missing figures: "
            f"{', '.join(comps.missing)}."
        )
    if extra_warnings:
        warnings.extend(extra_warnings)

    return {
        "company": company,
        "ticker": ticker,
        "currency": currency,
        "current_price": round(current_price, 2),
        "valuation": _valuation_block(method, valuation, assumptions, sensitivity),
        "value_per_share": round(value_per_share, 2),
        "upside_pct": round(upside_pct * 100, 1),
        "comps": _comps_block(comps),
        "ratios": _ratios_block(ratios),
        "warnings": warnings,
    }


def _valuation_block(
    method: str,
    valuation: ValuationResult,
    assumptions: dict[str, Any],
    sensitivity: Optional[dict[tuple[float, float], Optional[float]]],
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "method": method,
        "value_per_share": round(valuation.value_per_share, 2),
        "assumptions": assumptions,
        "warnings": list(valuation.warnings),
    }

    if hasattr(valuation, "enterprise_value"):
        block["bridge"] = {
            "pv_explicit": round(valuation.pv_explicit, 1),
            "terminal_value": round(valuation.terminal_value, 1),
            "pv_terminal": round(valuation.pv_terminal, 1),
            "enterprise_value": round(valuation.enterprise_value, 1),
            "equity_value": round(valuation.equity_value, 1),
            "terminal_value_pct_of_ev": round(valuation.tv_pct_of_ev * 100, 1),
        }
        if getattr(valuation, "implied_exit_multiple", None) is not None:
            block["bridge"]["implied_exit_multiple"] = round(valuation.implied_exit_multiple, 2)
    if hasattr(valuation, "implied_pb"):
        block["bridge"] = {
            "book_equity": round(valuation.book_equity_0, 1),
            "pv_excess_explicit": round(valuation.pv_excess_explicit, 1),
            "pv_excess_terminal": round(valuation.pv_excess_terminal, 1),
            "equity_value": round(valuation.equity_value, 1),
            "implied_pb": round(valuation.implied_pb, 2),
        }

    if sensitivity is not None:
        block["sensitivity"] = _sensitivity_read(valuation.value_per_share, sensitivity)

    return block


def _sensitivity_read(
    base_value: float,
    grid: dict[tuple[float, float], Optional[float]],
) -> dict[str, Any]:
    values = [v for v in grid.values() if v is not None]
    if not values:
        return {"text": "Sensitivity grid produced no valid scenarios."}
    low, high = min(values), max(values)
    return {
        "base_value_per_share": round(base_value, 2),
        "low_value_per_share": round(low, 2),
        "high_value_per_share": round(high, 2),
        "scenarios": len(grid),
        "text": (
            f"Flexing WACC/cost-of-equity and terminal growth across the grid, "
            f"value/share spans {low:,.0f} to {high:,.0f}."
        ),
    }


def _comps_block(comps: Optional[CompsResult]) -> Optional[dict[str, Any]]:
    if comps is None:
        return None
    return {
        "peers_used": [p.ticker for p in comps.peers],
        "median_ev_ebitda": _round(comps.median_ev_ebitda, 1),
        "median_pe": _round(comps.median_pe, 1),
        "implied_from_ev_ebitda": _round(comps.implied_from_ev_ebitda, 2),
        "implied_from_pe": _round(comps.implied_from_pe, 2),
        "value_per_share_low": round(comps.value_per_share_low, 2),
        "value_per_share_high": round(comps.value_per_share_high, 2),
    }


def _ratios_block(ratios: Union[Ratios, Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(ratios, Ratios):
        return {
            "revenue_cagr_pct": round(ratios.revenue_cagr * 100, 1),
            "ebit_margin_latest_pct": round(ratios.ebit_margin_latest * 100, 1),
            "ebit_margin_trend_pp_per_yr": round(ratios.ebit_margin_trend * 100, 2),
            "roce_pct": round(ratios.roce * 100, 1),
            "net_debt_to_ebitda": round(ratios.net_debt_to_ebitda, 2),
        }
    return dict(ratios)


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return round(value, digits) if value is not None else None


def system_prompt() -> str:
    """Frame the model as a sell-side analyst writing a one-page IC memo."""
    return (
        "You are a sell-side equity research analyst writing a one-page investment "
        "committee (IC) memo of roughly 500 words. You will be given a JSON payload of "
        "figures that have ALREADY been computed by a separate valuation engine.\n\n"
        "HARD RULES:\n"
        "- Write prose only. Never invent, estimate, extrapolate, or recompute any "
        "number. Use only the figures present in the payload, quoting them verbatim "
        "(you may round for readability but must not change a value's meaning).\n"
        "- If a figure is not in the payload, do not state it. Do not fill gaps.\n"
        "- Address every item in the payload's `warnings` list honestly in the Risks "
        "section — do not bury or soften them.\n"
        "- Adapt your terminology to `valuation.method`. For an FCFF DCF, speak of "
        "discounted free cash flow, WACC, and terminal value. For a bank/NBFC "
        "excess-return model, speak of excess returns over the cost of equity, the "
        "cost of equity (CAPM), and implied price-to-book — NEVER say 'DCF' or 'WACC' "
        "for a bank.\n\n"
        "STRUCTURE (use these section headers, in order):\n"
        "1. Recommendation & Target — a Buy/Hold/Sell call grounded in the upside vs. "
        "the current price, with the target (value per share).\n"
        "2. Business — one short paragraph on what the company is.\n"
        "3. Valuation — the method, the value per share, and the value bridge.\n"
        "4. Assumptions — the key inputs driving the number.\n"
        "5. Sensitivity — how the value moves across the scenario range.\n"
        "6. Risks — every warning, plus the obvious investment risks.\n"
        "7. A single-sentence closing line.\n\n"
        "Keep it tight, neutral, and decision-useful. Around 500 words total."
    )


def _user_prompt(payload: dict[str, Any]) -> str:
    return (
        "Write the IC memo from the following computed figures. Use only these "
        "numbers; do not introduce any others.\n\n"
        "```json\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n"
        "```"
    )


def write_memo(payload: dict[str, Any]) -> str:
    """Draft the IC memo from a payload via the Anthropic API and return the text."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt(),
        messages=[{"role": "user", "content": _user_prompt(payload)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_memo(payload: dict[str, Any]) -> str:
    """Produce the memo per ``MEMO_MODE`` (default ``offline``, never spends money).

    ``offline`` renders a deterministic template memo from the computed figures;
    ``live`` calls the Anthropic API via :func:`write_memo`. Live mode must be opted
    into explicitly so a default run or a test never incurs API cost.
    """
    mode = os.environ.get("MEMO_MODE", "offline").strip().lower()
    if mode == "offline":
        return render_offline_memo(payload)
    if mode == "live":
        return write_memo(payload)
    raise ValueError(f"Unknown MEMO_MODE {mode!r}; expected 'offline' or 'live'.")


def render_offline_memo(payload: dict[str, Any]) -> str:
    """Build the 7-section IC memo deterministically from the payload's numbers.

    No external call, no fabrication: every figure comes from ``payload``. Mirrors
    the structure the live system prompt asks for, adapts terminology to the
    valuation method, surfaces every warning under Risks, and prints a header
    marking the output as template-generated so it is never mistaken for the LLM memo.
    """
    company = payload["company"]
    ticker = payload["ticker"]
    currency = payload["currency"]
    price = payload["current_price"]
    value_per_share = payload["value_per_share"]
    upside_pct = payload["upside_pct"]
    val = payload["valuation"]
    bridge = val.get("bridge", {})
    method = val["method"]
    is_bank = (
        "implied_pb" in bridge
        or "book_equity" in bridge
        or "bank" in method.lower()
        or "excess" in method.lower()
    )

    rec, rationale = _recommendation(upside_pct)

    sections = [
        _offline_header(),
        f"# {company} ({ticker}) — Investment Committee Memo",
        f"_Monetary totals in {currency}; price and value per share are per-share figures._",
        _section_recommendation(rec, rationale, value_per_share, price, upside_pct),
        _section_business(company, ticker, payload["ratios"]),
        _section_valuation(is_bank, method, val, bridge, value_per_share, currency),
        _section_assumptions(is_bank, val.get("assumptions", {}), payload.get("comps")),
        _section_sensitivity(is_bank, val.get("sensitivity")),
        _section_risks(payload.get("warnings", []), is_bank, bridge, payload.get("comps"), upside_pct),
        _section_close(rec, company, value_per_share, price, upside_pct),
    ]
    return "\n\n".join(sections) + "\n"


def _offline_header() -> str:
    return (
        "> **OFFLINE MEMO — template-generated from the computed figures, "
        "NOT written by the Claude API.**  \n"
        "> Deterministic render of the payload. Set `MEMO_MODE=live` to generate "
        "the LLM-authored version instead."
    )


def _recommendation(upside_pct: float) -> tuple[str, str]:
    if upside_pct >= 15.0:
        return "BUY", "the model value sits well above the current price"
    if upside_pct <= -15.0:
        return "SELL", "the model value sits well below the current price"
    return "HOLD", "the model value is close to the current price"


def _section_recommendation(
    rec: str, rationale: str, value_per_share: float, price: float, upside_pct: float
) -> str:
    return (
        "## 1. Recommendation & Target\n\n"
        f"**{rec}.** Target value per share of **{value_per_share:,.2f}** against a "
        f"current price of **{price:,.2f}**, an implied **{upside_pct:+.1f}%** "
        f"upside — {rationale}."
    )


def _section_business(company: str, ticker: str, ratios: dict[str, Any]) -> str:
    body = [
        "## 2. Business\n",
        f"{company} ({ticker}). The computed operating and balance-sheet metrics:",
    ]
    body.extend(f"- {_humanize(key)}: {_format_value(value)}" for key, value in ratios.items())
    return "\n".join(body)


def _section_valuation(
    is_bank: bool,
    method: str,
    val: dict[str, Any],
    bridge: dict[str, Any],
    value_per_share: float,
    currency: str,
) -> str:
    body = ["## 3. Valuation\n"]
    if is_bank:
        body.append(
            f"Method: **excess-return (residual income)**. The franchise is worth its "
            f"book equity plus the present value of returns earned above the **cost of "
            f"equity**, discounted at the cost of equity (CAPM) — never a blended cost of capital."
        )
        if bridge:
            body.append(
                f"Book equity of {bridge['book_equity']:,.1f} {currency}, plus the PV of "
                f"excess returns — {bridge['pv_excess_explicit']:,.1f} explicit and "
                f"{bridge['pv_excess_terminal']:,.1f} terminal — gives an equity value of "
                f"{bridge['equity_value']:,.1f} {currency}, or **{value_per_share:,.2f} per "
                f"share** at an implied price-to-book of {bridge['implied_pb']:.2f}x."
            )
    else:
        body.append(
            f"Method: **discounted free cash flow (FCFF)**. Unlevered free cash flows are "
            f"discounted at the **WACC** and a terminal value caps the explicit window."
        )
        if bridge:
            body.append(
                f"The PV of explicit free cash flow is {bridge['pv_explicit']:,.1f} {currency} "
                f"and the terminal value {bridge['terminal_value']:,.1f} {currency} "
                f"(PV {bridge['pv_terminal']:,.1f}), for an enterprise value of "
                f"{bridge['enterprise_value']:,.1f} {currency}. Net of the debt bridge, equity "
                f"value is {bridge['equity_value']:,.1f} {currency} — **{value_per_share:,.2f} "
                f"per share**. Terminal value is {bridge['terminal_value_pct_of_ev']:.1f}% of EV."
            )
    return "\n".join(body)


def _section_assumptions(
    is_bank: bool, assumptions: dict[str, Any], comps: Optional[dict[str, Any]]
) -> str:
    body = ["## 4. Assumptions\n"]
    if assumptions:
        body.extend(f"- {_humanize(key)}: {_format_value(value)}" for key, value in assumptions.items())
    else:
        discount = "cost of equity" if is_bank else "WACC"
        body.append(f"- No explicit assumption set was provided (key drivers are the {discount} and terminal growth).")
    if comps:
        body.append(
            f"- Peer comps cross-check (peers: {', '.join(comps['peers_used'])}): implied "
            f"value/share {comps['value_per_share_low']:,.2f}–{comps['value_per_share_high']:,.2f}."
        )
    return "\n".join(body)


def _section_sensitivity(is_bank: bool, sensitivity: Optional[dict[str, Any]]) -> str:
    driver = "cost of equity" if is_bank else "WACC"
    if not sensitivity or "low_value_per_share" not in sensitivity:
        return (
            "## 5. Sensitivity\n\n"
            f"No sensitivity grid was provided; the value is most sensitive to the "
            f"{driver} and the terminal growth rate."
        )
    return (
        "## 5. Sensitivity\n\n"
        f"Across {sensitivity['scenarios']} scenarios flexing the {driver} and terminal "
        f"growth, value per share spans **{sensitivity['low_value_per_share']:,.2f}** to "
        f"**{sensitivity['high_value_per_share']:,.2f}**, around a base of "
        f"{sensitivity['base_value_per_share']:,.2f}."
    )


def _section_risks(
    warnings: list[str],
    is_bank: bool,
    bridge: dict[str, Any],
    comps: Optional[dict[str, Any]],
    upside_pct: float,
) -> str:
    body = ["## 6. Risks\n"]
    if warnings:
        body.extend(f"- {warning}" for warning in warnings)
    else:
        body.append("- No data-quality or model warnings were flagged.")

    if not is_bank and "terminal_value_pct_of_ev" in bridge:
        body.append(
            f"- Terminal value is {bridge['terminal_value_pct_of_ev']:.1f}% of enterprise "
            f"value, so the answer leans on long-run cash-flow assumptions."
        )
    if is_bank and "implied_pb" in bridge:
        body.append(
            f"- The case rests on a sustained ROE–cost-of-equity spread; the implied "
            f"{bridge['implied_pb']:.2f}x price-to-book compresses if that spread narrows."
        )
    if comps:
        body.append(
            f"- The comps range ({comps['value_per_share_low']:,.2f}–"
            f"{comps['value_per_share_high']:,.2f}) is a check on, not a substitute for, the "
            f"intrinsic value; divergence signals model or multiple risk."
        )
    if upside_pct <= 0:
        body.append(
            "- The model value sits below the current price: the market is pricing in more "
            "than these assumptions support, which may persist."
        )
    return "\n".join(body)


def _section_close(
    rec: str, company: str, value_per_share: float, price: float, upside_pct: float
) -> str:
    return (
        "## 7. Bottom Line\n\n"
        f"**{rec} {company}** — model value {value_per_share:,.2f} versus a price of "
        f"{price:,.2f} ({upside_pct:+.1f}%)."
    )


def _humanize(key: str) -> str:
    label = key.replace("_pct", " (%)").replace("_", " ")
    return label[:1].upper() + label[1:]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _demo_cipla() -> None:
    """Run the full memo pipeline on Cipla from its override file and print it."""
    from core.comps import CompsResult, PeerMultiples
    from core.ratios import compute_ratios
    from core.valuation.fcff import DCFInputs, run_dcf, sensitivity
    from data.fetch import load_override

    ticker = "CIPLA.NS"
    data = load_override(ticker)
    if data is None:
        raise RuntimeError(f"No override file for {ticker!r}; expected data/overrides/{ticker}.yaml")

    ratios = compute_ratios(data)

    net_debt = data["total_debt"] - data["cash"]
    inputs = DCFInputs(
        fcff_forecast=[4270.0, 4650.0, 5050.0, 5480.0, 5920.0, 6330.0, 6710.0],
        wacc=0.107,
        terminal_growth=0.05,
        net_debt=net_debt,
        shares_outstanding=data["shares"],
    )
    result = run_dcf(inputs)
    grid = sensitivity(inputs)

    latest = data["statements"][-1]
    ebitda = latest["ebit"] + latest["depreciation"]
    net_income = latest["pretax_income"] - latest["tax_expense"]
    median_ev_ebitda, median_pe = 16.0, 28.0
    implied_ev_ebitda = (median_ev_ebitda * ebitda - net_debt) / data["shares"]
    implied_pe = (median_pe * net_income) / data["shares"]
    comps = CompsResult(
        peers=[
            PeerMultiples("SUNPHARMA.NS", 18.0, 32.0),
            PeerMultiples("DRREDDY.NS", 14.0, 24.0),
        ],
        missing=[],
        median_ev_ebitda=median_ev_ebitda,
        median_pe=median_pe,
        implied_from_ev_ebitda=implied_ev_ebitda,
        implied_from_pe=implied_pe,
        value_per_share_low=min(implied_ev_ebitda, implied_pe),
        value_per_share_high=max(implied_ev_ebitda, implied_pe),
    )

    payload = build_payload(
        company=data["company"],
        ticker=ticker,
        currency=data["currency"],
        current_price=data["price"],
        method="FCFF DCF",
        valuation=result,
        assumptions={
            "forecast_years": len(inputs.fcff_forecast),
            "wacc_pct": round(inputs.wacc * 100, 2),
            "terminal_growth_pct": round(inputs.terminal_growth * 100, 2),
            "net_debt": net_debt,
            "shares_outstanding": inputs.shares_outstanding,
            "comps_median_ev_ebitda": median_ev_ebitda,
            "comps_median_pe": median_pe,
        },
        ratios=ratios,
        comps=comps,
        sensitivity=grid,
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    print("\n" + "=" * 70 + "\n")
    print(f"MEMO_MODE={os.environ.get('MEMO_MODE', 'offline')}\n")
    print(generate_memo(payload))


if __name__ == "__main__":
    _demo_cipla()
