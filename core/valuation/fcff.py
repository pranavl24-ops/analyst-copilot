"""Free-cash-flow-to-firm (FCFF) discounted cash flow valuation.

Projects unlevered free cash flows, discounts them at the WACC, and derives an
intrinsic equity value per share. Suited to non-financial companies. ``run_dcf``
is the verified, standalone engine; ``StandardFCFF`` wraps it behind the model
protocol.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from core.valuation import assumptions as macro
from core.valuation.base import ValuationModel

FCFF_GROWTH_CAP = 0.15


@dataclass
class FCFFDrivers:
    """The component-forecast recipe behind a DCFInputs.fcff_forecast.

    Carried alongside the forecast so the reverse-expectations solver can rebuild
    it for an alternative target margin or near-term revenue growth without
    re-reading the statements. Present only when the inputs were projected from
    statements; an override-supplied forecast leaves it unset.
    """

    latest_revenue: float
    revenue_cagr: float
    target_ebit_margin: float
    max_ebit_margin: float
    tax_rate: float
    dep_ratio: float
    capex_pct_latest: float
    nwc_intensity: float
    years: int


@dataclass
class DCFInputs:
    fcff_forecast: list[float]
    wacc: float
    terminal_growth: float
    net_debt: float
    shares_outstanding: float
    minority_interest: float = 0.0
    preferred_equity: float = 0.0
    investments_associates: float = 0.0
    terminal_ebitda: Optional[float] = None
    exit_multiple: Optional[float] = None
    forecast_drivers: Optional[FCFFDrivers] = None


@dataclass
class DCFResult:
    pv_explicit: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    tv_pct_of_ev: float
    implied_exit_multiple: Optional[float]
    warnings: list[str] = field(default_factory=list)


def _validate(inp: DCFInputs) -> list[str]:
    w = []
    if inp.terminal_growth >= inp.wacc:
        raise ValueError(
            f"terminal_growth ({inp.terminal_growth:.2%}) >= WACC ({inp.wacc:.2%}); "
            "Gordon Growth is undefined. Lower g or raise WACC."
        )
    if inp.terminal_growth > 0.07:
        w.append(f"terminal_growth {inp.terminal_growth:.2%} exceeds a sane long-run ceiling (~6-7%).")
    if not 0.05 <= inp.wacc <= 0.20:
        w.append(f"WACC {inp.wacc:.2%} outside the normal 5-20% band; recheck CAPM inputs.")
    if inp.shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive.")
    if len(inp.fcff_forecast) < 3:
        w.append("Fewer than 3 explicit forecast years; standard is 5-10.")
    return w


def run_dcf(inp: DCFInputs) -> DCFResult:
    warnings = _validate(inp)
    n = len(inp.fcff_forecast)

    pv_explicit = sum(
        fcff / (1 + inp.wacc) ** t
        for t, fcff in enumerate(inp.fcff_forecast, start=1)
    )

    fcff_terminal_year = inp.fcff_forecast[-1]
    tv = fcff_terminal_year * (1 + inp.terminal_growth) / (inp.wacc - inp.terminal_growth)
    pv_terminal = tv / (1 + inp.wacc) ** n

    ev = pv_explicit + pv_terminal
    equity_value = (
        ev - inp.net_debt - inp.minority_interest
        - inp.preferred_equity + inp.investments_associates
    )
    value_per_share = equity_value / inp.shares_outstanding

    tv_pct = pv_terminal / ev if ev else 0.0
    if tv_pct > 0.75:
        warnings.append(
            f"Terminal value is {tv_pct:.0%} of EV (>75%). The answer leans heavily on the "
            "terminal assumption; extend the explicit window or sanity-check g."
        )

    implied_exit = None
    if inp.terminal_ebitda:
        implied_exit = tv / inp.terminal_ebitda
        if inp.exit_multiple is not None:
            gap = abs(implied_exit - inp.exit_multiple) / inp.exit_multiple
            if gap > 0.25:
                warnings.append(
                    f"Gordon-implied exit multiple {implied_exit:.1f}x vs assumed "
                    f"{inp.exit_multiple:.1f}x differ by {gap:.0%}; the TV methods disagree."
                )

    return DCFResult(
        pv_explicit, tv, pv_terminal, ev, equity_value,
        value_per_share, tv_pct, implied_exit, warnings,
    )


def sensitivity(inp: DCFInputs,
                wacc_steps=(-0.01, -0.005, 0, 0.005, 0.01),
                g_steps=(-0.01, -0.005, 0, 0.005, 0.01)) -> dict:
    grid = {}
    for dw in wacc_steps:
        for dg in g_steps:
            trial = DCFInputs(**{**inp.__dict__})
            trial.wacc = inp.wacc + dw
            trial.terminal_growth = inp.terminal_growth + dg
            try:
                grid[(round(trial.wacc, 4), round(trial.terminal_growth, 4))] = \
                    round(run_dcf(trial).value_per_share, 1)
            except ValueError:
                grid[(round(trial.wacc, 4), round(trial.terminal_growth, 4))] = None
    return grid


def _nwc(period: dict[str, Any]) -> Optional[float]:
    receivables = period.get("receivables")
    inventory = period.get("inventory")
    payables = period.get("payables")
    if receivables is None or inventory is None or payables is None:
        return None
    return float(receivables) + float(inventory) - float(payables)


def _delta_nwc(period: dict[str, Any], prev: Optional[dict[str, Any]]) -> Optional[float]:
    """Increase in net working capital (a use of cash), positive when NWC grows.

    The cash-flow ``change_in_working_capital`` line carries the opposite sign (it is
    the cash impact), so it is negated to match the balance-sheet derivation.
    """
    reported = period.get("change_in_working_capital")
    if reported is not None:
        return -float(reported)
    if prev is None:
        return None
    current, previous = _nwc(period), _nwc(prev)
    if current is None or previous is None:
        return None
    return current - previous


def _historical_fcff(statements: list[dict[str, Any]], tax_rate: float) -> list[float]:
    fcff: list[float] = []
    prev: Optional[dict[str, Any]] = None
    for period in statements:
        ebit = period.get("ebit")
        depreciation = period.get("depreciation")
        capex = period.get("capex")
        delta_nwc = _delta_nwc(period, prev)
        prev = period
        if ebit is None or depreciation is None or capex is None or delta_nwc is None:
            continue
        fcff.append(
            float(ebit) * (1 - tax_rate)
            + float(depreciation)
            - abs(float(capex))
            - delta_nwc
        )
    return fcff


def _positive_revenues(statements: list[dict[str, Any]]) -> list[float]:
    return [
        float(p["revenue"])
        for p in statements
        if p.get("revenue") is not None and float(p["revenue"]) > 0
    ]


def _revenue_cagr(statements: list[dict[str, Any]]) -> float:
    revenues = _positive_revenues(statements)
    if len(revenues) < 2:
        macro.warn("Fewer than two positive revenue years; forecasting FCFF flat.")
        return 0.0
    return (revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1


def _last_n_mean(values: list[float], n: int = 3) -> float:
    recent = values[-n:]
    return sum(recent) / len(recent)


def _ratio_to_revenue(statements: list[dict[str, Any]], key: str, absolute: bool = False) -> list[float]:
    ratios: list[float] = []
    for period in statements:
        revenue = period.get("revenue")
        value = period.get(key)
        if revenue is None or value is None or float(revenue) <= 0:
            continue
        numerator = abs(float(value)) if absolute else float(value)
        ratios.append(numerator / float(revenue))
    return ratios


def _nwc_intensity(statements: list[dict[str, Any]]) -> float:
    """Mean of [YoY change in net working capital] / [YoY change in revenue].

    Clamped to [0, 0.25]; defaults to 0.05 when it cannot be computed from at least
    one usable year-over-year pair.
    """
    ratios: list[float] = []
    prev: Optional[dict[str, Any]] = None
    for period in statements:
        if prev is not None:
            nwc, prev_nwc = _nwc(period), _nwc(prev)
            revenue, prev_revenue = period.get("revenue"), prev.get("revenue")
            if nwc is not None and prev_nwc is not None and revenue is not None and prev_revenue is not None:
                delta_revenue = float(revenue) - float(prev_revenue)
                if delta_revenue != 0:
                    ratios.append((nwc - prev_nwc) / delta_revenue)
        prev = period
    if not ratios:
        return 0.05
    return min(max(sum(ratios) / len(ratios), 0.0), 0.25)


def _revenue_path(latest_revenue: float, near_term_growth: float,
                  terminal_growth: float, years: int) -> list[float]:
    """Compound the latest revenue at a rate fading linearly from ``near_term_growth``
    in year 1 to ``terminal_growth`` in the final year."""
    revenues: list[float] = []
    prev = latest_revenue
    for t in range(1, years + 1):
        growth = near_term_growth + (terminal_growth - near_term_growth) * (t - 1) / (years - 1)
        prev = prev * (1 + growth)
        revenues.append(prev)
    return revenues


def _faded_revenue_forecast(latest_revenue: float, revenue_cagr: float,
                            terminal_growth: float, years: int) -> list[float]:
    """Revenue path whose year-1 growth is the CAGR capped at ``FCFF_GROWTH_CAP``."""
    return _revenue_path(latest_revenue, min(revenue_cagr, FCFF_GROWTH_CAP), terminal_growth, years)


def _capex_pct_schedule(capex_pct_latest: float, dep_ratio: float, years: int) -> list[float]:
    """Capex/revenue tapering linearly from the latest actual ratio toward the
    depreciation ratio (maintenance capex), never falling below it."""
    return [
        max(capex_pct_latest + (dep_ratio - capex_pct_latest) * (t - 1) / (years - 1), dep_ratio)
        for t in range(1, years + 1)
    ]


def _extract_drivers(statements: list[dict[str, Any]], tax_rate: float) -> FCFFDrivers:
    ebit_ratios = _ratio_to_revenue(statements, "ebit")
    return FCFFDrivers(
        latest_revenue=_positive_revenues(statements)[-1],
        revenue_cagr=_revenue_cagr(statements),
        target_ebit_margin=_last_n_mean(ebit_ratios),
        max_ebit_margin=max(ebit_ratios),
        tax_rate=tax_rate,
        dep_ratio=_last_n_mean(_ratio_to_revenue(statements, "depreciation")),
        capex_pct_latest=_ratio_to_revenue(statements, "capex", absolute=True)[-1],
        nwc_intensity=_nwc_intensity(statements),
        years=macro.FORECAST_YEARS,
    )


def base_near_term_growth(drivers: FCFFDrivers) -> float:
    """The capped year-1 revenue growth the base-case forecast actually uses."""
    return min(drivers.revenue_cagr, FCFF_GROWTH_CAP)


def forecast_from_drivers(drivers: FCFFDrivers, terminal_growth: float, *,
                          ebit_margin: Optional[float] = None,
                          near_term_growth: Optional[float] = None) -> list[float]:
    """Component forecast: project revenue, then EBIT, NOPAT, depreciation, capex,
    and the working-capital draw separately, so a depressed or volatile historical
    FCFF base never anchors the forecast and capex converges to maintenance levels.

    ``ebit_margin`` and ``near_term_growth`` default to the base-case driver values;
    overriding either lets the reverse-expectations solver re-run the same forecast
    under a single altered assumption.
    """
    margin = drivers.target_ebit_margin if ebit_margin is None else ebit_margin
    g1 = base_near_term_growth(drivers) if near_term_growth is None else near_term_growth

    revenues = _revenue_path(drivers.latest_revenue, g1, terminal_growth, drivers.years)
    capex_pcts = _capex_pct_schedule(drivers.capex_pct_latest, drivers.dep_ratio, drivers.years)

    forecast: list[float] = []
    prev_revenue = drivers.latest_revenue
    for t, revenue in enumerate(revenues):
        nopat = revenue * margin * (1 - drivers.tax_rate)
        depreciation = revenue * drivers.dep_ratio
        capex = revenue * capex_pcts[t]
        delta_nwc = (revenue - prev_revenue) * drivers.nwc_intensity
        forecast.append(nopat + depreciation - capex - delta_nwc)
        prev_revenue = revenue
    return forecast


def _forecast_fcff(statements: list[dict[str, Any]], tax_rate: float,
                   terminal_growth: float) -> list[float]:
    return forecast_from_drivers(_extract_drivers(statements, tax_rate), terminal_growth)


def _wacc(market_data: dict[str, Any], tax_rate: float,
          profile: dict[str, Any], beta_value: float) -> float:
    equity = float(market_data["market_cap"])
    debt = market_data.get("total_debt")
    debt = float(debt) if debt is not None else 0.0
    total = equity + debt
    after_tax_kd = macro.cost_of_debt(profile) * (1 - tax_rate)
    return (equity / total) * macro.cost_of_equity(beta_value, profile) + (debt / total) * after_tax_kd


def build_dcf_inputs(statements: list[dict[str, Any]],
                     market_data: dict[str, Any],
                     profile: dict[str, Any]) -> DCFInputs:
    if market_data.get("market_cap") is None:
        raise ValueError("market_cap is required to weight the WACC and bridge to equity; none was available.")
    if market_data.get("shares") is None:
        raise ValueError("shares_outstanding is required for a per-share value; none was available.")

    tax_rate = macro.effective_tax_rate(statements, profile)
    history = _historical_fcff(statements, tax_rate)
    if len(history) < 2:
        raise ValueError(
            "Need at least two years of computable FCFF; each needs EBIT, depreciation, "
            "capex, and a working-capital change (change_in_working_capital, or "
            "receivables/inventory/payables for a year-over-year delta)."
        )

    terminal_growth = macro.terminal_growth(profile)
    drivers = _extract_drivers(statements, tax_rate)
    forecast = forecast_from_drivers(drivers, terminal_growth)
    beta_value = macro.beta_for_ticker(market_data.get("ticker"), market_data, profile)

    debt = market_data.get("total_debt")
    cash = market_data.get("cash")
    if debt is None:
        macro.warn("total_debt unavailable; treating as 0 in net debt.")
    if cash is None:
        macro.warn("cash unavailable; treating as 0 in net debt.")
    net_debt = (float(debt) if debt is not None else 0.0) - (float(cash) if cash is not None else 0.0)

    minority = float(macro.profile_assumptions(profile).get("minority_interest", 0.0))

    return DCFInputs(
        fcff_forecast=forecast,
        wacc=_wacc(market_data, tax_rate, profile, beta_value),
        terminal_growth=terminal_growth,
        net_debt=net_debt,
        shares_outstanding=float(market_data["shares"]),
        minority_interest=minority,
        forecast_drivers=drivers,
    )


class StandardFCFF(ValuationModel[DCFInputs]):
    """FCFF DCF for industrials, consumer, pharma, and IT names."""

    name = "fcff"

    def value(self, inputs: DCFInputs) -> DCFResult:
        return run_dcf(inputs)

    def from_statements(self, statements: list[dict[str, Any]],
                        market_data: dict[str, Any], profile: dict[str, Any]) -> DCFInputs:
        return build_dcf_inputs(statements, market_data, profile)
