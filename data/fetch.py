"""Fetch raw market data and financial statements for a ticker.

Market data comes from yfinance; statements come from the FMP REST API. A local
``data/overrides/<TICKER>.yaml`` file, when present, takes precedence over the
API on a per-field basis. Missing values are never coerced to zero: they are
returned as ``None`` and surfaced to the caller.

Indian tickers carry an exchange suffix: ``.NS`` for NSE, ``.BO`` for BSE.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import requests
import yaml
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FMP_BASE = "https://financialmodelingprep.com/stable"
OVERRIDES_DIR = Path(__file__).resolve().parent / "overrides"

MARKET_FIELDS = ("price", "market_cap", "shares", "beta", "total_debt", "cash")
STATEMENT_FIELDS = (
    "revenue",
    "ebit",
    "depreciation",
    "capex",
    "receivables",
    "inventory",
    "payables",
    "change_in_working_capital",
    "tax_expense",
    "pretax_income",
)

YF_LABELS: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "ebit": ("EBIT", "Operating Income"),
    "depreciation": (
        "Depreciation And Amortization",
        "Reconciled Depreciation",
        "Depreciation And Amortization In Income Statement",
        "Depreciation",
    ),
    "capex": ("Capital Expenditure",),
    "receivables": (
        "Accounts Receivable",
        "Net Receivables",
        "Gross Accounts Receivable",
        "Other Receivables",
    ),
    "inventory": ("Inventory",),
    "payables": ("Accounts Payable", "Payables"),
    "change_in_working_capital": ("Change In Working Capital",),
    "tax_expense": ("Tax Provision",),
    "pretax_income": ("Pretax Income",),
}


class DataError(RuntimeError):
    """Raised when required external data cannot be retrieved."""


def market_data(ticker: str) -> dict[str, Any]:
    """Return price, market cap, shares, beta, total debt, and cash from yfinance.

    Individual unavailable fields are ``None``; a wholesale failure to reach the
    provider raises :class:`DataError`.
    """
    info = yf.Ticker(ticker).info
    if not info or (info.get("symbol") is None and info.get("shortName") is None):
        raise DataError(f"yfinance returned no data for {ticker!r}")
    return {
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "market_cap": info.get("marketCap"),
        "shares": info.get("sharesOutstanding"),
        "beta": info.get("beta"),
        "total_debt": info.get("totalDebt"),
        "cash": info.get("totalCash"),
    }


def financial_statements(ticker: str) -> dict[str, Any]:
    """Return the last 5 annual periods from FMP, ascending by fiscal year.

    Each period is a dict over :data:`STATEMENT_FIELDS`. Raises :class:`DataError`
    if FMP is unreachable, unauthorized, or returns no overlapping periods.
    """
    income = {row["fiscalYear"]: row for row in _fmp_get("income-statement", ticker)}
    balance = {row["fiscalYear"]: row for row in _fmp_get("balance-sheet-statement", ticker)}
    cashflow = {row["fiscalYear"]: row for row in _fmp_get("cash-flow-statement", ticker)}

    years = sorted(set(income) & set(balance) & set(cashflow))
    if not years:
        raise DataError(f"FMP returned no overlapping annual periods for {ticker!r}")

    periods: list[dict[str, Any]] = []
    for year in years[-5:]:
        inc, bal, cfs = income[year], balance[year], cashflow[year]
        periods.append(
            {
                "year": int(year),
                "revenue": inc.get("revenue"),
                "ebit": inc.get("ebit"),
                "depreciation": inc.get("depreciationAndAmortization"),
                "capex": cfs.get("capitalExpenditure"),
                "receivables": bal.get("netReceivables"),
                "inventory": bal.get("inventory"),
                "payables": bal.get("accountPayables"),
                "change_in_working_capital": cfs.get("changeInWorkingCapital"),
                "tax_expense": inc.get("incomeTaxExpense"),
                "pretax_income": inc.get("incomeBeforeTax"),
            }
        )

    return {
        "ticker": ticker,
        "currency": income[years[-1]].get("reportedCurrency"),
        "periods": periods,
    }


def yfinance_statements(ticker: str) -> dict[str, Any] | None:
    """Return statements from yfinance normalized to the FMP schema, or ``None``.

    Used only as a fallback when FMP has no data. yfinance labels vary, so each
    field is mapped through a list of candidate labels (see :data:`YF_LABELS`).
    Absent or NaN values stay ``None``; ``change_in_working_capital`` is left
    ``None`` when unavailable so the DCF can derive ΔNWC from balance deltas.
    """
    handle = yf.Ticker(ticker)
    frames = [handle.income_stmt, handle.balance_sheet, handle.cashflow]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return None

    years = sorted({column.year for frame in frames for column in frame.columns})
    periods: list[dict[str, Any]] = []
    for year in years[-5:]:
        labels: dict[str, Any] = {}
        for frame in frames:
            column = _column_for_year(frame, year)
            if column is None:
                continue
            for label in frame.index:
                labels.setdefault(label, frame.loc[label, column])

        period: dict[str, Any] = {"year": year}
        for field, candidates in YF_LABELS.items():
            period[field] = _first_label(labels, candidates)
        if any(period[field] is not None for field in STATEMENT_FIELDS):
            periods.append(period)

    return {
        "ticker": ticker,
        "currency": handle.info.get("financialCurrency"),
        "periods": periods,
    }


def load_override(ticker: str) -> dict[str, Any] | None:
    """Return parsed ``data/overrides/<TICKER>.yaml`` if it exists, else ``None``."""
    path = OVERRIDES_DIR / f"{ticker}.yaml"
    if not path.exists():
        return None
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return None
    if not isinstance(data, dict):
        raise DataError(f"Override file {path} must contain a mapping at its top level")
    return data


def merged(ticker: str) -> dict[str, Any]:
    """Combine sources by a fixed priority cascade, recording per-field provenance.

    Priority: ``override`` (field by field) > ``fmp`` > ``yfinance`` (statements
    only, when FMP has none) > ``missing``. Market fields originate from yfinance;
    statements from the cascade. Nothing missing is ever zero-filled — it is
    flagged ``"missing"`` under ``_sources``.
    """
    override = load_override(ticker) or {}
    market = market_data(ticker)

    result: dict[str, Any] = {"ticker": ticker}
    sources: dict[str, str] = {}
    for field in MARKET_FIELDS:
        if field in override:
            result[field] = override[field]
            sources[field] = "override"
        elif market.get(field) is not None:
            result[field] = market[field]
            sources[field] = "yfinance"
        else:
            result[field] = None
            sources[field] = "missing"

    result["statements"], sources["statements"] = _statements_cascade(ticker, override)
    result["_sources"] = sources
    return result


def _statements_cascade(
    ticker: str, override: dict[str, Any]
) -> tuple[list[dict[str, Any]] | None, str]:
    if "statements" in override:
        return override["statements"], "override"

    try:
        periods = financial_statements(ticker)["periods"]
        if periods:
            return periods, "fmp"
    except DataError:
        pass

    fallback = yfinance_statements(ticker)
    if fallback and fallback["periods"]:
        return fallback["periods"], "yfinance"

    return None, "missing"


def _fmp_get(endpoint: str, ticker: str) -> list[dict[str, Any]]:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise DataError("FMP_API_KEY is not set in the environment")
    response = requests.get(
        f"{FMP_BASE}/{endpoint}",
        params={"symbol": ticker, "period": "annual", "limit": 5, "apikey": key},
        timeout=30,
    )
    if response.status_code != 200:
        raise DataError(
            f"FMP {endpoint} for {ticker!r} failed: HTTP {response.status_code} — "
            f"{response.text[:200].strip()}"
        )
    data = response.json()
    if not isinstance(data, list) or not data:
        raise DataError(f"FMP {endpoint} for {ticker!r} returned no rows")
    return data


def _column_for_year(frame: Any, year: int) -> Any | None:
    for column in frame.columns:
        if column.year == year:
            return column
    return None


def _first_label(labels: dict[str, Any], candidates: tuple[str, ...]) -> Any | None:
    for candidate in candidates:
        if candidate in labels:
            value = _clean(labels[candidate])
            if value is not None:
                return value
    return None


def _clean(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        return value
    return float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "\033[33mMISSING\033[0m"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _print_market(ticker: str) -> None:
    print(f"\n# market_data({ticker})")
    try:
        data = market_data(ticker)
    except DataError as error:
        print(f"  \033[33mWARNING\033[0m: {error}")
        return
    for field in MARKET_FIELDS:
        value = data[field]
        if value is None:
            print(f"  \033[33mWARNING\033[0m: {field} unavailable")
        else:
            print(f"  {field:<12} {_fmt(value)}")


def _print_statements(ticker: str) -> None:
    print(f"\n# financial_statements({ticker})")
    try:
        data = financial_statements(ticker)
    except DataError as error:
        print(f"  \033[33mWARNING\033[0m: {error}")
        return
    print(f"  currency: {data['currency']}")
    for period in data["periods"]:
        print(f"  --- FY{period['year']} ---")
        for field in STATEMENT_FIELDS:
            value = period[field]
            if value is None:
                print(f"    \033[33mWARNING\033[0m: {field} missing")
            else:
                print(f"    {field:<26} {_fmt(value)}")


def _print_merged(ticker: str) -> None:
    print(f"\n# merged({ticker})")
    try:
        data = merged(ticker)
    except DataError as error:
        print(f"  \033[33mWARNING\033[0m: {error}")
        return

    statements = data["statements"]
    if statements:
        print(f"  statements ({data['_sources']['statements']}):")
        for period in statements:
            print(f"    --- FY{period['year']} ---")
            for field in STATEMENT_FIELDS:
                value = period.get(field)
                if value is None:
                    print(f"      \033[33mWARNING\033[0m: {field} missing")
                else:
                    print(f"      {field:<26} {_fmt(value)}")

    print("  provenance map (_sources):")
    for field, source in data["_sources"].items():
        marker = "\033[33m" if source == "missing" else ""
        reset = "\033[0m" if marker else ""
        print(f"    {field:<12} -> {marker}{source}{reset}")


def _main(tickers: list[str]) -> None:
    for ticker in tickers:
        print("=" * 60)
        print(ticker)
        print("=" * 60)
        override = load_override(ticker)
        print(f"# load_override({ticker}) -> {'present' if override else 'none'}")
        _print_market(ticker)
        _print_statements(ticker)
        _print_merged(ticker)
        print()


if __name__ == "__main__":
    args = sys.argv[1:] or ["RELIANCE.NS"]
    _main(args)
