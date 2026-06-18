"""Macro assumptions shared by the statements -> inputs adapters.

Both the FCFF and bank adapters project the same India macro assumptions (risk-free
rate, equity risk premium, beta, effective tax) onto raw pulled statements. Keeping
them here keeps one consistent equity risk premium across every model, which the
valuation-correctness rules require, and lets a sector profile override any default.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

NIFTY_INDEX = "^NSEI"
BETA_LOOKBACK = "2y"
BETA_INTERVAL = "1wk"

RF_INDIA = 0.07
ERP_INDIA = 0.055
DEFAULT_BETA = 1.0
COST_OF_DEBT_DEFAULT = 0.08
TERMINAL_GROWTH_DEFAULT = 0.055
TAX_RATE_FLOOR = 0.15
TAX_RATE_CEIL = 0.40
DEFAULT_TAX_RATE = 0.25
FORECAST_YEARS = 10


def warn(message: str) -> None:
    """Surface a non-fatal data gap without zero-filling it silently."""
    print(f"\033[33mWARNING\033[0m: {message}", file=sys.stderr)


def profile_assumptions(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    assumptions = profile.get("assumptions")
    return assumptions if isinstance(assumptions, dict) else {}


def is_indian_ticker(ticker: str | None) -> bool:
    return bool(ticker) and str(ticker).upper().endswith((".NS", ".BO"))


def _default_beta(profile: dict[str, Any] | None) -> float:
    return float(profile_assumptions(profile).get("default_beta", DEFAULT_BETA))


def regression_beta(stock_returns: list[float], index_returns: list[float]) -> float:
    """Ordinary-least-squares beta: cov(stock, index) / var(index)."""
    n = len(stock_returns)
    if n < 2 or n != len(index_returns):
        raise ValueError("need at least two aligned return observations to regress beta")
    mean_stock = sum(stock_returns) / n
    mean_index = sum(index_returns) / n
    covariance = sum(
        (s - mean_stock) * (i - mean_index)
        for s, i in zip(stock_returns, index_returns)
    ) / (n - 1)
    variance = sum((i - mean_index) ** 2 for i in index_returns) / (n - 1)
    if variance == 0:
        raise ValueError("index variance is zero; cannot regress beta")
    return covariance / variance


@lru_cache(maxsize=None)
def _nifty_regression_beta(ticker: str) -> float:
    """Beta of an Indian stock against the Nifty 50 from 2y weekly returns.

    The yfinance ``beta`` field regresses against a US index and is unusable for
    Indian names (CIPLA reads ~0.12), so we regress against ^NSEI directly.
    """
    import yfinance as yf

    frame = yf.download(
        [ticker, NIFTY_INDEX],
        period=BETA_LOOKBACK,
        interval=BETA_INTERVAL,
        auto_adjust=True,
        progress=False,
    )
    closes = frame["Close"].dropna()
    if ticker not in closes or NIFTY_INDEX not in closes:
        raise ValueError("no overlapping ^NSEI price history")
    returns = closes[[ticker, NIFTY_INDEX]].pct_change().dropna()
    return regression_beta(returns[ticker].tolist(), returns[NIFTY_INDEX].tolist())


def beta_for_ticker(ticker: str | None,
                    market_data: dict[str, Any],
                    profile: dict[str, Any] | None) -> float:
    """Resolve beta: explicit profile override > ^NSEI regression (India) > field."""
    if profile and profile.get("beta") is not None:
        return float(profile["beta"])
    if is_indian_ticker(ticker):
        try:
            return _nifty_regression_beta(str(ticker))
        except Exception as error:
            fallback = _default_beta(profile)
            warn(f"Could not regress {ticker} beta vs {NIFTY_INDEX} ({error}); using default beta {fallback:.2f}.")
            return fallback
    field_beta = market_data.get("beta")
    if field_beta is not None:
        return float(field_beta)
    fallback = _default_beta(profile)
    warn(f"No beta available for {ticker or 'ticker'}; using default beta {fallback:.2f}.")
    return fallback


def cost_of_equity(beta_value: float, profile: dict[str, Any] | None) -> float:
    """CAPM cost of equity: Rf + beta * ERP, with India defaults."""
    assumptions = profile_assumptions(profile)
    rf = float(assumptions.get("rf", RF_INDIA))
    erp = float(assumptions.get("erp", ERP_INDIA))
    return rf + beta_value * erp


def cost_of_debt(profile: dict[str, Any] | None) -> float:
    return float(profile_assumptions(profile).get("cost_of_debt", COST_OF_DEBT_DEFAULT))


def terminal_growth(profile: dict[str, Any] | None) -> float:
    return float(profile_assumptions(profile).get("terminal_growth", TERMINAL_GROWTH_DEFAULT))


def effective_tax_rate(statements: list[dict[str, Any]], profile: dict[str, Any] | None) -> float:
    """Average historical tax_expense / pretax_income, clamped to a sane band."""
    assumptions = profile_assumptions(profile)
    fallback = float(assumptions.get("tax_rate", DEFAULT_TAX_RATE))
    ratios: list[float] = []
    for period in statements:
        pretax = period.get("pretax_income")
        tax = period.get("tax_expense")
        if pretax is None or tax is None or float(pretax) <= 0:
            continue
        ratios.append(float(tax) / float(pretax))
    rate = sum(ratios) / len(ratios) if ratios else fallback
    return min(max(rate, TAX_RATE_FLOOR), TAX_RATE_CEIL)
