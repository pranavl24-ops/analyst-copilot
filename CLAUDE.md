# Analyst Co-pilot

A CLI tool: a ticker goes in, an investment memo comes out. It pulls financials,
validates them, runs a valuation appropriate to the company's industry, and uses
the Claude API to draft a one-page IC memo from the COMPUTED numbers.

## Stack
Python 3.11+, yfinance + FMP (financials), pydantic (validation), anthropic SDK
(memo), matplotlib (chart), jinja2 + Playwright (PDF). Runs locally.

## Conventions (follow without exception)
- Clean code, NO inline comments. Names and structure carry the meaning. Docstrings
  only where a function's contract is non-obvious.
- Full type hints. Use dataclasses or pydantic models, not loose dicts, for anything
  structured.
- Small, single-purpose modules. No premature abstraction, no clever metaprogramming.
- Fail loudly: raise on bad input, never silently coerce or paper over missing data.

## Architecture
Valuation is a STRATEGY PATTERN, not one model:
- core/valuation/base.py defines a ValuationModel ABC.
- core/valuation/fcff.py — StandardFCFF (industrials, consumer, pharma, IT).
- core/valuation/financials.py — BankExcessReturn (banks/NBFCs: FCFF is WRONG for
  these; discount excess returns at cost of equity, NOT WACC).
- core/valuation/highgrowth.py — margin-convergence model for pre-profit names (v3).
- core/valuation/registry.py — maps sector -> model, with a manual override.
- profiles/*.yaml — per-sector defaults and sanity thresholds.
The tool REFUSES to run on REITs, commodities/cyclicals, and insurers, printing why.

## HARD RULES — valuation correctness
- Terminal value grows the FINAL explicit-year cash flow (year N), never the PV of
  year 1. Discount TV back from year N.
- Enforce g < discount rate; raise if violated. Flag g > 7%.
- FCFF WACC uses MARKET-VALUE weights, never hardcoded splits.
- One consistent equity risk premium across the whole model.
- EV->equity bridge subtracts net debt, minority interest, preferred; adds non-op
  investments; net cash adds back. Per-share uses DILUTED shares.
- Banks: discount at cost of equity (CAPM); value = book equity + PV(excess returns).

## HARD RULES — the LLM never does maths
Python computes every number. The Claude API receives finished figures and writes
prose only. The memo prompt forbids the model from inventing or recomputing anything.

## Data integrity
- Source cascade: override > FMP > yfinance fallback > flagged-missing. yfinance is
  the fallback for Indian (.NS/.BO) statements FMP's plan doesn't cover; it is
  normalized to the FMP schema and treated as lower-confidence.
- data/overrides/<TICKER>.yaml is the trust anchor for any name being pitched, and
  wins field-by-field.
- Fully-empty periods are dropped; genuinely-missing individual fields surface as
  MISSING warnings, never zero-filled.
- API keys come from environment variables (.env via python-dotenv). Never hardcode.

## Testing
pytest. Valuation correctness tests must pass after any change to valuation code.
