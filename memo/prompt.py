"""Prompt construction for the LLM-generated investment memo.

Assembles analysis results (ratios, comps, valuation) into a structured
prompt for the Anthropic API.
"""

from __future__ import annotations

from typing import Any

from core.models import Company


def build_prompt(company: Company, analysis: dict[str, Any]) -> str:
    """Build the memo-generation prompt from a company and its analysis."""
    ...


def system_prompt() -> str:
    """Return the system prompt that frames the analyst persona and format."""
    ...
