"""Render the investment memo to HTML.

Takes generated memo content plus structured analysis and renders it through
the Jinja2 template in ``templates/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_memo(context: dict[str, Any], template_dir: Path | None = None) -> str:
    """Render the memo template with the given context and return HTML."""
    ...


def write_memo(html: str, output_path: Path) -> Path:
    """Write rendered memo HTML to disk and return the output path."""
    ...
