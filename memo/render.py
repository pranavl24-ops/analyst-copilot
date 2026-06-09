"""Render the investment memo to a chart and a one-page PDF.

``football_field`` draws the valuation ranges (model, comps, 52-week) as a
horizontal bar chart with the current price marked. ``render_pdf`` fills the
Jinja2 template with the chart and the memo body and prints it to PDF with
Playwright. The memo arrives as markdown and is converted to HTML here so the
template stays presentation-only.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
TEMPLATE_NAME = "memo.html.j2"

_BAR_COLOR = "#4C72B0"
_PRICE_COLOR = "#C44E52"


def football_field(
    methods: Mapping[str, tuple[float, float]],
    current_price: float,
    out: Path,
) -> Path:
    """Draw one horizontal range bar per method and mark the current price.

    ``methods`` maps a label (e.g. ``"DCF (FCFF)"``, ``"Comps"``, ``"52-week"``)
    to its ``(low, high)`` value-per-share range. The current price is a dashed
    vertical line. Saves a PNG to ``out`` and returns the path.
    """
    if not methods:
        raise ValueError("football_field needs at least one method range")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = list(methods)
    positions = range(len(labels))

    lows = [min(methods[label]) for label in labels]
    highs = [max(methods[label]) for label in labels]
    span = max(max(highs) - min(lows), 1e-9)
    pad = 0.12 * span
    x_min, x_max = min(min(lows), current_price) - pad, max(max(highs), current_price) + pad
    gap = 0.015 * span

    fig, ax = plt.subplots(figsize=(8, 0.9 * len(labels) + 1.8))
    for y, label in zip(positions, labels):
        low, high = sorted(methods[label])
        width = max(high - low, 1e-9)
        ax.barh(y, width, left=low, height=0.5, color=_BAR_COLOR, alpha=0.85, zorder=2)
        ax.text(low - gap, y, f"{low:,.0f}", va="center", ha="right", fontsize=9, color="#333")
        ax.text(high + gap, y, f"{high:,.0f}", va="center", ha="left", fontsize=9, color="#333")

    ax.axvline(current_price, color=_PRICE_COLOR, linestyle="--", linewidth=1.6, zorder=3)
    ax.text(
        current_price,
        -0.7,
        f"price {current_price:,.0f}",
        color=_PRICE_COLOR,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="center",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(len(labels) - 0.5, -1.0)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Value per share")
    ax.set_title("Valuation football field")
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_pdf(memo_markdown: str, chart_png: Path, out: Path) -> Path:
    """Fill the template with the chart and memo body and print it to PDF."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    chart_uri = _png_data_uri(Path(chart_png))
    body_html = _markdown_to_html(memo_markdown)
    title = _first_heading(memo_markdown) or "Investment Committee Memo"

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
    )
    page_html = env.get_template(TEMPLATE_NAME).render(
        title=title,
        chart_data_uri=chart_uri,
        body_html=body_html,
    )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(page_html, wait_until="networkidle")
        page.pdf(
            path=str(out),
            format="A4",
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
        )
        browser.close()
    return out


def _png_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return _strip_inline(line[2:].strip())
    return None


def _markdown_to_html(markdown: str) -> str:
    """Convert the constrained memo markdown to HTML.

    Handles the constructs the memo uses: ``#``/``##`` headings, ``> `` blockquotes,
    ``- `` bullet lists, blank-line-separated paragraphs, and inline ``**bold**`` /
    ``_italic_``. Not a general markdown engine — just enough for the memo.
    """
    blocks: list[str] = []
    bullets: list[str] = []
    quote: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(f"<p>{' '.join(paragraph)}</p>")
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            items = "".join(f"<li>{item}</li>" for item in bullets)
            blocks.append(f"<ul>{items}</ul>")
            bullets.clear()

    def flush_quote() -> None:
        if quote:
            blocks.append(f"<blockquote>{'<br>'.join(quote)}</blockquote>")
            quote.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph(); flush_bullets(); flush_quote()
            continue
        if stripped.startswith("> "):
            flush_paragraph(); flush_bullets()
            quote.append(_strip_inline(stripped[2:].strip()))
            continue
        if stripped.startswith("## "):
            flush_paragraph(); flush_bullets(); flush_quote()
            blocks.append(f"<h2>{_strip_inline(stripped[3:].strip())}</h2>")
            continue
        if stripped.startswith("# "):
            flush_paragraph(); flush_bullets(); flush_quote()
            blocks.append(f"<h1>{_strip_inline(stripped[2:].strip())}</h1>")
            continue
        if stripped.startswith("- "):
            flush_paragraph(); flush_quote()
            bullets.append(_strip_inline(stripped[2:].strip()))
            continue

        flush_bullets(); flush_quote()
        paragraph.append(_strip_inline(stripped))

    flush_paragraph(); flush_bullets(); flush_quote()
    return "\n".join(blocks)


def _strip_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", escaped)
    return escaped
