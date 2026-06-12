"""Render the investment memo to a chart and a one-page PDF.

``football_field`` draws the valuation ranges (model, comps, 52-week) as a
horizontal bar chart with the current price marked. ``render_pdf`` fills the
Jinja2 template with the chart and the memo body and prints it to PDF with
Playwright. The memo arrives as markdown and is converted to HTML here so the
template stays presentation-only.
"""

from __future__ import annotations

import base64
import datetime
import html
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
TEMPLATE_NAME = "memo.html.j2"

_BAR_COLOR = "#1f3a5f"
_PRICE_COLOR = "#b3322c"

_EXCHANGES = {".NS": "NSE", ".BO": "BSE"}


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


def render_pdf(memo_markdown: str, payload: Mapping[str, Any], chart_png: Path, out: Path) -> Path:
    """Fill the equity-research template with computed figures, the chart, and the
    memo prose, then print it to a one-page PDF with Playwright."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    context = _build_context(memo_markdown, payload)
    context["chart_data_uri"] = _png_data_uri(Path(chart_png))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
    )
    page_html = env.get_template(TEMPLATE_NAME).render(**context)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(page_html, wait_until="networkidle")
        page.pdf(
            path=str(out),
            format="A4",
            print_background=True,
            margin={"top": "16mm", "bottom": "16mm", "left": "16mm", "right": "16mm"},
        )
        browser.close()
    return out


def _build_context(memo_markdown: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    valuation = payload["valuation"]
    bridge = valuation.get("bridge", {})
    assumptions = valuation.get("assumptions", {})
    ratios = payload.get("ratios", {})
    method = valuation["method"]
    is_bank = "implied_pb" in bridge or "book_equity" in bridge or "bank" in method.lower() or "excess" in method.lower()

    upside = float(payload["upside_pct"])
    sections = _parse_sections(memo_markdown)
    risks = next((body for title, body in sections if "risk" in title.lower()), None)
    prose = [
        {"title": _clean_title(title), "body_html": _markdown_to_html(body)}
        for title, body in sections
        if "risk" not in title.lower()
    ]

    as_of = datetime.date.today().isoformat()
    currency = str(payload.get("currency") or "").strip()

    return {
        "company": payload["company"],
        "ticker": payload["ticker"],
        "exchange": _exchange(payload["ticker"]),
        "as_of_date": as_of,
        "recommendation": _recommendation(upside),
        "offline": "OFFLINE MEMO" in memo_markdown,
        "stats": _stat_row(payload, method, upside),
        "summary_rows": _summary_rows(is_bank, bridge, assumptions, float(payload["value_per_share"])),
        "metric_cards": _metric_cards(is_bank, ratios, bridge),
        "sections": prose,
        "risks_html": _markdown_to_html(risks) if risks else _warnings_to_html(payload.get("warnings", [])),
        "footer_note": _footer_note(currency, as_of, payload.get("warnings", [])),
    }


def _stat_row(payload: Mapping[str, Any], method: str, upside: float) -> list[dict[str, str]]:
    return [
        {"label": "Current Price", "value": f"{float(payload['current_price']):,.2f}"},
        {"label": "Intrinsic Value / sh", "value": f"{float(payload['value_per_share']):,.2f}"},
        {"label": "Upside / Downside", "value": f"{upside:+.1f}%", "tone": "pos" if upside >= 0 else "neg"},
        {"label": "Valuation Method", "value": method},
    ]


def _summary_rows(is_bank: bool, bridge: Mapping[str, Any], assumptions: Mapping[str, Any],
                  value_per_share: float) -> list[dict[str, str]]:
    rows = [{"label": "Intrinsic value / share", "value": f"{value_per_share:,.2f}"}]
    if is_bank:
        _append(rows, "Equity value", bridge.get("equity_value"), _compact_money)
        _append(rows, "Book equity", bridge.get("book_equity"), _compact_money)
        _append(rows, "Cost of equity", assumptions.get("cost_of_equity_pct"), _pct)
        _append(rows, "Terminal growth", assumptions.get("terminal_growth_pct"), _pct)
        _append(rows, "Implied P/B", bridge.get("implied_pb"), _multiple)
    else:
        _append(rows, "Enterprise value", bridge.get("enterprise_value"), _compact_money)
        _append(rows, "Equity value", bridge.get("equity_value"), _compact_money)
        _append(rows, "WACC", assumptions.get("wacc_pct"), _pct)
        _append(rows, "Terminal growth", assumptions.get("terminal_growth_pct"), _pct)
        _append(rows, "Terminal value % of EV", bridge.get("terminal_value_pct_of_ev"), lambda v: f"{v:.0f}%")
        _append(rows, "Implied exit multiple", bridge.get("implied_exit_multiple"), _multiple)
    return rows


def _metric_cards(is_bank: bool, ratios: Mapping[str, Any], bridge: Mapping[str, Any]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    if is_bank:
        _append(cards, "ROE", ratios.get("roe_latest_pct"), _pct)
        _append(cards, "Book value CAGR", ratios.get("book_value_cagr_pct"), _pct)
        _append(cards, "ROE − Ke spread", ratios.get("roe_minus_ke_spread_pp"), lambda v: f"{v:+.1f} pp")
        _append(cards, "Implied P/B", bridge.get("implied_pb"), _multiple)
    else:
        _append(cards, "Revenue CAGR", ratios.get("revenue_cagr_pct"), _pct)
        _append(cards, "EBIT margin", ratios.get("ebit_margin_latest_pct"), _pct)
        _append(cards, "ROCE", ratios.get("roce_pct"), _pct)
        _append(cards, "Net debt / EBITDA", ratios.get("net_debt_to_ebitda"), _multiple)
    return cards


def _append(rows: list[dict[str, str]], label: str, value: Optional[float], fmt) -> None:
    if value is None:
        return
    rows.append({"label": label, "value": fmt(float(value))})


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _multiple(value: float) -> str:
    return f"{value:.2f}x"


def _compact_money(value: float) -> str:
    magnitude = abs(value)
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= divisor:
            return f"{value / divisor:,.1f}{suffix}"
    return f"{value:,.0f}"


def _exchange(ticker: str) -> str:
    upper = str(ticker).upper()
    for suffix, name in _EXCHANGES.items():
        if upper.endswith(suffix):
            return name
    return upper.split(".")[-1] if "." in upper else "—"


def _recommendation(upside_pct: float) -> str:
    if upside_pct >= 15.0:
        return "BUY"
    if upside_pct <= -15.0:
        return "SELL"
    return "HOLD"


def _footer_note(currency: str, as_of: str, warnings: list[str]) -> str:
    basis = currency or "as reported"
    return (
        f"Data from live pull or trust-anchor override, as of {as_of}. "
        f"Monetary totals on a {basis} basis; per-share figures are per diluted share."
    )


def _png_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _parse_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title: Optional[str] = None
    body: list[str] = []
    for line in markdown.splitlines():
        if line.strip().startswith("## "):
            if title is not None:
                sections.append((title, "\n".join(body).strip()))
            title = line.strip()[3:].strip()
            body = []
        elif title is not None:
            body.append(line)
    if title is not None:
        sections.append((title, "\n".join(body).strip()))
    return sections


def _clean_title(title: str) -> str:
    return re.sub(r"^\d+\.\s*", "", title)


def _warnings_to_html(warnings: list[str]) -> str:
    if not warnings:
        return "<p>No data-quality or model warnings were flagged.</p>"
    items = "".join(f"<li>{_strip_inline(w)}</li>" for w in warnings)
    return f"<ul>{items}</ul>"


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
