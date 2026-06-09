from memo.render import _markdown_to_html, football_field


def test_football_field_writes_png(tmp_path):
    out = tmp_path / "chart.png"
    result = football_field(
        {"DCF (FCFF)": (914.0, 1664.0), "Comps": (1475.0, 1511.0), "52-week": (1350.0, 1650.0)},
        current_price=1500.0,
        out=out,
    )
    assert result == out
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_football_field_rejects_empty(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        football_field({}, current_price=100.0, out=tmp_path / "x.png")


def test_markdown_to_html_handles_memo_constructs():
    md = (
        "> **OFFLINE MEMO**\n\n"
        "# Cipla Ltd (CIPLA.NS) — Investment Committee Memo\n\n"
        "## 1. Recommendation & Target\n\n"
        "**SELL.** Target value per share of **1,158.52**.\n\n"
        "## 2. Business\n\n"
        "- Revenue cagr (%): 9.50\n"
        "- Roce (%): 18.60\n"
    )
    html = _markdown_to_html(md)
    assert "<blockquote>" in html
    assert "<h1>Cipla Ltd (CIPLA.NS) — Investment Committee Memo</h1>" in html
    assert "<h2>1. Recommendation &amp; Target</h2>" in html
    assert "<strong>SELL.</strong>" in html
    assert "<ul><li>Revenue cagr (%): 9.50</li><li>Roce (%): 18.60</li></ul>" in html


def test_markdown_escapes_html():
    html = _markdown_to_html("A <script> & a value")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
