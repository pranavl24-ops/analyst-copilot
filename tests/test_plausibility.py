from core.valuation.plausibility import (
    book_equity_distortion_warnings,
    extreme_gap_warnings,
    unstable_fcff_warnings,
)


def test_extreme_gap_fires_on_large_downside():
    warnings = extreme_gap_warnings(value_per_share=343.0, current_price=1400.0)
    assert len(warnings) == 1
    assert "Extreme valuation gap" in warnings[0]


def test_extreme_gap_fires_on_large_upside():
    assert extreme_gap_warnings(value_per_share=200.0, current_price=100.0)


def test_extreme_gap_silent_within_band():
    assert extreme_gap_warnings(value_per_share=120.0, current_price=100.0) == []
    assert extreme_gap_warnings(value_per_share=70.0, current_price=100.0) == []


def test_extreme_gap_ignores_nonpositive_price():
    assert extreme_gap_warnings(value_per_share=100.0, current_price=0.0) == []


def test_unstable_fcff_fires_on_negative_year():
    warnings = unstable_fcff_warnings([5000.0, -1200.0, 4800.0, 5100.0])
    assert len(warnings) == 1
    assert "negative" in warnings[0]


def test_unstable_fcff_fires_on_high_dispersion():
    warnings = unstable_fcff_warnings([1000.0, 9000.0, 2000.0, 8000.0])
    assert len(warnings) == 1
    assert "coefficient of variation" in warnings[0]


def test_unstable_fcff_silent_on_stable_series():
    assert unstable_fcff_warnings([5000.0, 5200.0, 5400.0, 5600.0]) == []


def test_unstable_fcff_needs_two_years():
    assert unstable_fcff_warnings([-1000.0]) == []


def test_bank_equity_distortion_fires_on_doubling():
    warnings = book_equity_distortion_warnings([150000.0, 175000.0, 350000.0, 380000.0])
    assert len(warnings) == 1
    assert "normalized-ROE override" in warnings[0]


def test_bank_equity_distortion_silent_on_steady_growth():
    assert book_equity_distortion_warnings([150000.0, 175000.0, 200000.0, 225000.0]) == []


def test_bank_equity_distortion_skips_nonpositive_base():
    assert book_equity_distortion_warnings([0.0, 100000.0]) == []
