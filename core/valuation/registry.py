"""Sector -> valuation model dispatch, with a manual override.

``pick_model`` maps a company's sector to the model class that should value it.
Banks and NBFCs use the excess-return model; sectors the tool refuses to value
(REITs, real estate, oil & gas, metals & mining, insurance) raise ``NotSupported``
with the reason; everything else uses the FCFF DCF. A manual override always wins.
"""

from __future__ import annotations

from core.valuation.base import ValuationModel
from core.valuation.fcff import StandardFCFF
from core.valuation.financials import BankExcessReturn


class NotSupported(Exception):
    """Raised when the tool deliberately refuses to value a sector."""


_OVERRIDES: dict[str, type[ValuationModel]] = {
    "fcff": StandardFCFF,
    "standardfcff": StandardFCFF,
    "bank": BankExcessReturn,
    "financials": BankExcessReturn,
    "bankexcessreturn": BankExcessReturn,
}

_REFUSALS: list[tuple[tuple[str, ...], str]] = [
    (("reit",), "REITs are valued on NAV/FFO, not a DCF; refusing."),
    (("real estate",), "Real estate is valued on NAV, not a DCF; refusing."),
    (("oil", "gas"), "Oil & gas is a commodity/cyclical; point-in-time DCF is misleading; refusing."),
    (("metal", "mining"), "Metals & mining is a commodity/cyclical; point-in-time DCF is misleading; refusing."),
    (("insurance", "insurer"), "Insurers need an embedded-value/float model, not this engine; refusing."),
]

_BANK_KEYS: tuple[str, ...] = ("bank", "nbfc", "financial")


def pick_model(sector: str, override: str | None) -> type[ValuationModel]:
    """Return the valuation model class for a sector; an override always wins."""
    if override is not None:
        key = override.strip().lower()
        if key not in _OVERRIDES:
            raise ValueError(
                f"Unknown valuation override {override!r}; "
                f"expected one of {sorted(_OVERRIDES)}."
            )
        return _OVERRIDES[key]

    normalized = sector.strip().lower()

    for keywords, reason in _REFUSALS:
        if any(keyword in normalized for keyword in keywords):
            raise NotSupported(f"{sector}: {reason}")

    if any(keyword in normalized for keyword in _BANK_KEYS):
        return BankExcessReturn

    return StandardFCFF
