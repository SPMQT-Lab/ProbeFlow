"""Central typography: the system UI font family + a comfortable type scale.

All GUI font construction goes through :func:`ui_font` so size and family are
defined in one place.  The legacy code used ad-hoc point sizes (mostly 8–9 pt,
which read as cramped); :data:`_SIZE_MAP` lifts those onto a comfortable scale
while keeping control text small enough for the (bumped) fixed-height controls.

Tune the whole app's text here — adjust ``_SIZE_MAP`` or ``ui_family``.
"""

from __future__ import annotations

# NB: Qt is imported lazily inside the functions so that importing this module
# (and therefore styling.py) stays Qt-free — pure GUI helpers must import without
# pulling in PySide6 (see tests/test_layout_compatibility.py).

# Legacy point size → comfortable scale.
#   ≤8  → caption (10)      9–10 → body (11)      11 → section (12)
#   12  → title (14)        13–16 → larger titles
_SIZE_MAP = {
    6: 10, 7: 10, 8: 10,
    9: 11, 10: 11,
    11: 12,
    12: 14, 13: 15, 14: 16, 15: 17, 16: 18,
}

_UI_FAMILY: str | None = None


def ui_family() -> str:
    """Return the platform UI font family (San Francisco on macOS), cached."""
    global _UI_FAMILY
    if _UI_FAMILY is None:
        family = ""
        try:
            from PySide6.QtGui import QFontDatabase

            family = QFontDatabase.systemFont(QFontDatabase.GeneralFont).family()
        except Exception:
            family = ""
        _UI_FAMILY = family or "Helvetica"
    return _UI_FAMILY


def scaled(pt: int) -> int:
    """Map a legacy point size onto the comfortable scale."""
    return _SIZE_MAP.get(int(pt), int(pt))


def ui_font(pt: int, *, bold: bool = False, weight=None) -> "QFont":
    """Build a UI font at the scaled size in the system family."""
    from PySide6.QtGui import QFont

    f = QFont(ui_family(), scaled(pt))
    if weight is not None:
        f.setWeight(weight)
    elif bold:
        f.setBold(True)
    return f


__all__ = ["ui_family", "scaled", "ui_font"]
