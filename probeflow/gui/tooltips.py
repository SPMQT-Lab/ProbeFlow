"""App-wide tooltip behaviour: wrap long tooltips to a tidy column near the cursor.

Qt only word-wraps *rich-text* tooltips; plain-text tooltips render on a single
line and can stretch across the whole screen.  Installing one application event
filter (``install_global_tooltips``) reformats every tooltip into width-capped
rich text shown at the cursor, so behaviour is consistent everywhere without
touching the hundreds of individual ``setToolTip`` calls.
"""

from __future__ import annotations

import html

from PySide6.QtCore import QEvent, QObject
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QToolTip

# Max tooltip width in px before text wraps to the next row.
_WRAP_PX = 340


def _needs_wrap(text: str) -> bool:
    """True when some line of plain text would exceed the wrap width.

    Short tooltips must keep their natural size — a fixed-width table leaves
    a mostly-empty box around one short sentence.
    """
    try:
        metrics = QFontMetrics(QToolTip.font())
        return any(
            metrics.horizontalAdvance(line) > _WRAP_PX
            for line in text.splitlines()
        )
    except Exception:
        return True  # be safe: cap width rather than risk a screen-wide strip


def _wrap(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("<"):
        # Already rich text (e.g. model formulas): just cap the width.
        return (
            f'<table width="{_WRAP_PX}" border="0" cellspacing="0" cellpadding="0">'
            f"<tr><td>{text}</td></tr></table>"
        )
    inner = html.escape(text).replace("\n", "<br>")
    if not _needs_wrap(text):
        # Fits already — rich text (so styling matches) at its natural size.
        return f"<p style='margin:0'>{inner}</p>"
    return (
        f'<table width="{_WRAP_PX}" border="0" cellspacing="0" cellpadding="0">'
        f"<tr><td>{inner}</td></tr></table>"
    )


class _ToolTipWrapper(QObject):
    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.ToolTip:
            tip = ""
            try:
                tip = obj.toolTip()
            except (AttributeError, RuntimeError):
                tip = ""
            if tip:
                QToolTip.showText(event.globalPos(), _wrap(tip), obj)
                return True
        return False


_wrapper: "_ToolTipWrapper | None" = None


def install_global_tooltips(app) -> None:
    """Install the wrapping tooltip handler on the application (idempotent)."""
    global _wrapper
    if _wrapper is not None:
        return
    _wrapper = _ToolTipWrapper(app)
    app.installEventFilter(_wrapper)


__all__ = ["install_global_tooltips"]
