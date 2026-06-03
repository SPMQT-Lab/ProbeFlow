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
from PySide6.QtWidgets import QToolTip

# Max tooltip width in px before text wraps to the next row.
_WRAP_PX = 340


def _wrap(text: str) -> str:
    stripped = text.lstrip()
    # Honour tooltips that are already rich text (e.g. model formulas); just cap
    # their width.  Plain text is escaped and its newlines become line breaks.
    inner = text if stripped.startswith("<") else html.escape(text).replace("\n", "<br>")
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
