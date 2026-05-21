"""Shared helpers for the browse widgets."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QFrame

from probeflow.gui.models import SxmFile


def _card_compact_meta_str(entry: SxmFile) -> str:
    """Compact V/I-only line for small thumbnail cards."""
    v_str = f"V: {entry.bias_mv:.0f} mV" if entry.bias_mv is not None else "V: ?"
    i_str = f"I: {entry.current_pa:.0f} pA" if entry.current_pa is not None else "I: ?"
    return f"{v_str}  |  {i_str}"


_CARD_SIZE_PRESETS = {
    "large": {"CARD_W": 200, "CARD_H": 220, "IMG_W": 180, "IMG_H": 150},
    "small": {"CARD_W": 160, "CARD_H": 176, "IMG_W": 144, "IMG_H": 120},
}


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _is_deleted_qt_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "already deleted" in message
        or "object has been deleted" in message
        or "internal c++ object" in message
    )


def _browse_attr(name: str, default):
    browse_mod = sys.modules.get("probeflow.gui.browse")
    if browse_mod is None:
        return default
    return getattr(browse_mod, name, default)
