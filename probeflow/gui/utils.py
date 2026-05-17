"""Small GUI utility helpers with no Qt widget dependencies."""

from __future__ import annotations

import shutil
import subprocess
import webbrowser


def _open_url(url: str) -> None:
    """Open URL in default browser. Tries Qt first, then Windows (WSL), then webbrowser."""
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        if QDesktopServices.openUrl(QUrl(url)):
            return
    except Exception:
        pass
    if shutil.which("cmd.exe"):
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", "", url],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except Exception:
            pass
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _format_scan_conditions(entry) -> str:
    """Return a compact bias/current string for the image viewer header."""
    parts = []
    bias_mv = getattr(entry, "bias_mv", None)
    current_pa = getattr(entry, "current_pa", None)
    if bias_mv is not None:
        parts.append(f"{bias_mv / 1000:.4g} V")
    if current_pa is not None:
        if abs(current_pa) >= 1000:
            parts.append(f"{current_pa / 1000:.4g} nA")
        else:
            parts.append(f"{current_pa:.4g} pA")
    return ", ".join(parts)


__all__ = ["_open_url", "_format_scan_conditions"]
