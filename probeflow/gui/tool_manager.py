"""Drawing tool state machine for ProbeFlow image canvas."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

TOOLS = ("pan", "rectangle", "ellipse", "polygon", "freehand", "line", "point")

_TOOL_HINTS: dict[str, str] = {
    "pan":       "Pan: drag to scroll  •  Ctrl+scroll to zoom  •  drag active ROI to move",
    "rectangle": "Rectangle [R]: drag to draw  •  Esc to cancel",
    "ellipse":   "Ellipse [E]: drag to draw bounding box  •  Esc to cancel",
    "polygon":   "Polygon [P]: click vertices  •  double-click or Enter to close  •  Esc to cancel",
    "freehand":  "Freehand [F]: drag to draw  •  release to finish",
    "line":      "Line [L]: drag to draw  •  Esc to cancel",
    "point":     "Point [T]: click to place",
}


class ToolManager(QObject):
    """Tracks the active drawing tool and notifies subscribers of changes."""

    tool_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tool = "pan"

    def tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            tool = "pan"
        if tool != self._tool:
            self._tool = tool
            self.tool_changed.emit(tool)
