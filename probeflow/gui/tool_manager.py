"""Drawing tool constants for ProbeFlow image canvas."""
from __future__ import annotations

TOOLS = ("pan", "rectangle", "ellipse", "polygon", "freehand", "line", "point", "angle")

_TOOL_HINTS: dict[str, str] = {
    "pan":       "Pan: drag to scroll  •  Ctrl+scroll to zoom  •  drag active ROI to move",
    "rectangle": "Rectangle [R]: drag to draw  •  Esc to cancel",
    "ellipse":   "Ellipse [E]: drag to draw bounding box  •  Esc to cancel",
    "polygon":   "Polygon [P]: click vertices  •  double-click or Enter to close  •  Esc to cancel",
    "freehand":  "Freehand [F]: drag to draw  •  release to finish",
    "line":      "Line [L]: drag to draw  •  Esc to cancel",
    "point":     "Point [T]: click to place",
    "angle":     "Angle: click P1, P2, P3 — measures angle at P2  •  drag handles to adjust  •  Esc to cancel",
}
