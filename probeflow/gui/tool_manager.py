"""Drawing tool constants for ProbeFlow image canvas."""
from __future__ import annotations

TOOLS = ("pan", "rectangle", "ellipse", "polygon", "freehand", "line", "point", "angle")

_TOOL_HINTS: dict[str, str] = {
    "pan": (
        "Cursor: click an ROI to select it; drag to pan. "
        "Right-click for actions; Ctrl+scroll to zoom."
    ),
    "rectangle": (
        "Rectangle ROI: drag to draw an area. "
        "Release to finish. Esc cancels."
    ),
    "ellipse": (
        "Ellipse ROI: drag to draw the bounding box. "
        "Release to finish. Esc cancels."
    ),
    "polygon": (
        "Polygon ROI: click vertices. "
        "Double-click or press Enter to close. Esc cancels."
    ),
    "freehand": (
        "Freehand ROI: drag around an area. "
        "Release to finish. Esc cancels."
    ),
    "line": (
        "Line ROI: drag to draw a profile line. "
        "After selecting it, drag endpoint handles to adjust."
    ),
    "point": "Point ROI: click to place a point marker. Esc cancels.",
    "angle": (
        "Angle tool: click P1, P2, P3. "
        "The angle is measured at P2. Esc cancels."
    ),
}
