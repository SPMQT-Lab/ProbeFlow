"""Drawing tool constants for ProbeFlow image canvas."""
from __future__ import annotations

TOOLS = ("pan", "rectangle", "ellipse", "polygon", "freehand", "line", "point", "angle")

_TOOL_HINTS: dict[str, str] = {
    "pan": (
        "Cursor: click an ROI to select it; drag to pan. "
        "Right-click for actions; Ctrl+scroll to zoom."
    ),
    "rectangle": (
        "Rectangle selection: drag to select a region to process. "
        "Apply runs inside it. Right-click → Promote to ROI to keep it. Esc clears."
    ),
    "ellipse": (
        "Ellipse selection: drag the bounding box to select a region to process. "
        "Right-click → Promote to ROI to keep it. Esc clears."
    ),
    "polygon": (
        "Polygon selection: click vertices; double-click or Enter to close. "
        "Apply runs inside it. Right-click → Promote to ROI to keep it. Esc clears."
    ),
    "freehand": (
        "Freehand selection: drag around a region to process. "
        "Right-click → Promote to ROI to keep it. Esc clears."
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
