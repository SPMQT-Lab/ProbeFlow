"""Vulture whitelist for ProbeFlow.

Symbols listed here are intentionally "unused" from a static-analysis
perspective — they are called by a framework (Qt, pytest, argparse) rather
than by Python code, or they are part of the public API designed for external
consumers.

Usage:
    vulture probeflow/ tests/ whitelist.py --min-confidence 80

Add a new entry whenever vulture flags a false positive.  Group by reason.
"""

# ── Qt override methods ───────────────────────────────────────────────────────
# Qt calls these via its C++ event loop, not via Python call sites.

from probeflow.gui.image_canvas import (         # noqa: F401
    CrosshairItem, PixelInfoOverlay, ImageCanvas,
)
from probeflow.gui.roi_items import (            # noqa: F401
    RoiRectItem, RoiEllipseItem, RoiPolygonItem, RoiLineItem, RoiPointItem,
)

# paint(self, painter, _option, widget) — `_option` is a required Qt signature arg
# (renamed with _ prefix to suppress unused-variable warnings).
ImageCanvas.paint                                # type: ignore[attr-defined]
CrosshairItem.paint                              # type: ignore[attr-defined]
PixelInfoOverlay.paint                           # type: ignore[attr-defined]
RoiRectItem.paint                                # type: ignore[attr-defined]

# contextMenuEvent, wheelEvent, etc. — called by Qt event dispatch
ImageCanvas.contextMenuEvent                     # type: ignore[attr-defined]

from probeflow.gui.browse import (               # noqa: F401
    ScanCard, ThumbnailGrid,
)
ScanCard.contextMenuEvent                        # type: ignore[attr-defined]
ThumbnailGrid.contextMenuEvent                   # type: ignore[attr-defined]

from probeflow.gui.terminal import TerminalWidget  # noqa: F401
TerminalWidget.contextMenuEvent                  # type: ignore[attr-defined]
TerminalWidget._clamp_cursor                     # type: ignore[attr-defined]

# _on_finished(exit_code, exit_status) — exit_status is required by the
# QProcess.finished(int, QProcess.ExitStatus) signal; not accessed by our code.
TerminalWidget._on_finished                      # type: ignore[attr-defined]

# ── GUI attributes that serve as declared slots / future wiring points ────────
# These are set in __init__ and will be wired up as the refactor progresses.

from probeflow.gui.image_canvas import ImageCanvas   # noqa: F401  (duplicate ok)
ImageCanvas._selection_tool                      # type: ignore[attr-defined]
ImageCanvas.set_show_markers                     # type: ignore[attr-defined]
ImageCanvas.set_active_roi_id                    # type: ignore[attr-defined]
ImageCanvas.add_roi_item                         # type: ignore[attr-defined]
ImageCanvas.set_selection_tool                   # type: ignore[attr-defined]

# ── Plugin API — designed for external consumers ──────────────────────────────

from probeflow.plugins.registry import PluginRegistry  # noqa: F401
PluginRegistry.register                          # type: ignore[attr-defined]

from probeflow.plugins import api as _api        # noqa: F401
_api.PluginOperation.function                    # type: ignore[attr-defined]
_api.PluginOperation.input_types                 # type: ignore[attr-defined]
_api.PluginOperation.output_types                # type: ignore[attr-defined]

# ── gui/__init__.py lazy-loader ───────────────────────────────────────────────
# __getattr__ on a module is a documented Python pattern for lazy imports.
from probeflow.gui import __getattr__            # noqa: F401

# ── Histogram widget properties exposed for external consumers ────────────────
from probeflow.gui.viewer.histogram import ClipHistogramWidget  # noqa: F401
ClipHistogramWidget.set_clip_text                # type: ignore[attr-defined]
ClipHistogramWidget.update_drag_lines            # type: ignore[attr-defined]
ClipHistogramWidget.min_value                    # type: ignore[attr-defined]
ClipHistogramWidget.max_value                    # type: ignore[attr-defined]
ClipHistogramWidget.brightness_value             # type: ignore[attr-defined]
ClipHistogramWidget.contrast_value               # type: ignore[attr-defined]
