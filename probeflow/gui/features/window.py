"""Standalone floating Feature Counting window.

Lets users run particle/template/classify analyses while keeping the Browse
thumbnail grid open in the main window at the same time.

Usage (from ProbeFlowWindow)
----------------------------
    win = FeatureCountingWindow()
    win.load_from_browse_needed.connect(self._on_fc_load_from_browse)
    win.show()
    # When Browse selection changes or "Load" is clicked:
    win.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
"""

from __future__ import annotations

import numpy as np

import os as _os
_os.environ.setdefault("QT_API", "pyside6")

from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStatusBar,
)

from probeflow.gui.features import (
    FeaturesPanel,
    FeaturesSidebar,
)
from probeflow.gui.features.controller import FeatureCountingController


class FeatureCountingWindow(QMainWindow):
    """Floating Feature Counting window — runs alongside the Browse thumbnail grid.

    The main :class:`ProbeFlowWindow` owns this object and bridges the
    ``load_from_browse_needed`` signal so that the selected Browse scan is
    loaded here without switching tabs.

    Two-step workflow (mirrors UniMR)
    ----------------------------------
    Step 1 — Segmentation:
        Set threshold, paint exclusion zones (mask), then click "① Segment"
        in the sidebar.  This populates ``_panel._particles`` and shows
        contour overlays on the image.

    Step 2 — Analysis:
        Choose a mode (Particles / Template / Lattice / Classify) and click
        "② Run".  For Classify, first label a few particles by clicking them
        after Step 1, then Run classifies all remaining particles.
    """

    # Emitted when the user clicks "Load primary scan from Browse".
    # ProbeFlowWindow listens and calls load_entry() with the data.
    load_from_browse_needed = Signal()

    def __init__(self, parent=None, theme: dict | None = None):
        # Qt.Window ensures this is an independent top-level window with its own
        # taskbar entry on Windows, not a child that hides behind the main window.
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("ProbeFlow — Feature Counting")
        self.resize(1200, 760)

        self._pool = QThreadPool.globalInstance()

        # Dedicated 1-thread pool for live slider previews.  Using a separate
        # pool with maxThreadCount=1 lets us cancel a queued-but-not-started
        # preview worker when a newer one arrives, so the UI always shows the
        # most recent slider position instead of queuing stale results.
        self._preview_pool = QThreadPool()
        self._preview_pool.setMaxThreadCount(1)

        # ── Widgets ──────────────────────────────────────────────────────────
        # Theme dict matches the host main window so future theme-aware
        # widgets in FeaturesPanel / FeaturesSidebar see the same palette as
        # the in-tab Features view (review gui-arch #9).  Top-level styling
        # still comes from the QApplication-level QSS so this window picks
        # up Light/Dark mode without per-widget restyling.
        self._theme: dict = dict(theme) if theme else {}
        self._panel   = FeaturesPanel(self._theme)
        self._sidebar = FeaturesSidebar(self._theme)

        # ── Status bar (created before controller so status_cb is valid) ─────
        self._status_bar = QStatusBar()
        self._status_bar.setFont(ui_font(10))
        self.setStatusBar(self._status_bar)

        # ── Controller ───────────────────────────────────────────────────────
        self._ctrl = FeatureCountingController(
            self._panel, self._sidebar,
            self._pool,
            status_cb=self._status_bar.showMessage,
            preview_pool=self._preview_pool,
            parent_widget=self,
        )

        # ── Remaining signals not owned by the controller ────────────────────
        # load_from_browse_requested is bridged to the host's Browse selection.
        self._sidebar.load_from_browse_requested.connect(
            self.load_from_browse_needed.emit)
        # "← Browse" button hides this window (Browse is always in main window).
        self._panel.go_to_browse_requested.connect(self.hide)

        # ── Layout ───────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._panel)
        splitter.addWidget(self._sidebar)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([860, 340])
        self.setCentralWidget(splitter)

    # ── Public API (called by ProbeFlowWindow) ────────────────────────────────

    def load_entry(self, entry, plane_idx: int, arr: np.ndarray,
                   px_m: float, px_x_m: float, px_y_m: float, scan=None) -> None:
        """Load a scan plane from Browse into this window."""
        self._panel.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m, scan=scan)
        self._sidebar.set_status(
            f"Loaded {entry.stem}  (plane {plane_idx},  "
            f"px = {px_m * 1e12:.1f} pm)")
        self._status_bar.showMessage(f"Loaded {entry.stem}")

    def apply_theme(self, theme: dict) -> None:
        """Sync this window's theme with the host main window.

        ``ProbeFlowWindow._apply_theme`` calls this so the floating Feature
        Counting window stays in step with Light/Dark mode toggles.  Most
        visible styling is driven by the QApplication-level stylesheet
        which propagates automatically; this method updates the cached
        theme dict on the panel and sidebar so any future theme-aware
        widget added there reads current values.
        """
        self._theme = dict(theme) if theme else {}
        self._panel._t = self._theme
        self._sidebar._t = self._theme
