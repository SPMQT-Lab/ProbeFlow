"""GUI smoke tests — require Qt (PySide6) and run offscreen.

These are the highest blast-radius tests: they exercise widget construction
and basic interaction that unit tests cannot reach.  Skipped automatically
when PySide6 is unavailable.

Run with: pytest tests/test_gui_smoke.py -v
Or headless: QT_QPA_PLATFORM=offscreen pytest tests/test_gui_smoke.py
"""

from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 not installed")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ── ProcessingControlPanel ────────────────────────────────────────────────────

class TestProcessingControlPanel:
    def test_viewer_full_panel_constructs(self, qapp):
        from probeflow.gui import ProcessingControlPanel
        panel = ProcessingControlPanel("viewer_full")
        assert panel is not None

    def test_browse_quick_panel_constructs(self, qapp):
        from probeflow.gui import ProcessingControlPanel
        panel = ProcessingControlPanel("browse_quick")
        assert panel is not None

    def test_viewer_full_panel_state_roundtrip(self, qapp):
        from probeflow.gui import ProcessingControlPanel
        panel = ProcessingControlPanel("viewer_full")
        original_state = {"align_rows": "median", "smooth_sigma": 2.0}
        panel.set_state(original_state)
        state = panel.state()
        assert state.get("align_rows") == "median"

    def test_panel_set_state_empty_does_not_crash(self, qapp):
        from probeflow.gui import ProcessingControlPanel
        panel = ProcessingControlPanel("viewer_full")
        panel.set_state({})

    def test_panel_state_returns_dict(self, qapp):
        from probeflow.gui import ProcessingControlPanel
        panel = ProcessingControlPanel("viewer_full")
        state = panel.state()
        assert isinstance(state, dict)


# ── ThumbnailGrid ─────────────────────────────────────────────────────────────

class TestThumbnailGrid:
    @pytest.fixture
    def theme(self):
        from probeflow.gui import THEMES
        return THEMES["light"]

    def test_thumbnail_grid_constructs(self, qapp, theme):
        from probeflow.gui import ThumbnailGrid
        grid = ThumbnailGrid(t=theme)
        assert grid is not None

    def test_thumbnail_grid_is_widget(self, qapp, theme):
        from PySide6.QtWidgets import QWidget
        from probeflow.gui import ThumbnailGrid
        grid = ThumbnailGrid(t=theme)
        assert isinstance(grid, QWidget)


# ── Navbar ────────────────────────────────────────────────────────────────────

class TestNavbar:
    def test_navbar_constructs_light(self, qapp):
        from probeflow.gui import Navbar
        nav = Navbar(dark=False)
        assert nav is not None

    def test_navbar_constructs_dark(self, qapp):
        from probeflow.gui import Navbar
        nav = Navbar(dark=True)
        assert nav is not None
