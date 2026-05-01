"""Regression tests for the Browse/Viewer processing control ownership."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")

    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


def test_browse_quick_panel_emits_only_thumbnail_corrections(qapp):
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("browse_quick")
    panel.set_state({
        "align_rows": "median",
        "bg_order": 4,
        "stm_line_bg": "step_tolerant",
        "facet_level": True,
        "smooth_sigma": 3,
        "highpass_sigma": 12,
        "fft_mode": "high_pass",
    })

    assert panel.state() == {"align_rows": "median"}


def test_viewer_full_panel_round_trips_standard_processing_state(qapp):
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("viewer_full")
    panel.set_state({
        "align_rows": "mean",
        "bg_order": 4,
        "bg_step_tolerance": True,
        "stm_line_bg": "step_tolerant",
        "facet_level": True,
        "smooth_sigma": 3,
        "highpass_sigma": 12,
        "edge_method": "dog",
        "edge_sigma": 4,
        "fft_mode": "high_pass",
        "fft_cutoff": 0.25,
        "fft_soft_border": True,
    })

    state = panel.state()

    assert state["align_rows"] == "mean"
    assert state["bg_order"] == 4
    assert state["bg_step_tolerance"] is True
    assert state["stm_line_bg"] == "step_tolerant"
    assert state["facet_level"] is True
    assert state["smooth_sigma"] == 3
    assert state["highpass_sigma"] == 12
    assert state["edge_method"] == "dog"
    assert state["edge_sigma"] == 4
    assert state["edge_sigma2"] == 8
    assert state["fft_mode"] == "high_pass"
    assert state["fft_cutoff"] == 0.25
    assert state["fft_soft_border"] is True


def test_viewer_dialog_keeps_standard_processing_visible(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])

    assert dlg._processing_panel.isHidden() is False
    assert not hasattr(dlg, "_set_zero_btn")
    assert not hasattr(dlg, "_selection_widget")
    assert hasattr(dlg, "_selection_group")
    labels = {
        btn.property("selection_tool"): btn.text()
        for btn in dlg._selection_group.buttons()
    }
    assert labels == {
        "none": "Pointer",
        "rectangle": "Rect.",
        "ellipse": "Ellipse",
        "polygon": "Polygon",
        "line": "Line",
    }
    assert dlg._set_zero_plane_btn.isHidden() is False
    assert dlg._advanced_widget.isHidden() is True
    assert dlg._spec_overlay_widget.isHidden() is True
    assert dlg._spec_show_cb.isChecked() is False
    assert dlg._export_widget.isHidden() is True

    dlg.close()
    dlg.deleteLater()


def test_viewer_apply_merges_standard_and_advanced_processing(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._processing_panel.set_state({"align_rows": "median", "bg_order": 2})
    dlg._undistort_shear_spin.setValue(3.0)
    dlg._undistort_scale_spin.setValue(1.10)

    dlg._on_apply_processing()

    assert dlg._processing["align_rows"] == "median"
    assert dlg._processing["bg_order"] == 2
    assert dlg._processing["linear_undistort"] is True
    assert dlg._processing["undistort_shear_x"] == 3.0
    assert dlg._processing["undistort_scale_y"] == 1.10

    dlg.close()
    dlg.deleteLater()


def test_viewer_line_selection_rejected_for_processing(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.zeros((8, 8), dtype=float)
    dlg._on_selection_changed({
        "kind": "line",
        "points_frac": [(0.0, 0.0), (1.0, 1.0)],
    })
    dlg._scope_cb.setCurrentIndex(1)

    dlg._on_apply_processing()

    assert "display-only" in dlg._status_lbl.text()
    assert "processing_scope" not in dlg._processing

    dlg.close()
    dlg.deleteLater()


def test_viewer_clear_selection_refreshes_selection_processing(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    calls = {"refresh": 0}
    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(
        ImageViewerDialog,
        "_refresh_processing_display",
        lambda self: calls.__setitem__("refresh", calls["refresh"] + 1),
    )

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._processing = {
        "processing_scope": "roi",
        "roi_geometry": {"kind": "ellipse", "rect_px": (1, 1, 6, 6)},
        "smooth_sigma": 1.0,
    }
    dlg._selection_geometry = {"kind": "ellipse", "rect_px": (1, 1, 6, 6)}

    dlg._on_clear_roi()

    assert calls["refresh"] == 1
    assert "processing_scope" not in dlg._processing
    assert "roi_geometry" not in dlg._processing
    assert dlg._selection_geometry is None

    dlg.close()
    dlg.deleteLater()


def test_viewer_zero_plane_workflow_remains_available(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.zeros((10, 10), dtype=float)

    dlg._set_zero_plane_btn.setChecked(True)
    dlg._on_set_zero_pick(0.0, 0.0)
    dlg._on_set_zero_pick(0.5, 0.5)
    dlg._on_set_zero_pick(1.0, 1.0)

    assert dlg._processing["set_zero_plane_points"] == [(0, 0), (4, 4), (9, 9)]
    assert "set_zero_xy" not in dlg._processing
    assert dlg._set_zero_plane_btn.isChecked() is False

    dlg.close()
    dlg.deleteLater()


def test_viewer_zero_plane_cancel_clears_partial_markers(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    calls = {"markers": None}
    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.zeros((10, 10), dtype=float)
    dlg._zoom_lbl.set_zero_markers = lambda markers: calls.__setitem__("markers", markers)

    dlg._set_zero_plane_btn.setChecked(True)
    dlg._on_set_zero_pick(0.0, 0.0)
    assert dlg._zero_plane_points_px == [(0, 0)]

    dlg._set_zero_plane_btn.setChecked(False)

    assert dlg._zero_plane_points_px == []
    assert calls["markers"] == []

    dlg.close()
    dlg.deleteLater()


def test_viewer_clear_zero_references_keeps_leveling_processing(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    calls = {"refresh": 0, "markers": None}
    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(
        ImageViewerDialog,
        "_refresh_processing_display",
        lambda self: calls.__setitem__("refresh", calls["refresh"] + 1),
    )

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.zeros((10, 10), dtype=float)
    dlg._processing = {
        "set_zero_plane_points": [(0, 0), (4, 4), (9, 9)],
        "set_zero_patch": 1,
        "align_rows": "median",
    }
    dlg._zoom_lbl.set_zero_markers = lambda markers: calls.__setitem__("markers", markers)

    dlg._on_clear_set_zero()

    assert dlg._processing["set_zero_plane_points"] == [(0, 0), (4, 4), (9, 9)]
    assert dlg._processing["set_zero_patch"] == 1
    assert calls["markers"] == []
    assert calls["refresh"] == 0

    dlg.close()
    dlg.deleteLater()


def test_zoom_label_shift_constrains_area_selection_to_square(qapp):
    from PySide6.QtCore import Qt
    from probeflow.gui_viewer_widgets import _ZoomLabel

    label = _ZoomLabel()
    label.resize(200, 100)

    bounds = label._constrain_bounds(0.1, 0.1, 0.8, 0.3, Qt.ShiftModifier)
    width_px = abs(bounds[2] - bounds[0]) * label.width()
    height_px = abs(bounds[3] - bounds[1]) * label.height()

    assert abs(width_px - height_px) < 1e-9


def test_zoom_label_endpoint_drag_updates_existing_selection(qapp):
    from probeflow.gui_viewer_widgets import _ZoomLabel

    label = _ZoomLabel()
    label.resize(200, 100)
    label._selection_geometry = {
        "kind": "rectangle",
        "bounds_frac": (0.1, 0.1, 0.5, 0.5),
    }

    geometry = label._geometry_with_dragged_handle(2, (0.8, 0.4))

    assert geometry == {
        "kind": "rectangle",
        "bounds_frac": (0.1, 0.1, 0.8, 0.4),
    }


def test_zoom_label_line_nudge_moves_one_image_pixel_and_emits(qapp):
    from probeflow.gui_viewer_widgets import _ZoomLabel

    label = _ZoomLabel()
    label.resize(200, 100)
    label._selection_geometry = {
        "kind": "line",
        "points_frac": [(0.25, 0.50), (0.75, 0.50)],
    }
    previews = []
    commits = []
    label.selection_preview_changed.connect(lambda geometry: previews.append(geometry))
    label.selection_changed.connect(lambda geometry: commits.append(geometry))

    moved = label.nudge_line(1, -1, (101, 201))

    assert moved is True
    assert previews and commits
    points = label.current_selection()["points_frac"]
    np.testing.assert_allclose(points, [(0.255, 0.49), (0.755, 0.49)])


def test_viewer_line_profile_uses_display_array_and_physical_units(qapp):
    from probeflow.gui import ImageViewerDialog

    class FakePanel:
        def __init__(self):
            self.empty = None
            self.profile = None

        def show_empty(self, message="Draw a line to show profile.", theme=None):
            self.empty = message

        def plot_profile(self, distance_nm, values, *, y_label, theme=None):
            self.profile = (np.asarray(distance_nm), np.asarray(values), y_label)

    class FakeZoom:
        def selection_tool(self):
            return "line"

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._line_profile_panel = FakePanel()
    dlg._zoom_lbl = FakeZoom()
    dlg._display_arr = np.tile(np.arange(5, dtype=np.float64), (5, 1))
    dlg._raw_arr = None
    dlg._scan_range_m = (5e-9, 5e-9)
    dlg._t = {}
    dlg._current_array_shape = lambda: dlg._display_arr.shape
    dlg._channel_unit = lambda: (1.0, "V", "Test channel")

    dlg._refresh_line_profile({
        "kind": "line",
        "points_px": [(0, 2), (4, 2)],
    })

    distance_nm, values, y_label = dlg._line_profile_panel.profile
    np.testing.assert_allclose(distance_nm, np.linspace(0.0, 4.0, 5))
    np.testing.assert_allclose(values, np.arange(5, dtype=np.float64))
    assert y_label == "Test channel [V]"


def test_viewer_dialog_initializes_panel_from_thumbnail_processing(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, ProcessingControlPanel, SxmFile

    captured = {}

    def fake_build(self):
        self._processing_panel = ProcessingControlPanel("viewer_full")

    def fake_set_state(self, state):
        captured.update(state)

    monkeypatch.setattr(ImageViewerDialog, "_build", fake_build)
    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ProcessingControlPanel, "set_state", fake_set_state)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    ImageViewerDialog(
        entry,
        [entry],
        "gray",
        {},
        processing={"align_rows": "median", "bg_order": 2},
    )

    assert captured == {"align_rows": "median", "bg_order": 2}
