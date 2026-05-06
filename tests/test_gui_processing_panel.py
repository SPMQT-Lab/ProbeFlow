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

    assert panel.state() == {"align_rows": "median", "remove_bad_lines": None}


def test_viewer_full_panel_round_trips_standard_processing_state(qapp):
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("viewer_full")
    panel.set_state({
        "align_rows": "mean",
        "remove_bad_lines": "step",
        "remove_bad_lines_threshold": 7.5,
        "remove_bad_lines_polarity": "dark",
        "remove_bad_lines_min_segment_length_px": 8,
        "remove_bad_lines_max_adjacent_bad_lines": 2,
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
    assert state["remove_bad_lines"] == "step"
    assert state["remove_bad_lines_threshold"] == 7.5
    assert state["remove_bad_lines_polarity"] == "dark"
    assert state["remove_bad_lines_min_segment_length_px"] == 8
    assert state["remove_bad_lines_max_adjacent_bad_lines"] == 2
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
    assert hasattr(dlg, "_drawing_group")
    labels = {
        btn.property("drawing_tool"): btn.text()
        for btn in dlg._drawing_group.buttons()
    }
    assert set(labels.keys()) == {"pan", "rectangle", "ellipse", "polygon",
                                   "freehand", "line", "point"}
    assert labels["pan"] == "✋ Pan"
    assert labels["rectangle"] == "▭ Rect"
    assert labels["line"] == "— Line"
    assert dlg._set_zero_plane_btn.isHidden() is False
    assert dlg._advanced_widget.isHidden() is True
    assert dlg._spec_overlay_widget.isHidden() is True
    assert dlg._spec_show_cb.isChecked() is False
    assert dlg._export_widget.isHidden() is True

    dlg.close()
    dlg.deleteLater()


def test_viewer_dialog_layout_prioritises_image_and_bounds_side_panels(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    splitter = dlg._viewer_main.centralWidget()

    assert splitter.widget(0).minimumWidth() == 500
    assert splitter.widget(1).minimumWidth() == 300
    assert splitter.widget(1).maximumWidth() == 380
    assert dlg._roi_dock.minimumWidth() == 160
    assert dlg._roi_dock.maximumWidth() == 280
    assert dlg._canvas.minimumHeight() == 140
    assert dlg._canvas.maximumHeight() == 140
    assert dlg._canvas.sizePolicy().verticalPolicy().name == "Fixed"
    assert dlg._processing_panel._TWO_COL_THRESHOLD == 360

    dlg.close()
    dlg.deleteLater()


def test_viewer_dialog_menus_mirror_existing_controls(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    top_menu_names = [action.text() for action in dlg._viewer_main.menuBar().actions()]

    assert top_menu_names == ["File", "View", "Processing", "ROI", "Export", "Help"]

    def action(menu_name: str, text: str):
        top_action = next(
            item for item in dlg._viewer_main.menuBar().actions()
            if item.text() == menu_name
        )
        menu = top_action.menu()
        for item in menu.actions():
            if item.text() == text:
                return item
            submenu = item.menu()
            if submenu is not None:
                for subitem in submenu.actions():
                    if subitem.text() == text:
                        return subitem
        raise AssertionError(f"Missing menu action: {menu_name} > {text}")

    action("Processing", "Median").trigger()
    assert dlg._processing_panel._align_combo.currentText() == "Median"
    assert action("Processing", "Median").isChecked()

    action("Processing", "Step segments").trigger()
    assert dlg._processing_panel._bad_lines_combo.currentText() == "Step segments"

    action("Processing", "Gaussian").trigger()
    assert dlg._processing_panel._smooth_combo.currentText() == "Gaussian"
    assert action("Processing", "STM Background...").isEnabled() is True

    action("ROI", "Rectangle").trigger()
    assert dlg._zoom_lbl.tool() == "rectangle"
    assert action("ROI", "Rectangle").isChecked()
    action("ROI", "Pan").trigger()
    assert dlg._zoom_lbl.tool() == "pan"

    dlg.show()
    qapp.processEvents()
    dlg._roi_dock.close()
    qapp.processEvents()
    assert dlg._roi_dock.isVisible() is False
    action("ROI", "Show ROI Manager").trigger()
    qapp.processEvents()
    assert dlg._roi_dock.isVisible() is True

    assert action("ROI", "Rename ROI").isEnabled() is False
    assert action("ROI", "Delete ROI").isEnabled() is False
    assert action("Export", "Save PNG copy").isEnabled() is True
    assert action("Export", "Save processed image").isEnabled() is False
    assert action("Export", "Save provenance").isEnabled() is False

    action("Help", "Definitions").trigger()
    qapp.processEvents()
    assert dlg._definitions_dialog.isVisible()
    definitions_dialog = dlg._definitions_dialog
    action("Help", "Definitions").trigger()
    qapp.processEvents()
    assert dlg._definitions_dialog is definitions_dialog
    definitions_dialog.close()
    qapp.processEvents()
    assert dlg.isVisible()

    dlg.close()
    dlg.deleteLater()


def test_viewer_stm_background_apply_records_processing_state(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])

    dlg._on_stm_background_applied({
        "fit_region": "active_roi",
        "fit_roi_id": "roi-1",
        "line_statistic": "median",
        "model": "poly2",
        "linear_x_first": False,
        "blur_length": None,
        "jump_threshold": None,
        "preserve_level": "median",
        "applied_to": "whole_image",
    })

    assert dlg._processing["stm_background"]["fit_roi_id"] == "roi-1"
    assert dlg._processing["stm_background"]["model"] == "poly2"
    assert dlg._processing["stm_background"]["applied_to"] == "whole_image"

    dlg.close()
    dlg.deleteLater()


def test_bad_segment_overlay_rectangles_match_detector_output(qapp):
    from PySide6.QtGui import QPixmap
    from probeflow.gui.image_canvas import ImageCanvas
    from probeflow.processing import BadSegment

    canvas = ImageCanvas()
    canvas.set_source(QPixmap(20, 10), reset_zoom=False)
    segments = [BadSegment(4, 3, 9, 12.0, "step")]

    canvas.set_bad_segment_overlay(segments)

    assert len(canvas._bad_segment_items) == 1
    rect = canvas._bad_segment_items[0].rect()
    assert rect.x() == 3
    assert rect.y() == 4
    assert rect.width() == 6
    assert rect.height() == 1

    canvas.clear_bad_segment_overlay()
    assert canvas._bad_segment_items == []
    canvas.deleteLater()


def test_bad_line_preview_uses_panel_detection_parameters(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    from probeflow.processing import BadLineCorrectionInfo, BadSegment

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._display_arr = np.ones((8, 8), dtype=float)
    dlg._processing_panel.set_state({
        "remove_bad_lines": "step",
        "remove_bad_lines_threshold": 4.5,
        "remove_bad_lines_polarity": "dark",
        "remove_bad_lines_min_segment_length_px": 6,
        "remove_bad_lines_max_adjacent_bad_lines": 2,
    })
    segment = BadSegment(3, 2, 6, 9.0, "step")
    captured = {}

    def fake_detect(image, **kwargs):
        captured["detect_image"] = image.copy()
        captured["detect_kwargs"] = kwargs
        return [segment]

    def fake_repair(image, segments, **kwargs):
        captured["repair_image"] = image.copy()
        captured["repair_segments"] = tuple(segments)
        captured["repair_kwargs"] = kwargs
        return image.copy(), BadLineCorrectionInfo(
            segments=tuple(segments),
            skipped_segments=(),
            method="step",
            threshold=float(kwargs["threshold"]),
            corrected_segments=tuple(segments),
            polarity=str(kwargs["polarity"]),
            min_segment_length_px=int(kwargs["min_segment_length_px"]),
            max_adjacent_bad_lines=int(kwargs["max_adjacent_bad_lines"]),
        )

    monkeypatch.setattr("probeflow.processing.detect_bad_scanline_segments", fake_detect)
    monkeypatch.setattr("probeflow.processing.repair_bad_scanline_segments", fake_repair)

    dlg._on_preview_bad_lines()

    np.testing.assert_array_equal(captured["detect_image"], dlg._display_arr)
    assert captured["detect_kwargs"] == {
        "threshold": 4.5,
        "method": "step",
        "polarity": "dark",
        "min_segment_length_px": 6,
        "max_adjacent_bad_lines": 2,
    }
    assert captured["repair_segments"] == (segment,)
    assert captured["repair_kwargs"] == {
        "max_adjacent_bad_lines": 2,
        "threshold": 4.5,
        "polarity": "dark",
        "min_segment_length_px": 6,
    }
    assert dlg._processing_panel._bad_line_preview_lbl.text() == (
        "Detected 1 segment on 1 scan line"
    )

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


def test_viewer_apply_scopes_local_filter_to_active_area_roi(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 2.0, "y": 2.0, "width": 3.0, "height": 3.0})
    roi_set.add(roi)
    roi_set.set_active(roi.id)
    dlg._image_roi_set = roi_set
    dlg._processing_panel.set_state({"smooth_sigma": 1.0})
    dlg._scope_cb.setCurrentIndex(0)

    dlg._on_apply_processing()

    assert dlg._processing["processing_scope"] == "roi"
    assert dlg._processing["processing_roi_id"] == roi.id
    assert "roi_geometry" not in dlg._processing
    assert "roi_rect" not in dlg._processing

    dlg.close()
    dlg.deleteLater()


def test_viewer_apply_rejects_local_filter_for_active_non_area_roi(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    roi_set = ROISet(image_id="img1")
    roi = ROI.new("point", {"x": 2.0, "y": 2.0})
    roi_set.add(roi)
    roi_set.set_active(roi.id)
    dlg._image_roi_set = roi_set
    dlg._processing_panel.set_state({"smooth_sigma": 1.0})

    dlg._on_apply_processing()

    assert "not valid for area processing" in dlg._status_lbl.text()
    assert "processing_scope" not in dlg._processing

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
    from probeflow.gui.viewer.widgets import _ZoomLabel

    label = _ZoomLabel()
    label.resize(200, 100)

    bounds = label._constrain_bounds(0.1, 0.1, 0.8, 0.3, Qt.ShiftModifier)
    width_px = abs(bounds[2] - bounds[0]) * label.width()
    height_px = abs(bounds[3] - bounds[1]) * label.height()

    assert abs(width_px - height_px) < 1e-9


def test_zoom_label_endpoint_drag_updates_existing_selection(qapp):
    from probeflow.gui.viewer.widgets import _ZoomLabel

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
    from probeflow.gui.viewer.widgets import _ZoomLabel

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

        def plot_profile(self, x_vals, values, *, x_label="Distance [nm]",
                         y_label, theme=None):
            self.profile = (np.asarray(x_vals), np.asarray(values), x_label, y_label)

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

    x_vals, values, x_label, y_label = dlg._line_profile_panel.profile
    # Profile spans 4 pixels × 1 nm/pixel = 4 nm = 40 Å.
    # choose_display_unit picks Å for ~2.5 nm median magnitude.
    assert "Distance" in x_label
    assert x_vals[0] == pytest.approx(0.0)
    assert x_vals[-1] > 0
    np.testing.assert_allclose(values, np.arange(5, dtype=np.float64))
    assert y_label == "Test channel [V]"


def test_viewer_active_line_roi_detection(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog

    roi_set = ROISet(image_id="img1")
    rect = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0})
    line = ROI.new("line", {"x1": 0.0, "y1": 1.0, "x2": 4.0, "y2": 1.0})
    roi_set.add(rect)
    roi_set.add(line)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set

    roi_set.set_active(line.id)
    assert dlg._active_line_roi_id() == line.id

    roi_set.set_active(rect.id)
    assert dlg._active_line_roi_id() is None


def test_viewer_line_profile_sync_clears_for_non_line_active_roi(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog

    class FakePanel:
        def __init__(self):
            self.visible = None
            self.empty = None

        def setVisible(self, visible):
            self.visible = visible

        def show_empty(self, message="Draw a line to show profile.", theme=None):
            self.empty = message

    class FakeZoom:
        def selection_tool(self):
            return "pan"

    roi_set = ROISet(image_id="img1")
    rect = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0})
    roi_set.add(rect)
    roi_set.set_active(rect.id)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._line_profile_panel = FakePanel()
    dlg._zoom_lbl = FakeZoom()
    dlg._line_profile_geometry = {"kind": "line", "points_px": [(0, 0), (1, 1)]}
    dlg._t = {}

    dlg._sync_line_profile_visibility()

    assert dlg._line_profile_panel.visible is False
    assert dlg._line_profile_panel.empty == "Draw a line to show profile."
    assert dlg._line_profile_geometry is None


def test_viewer_line_profile_sync_uses_active_line_roi(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog

    class FakePanel:
        def __init__(self):
            self.visible = None

        def setVisible(self, visible):
            self.visible = visible

    class FakeZoom:
        def selection_tool(self):
            return "pan"

    roi_set = ROISet(image_id="img1")
    line = ROI.new("line", {"x1": 0.0, "y1": 1.0, "x2": 4.0, "y2": 1.0})
    roi_set.add(line)
    roi_set.set_active(line.id)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._line_profile_panel = FakePanel()
    dlg._zoom_lbl = FakeZoom()
    called = []
    dlg._on_roi_line_profile = lambda roi_id: called.append(roi_id)

    dlg._sync_line_profile_visibility()

    assert dlg._line_profile_panel.visible is True
    assert called == [line.id]


def test_line_profile_panel_empty_clears_source_title(qapp):
    from probeflow.gui.viewer.widgets import LineProfilePanel

    panel = LineProfilePanel()
    panel.plot_profile(
        np.asarray([0.0, 1.0]),
        np.asarray([2.0, 3.0]),
        y_label="Height [m]",
        theme={},
    )
    panel.set_source_label("Line ROI: line_1 (abc12345)", theme={})

    assert panel._ax.get_title() == "Line ROI: line_1 (abc12345)"

    panel.show_empty(theme={})

    assert panel._ax.get_title() == ""


def test_viewer_transform_updates_roi_set_coordinates(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog

    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 2.0, "y": 1.0, "width": 4.0, "height": 3.0})
    roi_set.add(roi)
    roi_set.set_active(roi.id)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = np.zeros((10, 20))
    dlg._raw_arr = None
    refreshed = []
    dlg._on_image_roi_set_changed = lambda: refreshed.append(True)

    dlg._transform_image_roi_set_for_display_op("rotate_90_cw")

    moved = roi_set.get(roi.id)
    assert moved.geometry["x"] == pytest.approx(6.0)
    assert moved.geometry["y"] == pytest.approx(2.0)
    assert moved.geometry["width"] == pytest.approx(3.0)
    assert moved.geometry["height"] == pytest.approx(4.0)
    assert roi_set.active_roi_id == roi.id
    assert refreshed == [True]


def test_viewer_arbitrary_transform_removes_rois(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    roi_set = ROISet(image_id="img1")
    roi = ROI.new("point", {"x": 2.0, "y": 1.0})
    roi_set.add(roi)
    roi_set.set_active(roi.id)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = np.zeros((10, 20))
    dlg._raw_arr = None
    dlg._status_lbl = FakeStatus()
    dlg._on_image_roi_set_changed = lambda: None

    dlg._transform_image_roi_set_for_display_op("rotate_arbitrary", {"angle_degrees": 30.0})

    assert roi_set.rois == []
    assert roi_set.active_roi_id is None
    assert "invalidated 1 ROI" in dlg._status_lbl.text


def test_viewer_refresh_display_array_passes_roi_set(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog
    import probeflow.gui._legacy as gui_mod

    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 0.0, "y": 0.0, "width": 2.0, "height": 2.0})
    roi_set.add(roi)
    seen = {}

    def fake_apply(arr, processing, roi_set=None):
        seen["roi_set"] = roi_set
        return arr + 1.0

    monkeypatch.setattr(gui_mod, "_apply_processing", fake_apply)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._display_arr = None
    dlg._raw_arr = np.zeros((2, 2))
    dlg._processing = {"bg_order": 1, "background_fit_roi_id": roi.id}
    dlg._image_roi_set = roi_set
    dlg._reset_zoom_on_next_pixmap = False
    dlg._processing_roi_error = ""

    dlg._refresh_display_array()

    assert seen["roi_set"] is roi_set
    np.testing.assert_allclose(dlg._display_arr, np.ones((2, 2)))


def test_viewer_refresh_display_array_blocks_stale_roi_reference(qapp, monkeypatch):
    from probeflow.core.roi import ROISet
    from probeflow.gui import ImageViewerDialog
    import probeflow.gui._legacy as gui_mod

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    called = []

    def fake_apply(arr, processing, roi_set=None):
        called.append(True)
        return arr + 1.0

    monkeypatch.setattr(gui_mod, "_apply_processing", fake_apply)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._display_arr = None
    dlg._raw_arr = np.zeros((2, 2))
    dlg._processing = {"bg_order": 1, "background_fit_roi_id": "missing-id"}
    dlg._image_roi_set = ROISet(image_id="img1")
    dlg._reset_zoom_on_next_pixmap = False
    dlg._status_lbl = FakeStatus()
    dlg._processing_roi_error = ""

    dlg._refresh_display_array()

    assert called == []
    np.testing.assert_allclose(dlg._display_arr, np.zeros((2, 2)))
    assert "missing ROI reference" in dlg._processing_roi_error
    assert "missing-id" in dlg._status_lbl.text


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
