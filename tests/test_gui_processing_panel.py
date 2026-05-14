"""Regression tests for the Browse/Viewer processing control ownership."""

from __future__ import annotations

import json
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


class _FakeMeasurementTable:
    def __init__(self):
        self.rows = []

    def next_measurement_id(self):
        return f"M{len(self.rows) + 1:04d}"

    def add_result(self, result):
        self.rows.append(result)


class _FakeStatus:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _FakeCanvas:
    def __init__(self):
        self.feature_points = None

    def set_feature_points(self, points):
        self.feature_points = list(points)


class _FakeSignal:
    def connect(self, _callback):
        return None


class _FakeFeaturePanel:
    def __init__(self, settings):
        self._settings = dict(settings)
        self.detectRequested = _FakeSignal()
        self.copyPointsRequested = _FakeSignal()
        self.exportCsvRequested = _FakeSignal()
        self.exportJsonRequested = _FakeSignal()
        self.exportMaskCsvRequested = _FakeSignal()
        self.computeFftRequested = _FakeSignal()
        self.exportFftCsvRequested = _FakeSignal()
        self.clearRequested = _FakeSignal()
        self.count = None
        self.message = ""

    def settings(self):
        return dict(self._settings)

    def mask_settings(self):
        return {
            "radius_px": int(self._settings.get("radius_px", 0)),
            "shape_mode": self._settings.get("shape_mode", "disk"),
        }

    def set_points_count(self, count, roi_name=None):
        self.count = count
        self.roi_name = roi_name

    def show_message(self, message):
        self.message = str(message)


def _make_feature_maxima_controller(image, feature_panel=None):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile
    from probeflow.gui.viewer import ImageMeasurementController

    roi_set = ROISet(image_id="img1")
    roi = ROI.new(
        "rectangle",
        {"x": 0, "y": 0, "width": image.shape[1], "height": image.shape[0]},
        name="all",
    )
    roi_set.add(roi)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = image
    dlg._entries = [
        SxmFile(
            path=Path("/tmp/example.sxm"),
            stem="example",
            Nx=image.shape[1],
            Ny=image.shape[0],
        )
    ]
    dlg._idx = 0
    dlg._channel_unit = lambda: (1.0, "nm", "Height")
    dlg._pixel_size_xy_m = lambda: (1e-9, 1e-9)
    dlg._status_lbl = _FakeStatus()
    dlg._zoom_lbl = _FakeCanvas()
    table = _FakeMeasurementTable()
    controller = ImageMeasurementController(
        dlg,
        table,
        feature_panel=feature_panel,
        point_mask_panel=feature_panel,
    )
    return controller, dlg, table, roi


def test_browse_quick_panel_emits_only_thumbnail_corrections(qapp):
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("browse_quick")
    panel.set_state({
        "align_rows": "median",
        "smooth_sigma": 3,
        "highpass_sigma": 12,
        "fft_mode": "high_pass",
    })

    assert panel.state() == {"align_rows": "median", "remove_bad_lines": None}


def test_viewer_full_panel_round_trips_standard_processing_state(qapp):
    from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("viewer_full")
    panel.set_state({
        "align_rows": "mean",
        "remove_bad_lines": "step",
        "remove_bad_lines_threshold": 7.5,
        "remove_bad_lines_polarity": "dark",
        "remove_bad_lines_min_segment_length_px": 8,
        "remove_bad_lines_max_adjacent_bad_lines": 2,
        "smooth_sigma": 3,
        "highpass_sigma": 12,
        "edge_method": "dog",
        "edge_sigma": 4,
    })

    state = panel.state()

    assert state["align_rows"] == "mean"
    assert state["remove_bad_lines"] == "step"
    assert state["remove_bad_lines_threshold"] == 7.5
    assert state["remove_bad_lines_polarity"] == "dark"
    assert state["remove_bad_lines_min_segment_length_px"] == 8
    assert state["remove_bad_lines_max_adjacent_bad_lines"] == 2
    assert state["smooth_sigma"] == 3
    assert state["highpass_sigma"] == 12
    assert state["edge_method"] == "dog"
    assert state["edge_sigma"] == 4
    assert state["edge_sigma2"] == 8
    labels = [label.text() for label in panel.findChildren(QLabel)]
    assert labels.index("Line corrections") < labels.index("Background") < labels.index("Filters")
    assert "Simple background" not in labels
    assert "Radial FFT" not in labels
    assert "Line offset:" not in labels
    assert not any(cb.text() == "Soft border" for cb in panel.findChildren(QCheckBox))
    assert any(
        btn.text() == "STM Background..."
        for btn in panel.findChildren(QPushButton)
    )


def test_viewer_dialog_keeps_standard_processing_visible(qapp, monkeypatch):
    from PySide6.QtWidgets import QCheckBox
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
    assert not hasattr(dlg, "_patch_roi_cb")
    assert not any(cb.text() == "Patch selection" for cb in dlg.findChildren(QCheckBox))
    assert dlg._advanced_widget.isHidden() is True
    assert dlg._advanced_fft_combo.currentText() == "None"
    assert dlg._advanced_fft_soft_cb.text() == "Soft border"
    assert dlg._spec_overlay_widget.isHidden() is True
    assert dlg._spec_show_cb.isChecked() is False
    assert dlg._export_widget.isHidden() is True
    assert dlg._sidebar_tabs.tabText(dlg._sidebar_tabs.currentIndex()) == "View"
    assert dlg._roi_dock.isVisible() is False
    assert dlg._measurement_dock.isVisible() is False

    dlg.close()
    dlg.deleteLater()


def test_viewer_dialog_layout_prioritises_image_and_bounds_side_panels(qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    splitter = dlg._viewer_main.centralWidget()

    assert splitter.widget(0).minimumWidth() == 500
    assert splitter.widget(1).minimumWidth() == 380
    assert splitter.widget(1).maximumWidth() == 420
    assert dlg._roi_dock.minimumWidth() == 160
    assert dlg._roi_dock.maximumWidth() == 280
    assert dlg._hist_panel._canvas.minimumHeight() == 140
    assert dlg._hist_panel._canvas.maximumHeight() == 140
    assert dlg._hist_panel._canvas.sizePolicy().verticalPolicy().name == "Fixed"
    assert [dlg._sidebar_tabs.tabText(i) for i in range(dlg._sidebar_tabs.count())] == [
        "View",
        "Process",
        "ROI",
        "Measure",
        "Export",
    ]
    assert dlg._sidebar_tabs.elideMode() == Qt.ElideNone
    assert dlg._sidebar_tabs.tabBar().usesScrollButtons() is False

    dlg.close()
    dlg.deleteLater()


def test_viewer_dialog_menus_mirror_existing_controls(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    top_menu_names = [action.text() for action in dlg._viewer_main.menuBar().actions()]

    assert top_menu_names == [
        "File",
        "View",
        "Processing",
        "ROI",
        "Measurements",
        "Export",
        "Help",
    ]

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
    with pytest.raises(AssertionError):
        action("Processing", "Radial FFT")
    with pytest.raises(AssertionError):
        action("Processing", "FFT soft border")
    assert action("Processing", "STM Background...").isEnabled() is True
    assert action("View", "Histogram / Contrast").isEnabled() is True
    assert action("View", "Processing panel").isEnabled() is True
    assert action("View", "ROI panel").isEnabled() is True
    assert action("View", "Measurements panel").isEnabled() is True
    assert action("View", "Export panel").isEnabled() is True
    assert action("View", "ROI Manager").isEnabled() is True
    assert action("View", "Measurements").isEnabled() is True
    assert action("Measurements", "Show measurements").isEnabled() is True
    assert action("Measurements", "Compute point-mask FFT").isEnabled() is False
    action("View", "Processing panel").trigger()
    assert dlg._sidebar_tabs.tabText(dlg._sidebar_tabs.currentIndex()) == "Process"
    action("View", "Histogram / Contrast").trigger()
    assert dlg._sidebar_tabs.tabText(dlg._sidebar_tabs.currentIndex()) == "View"
    dlg._display_arr = np.ones((8, 8), dtype=float)
    action("Processing", "STM Background...").trigger()
    qapp.processEvents()
    assert dlg._stm_background_dialog.windowTitle() == "STM Background"
    assert dlg._stm_background_dialog.isVisible()
    dlg._stm_background_dialog.close()
    qapp.processEvents()
    dlg._processing_panel._stm_background_btn.click()
    qapp.processEvents()
    assert dlg._stm_background_dialog.windowTitle() == "STM Background"
    assert dlg._stm_background_dialog.isVisible()
    dlg._stm_background_dialog.close()

    action("ROI", "Rectangle").trigger()
    assert dlg._zoom_lbl.tool() == "rectangle"
    assert action("ROI", "Rectangle").isChecked()
    assert dlg._sidebar_tabs.tabText(dlg._sidebar_tabs.currentIndex()) == "ROI"
    action("ROI", "Pan").trigger()
    assert dlg._zoom_lbl.tool() == "pan"
    action("ROI", "Line").trigger()
    assert dlg._zoom_lbl.tool() == "line"
    assert dlg._sidebar_tabs.tabText(dlg._sidebar_tabs.currentIndex()) == "Measure"
    assert dlg._measurement_panel.measurement_type() == "line_profile"
    action("ROI", "Pan").trigger()

    dlg.show()
    qapp.processEvents()
    dlg._roi_dock.close()
    qapp.processEvents()
    assert dlg._roi_dock.isVisible() is False
    action("ROI", "Show ROI Manager").trigger()
    qapp.processEvents()
    assert dlg._roi_dock.isVisible() is True
    assert dlg._sidebar_tabs.tabText(dlg._sidebar_tabs.currentIndex()) == "ROI"
    dlg._measurement_dock.close()
    qapp.processEvents()
    assert dlg._measurement_dock.isVisible() is False
    action("View", "Measurements").trigger()
    qapp.processEvents()
    assert dlg._measurement_dock.isVisible() is True
    action("View", "Export panel").trigger()
    assert dlg._sidebar_tabs.tabText(dlg._sidebar_tabs.currentIndex()) == "Export"

    assert action("ROI", "Rename ROI").isEnabled() is False
    assert action("ROI", "Delete ROI").isEnabled() is False
    assert action("Export", "Save PNG copy").isEnabled() is True
    assert action("Export", "Save processed image").isEnabled() is True
    assert action("Export", "Save provenance").isEnabled() is True

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


def test_viewer_align_rows_applies_immediately(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    refreshes = []
    monkeypatch.setattr(
        ImageViewerDialog,
        "_refresh_processing_display",
        lambda self: refreshes.append(dict(self._processing)),
    )

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])

    dlg._processing_panel._align_combo.setCurrentText("Mean")

    assert dlg._processing["align_rows"] == "mean"
    assert refreshes[-1]["align_rows"] == "mean"
    assert dlg._proc_undo_ctrl._undo_stack[-1] == {}

    dlg._processing_panel._align_combo.setCurrentText("None")

    assert "align_rows" not in dlg._processing
    assert refreshes[-1] == {}

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


def test_legacy_background_controls_are_not_in_active_processing_panel(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])

    assert not hasattr(dlg._processing_panel, "_bg_combo")
    assert not hasattr(dlg._processing_panel, "_stm_line_bg_combo")
    assert not hasattr(dlg._processing_panel, "_facet_cb")
    assert not hasattr(dlg, "_stm_background_dialog")

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
    dlg._processing_panel.set_state({"align_rows": "median"})
    dlg._undistort_shear_spin.setValue(3.0)
    dlg._undistort_scale_spin.setValue(1.10)
    dlg._advanced_fft_combo.setCurrentText("High-pass")
    dlg._advanced_fft_cutoff_spin.setValue(0.25)
    dlg._advanced_fft_soft_cb.setChecked(True)

    dlg._on_apply_processing()

    assert dlg._processing["align_rows"] == "median"
    assert "bg_order" not in dlg._processing
    assert dlg._processing["linear_undistort"] is True
    assert dlg._processing["undistort_shear_x"] == 3.0
    assert dlg._processing["undistort_scale_y"] == 1.10
    assert dlg._processing["fft_mode"] == "high_pass"
    assert dlg._processing["fft_cutoff"] == 0.25
    assert dlg._processing["fft_soft_border"] is True
    assert dlg._processing["fft_soft_mode"] == "high_pass"

    dlg.close()
    dlg.deleteLater()


def test_viewer_apply_keeps_whole_image_scope_with_active_area_roi(qapp, monkeypatch):
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

    assert dlg._processing["smooth_sigma"] == 1.0
    assert "processing_scope" not in dlg._processing
    assert "processing_roi_id" not in dlg._processing

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
    dlg._scope_cb.setCurrentIndex(1)

    dlg._on_apply_processing()

    assert dlg._processing["processing_scope"] == "roi"
    assert dlg._processing["processing_roi_id"] == roi.id

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
    dlg._scope_cb.setCurrentIndex(1)

    dlg._on_apply_processing()

    assert "not valid for area processing" in dlg._status_lbl.text()
    assert "processing_scope" not in dlg._processing

    dlg.close()
    dlg.deleteLater()


def test_viewer_reset_clears_roi_filter_scope(qapp, monkeypatch):
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
        "processing_roi_id": "roi-1",
        "smooth_sigma": 1.0,
    }

    dlg._on_reset_processing()

    assert calls["refresh"] == 1
    assert "processing_scope" not in dlg._processing
    assert "processing_roi_id" not in dlg._processing

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
    assert dlg._zero_ctrl.points == [(0, 0)]

    dlg._set_zero_plane_btn.setChecked(False)

    assert dlg._zero_ctrl.points == []
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


def test_viewer_line_profile_uses_display_array_and_physical_units(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog

    class FakePanel:
        def __init__(self):
            self.empty = None
            self.profile = None
            self.visible = None
            self.source = None

        def show_empty(self, message="Draw a line to show profile.", theme=None):
            self.empty = message

        def setVisible(self, visible):
            self.visible = visible

        def set_width(self, width):
            pass

        def plot_profile(self, x_vals, values, *, x_label="Distance [nm]",
                         y_label, theme=None):
            self.profile = (np.asarray(x_vals), np.asarray(values), x_label, y_label)

        def set_source_label(self, label, *, theme=None):
            self.source = label

    roi_set = ROISet(image_id="img1")
    line = ROI.new("line", {"x1": 0.0, "y1": 2.0, "x2": 4.0, "y2": 2.0})
    roi_set.add(line)
    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._line_profile_panel = FakePanel()
    dlg._display_arr = np.tile(np.arange(5, dtype=np.float64), (5, 1))
    dlg._raw_arr = None
    dlg._scan_range_m = (5e-9, 5e-9)
    dlg._t = {}
    dlg._current_array_shape = lambda: dlg._display_arr.shape
    dlg._channel_unit = lambda: (1.0, "V", "Test channel")

    dlg._on_roi_line_profile(line.id)

    x_vals, values, x_label, y_label = dlg._line_profile_panel.profile
    # Profile spans 4 pixels × 1 nm/pixel = 4 nm = 40 Å.
    # choose_display_unit picks Å for ~2.5 nm median magnitude.
    assert "Distance" in x_label
    assert x_vals[0] == pytest.approx(0.0)
    assert x_vals[-1] > 0
    np.testing.assert_allclose(values, np.arange(5, dtype=np.float64))
    assert y_label == "Test channel [V]"
    assert dlg._line_profile_panel.visible is True
    assert dlg._line_profile_panel.source.startswith("Line ROI:")


def test_viewer_adds_roi_statistics_to_measurement_table(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile
    from probeflow.gui.viewer import ImageMeasurementController

    class FakeTable:
        def __init__(self):
            self.rows = []

        def next_measurement_id(self):
            return f"M{len(self.rows) + 1:04d}"

        def add_result(self, result):
            self.rows.append(result)

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 2}, name="terrace")
    roi_set.add(roi)
    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = np.arange(16, dtype=float).reshape(4, 4) * 1e-9
    dlg._entries = [SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=4, Ny=4)]
    dlg._idx = 0
    dlg._channel_unit = lambda: (1e9, "nm", "Height")
    dlg._pixel_size_xy_m = lambda: (2e-9, 3e-9)
    table = FakeTable()
    dlg._measurement_table = table
    dlg._status_lbl = FakeStatus()
    controller = ImageMeasurementController(dlg, table)

    controller.add_roi_stats_measurement(roi.id)

    result = table.rows[0]
    assert result.kind == "roi_stats"
    assert result.values["mean_height"] == pytest.approx(7.5)
    assert result.values["area"] == pytest.approx(24.0)
    assert result.context["roi_name"] == "terrace"
    assert result.context["height_unit"] == "nm"
    assert "M0001" in dlg._status_lbl.text


def test_viewer_adds_step_height_from_two_rois(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile
    from probeflow.gui.viewer import ImageMeasurementController

    class FakeTable:
        def __init__(self):
            self.rows = []

        def next_measurement_id(self):
            return f"M{len(self.rows) + 1:04d}"

        def add_result(self, result):
            self.rows.append(result)

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    image = np.zeros((4, 4), dtype=float)
    image[:, 2:] = 2.5e-9
    roi_set = ROISet(image_id="img1")
    lower = ROI.new("rectangle", {"x": 0, "y": 0, "width": 2, "height": 4}, name="lower")
    upper = ROI.new("rectangle", {"x": 2, "y": 0, "width": 2, "height": 4}, name="upper")
    roi_set.add(lower)
    roi_set.add(upper)
    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = image
    dlg._entries = [SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=4, Ny=4)]
    dlg._idx = 0
    dlg._channel_unit = lambda: (1e9, "nm", "Height")
    table = FakeTable()
    dlg._measurement_table = table
    dlg._status_lbl = FakeStatus()
    controller = ImageMeasurementController(dlg, table)

    controller.add_step_height_measurement_for_rois([lower.id, upper.id])

    result = table.rows[0]
    assert result.kind == "step_height"
    assert result.values["height_difference"] == pytest.approx(2.5)
    assert result.context["roi_a_name"] == "lower"
    assert result.context["roi_b_name"] == "upper"
    assert result.context["height_unit"] == "nm"


def test_viewer_adds_line_profile_summary_to_measurement_table(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile
    from probeflow.gui.viewer import ImageMeasurementController

    class FakeTable:
        def __init__(self):
            self.rows = []

        def next_measurement_id(self):
            return f"M{len(self.rows) + 1:04d}"

        def add_result(self, result):
            self.rows.append(result)

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    roi_set = ROISet(image_id="img1")
    line = ROI.new("line", {"x1": 0.0, "y1": 2.0, "x2": 4.0, "y2": 2.0}, name="step")
    roi_set.add(line)
    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = np.tile(np.arange(5, dtype=np.float64), (5, 1)) * 1e-9
    dlg._entries = [SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=5, Ny=5)]
    dlg._idx = 0
    dlg._channel_unit = lambda: (1e9, "nm", "Height")
    dlg._pixel_size_xy_m = lambda: (1e-9, 1e-9)
    table = FakeTable()
    dlg._measurement_table = table
    dlg._status_lbl = FakeStatus()
    dlg._on_roi_line_profile = lambda _roi_id: None
    controller = ImageMeasurementController(dlg, table)

    controller.add_line_profile_measurement_for_roi(line.id)

    result = table.rows[0]
    assert result.kind == "line_profile"
    assert result.values["height_peak_to_peak"] == pytest.approx(4.0)
    assert result.values["x2"] == pytest.approx(4.0)
    assert result.values["y2"] == pytest.approx(2.0)
    assert result.context["roi_id"] == line.id
    assert result.context["roi_name"] == "step"
    assert result.y_unit == "nm"


def test_viewer_feature_maxima_action_adds_summary_measurement(
    qapp, monkeypatch, tmp_path,
):
    from PySide6.QtWidgets import QFileDialog
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile
    from probeflow.gui.viewer import ImageMeasurementController

    class FakeTable:
        def __init__(self):
            self.rows = []

        def next_measurement_id(self):
            return f"M{len(self.rows) + 1:04d}"

        def add_result(self, result):
            self.rows.append(result)

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    image = np.zeros((7, 7), dtype=float)
    image[2, 2] = 10.0
    image[5, 5] = 8.0
    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 0, "y": 0, "width": 7, "height": 7}, name="all")
    roi_set.add(roi)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = image
    dlg._entries = [SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=7, Ny=7)]
    dlg._idx = 0
    dlg._channel_unit = lambda: (1.0, "nm", "Height")
    dlg._pixel_size_xy_m = lambda: (1e-9, 1e-9)
    table = FakeTable()
    dlg._measurement_table = table
    dlg._status_lbl = FakeStatus()
    controller = ImageMeasurementController(dlg, table)

    controller.detect_feature_maxima_for_roi(
        roi.id,
        settings={
            "threshold_mode": "percentile",
            "threshold_value": 99.0,
            "min_distance_px": 1,
        },
    )

    result = table.rows[0]
    assert result.kind == "feature_maxima"
    assert result.values["n_points"] >= 1
    assert result.context["roi_id"] == roi.id
    assert "Detected" in dlg._status_lbl.text

    controller.copy_feature_points()
    assert "point_id,x_px,y_px" in qapp.clipboard().text()

    csv_path = tmp_path / "points.csv"
    json_path = tmp_path / "points.json"
    paths = iter([str(csv_path), str(json_path)])
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (next(paths), ""),
    )

    controller.export_feature_points_csv()
    controller.export_feature_points_json()

    assert "P0001" in csv_path.read_text(encoding="utf-8")
    assert "x_unit,y_unit,z_unit" in csv_path.read_text(encoding="utf-8")
    assert "probeflow_feature_points" in json_path.read_text(encoding="utf-8")


def test_viewer_feature_maxima_roi_action_uses_panel_settings(qapp):
    from PySide6.QtWidgets import QApplication

    image = np.zeros((7, 7), dtype=float)
    image[2, 2] = 10.0
    image[5, 5] = 8.0
    panel = _FakeFeaturePanel({
        "threshold_mode": "absolute",
        "threshold_value": 9.0,
        "min_distance_px": 1,
    })
    controller, dlg, table, roi = _make_feature_maxima_controller(image, panel)

    controller.detect_feature_maxima_for_roi(roi.id)

    assert len(controller.feature_points) == 1
    assert table.rows[0].values["n_points"] == 1
    assert panel.count == 1
    assert "Detected 1 maxima" in dlg._status_lbl.text

    controller.copy_feature_points()
    assert ",nm,nm,nm," in QApplication.clipboard().text()


def test_viewer_feature_maxima_failure_clears_stale_overlay(qapp):
    image = np.zeros((7, 7), dtype=float)
    image[2, 2] = 10.0
    controller, dlg, table, roi = _make_feature_maxima_controller(image)

    controller.detect_feature_maxima_for_roi(
        roi.id,
        settings={
            "threshold_mode": "absolute",
            "threshold_value": 1.0,
            "min_distance_px": 1,
        },
    )
    assert len(controller.feature_points) == 1
    assert len(dlg._zoom_lbl.feature_points) == 1

    controller.detect_feature_maxima_for_roi(
        roi.id,
        settings={
            "threshold_mode": "not-a-mode",
            "threshold_value": 1.0,
            "min_distance_px": 1,
        },
    )

    assert controller.feature_points == []
    assert dlg._zoom_lbl.feature_points == []
    assert len(table.rows) == 1
    assert "Could not detect maxima" in dlg._status_lbl.text


def test_viewer_feature_maxima_exports_point_mask_and_fft(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    image = np.zeros((8, 8), dtype=float)
    image[2, 2] = 10.0
    image[6, 6] = 8.0
    panel = _FakeFeaturePanel({
        "threshold_mode": "absolute",
        "threshold_value": 1.0,
        "min_distance_px": 1,
        "radius_px": 1,
        "shape_mode": "square",
    })
    controller, dlg, table, roi = _make_feature_maxima_controller(image, panel)
    controller.detect_feature_maxima_for_roi(roi.id)

    controller.compute_point_mask_fft(show_dialog=False)
    assert table.rows[-1].kind == "point_fft"
    assert table.rows[-1].context["shape_mode"] == "square"
    assert table.rows[-1].context["radius_px"] == 1

    mask_path = tmp_path / "mask.csv"
    fft_path = tmp_path / "fft.csv"
    paths = iter([str(mask_path), str(fft_path)])
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (next(paths), ""),
    )

    controller.export_point_mask_csv()
    controller.export_point_fft_csv()

    assert "1" in mask_path.read_text(encoding="utf-8")
    fft_text = fft_path.read_text(encoding="utf-8")
    assert "qx,qy,magnitude,unit" in fft_text
    assert "cycles/nm" in fft_text


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
    dlg._t = {}

    dlg._sync_line_profile_visibility()

    assert dlg._line_profile_panel.visible is False
    assert dlg._line_profile_panel.empty == "Draw a line to show profile."


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
    assert panel._add_measurement_btn.isEnabled()

    panel.show_empty(theme={})

    assert panel._ax.get_title() == ""
    assert not panel._add_measurement_btn.isEnabled()


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
    dlg._processing = {
        "stm_background": {
            "fit_region": "active_roi",
            "fit_roi_id": roi.id,
            "line_statistic": "median",
            "model": "linear",
        }
    }
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
    dlg._processing = {
        "stm_background": {
            "fit_region": "active_roi",
            "fit_roi_id": "missing-id",
            "line_statistic": "median",
            "model": "linear",
        }
    }
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
        processing={
            "align_rows": "median",
            "stm_background": {"model": "linear", "line_statistic": "median"},
        },
    )

    assert captured == {
        "align_rows": "median",
        "stm_background": {"model": "linear", "line_statistic": "median"},
    }


def test_viewer_save_provenance_action_writes_json(qapp, monkeypatch, tmp_path):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    from probeflow.provenance import ProcessingHistory, SourceRecord
    import probeflow.gui._legacy as gui_mod

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    history = ProcessingHistory(SourceRecord(
        source_filename="example.sxm",
        source_path="/tmp/example.sxm",
        source_file_type="Nanonis .sxm",
        channel="Z forward",
        loader_name="Nanonis .sxm reader",
        loader_version="0.0.0",
    ))
    history.append_step(
        operation_id="file_load",
        operation_name="Loaded Nanonis .sxm",
        parameters={},
    )
    dlg._processing_history = history
    out = tmp_path / "example.probeflow.json"
    monkeypatch.setattr(
        gui_mod.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(out), ""),
    )

    dlg._on_save_provenance()

    data = json.loads(out.read_text(encoding="utf-8"))
    ops = [step["operation_id"] for step in data["processing_history"]["steps"]]
    assert ops[-1] == "export_provenance_json"
    assert "Saved provenance" in dlg._status_lbl.text()

    dlg.close()
    dlg.deleteLater()


def test_viewer_save_processed_image_action_dispatches_writer(qapp, monkeypatch, tmp_path):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    import probeflow.gui._legacy as gui_mod

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    out = tmp_path / "processed.csv"
    monkeypatch.setattr(
        gui_mod.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(out), ""),
    )
    calls = []

    class FakeScan:
        def save_csv(self, path, plane_idx=0, provenance=None):
            calls.append(("csv", Path(path), plane_idx, provenance))

    monkeypatch.setattr(
        dlg,
        "_processed_scan_for_export",
        lambda: (FakeScan(), 2),
    )
    monkeypatch.setattr(
        dlg,
        "_processed_export_provenance",
        lambda scan, path, plane_idx: "prov",
    )
    monkeypatch.setattr(
        dlg,
        "_preflight_processed_export_sidecar",
        lambda path: calls.append(("preflight", Path(path))),
    )

    dlg._on_save_processed_image()

    assert calls == [("preflight", out), ("csv", out, 2, "prov")]
    assert "Saved processed image" in dlg._status_lbl.text()

    dlg.close()
    dlg.deleteLater()
