"""Regression tests for the Browse/Viewer processing control ownership."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

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


def test_align_rows_exposes_linear_option(qapp):
    # The backend align_rows supports median/mean/linear; the GUI must offer all
    # three so the documented 'linear' option is actually reachable.
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("viewer_full")
    items = [panel._align_combo.itemText(i) for i in range(panel._align_combo.count())]
    assert items == ["None", "Median", "Mean", "Linear"]

    panel.set_state({"align_rows": "linear"})
    assert panel._align_combo.currentIndex() == 3
    assert panel.state()["align_rows"] == "linear"


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


def test_sidebar_merges_roi_mask_and_orders_tabs(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    try:
        tabs = dlg._sidebar_tabs
        assert [tabs.tabText(i) for i in range(tabs.count())] == [
            "View", "Process", "Measure", "ROI/Mask", "Export",
        ]
        # The old "masks" key aliases onto the merged tab (no separate tab).
        assert dlg._sidebar_tab_indices["masks"] == dlg._sidebar_tab_indices["roi"]
        # ROI section open, Masks section collapsed by default.
        assert dlg._mask_section_btn.isChecked() is False
        # Navigating to "masks" selects the merged tab and reveals the section.
        dlg._show_sidebar_tab("masks")
        assert tabs.tabText(tabs.currentIndex()) == "ROI/Mask"
        assert dlg._mask_section_btn.isChecked() is True
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_viewer_entry_selection_distinguishes_matching_stems():
    from probeflow.gui import SxmFile
    from probeflow.gui.dialogs.image_viewer import _viewer_entry_index

    dat = SxmFile(path=Path("/tmp/scan.dat"), stem="scan")
    sxm = SxmFile(path=Path("/tmp/scan.sxm"), stem="scan")
    equivalent_sxm = SxmFile(path=Path("/tmp/scan.sxm"), stem="scan")

    assert _viewer_entry_index(sxm, [dat, sxm]) == 1
    assert _viewer_entry_index(equivalent_sxm, [dat, sxm]) == 1


def test_threshold_dialog_applies_equal_band_for_constant_image(qapp):
    from probeflow.gui.dialogs.threshold_dialog import ThresholdDialog

    dialog = ThresholdDialog(np.full((3, 3), 1.25))
    applied = []
    dialog.applied.connect(applied.append)
    try:
        dialog._do_apply()
        assert applied == [{"mode": "clip", "lower": 1.25, "upper": 1.25}]
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_format_gaussian_readout_pure():
    from probeflow.gui.processing import format_gaussian_readout

    # Calibrated: σ, FWHM and kernel extent reported in nm (σ_nm = σ_px × px_nm,
    # FWHM = 2.3548·σ, kernel half-width = int(4σ+0.5) to match scipy truncate=4).
    text = format_gaussian_readout(1.0, 0.5)
    assert "σ 1.0 px" in text
    assert "0.5 nm" in text          # σ_nm
    assert "FWHM 1.18 nm" in text     # 2.3548 * 0.5
    assert "kernel ±4 px" in text
    assert "±2 nm" in text            # 4 px × 0.5 nm

    # Uncalibrated: px-only, no nm.
    px_only = format_gaussian_readout(2.0, None)
    assert "nm" not in px_only
    assert "kernel ±8 px" in px_only


def test_viewer_full_smooth_sigma_is_subpixel_float(qapp):
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("viewer_full")

    # Sub-pixel σ now round-trips through the GUI slider.
    panel.set_state({"smooth_sigma": 0.5})
    assert panel.state()["smooth_sigma"] == pytest.approx(0.5)

    # Physical readout reflects the calibration once the pixel size is known.
    panel.set_pixel_size_nm(0.5)
    assert "nm" in panel._smooth_readout_lbl.text()
    assert "σ 0.5 px" in panel._smooth_readout_lbl.text()

    # Without calibration the readout falls back to pixels only.
    panel.set_pixel_size_nm(None)
    assert "nm" not in panel._smooth_readout_lbl.text()


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
    assert "processing.image_operations" in dlg._viewer_command_actions
    assert "measure.particle_statistics" not in dlg._viewer_command_actions

    dlg._processing_panel._align_combo.setCurrentText("Mean")

    assert dlg._processing["align_rows"] == "mean"
    assert refreshes[-1]["align_rows"] == "mean"
    assert dlg._proc_undo_ctrl._undo_stack[-1]["processing"] == {}

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


class _FakeFFTDialog:
    """Captures the kwargs the opener passes; behaves enough like a dialog."""

    last_kwargs: dict = {}

    def __init__(self, *args, **kwargs):
        _FakeFFTDialog.last_kwargs = dict(kwargs)
        _FakeFFTDialog.last_args = tuple(args)

    def show(self):
        pass

    def raise_(self):
        pass

    def activateWindow(self):
        pass


def test_open_fft_viewer_passes_active_area_roi_bounds(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(
        "probeflow.gui.viewer.image_viewer_tools_mixin.FFTViewerDialog",
        _FakeFFTDialog,
    )
    monkeypatch.setattr(ImageViewerDialog, "_track_modeless_child", lambda self, d: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=10, Ny=10)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.ones((10, 10), dtype=float)
    dlg._display_arr = dlg._raw_arr
    dlg._scan_range_m = (10e-9, 10e-9)
    dlg._display_scan_range_m = (10e-9, 10e-9)
    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 4.0, "height": 5.0},
                  name="region A")
    roi_set.add(roi)
    roi_set.set_active(roi.id)
    dlg._image_roi_set = roi_set

    dlg._on_open_fft_viewer()

    kw = _FakeFFTDialog.last_kwargs
    # inclusive bounds: rows 2..6, cols 1..4
    assert kw["roi_bounds_px"] == (2, 6, 1, 4)
    assert kw["roi_id"] == roi.id
    assert kw["roi_name"] == "region A"
    assert callable(kw["new_image_fn"])

    dlg.close()
    dlg.deleteLater()


def test_open_fft_viewer_passes_no_roi_when_none_active(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(
        "probeflow.gui.viewer.image_viewer_tools_mixin.FFTViewerDialog",
        _FakeFFTDialog,
    )
    monkeypatch.setattr(ImageViewerDialog, "_track_modeless_child", lambda self, d: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=10, Ny=10)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.ones((10, 10), dtype=float)
    dlg._display_arr = dlg._raw_arr
    dlg._scan_range_m = (10e-9, 10e-9)
    # Only a line ROI is active — not an area ROI, so no ROI source.
    roi_set = ROISet(image_id="img1")
    line = ROI.new("line", {"x1": 0, "y1": 0, "x2": 5, "y2": 5})
    roi_set.add(line)
    roi_set.set_active(line.id)
    dlg._image_roi_set = roi_set

    dlg._on_open_fft_viewer()

    kw = _FakeFFTDialog.last_kwargs
    assert kw["roi_bounds_px"] is None
    assert kw["roi_id"] is None
    assert kw["roi_name"] is None

    dlg.close()
    dlg.deleteLater()


def test_image_arithmetic_dialog_builds_constant_spec(qapp):
    from probeflow.gui import SxmFile
    from probeflow.gui.dialogs.image_arithmetic import ImageArithmeticDialog

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageArithmeticDialog(
        [entry],
        current_entry_index=0,
        current_plane_idx=0,
        current_shape=(8, 8),
        current_scan_range_m=(10e-9, 10e-9),
        display_scale=1e9,
        display_unit="nm",
    )

    dlg._constant_spin.setValue(5.0)
    dlg.accept()
    spec = dlg.operation_spec()

    assert spec["op"] == "arithmetic"
    assert spec["params"]["operation"] == "add"
    assert spec["params"]["operand_type"] == "constant"
    assert spec["params"]["value_si"] == pytest.approx(5e-9)

    dlg.deleteLater()


def test_image_arithmetic_dialog_builds_generated_pattern_spec(qapp):
    from probeflow.gui import SxmFile
    from probeflow.gui.dialogs.image_arithmetic import ImageArithmeticDialog

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageArithmeticDialog(
        [entry],
        current_entry_index=0,
        current_plane_idx=0,
        current_shape=(8, 8),
        current_scan_range_m=(10e-9, 10e-9),
        display_scale=1e9,
        display_unit="nm",
    )

    dlg._operand_type_combo.setCurrentIndex(
        dlg._operand_type_combo.findData("generated")
    )
    dlg._pattern_combo.setCurrentIndex(dlg._pattern_combo.findData("impulse_grid"))
    dlg._amplitude_spin.setValue(5.0)
    dlg._period_spin.setValue(4)
    dlg.accept()
    spec = dlg.operation_spec()

    assert spec["op"] == "arithmetic"
    assert spec["params"]["operation"] == "add"
    assert spec["params"]["operand_type"] == "generated"
    assert spec["params"]["pattern"] == "impulse_grid"
    assert spec["params"]["amplitude_si"] == pytest.approx(5e-9)
    assert spec["params"]["display_amplitude"] == pytest.approx(5.0)
    assert spec["params"]["display_unit"] == "nm"
    assert spec["params"]["period_px"] == 4

    dlg.deleteLater()


def test_viewer_generated_arithmetic_appends_roi_scoped_step(qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    class FakeImageArithmeticDialog:
        ACTIVE_AREA_ROI = "active_area_roi"

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def exec(self):
            return QDialog.Accepted

        def operation_spec(self):
            return {
                "op": "arithmetic",
                "params": {
                    "operation": "add",
                    "operand_type": "generated",
                    "pattern": "checkerboard",
                    "amplitude_si": 1.0,
                    "period_px": 2,
                },
            }

        def scope(self):
            return self.ACTIVE_AREA_ROI

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)
    monkeypatch.setattr(
        "probeflow.gui.dialogs.image_arithmetic.ImageArithmeticDialog",
        FakeImageArithmeticDialog,
    )

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.ones((8, 8), dtype=float)
    dlg._display_arr = dlg._raw_arr
    dlg._scan_range_m = (8e-9, 8e-9)
    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 2.0, "y": 2.0, "width": 3.0, "height": 3.0})
    roi_set.add(roi)
    roi_set.set_active(roi.id)
    dlg._image_roi_set = roi_set

    dlg._on_open_image_operations()

    op_spec = dlg._processing["arithmetic_ops"][0]
    assert op_spec["op"] == "arithmetic"
    assert op_spec["roi_id"] == roi.id
    assert op_spec["params"]["operand_type"] == "generated"
    assert op_spec["params"]["pattern"] == "checkerboard"
    assert op_spec["params"]["amplitude_si"] == 1.0

    dlg.close()
    dlg.deleteLater()


def test_viewer_image_arithmetic_rejects_non_area_roi_scope(qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    class FakeImageArithmeticDialog:
        ACTIVE_AREA_ROI = "active_area_roi"

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.Accepted

        def operation_spec(self):
            return {
                "op": "arithmetic",
                "params": {
                    "operation": "add",
                    "operand_type": "constant",
                    "value_si": 1.0,
                },
            }

        def scope(self):
            return self.ACTIVE_AREA_ROI

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)
    monkeypatch.setattr(
        "probeflow.gui.dialogs.image_arithmetic.ImageArithmeticDialog",
        FakeImageArithmeticDialog,
    )

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.ones((8, 8), dtype=float)
    dlg._display_arr = dlg._raw_arr
    roi_set = ROISet(image_id="img1")
    roi = ROI.new("point", {"x": 2.0, "y": 2.0})
    roi_set.add(roi)
    roi_set.set_active(roi.id)
    dlg._image_roi_set = roi_set

    dlg._on_open_image_operations()

    assert "arithmetic_ops" not in dlg._processing
    assert "not valid for image arithmetic" in dlg._status_lbl.text()

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

    dlg._on_apply_processing()

    # No quick selection and no ROI set as filter scope → whole-image filter.
    assert dlg._processing["smooth_sigma"] == 1.0
    assert not dlg._processing.get("roi_filter_ops")
    assert "processing_scope" not in dlg._processing
    assert "processing_roi_id" not in dlg._processing

    dlg.close()
    dlg.deleteLater()


def test_viewer_apply_scopes_local_filter_to_roi_filter_scope(qapp, monkeypatch):
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
    # Opt in to ROI scoping (the retired dropdown's replacement).
    dlg._roi_filter_scope_id = roi.id

    dlg._on_apply_processing()

    # ROI-scoped filters are committed durably with frozen geometry, not as a
    # single global processing_scope/_roi_id.
    assert "processing_scope" not in dlg._processing
    assert "processing_roi_id" not in dlg._processing
    committed = dlg._processing["roi_filter_ops"]
    assert len(committed) == 1
    assert committed[0]["op"] == "smooth"
    assert committed[0]["roi_id"] == roi.id
    assert committed[0].get("scope_kind") in (None, "roi")
    assert committed[0]["frozen_geometry"]["kind"] == "rectangle"
    assert "smooth_sigma" not in dlg._processing

    dlg.close()
    dlg.deleteLater()


def test_viewer_apply_scopes_filter_to_quick_selection_as_region(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._processing_panel.set_state({"smooth_sigma": 1.0})
    # A drawn quick selection (not a managed ROI).
    dlg._zoom_lbl.set_selection("rectangle",
                                {"x": 1.0, "y": 1.0, "width": 4.0, "height": 4.0})

    dlg._on_apply_processing()

    committed = dlg._processing["roi_filter_ops"]
    assert len(committed) == 1
    assert committed[0]["scope_kind"] == "region"
    assert "roi_id" not in committed[0]
    assert committed[0]["frozen_geometry"]["kind"] == "rectangle"
    # No ROI was created.
    assert dlg._image_roi_set is None or not dlg._image_roi_set.rois
    # Selection persists for further filtering (ImageJ-style).
    assert dlg._active_quick_selection() is not None

    dlg.close()
    dlg.deleteLater()


def test_viewer_promote_selection_creates_roi_and_clears_marquee(qapp, monkeypatch):
    from probeflow.core.roi import ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._image_roi_set = ROISet(image_id="img1")
    dlg._zoom_lbl.set_selection("rectangle",
                                {"x": 1.0, "y": 1.0, "width": 4.0, "height": 4.0})

    dlg._promote_selection_to_roi()

    assert len(dlg._image_roi_set.rois) == 1
    assert dlg._image_roi_set.rois[0].kind == "rectangle"
    assert dlg._active_quick_selection() is None  # marquee consumed

    dlg.close()
    dlg.deleteLater()


def test_viewer_apply_rejects_local_filter_for_non_area_roi_scope(qapp, monkeypatch):
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
    dlg._roi_filter_scope_id = roi.id

    dlg._on_apply_processing()

    assert "not an area ROI" in dlg._status_lbl.text()
    assert not dlg._processing.get("roi_filter_ops")

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

    dlg._on_undo_processing()
    assert "set_zero_plane_points" not in dlg._processing
    assert dlg._zero_ctrl.points == []
    assert dlg._set_zero_plane_btn.isChecked() is False

    dlg._on_redo_processing()
    assert dlg._processing["set_zero_plane_points"] == [(0, 0), (4, 4), (9, 9)]
    assert dlg._zero_ctrl.points == [(0, 0), (4, 4), (9, 9)]

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


def test_line_profile_panel_scales_to_negative_profile_range(qapp):
    from probeflow.gui.viewer.widgets import LineProfilePanel

    panel = LineProfilePanel()
    panel.plot_profile(
        np.arange(4, dtype=np.float64),
        np.array([-138.6, -138.4, -138.5, -138.3], dtype=np.float64),
        y_label="Z [nm]",
        theme={},
    )

    ymin, ymax = panel._ax.get_ylim()
    assert ymin < -138.6
    assert ymax > -138.3
    assert ymax < -130.0

    panel.deleteLater()


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
    assert result.values["height_difference"] == pytest.approx(4.0)
    assert "height_peak_to_peak" not in result.values
    assert result.values["x2"] == pytest.approx(4.0)
    assert result.values["y2"] == pytest.approx(2.0)
    assert result.context["roi_id"] == line.id
    assert result.context["roi_name"] == "step"
    assert result.y_unit == "nm"


def test_periodicity_secondary_exports_include_method_context(qapp, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import ImageViewerDialog, SxmFile
    from probeflow.gui.viewer import ImageMeasurementController

    class FakeLinePeriodicityPanel:
        def __init__(self):
            self.findPeriodicityRequested = _FakeSignal()
            self.copyResultRequested = _FakeSignal()
            self.exportProfileCsvRequested = _FakeSignal()
            self.saveStructureRequested = _FakeSignal()
            self.result = None

        def settings(self):
            return {
                "method": "fft",
                "background": "none",
                "smoothing": "none",
                "width_px": 3.0,
                "min_period_m": 5e-9,
                "max_period_m": 20e-9,
            }

        def set_result(self, result):
            self.result = result

        def show_message(self, _message):
            return None

    x = np.arange(128, dtype=np.float64)
    image = np.tile(np.sin(2.0 * np.pi * x / 12.0), (8, 1))
    roi_set = ROISet(image_id="img1")
    line = ROI.new("line", {"x1": 0.0, "y1": 4.0, "x2": 127.0, "y2": 4.0}, name="wave")
    roi_set.add(line)
    roi_set.set_active(line.id)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._image_roi_set = roi_set
    dlg._display_arr = image
    dlg._entries = [SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=128, Ny=8)]
    dlg._idx = 0
    dlg._channel_unit = lambda: (1.0, "nm", "Height")
    dlg._pixel_size_xy_m = lambda: (1e-9, 1e-9)
    dlg._active_line_roi_id = lambda: line.id
    dlg._on_roi_line_profile = lambda _roi_id: None
    dlg._status_lbl = _FakeStatus()

    table = _FakeMeasurementTable()
    panel = FakeLinePeriodicityPanel()
    controller = ImageMeasurementController(dlg, table, line_periodicity_panel=panel)
    controller._show_periodicity_plot_dialog = lambda *_args, **_kwargs: None

    controller.find_periodicity_for_active_line_roi()
    controller.copy_periodicity_result()
    copied = qapp.clipboard().text()

    assert "Method: fft" in copied
    assert "ROI: wave" in copied
    assert "Source: example:Height" in copied
    assert "Width: 3 px" in copied
    assert "Period bounds: 5 nm to 20 nm" in copied
    assert "Quality:" in copied

    csv_path = tmp_path / "periodicity_profile.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(csv_path), ""),
    )
    controller.export_periodicity_profile_csv()

    text = csv_path.read_text(encoding="utf-8")
    assert "# export_type,probeflow_line_periodicity_profile" in text
    assert "# roi_name,wave" in text
    assert "# method,fft" in text
    assert "# width_px,3" in text
    assert "# min_period_m,5e-09" in text
    assert "s_m,s_nm,z_raw,z_processed" in text


def test_periodicity_result_can_be_saved_as_known_structure(monkeypatch):
    from probeflow.gui.viewer import ImageMeasurementController

    class FakeResult:
        period_m = 2.46e-10

    saved = []
    monkeypatch.setattr(
        "probeflow.gui.viewer.image_measurements.load_known_structures",
        lambda: [],
    )
    monkeypatch.setattr(
        "probeflow.gui.viewer.image_measurements.save_known_structures",
        lambda structures: saved.extend(structures),
    )

    dlg = type("FakeViewer", (), {})()
    dlg._status_lbl = _FakeStatus()
    controller = ImageMeasurementController(dlg, _FakeMeasurementTable())
    controller._last_periodicity_result = FakeResult()
    controller._last_periodicity_settings = {"roi_name": "row"}

    controller.save_periodicity_as_known_structure("row spacing", "hexagonal")

    assert saved
    assert saved[0].name == "row spacing"
    assert saved[0].symmetry == "hexagonal"
    assert saved[0].a_nm == pytest.approx(0.246)
    assert "Saved known structure" in dlg._status_lbl.text


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


def test_viewer_point_source_collectors_include_measure_tab_feature_points(qapp):

    from probeflow.gui import ImageViewerDialog
    from probeflow.measurements.models import FeaturePoint

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._pixel_size_xy_m = lambda: (2e-9, 3e-9)
    dlg._image_roi_set = None
    dlg._image_measurements = SimpleNamespace(
        feature_points=[
            FeaturePoint(
                point_id="P0001",
                x_px=4.0,
                y_px=5.0,
                x_phys=4.0,
                y_phys=5.0,
                z_value=1.0,
                channel="Height",
                source_label="example",
            ),
        ]
    )

    px_sources = dlg._collect_point_sources_px()
    m_sources = dlg._collect_point_sources_m()

    assert "Detected feature maxima" in px_sources
    np.testing.assert_allclose(px_sources["Detected feature maxima"], [[4.0, 5.0]])
    np.testing.assert_allclose(m_sources["Detected feature maxima"], [[8e-9, 15e-9]])




def test_legacy_measurement_context_survives_dock_conversion(qapp):
    # The legacy ``probeflow.analysis.measurements.MeasurementResult``
    # dataclass was removed in arch-backend #1 (2026-05-28); a
    # SimpleNamespace stub exercises the duck-typed adapter path.
    from probeflow.gui import ImageViewerDialog

    legacy = SimpleNamespace(
        id="M?",
        kind="pair_corr",
        source="scan:Height",
        channel="Height",
        roi_id=None,
        summary="Points: 25",
        values={"n_points": 25, "quality": "good"},
        units={},
        notes="",
        context={
            "source_path": "/tmp/scan.sxm",
            "point_source": "Detected feature maxima",
            "edge_correction": "not_applied",
            "bin_width_m": 5e-10,
        },
    )
    dlg = ImageViewerDialog.__new__(ImageViewerDialog)

    result = dlg._to_dock_result(legacy, "M0001")

    assert result.source_path == "/tmp/scan.sxm"
    assert result.context["point_source"] == "Detected feature maxima"
    assert result.context["edge_correction"] == "not_applied"
    assert result.context["bin_width_m"] == pytest.approx(5e-10)


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

    mask_text = mask_path.read_text(encoding="utf-8")
    assert "# export_type,probeflow_feature_point_mask" in mask_text
    assert "# source_path,example.sxm" in mask_text
    assert "# threshold_mode,absolute" in mask_text
    assert "# radius_px,1" in mask_text
    assert "1" in mask_text
    fft_text = fft_path.read_text(encoding="utf-8")
    assert "# export_type,probeflow_point_mask_fft" in fft_text
    assert "# source_path,example.sxm" in fft_text
    assert "# mask_shape_y,8" in fft_text
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
    import probeflow.gui.dialogs.image_viewer as _iv_mod

    roi_set = ROISet(image_id="img1")
    roi = ROI.new("rectangle", {"x": 0.0, "y": 0.0, "width": 2.0, "height": 2.0})
    roi_set.add(roi)
    seen = {}

    def fake_apply(arr, state, passed_roi_set, *, mask_set=None, scan_range_m=None):
        seen["roi_set"] = passed_roi_set
        return arr + 1.0, scan_range_m

    monkeypatch.setattr(_iv_mod, "apply_processing_state_with_calibration", fake_apply)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._display_arr = None
    dlg._raw_arr = np.zeros((2, 2))
    dlg._scan_range_m = None
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
    import probeflow.gui.dialogs.image_viewer as _iv_mod

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    called = []

    def fake_apply(arr, state, passed_roi_set, *, mask_set=None, scan_range_m=None):
        called.append(True)
        return arr + 1.0, scan_range_m

    monkeypatch.setattr(_iv_mod, "apply_processing_state_with_calibration", fake_apply)

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


def test_viewer_refresh_display_array_blocks_export_after_processing_error(qapp, monkeypatch):
    from probeflow.core.roi import ROISet
    from probeflow.gui import ImageViewerDialog
    import probeflow.gui.dialogs.image_viewer as _iv_mod

    class FakeStatus:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    def fail_apply(arr, state, passed_roi_set, *, mask_set=None, scan_range_m=None):
        raise RuntimeError("bad processing setting")

    monkeypatch.setattr(_iv_mod, "apply_processing_state_with_calibration", fail_apply)

    dlg = ImageViewerDialog.__new__(ImageViewerDialog)
    dlg._display_arr = None
    dlg._raw_arr = np.zeros((2, 2))
    dlg._scan_range_m = None
    dlg._processing = {"smooth_sigma": 1.0}
    dlg._image_roi_set = ROISet(image_id="img1")
    dlg._reset_zoom_on_next_pixmap = False
    dlg._status_lbl = FakeStatus()
    dlg._processing_roi_error = ""
    dlg._processing_error = ""

    dlg._refresh_display_array()

    np.testing.assert_allclose(dlg._display_arr, np.zeros((2, 2)))
    assert "bad processing setting" in dlg._processing_error
    assert dlg._assert_exportable_processing() is False
    assert "Export blocked" in dlg._status_lbl.text


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
    import probeflow.gui.compat as gui_mod

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
    import probeflow.gui.compat as gui_mod
    import probeflow.gui.viewer.image_viewer_processing_export_mixin as export_mixin

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    out = tmp_path / "processed.csv"
    monkeypatch.setattr(
        gui_mod.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(out), ""),
    )

    class FakeScan:
        processing_state = type("PS", (), {"steps": []})()

    saved_calls = []

    def fake_save_processed_image(scan, plane_idx, path, **kwargs):
        saved_calls.append((plane_idx, path))
        return f"Saved processed image -> {path.name}"

    monkeypatch.setattr(
        dlg,
        "_processed_scan_for_export",
        lambda: (FakeScan(), 2),
    )
    monkeypatch.setattr(export_mixin, "save_processed_image", fake_save_processed_image)

    dlg._on_save_processed_image()

    assert saved_calls == [(2, out)]
    assert "Saved processed image" in dlg._status_lbl.text()

    dlg.close()
    dlg.deleteLater()


def test_viewer_export_tab_shows_summary_and_format_controls(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    long_stem = "createc_scan_overview_240nm_pos_with_a_really_long_filename"
    entry = SxmFile(
        path=Path("/tmp/example.sxm"),
        stem=long_stem,
        Nx=7,
        Ny=5,
        bias_mv=-50.0,
        current_pa=1500.0,
        source_format="dat",
    )
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.zeros((5, 7), dtype=float)
    dlg._display_arr = None
    dlg._processing = {
        "geometric_ops": [
            {"op": "quantize_bit_depth", "params": {"bits": 16}},
        ],
    }
    dlg._export_png_file_lbl.resize(80, dlg._export_png_file_lbl.height())
    dlg._update_export_summary()
    full_name = f"{long_stem}_viewer.png"

    assert dlg._export_png_size_lbl.text() == "7x5 px"
    assert dlg._export_png_file_lbl.text() != full_name
    assert dlg._export_png_file_lbl.toolTip() == full_name
    assert dlg._export_bias_lbl.text() == "-50 mV"
    assert dlg._export_current_lbl.text() == "1.5 nA"
    assert dlg._export_precision_lbl.text() == "16-bit quantized"
    assert dlg._export_provenance_chk.isChecked()
    assert dlg._export_scalebar_chk.isChecked()
    assert dlg._save_png_btn.text().endswith("Save PNG copy…")
    assert dlg._save_pdf_btn.text() == "Save PDF copy…"
    assert dlg._save_sxm_btn.isEnabled()
    assert dlg._save_gwy_btn.text() == "Save GWY copy…"

    dlg.close()
    dlg.deleteLater()


def test_viewer_export_tab_disables_sxm_for_sm4_sources(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(
        path=Path("/tmp/example.sm4"),
        stem="example",
        Nx=8,
        Ny=8,
        source_format="sm4",
    )
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._update_export_summary()

    assert not dlg._save_sxm_btn.isEnabled()
    assert "Createc .dat" in dlg._save_sxm_btn.toolTip()

    dlg.close()
    dlg.deleteLater()


def test_viewer_save_png_action_uses_export_checkboxes(qapp, monkeypatch, tmp_path):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    import probeflow.gui.compat as gui_mod
    import probeflow.gui.viewer.image_viewer_processing_export_mixin as export_mixin

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._display_arr = np.ones((8, 8), dtype=float)
    dlg._export_provenance_chk.setChecked(False)
    dlg._export_scalebar_chk.setChecked(False)
    out = tmp_path / "current_view.png"
    monkeypatch.setattr(
        gui_mod.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(out), ""),
    )

    saved_calls = []

    def fake_save_viewer_png(*args, **kwargs):
        saved_calls.append((args, kwargs))
        return f"Saved -> {Path(args[1]).name}"

    monkeypatch.setattr(dlg, "_assert_exportable_processing", lambda: True)
    monkeypatch.setattr(export_mixin, "save_viewer_png", fake_save_viewer_png)

    dlg._on_save_png()

    assert saved_calls[0][1]["add_scalebar"] is False
    assert saved_calls[0][1]["include_provenance"] is False
    assert saved_calls[0][1]["scan_range_m"] == dlg._processed_scan_range_m()
    assert "Saved" in dlg._status_lbl.text()

    dlg.close()
    dlg.deleteLater()


def test_viewer_save_pdf_action_dispatches_current_view_writer(qapp, monkeypatch, tmp_path):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    import probeflow.gui.compat as gui_mod
    import probeflow.gui.viewer.image_viewer_processing_export_mixin as export_mixin

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._display_arr = np.ones((8, 8), dtype=float)
    dlg._export_provenance_chk.setChecked(False)
    dlg._export_scalebar_chk.setChecked(False)
    out_without_suffix = tmp_path / "current_view"
    captured_dialog = {}
    monkeypatch.setattr(
        gui_mod.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (
            captured_dialog.setdefault("default", args[2]) and str(out_without_suffix),
            "",
        ),
    )

    class FakeScan:
        processing_state = type("PS", (), {"steps": []})()

    saved_calls = []

    def fake_save_processed_image(scan, plane_idx, path, **kwargs):
        saved_calls.append((plane_idx, path, kwargs))
        return f"Saved processed image -> {path.name}"

    monkeypatch.setattr(dlg, "_assert_exportable_processing", lambda: True)
    monkeypatch.setattr(dlg, "_processed_scan_for_export", lambda: (FakeScan(), 1))
    monkeypatch.setattr(export_mixin, "save_processed_image", fake_save_processed_image)

    dlg._on_save_pdf()

    assert captured_dialog["default"].endswith("example_viewer.pdf")
    assert saved_calls[0][0] == 1
    assert saved_calls[0][1] == out_without_suffix.with_suffix(".pdf")
    assert saved_calls[0][2]["include_provenance"] is False
    assert saved_calls[0][2]["add_scalebar"] is False
    assert saved_calls[0][2]["display_settings"] is None
    assert "Saved processed image" in dlg._status_lbl.text()

    dlg.close()
    dlg.deleteLater()


@pytest.mark.parametrize(
    "method_name,suffix",
    [
        ("_on_save_sxm", ".sxm"),
        ("_on_save_gwy", ".gwy"),
    ],
)
def test_viewer_direct_data_export_actions_dispatch_current_view_writer(
    qapp,
    monkeypatch,
    tmp_path,
    method_name,
    suffix,
):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    import probeflow.gui.compat as gui_mod
    import probeflow.gui.viewer.image_viewer_processing_export_mixin as export_mixin

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(
        path=Path("/tmp/example.dat"),
        stem="example",
        Nx=8,
        Ny=8,
        source_format="dat",
    )
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._display_arr = np.ones((8, 8), dtype=float)
    out_without_suffix = tmp_path / f"current_view_{suffix.lstrip('.')}"
    captured_dialog = {}
    monkeypatch.setattr(
        gui_mod.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (
            captured_dialog.setdefault("default", args[2]) and str(out_without_suffix),
            "",
        ),
    )

    class FakeScan:
        processing_state = type("PS", (), {"steps": []})()

    saved_calls = []

    def fake_save_processed_image(scan, plane_idx, path, **kwargs):
        saved_calls.append((plane_idx, path, kwargs))
        return f"Saved processed image -> {path.name}"

    monkeypatch.setattr(dlg, "_assert_exportable_processing", lambda: True)
    monkeypatch.setattr(dlg, "_processed_scan_for_export", lambda: (FakeScan(), 1))
    monkeypatch.setattr(export_mixin, "save_processed_image", fake_save_processed_image)

    getattr(dlg, method_name)()

    assert captured_dialog["default"].endswith(f"example_viewer{suffix}")
    assert saved_calls[0][0] == 1
    assert saved_calls[0][1] == out_without_suffix.with_suffix(suffix)
    assert saved_calls[0][2]["include_provenance"] is True
    assert "Saved processed image" in dlg._status_lbl.text()

    dlg.close()
    dlg.deleteLater()


def test_tv_attribution_marks_feature_as_experimental_and_links_upstream():
    from probeflow.gui.tv import AISURF_PROJECT_URL, TV_ATTRIBUTION_HTML

    assert AISURF_PROJECT_URL in TV_ATTRIBUTION_HTML
    assert "adapted from" in TV_ATTRIBUTION_HTML
    assert "not been rigorously validated" in TV_ATTRIBUTION_HTML
    assert "testing purposes" in TV_ATTRIBUTION_HTML


def test_tv_load_from_browse_reuses_processed_scan_helper(qapp, monkeypatch):
    from probeflow.gui import SxmFile
    from probeflow.gui.app import ProbeFlowWindow

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    processed = np.ones((4, 4), dtype=float) * 7.0
    statuses = []
    loaded = []

    class FakeGrid:
        def get_primary_entry(self):
            return entry

    class FakeSidebar:
        def plane_index(self):
            return 3

        def set_running(self, _running):
            pass

        def set_status(self, text):
            statuses.append(text)

    class FakePanel:
        def load_entry(self, *args):
            loaded.append(args)

    win = ProbeFlowWindow.__new__(ProbeFlowWindow)
    win._grid = FakeGrid()
    win._tv_sidebar = FakeSidebar()
    win._tv_panel = FakePanel()
    win._status_bar = type("Status", (), {"showMessage": lambda self, text: None})()

    def fake_load_scan_plane_for_analysis(self, got_entry, plane_idx):
        assert got_entry is entry
        assert plane_idx == 3
        # Helper now also returns the source Scan (for export provenance); the
        # TV path ignores it. None is fine here.
        return processed, 2.5e-10, 2.0e-10, 3.0e-10, 1, None

    monkeypatch.setattr(
        ProbeFlowWindow,
        "_load_scan_plane_for_analysis",
        fake_load_scan_plane_for_analysis,
    )

    ProbeFlowWindow._on_tv_load_from_browse(win)

    assert len(loaded) == 1
    assert loaded[0][0] is entry
    assert loaded[0][1] == 1
    assert loaded[0][2] is processed
    assert loaded[0][3] == 2.5e-10
    assert "Loaded example" in statuses[-1]


def test_tv_completion_is_ignored_after_a_different_scan_is_loaded():
    """A late worker must not attach scan A's pixels to scan B."""
    from probeflow.gui.app import ProbeFlowWindow

    old = np.zeros((3, 3))
    current = np.ones((3, 3))

    class FakePanel:
        result = None

        def current_array(self): return current
        def current_plane_idx(self): return 2
        def set_denoised(self, result): self.result = result

    class FakeSidebar:
        running_calls = []

        def set_running(self, running): self.running_calls.append(running)
        def set_status(self, _text): pass

    win = ProbeFlowWindow.__new__(ProbeFlowWindow)
    win._tv_panel = FakePanel()
    win._tv_sidebar = FakeSidebar()
    win._tv_run_generation = 8

    ProbeFlowWindow._on_tv_finished(
        win, np.full((3, 3), 9.0), "", (7, id(old), 0)
    )

    assert win._tv_panel.result is None
    assert win._tv_sidebar.running_calls == []


def test_tv_completion_is_accepted_for_current_run():
    from probeflow.gui.app import ProbeFlowWindow

    current = np.ones((3, 3))
    result = np.full((3, 3), 2.0)

    class FakePanel:
        denoised = None

        def current_array(self): return current
        def current_plane_idx(self): return 1
        def set_denoised(self, value): self.denoised = value

    class FakeSidebar:
        running = True

        def set_running(self, running): self.running = running
        def set_status(self, _text): pass

    win = ProbeFlowWindow.__new__(ProbeFlowWindow)
    win._tv_panel = FakePanel()
    win._tv_sidebar = FakeSidebar()
    win._tv_run_generation = 4

    ProbeFlowWindow._on_tv_finished(win, result, "", (4, id(current), 1))

    assert win._tv_panel.denoised is result
    assert win._tv_sidebar.running is False


def test_roi_histogram_uses_only_finite_pixels(monkeypatch, qapp):
    from PySide6.QtWidgets import QMessageBox
    from probeflow.gui.viewer.roi_analysis import show_roi_histogram

    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, text: messages.append((title, text)),
    )
    roi = SimpleNamespace(
        name="mixed",
        to_mask=lambda _shape: np.ones((2, 2), dtype=bool),
    )

    show_roi_histogram(
        roi,
        np.array([[1.0, np.nan], [3.0, np.inf]]),
        lambda: (1.0, "m", "Height"),
    )

    assert "2 finite" in messages[0][1]
    assert "Min:  1 m" in messages[0][1]
    assert "Max:  3 m" in messages[0][1]
    assert "Mean: 2 m" in messages[0][1]


def test_roi_manager_disables_area_algebra_for_line_rois(qapp):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui.roi_manager_dock import ROIManagerPanel

    roi_set = ROISet(image_id="img")
    line = ROI.new("line", {"x1": 0, "y1": 0, "x2": 5, "y2": 5})
    rect = ROI.new("rectangle", {"x": 1, "y": 1, "width": 3, "height": 3})
    roi_set.add(line)
    roi_set.add(rect)
    panel = ROIManagerPanel(
        lambda: roi_set,
        {"get_image_shape": lambda: (10, 10)},
    )
    panel.refresh(roi_set)

    panel._list.item(0).setSelected(True)
    panel._on_item_selection_changed()
    assert panel._invert_btn.isEnabled() is False
    panel._on_invert()
    assert len(roi_set.rois) == 2

    panel._list.item(1).setSelected(True)
    panel._on_item_selection_changed()
    assert panel._combine_btn.isEnabled() is False
    panel._on_combine()
    assert len(roi_set.rois) == 2

    panel.deleteLater()


def test_open_viewer_tracking_reaps_destroyed_dialog():
    from probeflow.gui.app import ProbeFlowWindow

    class FakeSignal:
        def __init__(self):
            self._callbacks = []

        def connect(self, callback):
            self._callbacks.append(callback)

        def emit(self):
            for callback in list(self._callbacks):
                callback(None)

    class FakeDialog:
        def __init__(self):
            self.destroyed = FakeSignal()

    win = ProbeFlowWindow.__new__(ProbeFlowWindow)
    win._open_viewers = set()
    dlg = FakeDialog()

    ProbeFlowWindow._track_open_viewer(win, dlg)
    assert dlg in win._open_viewers

    dlg.destroyed.emit()
    assert dlg not in win._open_viewers


def test_measure_tab_menu_lists_feature_maxima_and_point_statistics():
    """The curated Measure-tab menu must expose the point-pattern tools."""
    from probeflow.gui.widgets.image_measurements_panel import ImageMeasurementsPanel

    keys = {
        key
        for _group, tools in ImageMeasurementsPanel._TOOL_GROUPS
        for _label, key, _kind in tools
    }
    assert "feature_maxima" in keys
    assert "pair_correlation" in keys
    # Point statistics opens as a dialog via the existing request signal.
    assert (
        ImageMeasurementsPanel._DIALOG_SIGNALS["pair_correlation"]
        == "pairCorrelationRequested"
    )
