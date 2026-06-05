"""Offscreen-Qt tests for the Advanced Edge Detection dialog."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

N = 64


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


def _step_image() -> np.ndarray:
    img = np.zeros((N, N), dtype=np.float64)
    img[:, N // 2:] = 1.0
    return img


def _dialog(qapp, **kw):
    from probeflow.gui.dialogs.edge_detection import EdgeDetectionDialog
    dlg = EdgeDetectionDialog(_step_image(), **kw)
    dlg._debounce.stop()  # run recompute synchronously in tests
    dlg._recompute()
    return dlg


def test_dialog_builds_and_previews_canny(qapp):
    dlg = _dialog(qapp)
    assert dlg._result is not None
    assert dlg._result.method == "canny"
    assert dlg._result.edge_mask.any()
    dlg.deleteLater()


def test_method_switch_swaps_panel_and_recomputes(qapp):
    dlg = _dialog(qapp)
    dlg._method_combo.setCurrentIndex(1)  # Sobel / Scharr
    dlg._recompute()
    assert dlg._stack.currentIndex() == 1
    assert dlg._result.method == "sobel"
    assert dlg._result.gradient_magnitude is not None
    dlg.deleteLater()


def test_preset_fills_canny_fields(qapp):
    dlg = _dialog(qapp)
    dlg._canny_preset.setCurrentText("Step edges / islands")
    assert dlg._canny_sigma.value() == pytest.approx(2.0)
    assert dlg._canny_low.value() == pytest.approx(60.0)
    assert dlg._canny_high.value() == pytest.approx(85.0)
    dlg.deleteLater()


def test_mask_created_signal_emits_image_mask(qapp):
    dlg = _dialog(qapp)
    captured = []
    dlg.mask_created.connect(captured.append)
    dlg._emit_mask()
    assert len(captured) == 1
    from probeflow.core.mask import ImageMask
    assert isinstance(captured[0], ImageMask)
    assert captured[0].count() > 0
    dlg.deleteLater()


def test_overlay_signal_emits_result(qapp):
    dlg = _dialog(qapp)
    captured = []
    dlg.overlay_requested.connect(captured.append)
    dlg._emit_overlay()
    assert len(captured) == 1
    assert captured[0] is dlg._result
    dlg.deleteLater()


def test_image_created_signal_carries_provenance(qapp):
    dlg = _dialog(qapp)
    captured = []
    dlg.image_created.connect(lambda arr, prov: captured.append((arr, prov)))
    dlg._emit_image()
    assert len(captured) == 1
    arr, prov = captured[0]
    assert arr.shape == (N, N)
    assert prov["op"] == "advanced_edge_detection"
    assert prov["method"] == "canny"
    dlg.deleteLater()


def test_rois_created_from_canny_mask(qapp):
    # A filled disk yields a closed region that converts to a polygon ROI.
    from probeflow.gui.dialogs.edge_detection import EdgeDetectionDialog
    yy, xx = np.mgrid[0:N, 0:N]
    disk = (((yy - 32) ** 2 + (xx - 32) ** 2) <= 12 ** 2).astype(np.float64)
    dlg = EdgeDetectionDialog(disk)
    dlg._debounce.stop()
    dlg._method_combo.setCurrentIndex(1)
    dlg._grad_threshold_cb.setChecked(True)
    dlg._recompute()
    dlg._roi_min_size.setValue(0)
    captured = []
    dlg.rois_created.connect(captured.append)
    dlg._emit_rois()
    # Either ROIs were produced, or an info box (no closed region) — both valid;
    # for a thresholded disk gradient we expect at least one ROI.
    assert captured and len(captured[0]) >= 1
    dlg.deleteLater()


def test_sobel_without_threshold_has_no_mask(qapp):
    dlg = _dialog(qapp)
    dlg._method_combo.setCurrentIndex(1)
    dlg._recompute()
    assert dlg._result.edge_mask is None
    dlg.deleteLater()
