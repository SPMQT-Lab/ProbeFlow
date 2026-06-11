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


# ── Mask alignment seams (2026-06-11 review: mask-from-edge-detection) ────────

def test_mask_created_is_aligned_and_roi_restricted(qapp):
    """The emitted mask must have the input array's exact shape and contain
    edges only inside the ROI restriction — the doc's 'mask output shifted
    relative to displayed image' / 'masks align with the displayed region'
    checks."""
    import numpy as np

    roi = np.zeros((N, N), dtype=bool)
    roi[:, : N // 2 + 4] = True  # includes the step edge at N//2

    dlg = _dialog(qapp, active_roi_mask=roi)
    captured = []
    dlg.mask_created.connect(captured.append)
    dlg._emit_mask()

    assert len(captured) == 1
    mask = captured[0]
    assert mask.shape == (N, N), "mask raster shape differs from input array"
    assert mask.data.any(), "step edge inside the ROI was not detected"
    assert not (mask.data & ~roi).any(), "mask leaked outside the ROI"
    dlg.deleteLater()


def _mask_host(tmp_path):
    """Minimal viewer stand-in for the mask-mixin handlers."""
    import numpy as np
    from types import SimpleNamespace

    from probeflow.gui.viewer.image_viewer_mask_mixin import ImageViewerMaskMixin

    class Host(ImageViewerMaskMixin):
        def __init__(self):
            self._entries = [SimpleNamespace(path=tmp_path / "scan_0001.sxm",
                                             stem="scan_0001")]
            self._idx = 0
            self._image_mask_set = None
            self._display_arr = np.zeros((16, 16))
            self._raw_arr = None
            self.overlays: list = []
            self.cleared = 0
            self._zoom_lbl = SimpleNamespace(
                set_mask_overlay=lambda data, **kw: self.overlays.append(data),
                clear_mask_overlay=lambda: setattr(
                    self, "cleared", self.cleared + 1),
            )
            self.statuses: list[str] = []
            self._status_lbl = SimpleNamespace(setText=self.statuses.append)

        def _channel_unit(self):
            return 1.0, "m", "Z forward"

    return Host()


def test_edge_mask_handler_stamps_context_activates_and_persists(qapp, tmp_path):
    import numpy as np

    from probeflow.core.mask import ImageMask

    host = _mask_host(tmp_path)
    raster = np.zeros((16, 16), dtype=bool)
    raster[4:8, 4:8] = True

    host._on_edge_mask_created(ImageMask.new(raster, name="edges"))

    ms = host._image_mask_set
    assert ms is not None and len(ms.masks) == 1
    mask = ms.masks[0]
    assert ms.active() is mask
    # Source context recorded so a processed-channel mask is not mistaken
    # for raw-data-derived later.
    assert mask.parameters["source_channel"] == "Z forward"
    assert mask.parameters["data_basis"] == "processed_image"
    assert mask.parameters["source_path"].endswith("scan_0001.sxm")
    # Persisted to the sidecar and shown as the overlay.
    assert (tmp_path / "scan_0001.masks.json").exists()
    assert len(host.overlays) == 1


def test_active_mask_array_guards_shape_mismatch(qapp, tmp_path):
    """A mask whose raster no longer matches the displayed array (e.g. after
    a shape-changing step) must read as None — never applied misaligned."""
    import numpy as np

    from probeflow.core.mask import ImageMask, MaskSet

    host = _mask_host(tmp_path)
    host._image_mask_set = MaskSet(image_id="img")
    mask = ImageMask.new(np.ones((16, 16), dtype=bool), name="m")
    host._image_mask_set.add(mask)
    host._image_mask_set.set_active(mask.id)

    assert host._active_mask_array() is not None
    host._display_arr = np.zeros((32, 32))  # shape-changing step happened
    assert host._active_mask_array() is None
    host._refresh_mask_overlay()
    assert host.cleared >= 1, "stale-shape mask left visible as overlay"


def test_channel_mismatch_warning_is_surfaced(qapp, tmp_path):
    import numpy as np

    from probeflow.core.mask import ImageMask, MaskSet

    host = _mask_host(tmp_path)
    host._image_mask_set = MaskSet(image_id="img")
    mask = ImageMask.new(np.ones((16, 16), dtype=bool), name="m",
                         parameters={"source_channel": "Current forward"})
    host._image_mask_set.add(mask)
    host._image_mask_set.set_active(mask.id)

    host._refresh_mask_overlay()

    assert any("was made on channel" in s for s in host.statuses)
