"""Scale/shear handling for ROIs, masks, and quick selections (doc Test 4).

The 2026-06-11 review found scale_image and shear bypassed the overlay
transform path entirely: ROI vector geometry rasterises against any array
shape, so after a resample every ROI was silently mislocated (no shape guard
saves vector geometry, unlike masks), and shear didn't even warn. Policy now:

* ``scale_image`` — ROIs and the quick selection scale exactly (vector
  geometry is lossless under resampling); raster masks are invalidated.
* ``shear`` — ROIs, masks, and the selection are invalidated, matching the
  rotate_arbitrary precedent (a rectangle cannot represent a sheared shape).

Also pins the ImageMask.transform doc/code mismatch fix: scale/shear/undistort/affine
now return None (invalidate) as documented instead of raising ValueError.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from probeflow.core.mask import ImageMask, MaskSet
from probeflow.core.roi import ROI, ROISet
from probeflow.processing.geometry import shear

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SHAPE = (64, 64)  # (Ny, Nx)
SCALE = {"new_width": 128, "new_height": 96}  # sx=2.0, sy=1.5
SHEAR = {"shear_x": 0.1, "shear_y": 0.0}


# ── Core: ROI.transform ───────────────────────────────────────────────────────

class TestRoiScaleTransform:
    @pytest.mark.parametrize("kind,geometry,expected", [
        ("rectangle", {"x": 4, "y": 6, "width": 10, "height": 8},
         {"x": 8.0, "y": 9.0, "width": 20.0, "height": 12.0}),
        ("ellipse", {"cx": 20, "cy": 22, "rx": 8, "ry": 5},
         {"cx": 40.0, "cy": 33.0, "rx": 16.0, "ry": 7.5}),
        ("polygon", {"vertices": [[5, 4], [25, 8], [18, 24]]},
         {"vertices": [[10.0, 6.0], [50.0, 12.0], [36.0, 36.0]]}),
        ("freehand", {"vertices": [[2, 2], [10, 2], [10, 10]]},
         {"vertices": [[4.0, 3.0], [20.0, 3.0], [20.0, 15.0]]}),
        ("line", {"x1": 2, "y1": 4, "x2": 30, "y2": 40},
         {"x1": 4.0, "y1": 6.0, "x2": 60.0, "y2": 60.0}),
        ("point", {"x": 16, "y": 16}, {"x": 32.0, "y": 24.0}),
    ])
    def test_scale_is_exact_per_axis(self, kind, geometry, expected):
        roi = ROI.new(kind, geometry)
        out = roi.transform("scale_image", SCALE, SHAPE)
        assert out is not None
        assert out.kind == kind
        assert out.id == roi.id
        for key, value in expected.items():
            if key == "vertices":
                np.testing.assert_allclose(out.geometry[key], value)
            else:
                assert out.geometry[key] == pytest.approx(value)

    def test_scale_multipolygon_scales_rings_and_holes(self):
        roi = ROI.new("multipolygon", {"components": [{
            "exterior": [[0, 0], [10, 0], [10, 10]],
            "holes": [[[2, 2], [4, 2], [4, 4]]],
        }]})
        out = roi.transform("scale_image", SCALE, SHAPE)
        comp = out.geometry["components"][0]
        assert comp["exterior"][1] == [20.0, 0.0]
        assert comp["holes"][0][2] == [8.0, 6.0]

    def test_scale_roundtrip_preserves_mask_fraction(self):
        """Scaling there and back must land the rasterised region in the
        same place — the practical 'overlay still over the same atoms' check."""
        roi = ROI.new("rectangle", {"x": 8, "y": 8, "width": 16, "height": 16})
        up = roi.transform("scale_image", SCALE, SHAPE)
        back = up.transform(
            "scale_image", {"new_width": 64, "new_height": 64}, (96, 128))
        np.testing.assert_array_equal(back.to_mask(SHAPE), roi.to_mask(SHAPE))

    def test_scale_invalid_params_raise(self):
        roi = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 2})
        with pytest.raises(ValueError, match="new_width"):
            roi.transform("scale_image", {}, SHAPE)

    @pytest.mark.parametrize("kind,geometry", [
        ("rectangle", {"x": 4, "y": 6, "width": 10, "height": 8}),
        ("ellipse", {"cx": 20, "cy": 22, "rx": 8, "ry": 5}),
        ("polygon", {"vertices": [[5, 4], [25, 8], [18, 24]]}),
        ("line", {"x1": 2, "y1": 4, "x2": 30, "y2": 40}),
    ])
    def test_shear_invalidates_every_kind(self, kind, geometry):
        roi = ROI.new(kind, geometry)
        assert roi.transform("shear", SHEAR, SHAPE) is None


# ── Core: ImageMask.transform ─────────────────────────────────────────────────

class TestMaskResamplingOps:
    @pytest.mark.parametrize("op,params", [
        ("scale_image", SCALE),
        ("shear", SHEAR),
        ("rotate_arbitrary", {"angle_degrees": 10.0}),
        ("linear_undistort", {"shear_x": 1.0, "scale_y": 1.0}),
        ("affine_lattice_correction", {"matrix": [[1, 0], [0, 1]]}),
    ])
    def test_resampling_ops_invalidate_not_raise(self, op, params):
        """The docstring always promised None for resampling ops; the code
        raised ValueError for scale/shear/affine. Pins the agreement."""
        raster = np.zeros(SHAPE, dtype=bool)
        raster[8:24, 8:24] = True
        mask = ImageMask.new(raster, name="m")
        assert mask.transform(op, params, SHAPE) is None

    def test_truly_unknown_op_still_raises(self):
        mask = ImageMask.new(np.ones(SHAPE, dtype=bool), name="m")
        with pytest.raises(ValueError, match="unknown operation"):
            mask.transform("bogus_op", {}, SHAPE)


@pytest.mark.parametrize(
    "shear_x,shear_y,error",
    [
        (1.0, 1.0, "singular"),
        (float("nan"), 0.0, "finite"),
        (0.0, float("inf"), "finite"),
    ],
)
def test_shear_kernel_rejects_invalid_matrices(shear_x, shear_y, error):
    with pytest.raises(ValueError, match=error):
        shear(np.ones((4, 4)), shear_x=shear_x, shear_y=shear_y)


# ── Viewer wiring ─────────────────────────────────────────────────────────────

@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"QApplication unavailable: {exc}")


def _viewer_host(qapp):
    from PySide6.QtGui import QPixmap

    from probeflow.gui.image_canvas import ImageCanvas
    from probeflow.gui.viewer.image_viewer_processing_export_mixin import (
        ImageViewerProcessingExportMixin,
    )
    from probeflow.gui.viewer.image_viewer_selection_mixin import (
        ImageViewerSelectionMixin,
    )
    from probeflow.gui.viewer.processing_undo import ProcessingUndoController

    canvas = ImageCanvas()
    pm = QPixmap(64, 64)
    pm.fill()
    canvas._view_scale_mode = "one_to_one"
    canvas.set_source(pm, reset_zoom=True)

    class Host(ImageViewerProcessingExportMixin, ImageViewerSelectionMixin):
        def __init__(self):
            self._zoom_lbl = canvas
            self._processing = {}
            self._image_roi_set = ROISet(image_id="img")
            self._image_mask_set = MaskSet(image_id="img")
            self.statuses: list[str] = []
            self._status_lbl = SimpleNamespace(setText=self.statuses.append)
            self.refreshes = 0
            self._proc_undo_ctrl = ProcessingUndoController(
                None, None, self._sync_viewer_menu_actions
            )

        def _refresh_processing_display(self):
            self.refreshes += 1

        def _sync_viewer_menu_actions(self):
            pass

        def _on_image_roi_set_changed(self):
            pass

        def _on_image_mask_set_changed(self):
            pass

        def _current_array_shape(self):
            return SHAPE

    host = Host()
    host._image_roi_set.add(
        ROI.new("rectangle", {"x": 4, "y": 6, "width": 10, "height": 8},
                name="r1"))
    raster = np.zeros(SHAPE, dtype=bool)
    raster[8:24, 8:24] = True
    host._image_mask_set.add(ImageMask.new(raster, name="m1"))
    canvas.set_selection("rectangle", {"x": 2, "y": 2, "width": 8, "height": 8})
    return host, canvas


def test_shear_dialog_keeps_singular_settings_open(qapp):
    from probeflow.gui.dialogs.shear_dialog import ShearDialog

    dialog = ShearDialog()
    applied = []
    dialog.applied.connect(applied.append)
    dialog._shear_x_spin.setValue(1.0)
    dialog._shear_y_spin.setValue(1.0)
    try:
        dialog._do_apply()
        assert applied == []
        assert "singular" in dialog._error_lbl.text().lower()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


class TestViewerScaleShearWiring:
    def test_scale_rescales_rois_and_selection_invalidates_masks(self, qapp):
        host, canvas = _viewer_host(qapp)

        host._on_scale_image_applied(dict(SCALE))

        roi = host._image_roi_set.rois[0]
        assert roi.geometry["x"] == pytest.approx(8.0)
        assert roi.geometry["height"] == pytest.approx(12.0)
        sel = canvas.selection()
        assert sel is not None
        assert sel["geometry"]["x"] == pytest.approx(4.0)
        assert sel["geometry"]["height"] == pytest.approx(12.0)
        assert host._image_mask_set.masks == [], (
            "raster mask survived a resample it cannot follow"
        )
        assert host._processing["geometric_ops"][-1]["op"] == "scale_image"
        assert host.refreshes == 1

    def test_shear_invalidates_rois_masks_and_selection(self, qapp):
        host, canvas = _viewer_host(qapp)

        host._on_shear_applied(dict(SHEAR))

        assert host._image_roi_set.rois == []
        assert host._image_mask_set.masks == []
        assert canvas.selection() is None
        assert host._processing["geometric_ops"][-1]["op"] == "shear"
        # The user is told overlays were dropped, not left to discover it.
        assert any("removed" in s for s in host.statuses)

    def test_shear_with_no_overlays_reports_plainly(self, qapp):
        host, canvas = _viewer_host(qapp)
        host._image_roi_set.rois = []
        host._image_mask_set.masks = []
        canvas.clear_selection(emit=False)

        host._on_shear_applied(dict(SHEAR))

        assert not any("removed" in s for s in host.statuses)
        assert any("Shear applied" in s for s in host.statuses)
