"""Tests for the active-mask layer: ImageMask, MaskSet, mask_name, sidecar, mask→ROI."""
from __future__ import annotations

import numpy as np
import pytest

from probeflow.core.mask import ImageMask, MaskSet, mask_name
from probeflow.core.roi import roi_from_mask


def _mask(n: int = 16) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[4:12, 4:12] = True
    return m


# ── ImageMask ─────────────────────────────────────────────────────────────────

class TestImageMask:
    def test_new_generates_id_and_name(self):
        m = ImageMask.new(_mask(), method="canny", parameters={"sigma": 1.0})
        assert m.id
        assert m.name == "Canny_sigma1_p70-90"
        assert m.count() == 64

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            ImageMask.new(np.zeros((2, 2, 2), dtype=bool))

    def test_packbits_roundtrip_preserves_shape_and_values(self):
        rng = np.random.default_rng(0)
        data = rng.random((23, 17)) > 0.5  # non-byte-aligned dims
        m = ImageMask.new(data, name="rand")
        back = ImageMask.from_dict(m.to_dict())
        assert back.shape == (23, 17)
        assert np.array_equal(back.data, data)
        assert back.name == "rand"

    def test_shape_property(self):
        assert ImageMask.new(_mask(16)).shape == (16, 16)


# ── mask_name ─────────────────────────────────────────────────────────────────

class TestMaskName:
    def test_canny_percentile(self):
        assert mask_name("canny", {"sigma": 1.5, "low": 60, "high": 85}) == "Canny_sigma1.5_p60-85"

    def test_canny_absolute_drops_percentile_suffix(self):
        assert mask_name("canny", {"sigma": 1.0, "threshold_mode": "absolute"}) == "Canny_sigma1"

    def test_sobel_magnitude_thresholded(self):
        assert mask_name("sobel", {"output": "magnitude", "threshold_to_mask": True,
                                    "threshold": 90}) == "Sobel_magnitude_p90"

    def test_scharr_directional(self):
        assert mask_name("scharr", {"output": "x"}) == "Scharr_x_gradient"


# ── MaskSet ───────────────────────────────────────────────────────────────────

class TestMaskSet:
    def test_add_active_remove(self):
        ms = MaskSet(image_id="img")
        m = ImageMask.new(_mask(), name="a")
        ms.add(m)
        ms.set_active(m.id)
        assert ms.active() is m
        ms.remove(m.id)
        assert ms.active_mask_id is None
        assert ms.get(m.id) is None

    def test_set_active_unknown_raises(self):
        ms = MaskSet(image_id="img")
        with pytest.raises(ValueError):
            ms.set_active("nope")

    def test_replace_data_in_place(self):
        ms = MaskSet(image_id="img")
        m = ImageMask.new(_mask())
        ms.add(m)
        ms.replace(m.id, np.zeros((16, 16), dtype=bool))
        assert ms.get(m.id).count() == 0

    def test_roundtrip(self):
        ms = MaskSet(image_id="img")
        m1 = ImageMask.new(_mask(), name="one")
        m2 = ImageMask.new(~_mask(), name="two")
        ms.add(m1)
        ms.add(m2)
        ms.set_active(m2.id)
        back = MaskSet.from_dict(ms.to_dict())
        assert back.image_id == "img"
        assert [m.name for m in back.masks] == ["one", "two"]
        assert back.active_mask_id == m2.id
        assert np.array_equal(back.get(m1.id).data, m1.data)

    def test_roundtrip_drops_dangling_active(self):
        d = MaskSet(image_id="img").to_dict()
        d["active_mask_id"] = "ghost"
        assert MaskSet.from_dict(d).active_mask_id is None


# ── geometric transforms ─────────────────────────────────────────────────────────

class TestMaskTransform:
    def _corner_mask(self):
        d = np.zeros((4, 6), dtype=bool)
        d[0, 0] = True  # top-left
        return ImageMask.new(d, name="corner")

    def test_flip_horizontal_moves_pixels(self):
        out = self._corner_mask().transform("flip_horizontal", {}, (4, 6))
        assert out.data[0, 5] and not out.data[0, 0]

    def test_flip_vertical_moves_pixels(self):
        out = self._corner_mask().transform("flip_vertical", {}, (4, 6))
        assert out.data[3, 0]

    def test_rot180_moves_pixels(self):
        out = self._corner_mask().transform("rotate_180", {}, (4, 6))
        assert out.data[3, 5]

    def test_rot90_cw_changes_shape(self):
        out = self._corner_mask().transform("rotate_90_cw", {}, (4, 6))
        assert out.data.shape == (6, 4)

    def test_crop_slices_inclusive(self):
        out = self._corner_mask().transform("crop", {"x0": 0, "y0": 0, "x1": 1, "y1": 1}, (4, 6))
        assert out.data.shape == (2, 2)
        assert out.data[0, 0]

    def test_rotate_arbitrary_invalidates(self):
        assert self._corner_mask().transform("rotate_arbitrary", {}, (4, 6)) is None

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="unknown operation"):
            self._corner_mask().transform("warp", {}, (4, 6))

    def test_identity_fields_preserved(self):
        m = self._corner_mask()
        out = m.transform("flip_horizontal", {}, (4, 6))
        assert out.id == m.id and out.name == m.name and out.method == m.method

    def test_transform_all_returns_invalidated_and_applies(self):
        m = self._corner_mask()
        ms = MaskSet(image_id="img")
        ms.add(m)
        ms.set_active(m.id)
        assert ms.transform_all("flip_horizontal", {}, (4, 6)) == []
        assert ms.masks[0].data[0, 5]
        assert ms.transform_all("rotate_arbitrary", {}, (4, 6)) == [m.id]
        # invalidated masks are kept in place; caller decides removal
        assert len(ms.masks) == 1


# ── mask → ROI ──────────────────────────────────────────────────────────────────

class TestRoiFromMask:
    def test_single_square_roundtrips(self):
        m = _mask(32)
        rois = roi_from_mask(m)
        assert len(rois) == 1
        rt = rois[0].to_mask(m.shape)
        # Allow the half-pixel contour expansion; most original pixels recovered.
        iou = (rt & m).sum() / (rt | m).sum()
        assert iou > 0.8
        # Nearly all original pixels recovered (contour tracing may clip a corner).
        assert (rt & m).sum() >= m.sum() - 1

    def test_two_components_split(self):
        m = np.zeros((40, 40), dtype=bool)
        m[5:12, 5:12] = True
        m[25:35, 25:35] = True
        rois = roi_from_mask(m, one_per_component=True)
        assert len(rois) == 2

    def test_min_size_filters_small_blobs(self):
        m = np.zeros((40, 40), dtype=bool)
        m[5:30, 5:30] = True   # big
        m[0, 0] = True         # 1-px speck
        rois = roi_from_mask(m, min_size_px=10)
        assert len(rois) == 1

    def test_union_returns_single_roi(self):
        m = np.zeros((40, 40), dtype=bool)
        m[5:12, 5:12] = True
        m[25:35, 25:35] = True
        rois = roi_from_mask(m, one_per_component=False)
        assert len(rois) == 1
        assert rois[0].kind == "multipolygon"

    def test_empty_mask_returns_empty(self):
        assert roi_from_mask(np.zeros((10, 10), dtype=bool)) == []
