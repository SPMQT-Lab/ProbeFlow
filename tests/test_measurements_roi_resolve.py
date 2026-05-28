"""Tests for probeflow.measurements.roi_resolve.

Consolidates the two near-identical ROI→mask resolvers that
arch-backend #11 (2026-05-27 deep review) flagged.
"""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.core.roi import ROI
from probeflow.measurements.roi_resolve import resolve_roi_to_mask


class TestResolveRoiToMask:
    def test_no_roi_no_mask_returns_all_true(self):
        out = resolve_roi_to_mask((4, 6))
        assert out.shape == (4, 6)
        assert out.dtype == bool
        assert out.all()

    def test_precomputed_mask_returned_as_copy(self):
        m = np.zeros((4, 6), dtype=bool)
        m[1, 2] = True
        out = resolve_roi_to_mask((4, 6), mask=m)
        # Mutating the returned mask must not affect the input.
        out[0, 0] = True
        assert not m[0, 0]
        assert out[1, 2]

    def test_mask_takes_precedence_over_roi(self):
        roi = ROI.new("rectangle", {"x": 0, "y": 0, "width": 2, "height": 2})
        m = np.zeros((4, 6), dtype=bool)
        m[3, 5] = True
        out = resolve_roi_to_mask((4, 6), roi=roi, mask=m)
        # Came from the mask, not the ROI rasterization.
        assert out[3, 5]
        # ROI rectangle would have set [0:2, 0:2]
        assert not out[0, 0]

    def test_roi_to_mask_used_when_no_precomputed_mask(self):
        roi = ROI.new("rectangle", {"x": 1, "y": 0, "width": 2, "height": 2})
        out = resolve_roi_to_mask((4, 6), roi=roi)
        assert out.dtype == bool
        # Rectangle at x=1, y=0, w=2, h=2 → cells (0,1), (0,2), (1,1), (1,2)
        assert out[0, 1] and out[1, 2]
        assert not out[3, 5]

    def test_mismatched_mask_shape_raises(self):
        bad = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="must match image shape"):
            resolve_roi_to_mask((4, 6), mask=bad)

    def test_higher_dim_shape_truncated(self):
        """Callers historically pass arr.shape (which can be 3-D for
        multi-channel arrays).  The helper takes the first two."""
        out = resolve_roi_to_mask((4, 6, 3))
        assert out.shape == (4, 6)


class TestLegacyWrappersDelegate:
    """The local helpers ``_mask_from_roi_or_mask`` and ``_roi_mask`` are
    now thin wrappers over ``resolve_roi_to_mask`` — verify they still
    behave identically to the canonical helper."""

    def test_image_module_helper(self):
        from probeflow.measurements.image import _mask_from_roi_or_mask
        m = np.zeros((4, 4), dtype=bool)
        m[0, 0] = True
        out_a = _mask_from_roi_or_mask((4, 4), mask=m)
        out_b = resolve_roi_to_mask((4, 4), mask=m)
        np.testing.assert_array_equal(out_a, out_b)

    def test_features_module_helper(self):
        from probeflow.measurements.features import _roi_mask
        roi = ROI.new("rectangle", {"x": 0, "y": 0, "width": 2, "height": 3})
        out_a = _roi_mask((6, 6), roi=roi, roi_mask=None)
        out_b = resolve_roi_to_mask((6, 6), roi=roi)
        np.testing.assert_array_equal(out_a, out_b)
