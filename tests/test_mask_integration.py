"""Qt-free integration: the active mask flows into statistics and background exclusion.

These exercise the backend seams the viewer relies on (``roi_statistics(mask=…)``
and ``subtract_background`` exclusion via the mask→ROI bridge), without Qt.
"""
from __future__ import annotations

import numpy as np

from probeflow.core.mask import ImageMask, MaskSet
from probeflow.core.roi import roi_from_mask
from probeflow.measurements.image import roi_statistics
from probeflow.processing import subtract_background


def test_active_mask_restricts_statistics():
    # Left half = 0, right half = 10. A mask over the right half should report
    # mean 10, not 5.
    arr = np.zeros((32, 32), dtype=np.float64)
    arr[:, 16:] = 10.0
    mask = np.zeros((32, 32), dtype=bool)
    mask[:, 16:] = True

    ms = MaskSet(image_id="img")
    ms.add(ImageMask.new(mask, name="right"))
    ms.set_active(ms.masks[0].id)

    result = roi_statistics(
        arr, measurement_id="m1", source_label="t",
        mask=ms.active().data, pixel_size_x=1.0, pixel_size_y=1.0,
    )
    assert result.values["mean_height"] == 10.0
    assert result.values["n_finite_pixels"] == 16 * 32


def test_mask_to_roi_excludes_region_from_plane_fit():
    # The active mask excludes regions from a plane fit *via the mask→ROI
    # bridge* (subtract_background takes an ROI, not a mask). A tilted plane
    # plus a bright contaminated blob: excluding the blob recovers a near-flat
    # residual; including it skews the fit.
    yy, xx = np.mgrid[0:64, 0:64]
    plane = 0.1 * xx + 0.05 * yy
    img = plane.astype(np.float64).copy()
    blob = np.zeros_like(img, dtype=bool)
    blob[10:20, 10:20] = True
    img[blob] += 50.0  # contamination

    rois = roi_from_mask(blob, min_size_px=0)
    assert rois  # mask converts to at least one ROI

    excluded = subtract_background(img, order=1, exclude_roi=rois[0])
    included = subtract_background(img, order=1)

    # Residual std away from the blob is much smaller when the blob is excluded.
    clean = ~blob
    assert np.std(excluded[clean]) < np.std(included[clean])
    assert np.std(excluded[clean]) < 1.0


def test_active_mask_array_shape_guard_via_maskset():
    ms = MaskSet(image_id="img")
    ms.add(ImageMask.new(np.ones((8, 8), dtype=bool)))
    ms.set_active(ms.masks[0].id)
    assert ms.active().shape == (8, 8)
