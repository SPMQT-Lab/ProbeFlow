"""Tests for per-region brightness/contrast compositing in render_scan_image."""

from __future__ import annotations

import numpy as np

from probeflow.gui import rendering as R


def _arr():
    return np.linspace(0.0, 1.0, 16).reshape(4, 4).astype(float)


def test_region_levels_remaps_only_masked_pixels():
    arr = _arr()
    mask = np.zeros((4, 4), dtype=bool)
    mask[2:, :] = True  # bottom half

    plain = np.asarray(
        R.render_scan_image(arr=arr, colormap="gray", vmin=0.0, vmax=1.0, size=None)
    )
    region = np.asarray(
        R.render_scan_image(
            arr=arr, colormap="gray", vmin=0.0, vmax=1.0, size=None,
            region_levels=[(mask, 0.4, 0.6)],
        )
    )

    # Pixels outside the mask are untouched by the region levels.
    assert np.array_equal(plain[~mask], region[~mask])

    # Pixels inside the mask are mapped with the region's own (vmin, vmax).
    lut = R._get_lut("gray")
    expected = lut[R._array_to_uint8(arr, vmin=0.4, vmax=0.6)]
    assert np.array_equal(region[mask], expected[mask])

    # The region scaling actually changed those pixels relative to the global map.
    assert not np.array_equal(plain[mask], region[mask])


def test_region_levels_none_matches_plain_render():
    arr = _arr()
    plain = np.asarray(
        R.render_scan_image(arr=arr, colormap="viridis", vmin=0.0, vmax=1.0, size=None)
    )
    none = np.asarray(
        R.render_scan_image(
            arr=arr, colormap="viridis", vmin=0.0, vmax=1.0, size=None, region_levels=None
        )
    )
    assert np.array_equal(plain, none)


def test_region_levels_skips_shape_mismatch_and_empty_mask():
    arr = _arr()
    plain = np.asarray(
        R.render_scan_image(arr=arr, colormap="gray", vmin=0.0, vmax=1.0, size=None)
    )
    bad = np.asarray(
        R.render_scan_image(
            arr=arr, colormap="gray", vmin=0.0, vmax=1.0, size=None,
            region_levels=[
                (np.zeros((4, 4), dtype=bool), 0.2, 0.8),   # empty mask
                (np.ones((2, 2), dtype=bool), 0.2, 0.8),    # wrong shape
                (None, 0.2, 0.8),                            # no mask
            ],
        )
    )
    assert np.array_equal(plain, bad)
