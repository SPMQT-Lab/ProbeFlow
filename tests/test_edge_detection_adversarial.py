"""Adversarial stress tests for advanced edge detection (review focus #4).

Verifies the hardening guarantees the module documents — flat images, hot
pixels, NaN rims, anisotropic pixels, ROI-restricted statistics — and pins
the orientation convention (image coordinates, y down).

The strict xfail documents a confirmed bug: ROI-restricted Canny inherits
skimage's whole-image quantile thresholds, so the near-zero gradient outside
the ROI dilutes the percentile cut and over-detects edges inside the ROI.
"""
from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.edge_detection import canny_edges, gradient_filter

N = 64


def _step_image() -> np.ndarray:
    """Vertical step: 0 on the left half, 1 on the right half."""
    img = np.zeros((N, N))
    img[:, N // 2:] = 1.0
    return img


# ── Flat and sparse images ────────────────────────────────────────────────────

class TestFlatAndSparse:
    def test_flat_zero_image_gradient_mask_is_empty(self):
        res = gradient_filter(np.zeros((N, N)),
                              threshold_to_mask=True, threshold=90.0)
        assert res.edge_mask is not None
        assert not res.edge_mask.any()

    def test_flat_nonzero_image_gradient_mask_is_empty(self):
        """Sobel on a constant non-zero image leaves ~eps·|data| float
        residue (was: the residue defeated the strict ``> 0.0`` guard and the
        ENTIRE image became an edge mask); the guard is now a relative floor
        scaled by data magnitude and pixel size."""
        res = gradient_filter(np.full((N, N), 3.7),
                              threshold_to_mask=True, threshold=90.0)
        assert res.edge_mask is not None
        assert not res.edge_mask.any(), (
            "flat zero-gradient background became an edge mask"
        )

    def test_flat_large_values_small_pixels_mask_is_empty(self):
        """The residue floor must track data scale and the pixel-size
        division: huge values on sub-nm pixels still yield no edges."""
        res = gradient_filter(np.full((N, N), 1.0e9),
                              threshold_to_mask=True, threshold=90.0,
                              pixel_size_x_nm=0.1, pixel_size_y_nm=0.1)
        assert not res.edge_mask.any()

    def test_weak_real_step_survives_the_residue_floor(self):
        """A genuinely tiny step (1e-12 amplitude) is real data, not float
        noise — the floor scales with the data so it must still be found."""
        img = np.zeros((N, N))
        img[:, N // 2:] = 1e-12
        res = gradient_filter(img, threshold_to_mask=True, threshold=90.0)
        assert res.edge_mask.any()
        ys, xs = np.nonzero(res.edge_mask)
        assert np.all(np.abs(xs - N // 2) <= 2)

    def test_flat_image_canny_is_empty(self):
        res = canny_edges(np.full((N, N), 3.7), sigma=1.5)
        assert not res.edge_mask.any()

    def test_single_step_mask_localised_to_step(self):
        res = gradient_filter(_step_image(),
                              threshold_to_mask=True, threshold=90.0)
        ys, xs = np.nonzero(res.edge_mask)
        assert len(xs) > 0, "step edge not detected"
        # Sobel footprint is 3 px; everything flagged must hug the step column.
        assert np.all(np.abs(xs - N // 2) <= 2), (
            "mask pixels far from the only step in the image"
        )

    def test_isolated_hot_pixel_does_not_flag_background(self):
        img = np.zeros((N, N))
        img[10, 10] = 100.0
        res = gradient_filter(img, threshold_to_mask=True, threshold=90.0)
        ys, xs = np.nonzero(res.edge_mask)
        assert len(xs) > 0
        assert np.all((np.abs(ys - 10) <= 2) & (np.abs(xs - 10) <= 2)), (
            "threshold mask leaked beyond the hot pixel's operator footprint"
        )

    def test_all_nan_image_yields_empty_outputs(self):
        img = np.full((N, N), np.nan)
        g = gradient_filter(img, threshold_to_mask=True)
        assert not g.edge_mask.any()
        assert np.all(np.isnan(g.display_image))
        c = canny_edges(img)
        assert not c.edge_mask.any()


# ── NaN holes ─────────────────────────────────────────────────────────────────

class TestNaNHoles:
    @staticmethod
    def _flat_with_hole() -> tuple[np.ndarray, np.ndarray]:
        img = np.zeros((N, N))
        hole = np.zeros((N, N), dtype=bool)
        hole[20:30, 20:30] = True
        img[hole] = np.nan
        return img, hole

    def test_nan_rim_not_detected_as_edge(self):
        """The mean-fill step at the hole rim must not produce edges on an
        otherwise flat image."""
        img, _hole = self._flat_with_hole()
        res = gradient_filter(img, threshold_to_mask=True, threshold=90.0)
        assert not res.edge_mask.any(), "NaN rim flagged as a real edge"

        c = canny_edges(img, sigma=1.5)
        assert not c.edge_mask.any(), "Canny detected the NaN rim as an edge"

    def test_rim_band_is_invalid_in_outputs(self):
        """Magnitude is nulled and orientation NaN for every pixel whose 3×3
        footprint touches the hole (the documented 1-px invalid band)."""
        img, hole = self._flat_with_hole()
        # Add a real step elsewhere so the image is not degenerate.
        img[:, 50:] = 5.0
        res = gradient_filter(img)
        from scipy.ndimage import binary_dilation
        band = binary_dilation(hole, structure=np.ones((3, 3), dtype=bool))
        assert np.all(res.gradient_magnitude[band] == 0.0)
        assert np.all(np.isnan(res.gradient_orientation[band]))
        assert np.all(np.isnan(res.display_image[band]))

    def test_nan_restored_in_display_only_inside_hole(self):
        img, hole = self._flat_with_hole()
        res = canny_edges(img, sigma=1.0)
        assert np.all(np.isnan(res.display_image[hole]))
        assert np.all(np.isfinite(res.display_image[~hole]))


# ── Anisotropic pixels ────────────────────────────────────────────────────────

class TestAnisotropicPixels:
    def test_anisotropic_orientation_and_relative_magnitude(self):
        """A plane z = x + y (in pixel units) on dx=2 nm, dy=1 nm pixels has
        physical slopes ∂z/∂x = 0.5, ∂z/∂y = 1.0 — orientation must steepen
        toward y, and the magnitude must scale like √(0.25 + 1) relative to
        the isotropic dx=dy=1 case."""
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        img = xx + yy
        aniso = gradient_filter(img, normalize=False,
                                pixel_size_x_nm=2.0, pixel_size_y_nm=1.0)
        iso = gradient_filter(img, normalize=False,
                              pixel_size_x_nm=1.0, pixel_size_y_nm=1.0)
        interior = (slice(4, N - 4), slice(4, N - 4))
        np.testing.assert_allclose(
            aniso.gradient_orientation[interior], np.arctan2(1.0, 0.5),
            rtol=1e-6)
        np.testing.assert_allclose(
            aniso.gradient_magnitude[interior] / iso.gradient_magnitude[interior],
            np.hypot(0.5, 1.0) / np.hypot(1.0, 1.0), rtol=1e-6)

    @pytest.mark.parametrize("operator", ["sobel", "scharr"])
    def test_gradient_magnitude_equals_physical_slope(self, operator):
        """gradient_magnitude is the true physical slope: skimage's kernels
        respond 2.0 per unit slope (two-pixel difference span), which the
        module now halves before the pixel-size division."""
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        img = xx + yy
        res = gradient_filter(img, operator=operator, normalize=False,
                              pixel_size_x_nm=2.0, pixel_size_y_nm=1.0)
        interior = (slice(4, N - 4), slice(4, N - 4))
        np.testing.assert_allclose(
            res.gradient_magnitude[interior], np.hypot(0.5, 1.0), rtol=1e-6)

    def test_isotropic_pixel_size_fallback_matches_explicit(self):
        img = _step_image()
        a = gradient_filter(img, normalize=False, pixel_size_nm=1.5)
        b = gradient_filter(img, normalize=False,
                            pixel_size_x_nm=1.5, pixel_size_y_nm=1.5)
        np.testing.assert_allclose(a.gradient_magnitude, b.gradient_magnitude)

    def test_orientation_is_image_coords_y_down(self):
        """A surface increasing downward (toward larger row index) must have
        orientation +π/2 in the documented image convention — not -π/2 as the
        math y-up convention would give."""
        yy = np.mgrid[0:N, 0:N][0].astype(float)
        res = gradient_filter(yy, normalize=False)
        interior = res.gradient_orientation[4:-4, 4:-4]
        np.testing.assert_allclose(interior, np.pi / 2, rtol=1e-6)


# ── ROI restriction ───────────────────────────────────────────────────────────

class TestROIRestriction:
    def test_outside_roi_gradient_excluded_from_stats_and_mask(self):
        """A huge edge OUTSIDE the ROI must neither appear in the mask nor
        swamp the normalisation/threshold statistics for the ROI interior."""
        rng = np.random.default_rng(3)
        img = np.zeros((N, N))
        img[:, 50:] = 1000.0                      # giant step outside ROI
        img[8:24, 8:24] = rng.normal(scale=0.1, size=(16, 16))  # weak texture
        roi = np.zeros((N, N), dtype=bool)
        roi[8:24, 8:24] = True

        res = gradient_filter(img, threshold_to_mask=True, threshold=80.0,
                              roi_mask=roi)
        assert res.edge_mask is not None
        assert not (res.edge_mask & ~roi).any(), "mask leaked outside the ROI"
        assert res.edge_mask.any(), (
            "weak in-ROI texture undetected — outside-ROI step entered the "
            "percentile statistics"
        )
        # Outside-ROI magnitude must be nulled before normalisation, so the
        # in-ROI normalised display can actually reach the [0, 1] peak.
        assert float(np.nanmax(res.display_image[roi])) == pytest.approx(1.0)

    def test_orientation_nan_outside_roi(self):
        img = _step_image()
        roi = np.zeros((N, N), dtype=bool)
        roi[:, : N // 2 + 4] = True
        res = gradient_filter(img, roi_mask=roi)
        assert np.all(np.isnan(res.gradient_orientation[~roi]))

    def test_roi_mask_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            gradient_filter(_step_image(), roi_mask=np.ones((8, 8), dtype=bool))
        with pytest.raises(ValueError, match="shape"):
            canny_edges(_step_image(), roi_mask=np.ones((8, 8), dtype=bool))

    def test_canny_roi_thresholds_match_standalone_crop(self):
        """Quantile thresholds must come from the restricted region only —
        skimage computes them over the whole magnitude array, where the
        ~zero gradient outside a small ROI diluted the cut to ~2x
        over-detection (432 vs 218 edge px before the fix)."""
        rng = np.random.default_rng(2)
        img = np.zeros((200, 200))
        img[80:120, 80:120] = rng.normal(size=(40, 40))
        roi = np.zeros((200, 200), dtype=bool)
        roi[80:120, 80:120] = True

        restricted = canny_edges(img, sigma=1.5, low=70, high=90, roi_mask=roi)
        standalone = canny_edges(img[80:120, 80:120], sigma=1.5, low=70, high=90)

        n_restricted = int(restricted.edge_mask[80:120, 80:120].sum())
        n_standalone = int(standalone.edge_mask.sum())
        # Identical data and parameters: allow boundary-effect slack, but the
        # ROI-restricted run must not systematically over-detect.
        assert n_restricted <= int(1.2 * n_standalone) + 5, (
            f"ROI-restricted Canny found {n_restricted} edge px vs "
            f"{n_standalone} standalone — outside-ROI zeros diluted the "
            "percentile thresholds"
        )


# ── Threshold-mask contract ───────────────────────────────────────────────────

class TestThresholdMaskContract:
    def test_threshold_uses_magnitude_regardless_of_output(self):
        """threshold_to_mask thresholds the magnitude even when output='x',
        so the mask is identical across output choices."""
        img = _step_image() + np.linspace(0, 0.5, N)[:, None]
        masks = [
            gradient_filter(img, output=out, threshold_to_mask=True,
                            threshold=85.0).edge_mask
            for out in ("magnitude", "x", "y", "orientation")
        ]
        for m in masks[1:]:
            np.testing.assert_array_equal(masks[0], m)

    def test_orientation_output_never_rescaled(self):
        # Downward ramp: every valid orientation is +π/2 ≈ 1.57 — outside
        # [-1, 1], so any normalisation of the orientation output is visible.
        img = np.mgrid[0:N, 0:N][0].astype(float)
        res = gradient_filter(img, output="orientation", normalize=True)
        vals = res.gradient_orientation[np.isfinite(res.gradient_orientation)]
        assert float(np.max(np.abs(vals))) == pytest.approx(np.pi / 2), (
            "orientation appears rescaled"
        )
