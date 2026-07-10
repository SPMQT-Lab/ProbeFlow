"""Tests for probeflow.analysis.step_edges — algorithmic step-edge detection.

The kernel detects the substrate step-edge region so molecules sitting on a step
can be excluded reproducibly (instead of being painted over by hand).  Fixtures
use realistic STM scales: a 0.235 nm Au monatomic step, 0.15 nm/px sampling,
~0.15 nm molecules, a slowly-varying tilt+bowl background, and 10-20 pm noise.

This file holds the kernel tests (mask behaviour).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from probeflow.analysis.step_edges import step_edge_mask

PX = 0.15e-9            # 0.15 nm / px
H_STEP = 0.235e-9       # Au(111) monatomic step
H_MOL = 0.15e-9         # molecule height
MOL_DIAM = 1.0e-9       # molecule diameter (suppression window source)
N = 256


# ─── Fixture builders ───────────────────────────────────────────────────────

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _grid():
    return np.mgrid[:N, :N]   # Y, X


def _gauss_bump(cx, cy, h, fwhm_nm=1.0):
    Y, X = _grid()
    sig = (fwhm_nm * 1e-9 / PX) / 2.3548
    return h * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2))


def _disk(cx, cy, r_px, h):
    Y, X = _grid()
    return np.where((X - cx) ** 2 + (Y - cy) ** 2 <= r_px * r_px, h, 0.0)


def _tilt(amp):
    Y, X = _grid()
    return amp * (X / (N - 1) + 0.5 * Y / (N - 1))


def _bowl(amp):
    Y, X = _grid()
    return amp * (((X - N / 2) / N) ** 2 + ((Y - N / 2) / N) ** 2)


def _vertical_step(col=128, height=H_STEP):
    a = np.zeros((N, N), dtype=np.float64)
    a[:, col:] = height
    return a


def _raw_slope_band(a, threshold_deg=20.0):
    """Slope-threshold band WITHOUT molecule suppression (for contrast tests)."""
    gy, gx = np.gradient(np.where(np.isfinite(a), a, 0.0))
    slope = np.sqrt((gx / PX) ** 2 + (gy / PX) ** 2)
    return slope > math.tan(math.radians(threshold_deg))


def _mask(a, **kw):
    kw.setdefault("molecule_diameter_m", MOL_DIAM)
    return step_edge_mask(a, pixel_size_x_m=PX, pixel_size_y_m=PX, **kw)


# ─── Kernel behaviour ────────────────────────────────────────────────────────

class TestStepEdgeMask:
    def test_step_flagged_and_localized(self):
        """A vertical step is detected and the band hugs the step, not the terraces."""
        a = _vertical_step() + _tilt(0.1e-9) + _rng(0).normal(0, 15e-12, (N, N))
        m = _mask(a, dilate_m=0.3e-9)
        assert m.any(), "step not detected"
        # Band is localized: its flagged columns cluster around the step (col 128).
        cols = np.where(m.any(axis=0))[0]
        assert cols.min() >= 120 and cols.max() <= 136, (cols.min(), cols.max())
        # Terrace interiors (far from the step) carry no band.
        assert not m[:, :100].any()
        assert not m[:, 156:].any()
        # And the band is a small fraction of the image.
        assert m.mean() < 0.06

    def test_curvature_and_tilt_not_flagged(self):
        """A smooth tilt+bowl background (no step) produces (essentially) no band."""
        a = _tilt(0.1e-9) + _bowl(0.12e-9)
        for cx, cy in [(60, 60), (180, 90), (120, 190)]:
            a = a + _gauss_bump(cx, cy, H_MOL)
        a = a + _rng(1).normal(0, 15e-12, (N, N))
        m = _mask(a, dilate_m=0.3e-9)
        assert m.mean() < 1e-3, f"curvature falsely flagged: {m.mean()*100:.3f}%"

    def test_flat_surface_not_flagged(self):
        """Flat surface + molecules + noise → no step band."""
        a = np.zeros((N, N))
        for cx, cy in [(50, 50), (150, 150), (90, 200)]:
            a = a + _gauss_bump(cx, cy, H_MOL)
        a = a + _rng(2).normal(0, 15e-12, (N, N))
        m = _mask(a, dilate_m=0.3e-9)
        assert m.mean() < 1e-3

    def test_molecule_suppression_is_doing_the_work(self):
        """Contrast: raw slope flags molecule perimeters; the kernel does not.

        Disk molecules (sharp, tall) trip a naive slope threshold across their
        own rims.  The kernel's morphological suppression removes them first, so
        a molecule on a flat terrace is NOT flagged — proving suppression, not
        luck, is what keeps terrace molecules out of the band.
        """
        a = np.zeros((N, N))
        mols = [(50, 60), (200, 180)]
        for cx, cy in mols:
            a = np.maximum(a, _disk(cx, cy, 5, 0.3e-9))
        a = a + _rng(3).normal(0, 5e-12, (N, N))

        raw = _raw_slope_band(a)
        supp = _mask(a, dilate_m=0.0)

        def frac(m, cx, cy, r=6):
            return m[cy - r:cy + r, cx - r:cx + r].mean()

        raw_flagged = np.mean([frac(raw, cx, cy) for cx, cy in mols])
        supp_flagged = np.mean([frac(supp, cx, cy) for cx, cy in mols])
        assert raw_flagged > 0.2, "raw slope should flag sharp molecule rims"
        assert supp_flagged == 0.0, "suppressed mask must not flag terrace molecules"

    def test_diagonal_step_detected(self):
        """Detection is rotation-agnostic (slope magnitude): a diagonal step works."""
        Y, X = _grid()
        a = np.where((X + Y) > N, H_STEP, 0.0) + _rng(4).normal(0, 5e-12, (N, N))
        m = _mask(a)
        assert m.any()
        # Band lies along the X+Y≈N anti-diagonal.
        ys, xs = np.where(m)
        assert np.abs((xs + ys).mean() - N) < 12

    def test_height_gate_drops_short_step_keeps_tall(self):
        """With min_step_height_m, a short (but slope-passing) step is dropped."""
        a = np.zeros((N, N))
        a[:, 80:] += 0.13e-9     # short step, still steep enough to pass slope
        a[:, 170:] += 0.30e-9    # tall step
        a = a + _rng(5).normal(0, 5e-12, (N, N))

        slope_only = _mask(a, dilate_m=0.0)
        gated = _mask(a, dilate_m=0.0, min_step_height_m=0.20e-9)

        cols_slope = np.where(slope_only.any(axis=0))[0]
        cols_gated = np.where(gated.any(axis=0))[0]
        assert cols_slope.min() < 90, "short step should appear without the gate"
        assert cols_gated.min() > 150, "gate should drop the short step, keep the tall one"

    def test_height_gate_off_keeps_both(self):
        a = np.zeros((N, N))
        a[:, 80:] += 0.13e-9
        a[:, 170:] += 0.30e-9
        a = a + _rng(5).normal(0, 5e-12, (N, N))
        from scipy.ndimage import label
        assert label(_mask(a, dilate_m=0.0))[1] == 2

    def test_suppress_dark_detects_step_under_pits(self):
        """Dark-feature mode: closing suppresses depressions, step still found."""
        a = _vertical_step()
        # Add dark pits (depressions) on both terraces.
        for cx, cy in [(60, 60), (190, 190)]:
            a = a - _disk(cx, cy, 5, 0.2e-9)
        a = a + _rng(6).normal(0, 5e-12, (N, N))
        m = _mask(a, dilate_m=0.0, suppress_dark=True)
        assert m.any()
        cols = np.where(m.any(axis=0))[0]
        assert cols.min() >= 120 and cols.max() <= 136
        # Pit rims must not be flagged as steps.
        assert m[54:66, 54:66].mean() == 0.0

    def test_dilation_widens_band(self):
        a = _vertical_step() + _rng(7).normal(0, 5e-12, (N, N))
        narrow = _mask(a, dilate_m=0.0).sum()
        wide = _mask(a, dilate_m=0.6e-9).sum()
        assert wide > narrow

    def test_deterministic(self):
        a = _vertical_step() + _tilt(0.1e-9) + _rng(8).normal(0, 15e-12, (N, N))
        m1 = _mask(a, dilate_m=0.3e-9, min_step_height_m=0.1e-9)
        m2 = _mask(a, dilate_m=0.3e-9, min_step_height_m=0.1e-9)
        assert np.array_equal(m1, m2)

    def test_nan_patch_handled(self):
        a = _vertical_step() + _rng(9).normal(0, 5e-12, (N, N))
        a[100:110, 60:70] = np.nan
        m = _mask(a)
        assert not m[100:110, 60:70].any(), "NaN pixels must be False"
        assert m.any(), "finite step still detected"

    def test_rectangular_pixels(self):
        """Per-axis suppression window keeps detection correct on non-square pixels."""
        a = _vertical_step() + _rng(10).normal(0, 5e-12, (N, N))
        m = step_edge_mask(
            a, pixel_size_x_m=PX, pixel_size_y_m=PX * 3,
            molecule_diameter_m=MOL_DIAM, dilate_m=0.0,
        )
        assert m.any()
        cols = np.where(m.any(axis=0))[0]
        assert cols.min() >= 120 and cols.max() <= 136

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            step_edge_mask(np.zeros((8, 8, 2)), pixel_size_x_m=PX,
                           pixel_size_y_m=PX, molecule_diameter_m=MOL_DIAM)
        with pytest.raises(ValueError):
            step_edge_mask(np.zeros((8, 8)), pixel_size_x_m=PX,
                           pixel_size_y_m=PX, molecule_diameter_m=0.0)

    def test_tiny_and_allnan_return_empty(self):
        assert not step_edge_mask(np.zeros((2, 2)), pixel_size_x_m=PX,
                                  pixel_size_y_m=PX, molecule_diameter_m=MOL_DIAM).any()
        allnan = np.full((32, 32), np.nan)
        assert not step_edge_mask(allnan, pixel_size_x_m=PX, pixel_size_y_m=PX,
                                  molecule_diameter_m=MOL_DIAM).any()
