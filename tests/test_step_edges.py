"""Tests for probeflow.analysis.step_edges — algorithmic step-edge detection.

The kernel detects the substrate step-edge region so molecules sitting on a step
can be excluded reproducibly (instead of being painted over by hand).  Fixtures
use realistic STM scales: a 0.235 nm Au monatomic step, 0.15 nm/px sampling,
~0.15 nm molecules, a slowly-varying tilt+bowl background, and 10-20 pm noise.

This file holds the kernel tests (mask behaviour).  The integration tests that
run the mask through ``segment_particles`` to drop whole particles live in the
``TestStepExclusionIntegration`` class, added alongside.
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


# ─── Integration: exclude_mask → segment_particles whole-particle rejection ──
#
# The exclusion *mechanics* (overlap rejection, the 0.25 threshold) are tested
# deterministically on a clean flat scene that segments to an exact count, with
# a constructed band — segmentation of molecules on a real stepped surface is a
# separate, harder problem and would make these assertions flaky. One end-to-end
# test then runs a real ``step_edge_mask`` on a stepped scene with robust
# *relative* assertions.

cv2 = pytest.importorskip("cv2")
from probeflow.analysis.features import segment_particles   # noqa: E402


def _seg(a, **kw):
    return segment_particles(
        a, PX, pixel_size_x_m=PX, pixel_size_y_m=PX,
        threshold="otsu", min_area_nm2=0.3, size_sigma_clip=None, **kw,
    )


def _flat_molecule_grid(cols, rows, *, r=5, h=5.0, seed=0):
    """Disk molecules on a flat background — segments to exactly len(cols)*len(rows)."""
    a = np.zeros((N, N))
    centres = [(cx, cy) for cx in cols for cy in rows]
    for cx, cy in centres:
        a = np.maximum(a, _disk(cx, cy, r, h))
    a = a + _rng(seed).normal(0, 0.05, (N, N))
    return a, centres


class TestExcludeMaskMechanics:
    def test_excludes_particles_under_band(self):
        """Particles whose footprint lies under the band are dropped; others kept."""
        cols, rows = [40, 80, 120, 160, 200], [60, 120, 180]
        a, centres = _flat_molecule_grid(cols, rows)
        assert len(_seg(a)) == len(centres), "flat scene must segment cleanly"

        band = np.zeros((N, N), dtype=bool)
        band[:, 115:126] = True              # a vertical stripe over the col=120 molecules
        kept = _seg(a, exclude_mask=band, max_exclude_overlap=0.25)

        assert len(kept) == len(centres) - len(rows)   # the 3 col=120 molecules gone
        assert not any(abs(p.centroid_x_m - 120 * PX) < 6 * PX for p in kept)

    def test_overlap_threshold_crossover(self):
        """A particle half-under the band is kept above its overlap, dropped below.

        Uses a small, dense scene (one big disk on a 64² field) so Otsu segments
        it cleanly — a lone disk on a large sparse field would make Otsu split
        the noise instead.
        """
        n = 64
        Y, X = np.mgrid[:n, :n]
        a = np.where((X - 32) ** 2 + (Y - 32) ** 2 <= 12 * 12, 5.0, 0.0)
        a = a + _rng(0).normal(0, 0.05, (n, n))
        band = np.zeros((n, n), dtype=bool)
        band[:, 32:] = True                  # covers the x≥32 half of the disk

        def seg64(**kw):
            return segment_particles(a, PX, pixel_size_x_m=PX, pixel_size_y_m=PX,
                                     threshold="otsu", min_area_nm2=0.3,
                                     size_sigma_clip=None, **kw)

        p = max(seg64(), key=lambda q: q.n_pixels)
        x0, y0, x1, y1 = p.bbox_px
        pm = np.zeros((n, n), dtype=bool)
        pm[y0:y1, x0:x1] = True
        frac = (pm & band).sum() / pm.sum()
        assert 0.2 < frac < 0.8, f"need a partial overlap, got {frac:.2f}"

        kept_lo = seg64(exclude_mask=band, max_exclude_overlap=min(0.99, frac + 0.2))
        kept_hi = seg64(exclude_mask=band, max_exclude_overlap=max(0.0, frac - 0.2))
        assert len(kept_lo) == 1 and len(kept_hi) == 0

    def test_empty_mask_excludes_nothing(self):
        cols, rows = [60, 120, 180], [60, 120, 180]
        a, centres = _flat_molecule_grid(cols, rows)
        empty = np.zeros((N, N), dtype=bool)
        assert len(_seg(a, exclude_mask=empty)) == len(_seg(a)) == len(centres)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            _seg(np.zeros((N, N)), exclude_mask=np.zeros((N, N // 2), dtype=bool))


_TERRACE = [(40, 60), (60, 180), (95, 110), (190, 60), (200, 175), (210, 120)]
_STEP = [(128, 45), (128, 128), (128, 210)]


class TestStepExclusionEndToEnd:
    """End-to-end, mirroring the GUI design: the step mask is computed from the
    RAW topography (which carries the step), while segmentation runs on the
    flattened image the user actually analyses.  Molecule positions are shared,
    so the band at the step lines up with the step-decorating molecules.
    """

    def _raw_topography(self, seed, *, include_step_mols=True):
        """Raw stepped surface: 0.235 nm step + tilt + bowl + low molecules + noise."""
        Y, X = _grid()
        a = _vertical_step() + _tilt(0.1e-9) + _bowl(0.1e-9)
        for cx, cy in _TERRACE + (_STEP if include_step_mols else []):
            a = a + _gauss_bump(cx, cy, 0.2e-9)
        return a + _rng(seed).normal(0, 12e-12, (N, N))

    def _flattened(self, seed, *, include_step_mols=True):
        """Terrace-leveled image the user segments: molecules on a flat field."""
        a = np.zeros((N, N))
        for cx, cy in _TERRACE + (_STEP if include_step_mols else []):
            a = np.maximum(a, _disk(cx, cy, 8, 5.0))   # r=8 → dense enough for Otsu
        return a + _rng(seed + 100).normal(0, 0.05, (N, N))

    @staticmethod
    def _near(parts, cx, cy, r=6):
        return any(abs(p.centroid_x_m / PX - cx) < r and abs(p.centroid_y_m / PX - cy) < r
                   for p in parts)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_step_decorating_molecules_dropped_terrace_survive(self, seed):
        raw = self._raw_topography(seed)
        mask = _mask(raw, threshold_deg=20.0, dilate_m=0.6e-9)
        assert mask.any(), "step must be detected on the raw topography"

        flat = self._flattened(seed)
        base = _seg(flat)
        kept = _seg(flat, exclude_mask=mask, max_exclude_overlap=0.25)

        assert len(base) == len(_TERRACE) + len(_STEP), "flat scene must segment cleanly"
        assert len(kept) == len(_TERRACE), f"got {len(kept)} (seed={seed})"
        assert all(not self._near(kept, *c) for c in _STEP), "step molecules must drop"
        assert all(self._near(kept, *c) for c in _TERRACE), "terrace molecules must survive"

    def test_no_step_molecules_keeps_all(self):
        raw = self._raw_topography(1, include_step_mols=False)
        mask = _mask(raw, threshold_deg=20.0, dilate_m=0.6e-9)
        flat = self._flattened(1, include_step_mols=False)
        assert len(_seg(flat, exclude_mask=mask)) == len(_seg(flat)) == len(_TERRACE)
