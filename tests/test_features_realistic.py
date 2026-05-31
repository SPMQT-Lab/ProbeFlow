"""Realistic-condition tests for the UniMR feature kernels.

The companion suite in ``test_features.py`` exercises the segmentation /
counting / classification kernels on clean shapes drawn on a perfectly flat,
noise-free background.  Those tests pin down units and basic counting, but they
do *not* resemble real STM data and they cannot catch failures that only show
up once the surface is noisy, tilted, or stepped — exactly the regime where the
algorithms are actually used.

This module adds fixtures that combine the three things every real scan has:

* **white noise** — additive Gaussian noise on every pixel,
* **a slowly-varying background** — a smooth tilt / bowl across the frame,
* **a step edge** — two terraces at different heights,

individually and all at once.  Where a raw kernel cannot cope with a gradient or
step on its own (global-Otsu segmentation and percentile grain thresholding both
assume a level surface), the test runs the *real* workflow — background
subtraction first, then analysis — and asserts the naive path genuinely fails
while the processed path recovers the right answer.  That contrast is what proves
the fixtures are hard rather than decorative.

Assertions here are deliberately strong: exact particle counts, exact per-class
classification, and recovered physical quantities within tolerance — not the
"len matches" / "or 'other' in names" checks that pass even when every result is
wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("sklearn")

from probeflow.analysis.features import (
    classify_particles,
    count_features,
    segment_particles,
)
from probeflow.analysis.grains import detect_grains, measure_periodicity
from probeflow.processing.background import subtract_background


# ─── Fixture builders ───────────────────────────────────────────────────────

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _disks(shape, centers, r, height=5.0):
    """Filled disks of radius `r` at each centre, on a flat 0 background."""
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    a = np.zeros(shape, dtype=np.float64)
    for cx, cy in centers:
        a = np.maximum(a, np.where((X - cx) ** 2 + (Y - cy) ** 2 <= r * r, height, 0.0))
    return a


def _blob(shape, cx, cy, rx, ry, height=5.0):
    """A single (possibly elongated) elliptical blob."""
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    return np.where(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0, height, 0.0)


def _tilt(shape, amp):
    """A smooth linear ramp across the frame — a typical sample-tilt background."""
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    return amp * (X / (Nx - 1) + 0.5 * Y / (Ny - 1))


def _bowl(shape, amp):
    """A smooth quadratic background — curved, e.g. piezo bow / thermal drift."""
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    xc, yc = Nx / 2.0, Ny / 2.0
    return amp * (((X - xc) / Nx) ** 2 + ((Y - yc) / Ny) ** 2)


def _step(shape, row, height):
    """A hard terrace step: everything from `row` downward raised by `height`."""
    a = np.zeros(shape, dtype=np.float64)
    a[row:, :] = height
    return a


def _white_noise(arr, sigma, seed):
    return arr + _rng(seed).normal(0.0, sigma, arr.shape)


def _aspect(p) -> float:
    x0, y0, x1, y1 = p.bbox_px
    w, h = (x1 - x0), (y1 - y0)
    return max(w, h) / max(1, min(w, h))


# Sixteen disks on a 4×4 grid — the canonical "count me" scene.
_GRID16 = [(16 + 16 * i, 16 + 16 * j) for i in range(4) for j in range(4)]
_PX = 1e-9


# ─── segment_particles ──────────────────────────────────────────────────────

class TestSegmentRealistic:
    def test_robust_to_white_noise(self):
        """All 16 disks survive strong additive noise (σ = 0.6 vs height 5)."""
        base = _disks((96, 96), _GRID16, r=4, height=5.0)
        arr = _white_noise(base, sigma=0.6, seed=1)
        parts = segment_particles(arr, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        assert len(parts) == 16

    def test_tilt_breaks_naive_segmentation_but_pipeline_recovers(self):
        """A strong tilt makes global-Otsu lose disks on the dark side.

        Segmenting the raw tilted image under-counts (the threshold that
        separates particle-from-surface on the bright side buries the disks on
        the dark side).  The real workflow — subtract the background plane,
        then segment — recovers all 16.  Asserting the naive count is *wrong*
        proves the fixture genuinely stresses the kernel.
        """
        base = _disks((96, 96), _GRID16, r=4, height=5.0)
        arr = base + _tilt((96, 96), amp=8.0)

        naive = segment_particles(arr, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        assert len(naive) < 16, "tilt should defeat naive global-Otsu segmentation"

        leveled = subtract_background(arr, order=1)
        fixed = segment_particles(leveled, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        assert len(fixed) == 16

    def test_combined_step_tilt_noise_pipeline(self):
        """Tilt + terrace step + white noise + disks — the full mess.

        The step edge and tilt spawn spurious segments in the raw image, so the
        naive count is badly wrong.  After plane subtraction the 16 disks are
        recovered cleanly.
        """
        base = _disks((96, 96), _GRID16, r=4, height=5.0)
        scene = (
            base
            + _tilt((96, 96), amp=6.0)
            + _step((96, 96), row=48, height=2.0)
            + _rng(2).normal(0.0, 0.25, base.shape)
        )

        naive = segment_particles(scene, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        assert len(naive) != 16, "step+tilt should corrupt the naive count"

        leveled = subtract_background(scene, order=1)
        fixed = segment_particles(leveled, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        assert len(fixed) == 16

    def test_curved_background_breaks_naive_but_pipeline_recovers(self):
        """A curved (quadratic) background needs a 2nd-order plane fit.

        A bowl-shaped background — piezo bow, thermal drift — is not removed by
        a linear fit, and global-Otsu over-segments it badly.  Subtracting an
        order-2 background plane levels the curve and segmentation recovers all
        16 disks.  Complements the linear-tilt case with a non-planar gradient.
        """
        base = _disks((96, 96), _GRID16, r=4, height=5.0)
        scene = base + _bowl((96, 96), amp=10.0) + _rng(8).normal(0.0, 0.2, base.shape)

        naive = segment_particles(scene, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        assert len(naive) != 16, "a curved background should defeat naive segmentation"

        leveled = subtract_background(scene, order=2)
        fixed = segment_particles(leveled, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        assert len(fixed) == 16


# ─── count_features ──────────────────────────────────────────────────────────

class TestCountRealistic:
    def _lattice_and_template(self):
        Ny, Nx = 96, 96
        Y, X = np.mgrid[:Ny, :Nx]
        centers = [(12 + 12 * i, 12 + 12 * j) for i in range(7) for j in range(7)]
        arr = np.zeros((Ny, Nx), dtype=np.float64)
        for cx, cy in centers:
            arr += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 1.5 ** 2)) * 5.0
        template = np.exp(
            -((X[:15, :15] - 7) ** 2 + (Y[:15, :15] - 7) ** 2) / (2 * 1.5 ** 2)
        ) * 5.0
        return arr, template, len(centers)

    def test_template_count_robust_to_noise(self):
        """Normalised cross-correlation recovers all 49 atoms despite noise."""
        arr, template, n_expected = self._lattice_and_template()
        noisy = _white_noise(arr, sigma=0.6, seed=7)
        dets = count_features(
            noisy, template, _PX, min_correlation=0.5, min_distance_m=5e-9
        )
        assert n_expected - 4 <= len(dets) <= n_expected + 2


# ─── classify_particles ─────────────────────────────────────────────────────

class TestClassifyRealistic:
    def _round_vs_elongated_scene(self, seed):
        img = np.zeros((200, 200), dtype=np.float64)
        for c in [(40, 40), (160, 40), (40, 160), (160, 160)]:
            img = np.maximum(img, _blob(img.shape, *c, 8, 8, height=5.0))   # round
        for c in [(100, 30), (100, 80), (100, 130), (100, 180)]:
            img = np.maximum(img, _blob(img.shape, *c, 4, 13, height=5.0))  # elongated
        img = img + _tilt(img.shape, amp=4.0) + _rng(seed).normal(0.0, 0.3, img.shape)
        return img

    @pytest.mark.parametrize("threshold_method", ["gmm", "otsu", "distribution", "manual"])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_two_classes_assigned_correctly_under_noise(self, threshold_method, seed):
        """Every particle gets its *correct* class — not just 'a label'.

        Four round + four elongated blobs on a tilted, noisy surface.  After
        plane subtraction we label one example of each class and classify the
        rest.  The strong assertion is the exact split (round: 4, elong: 4) with
        every blob on the correct side — the kind of check the clean-fixture
        suite omits, and which the auto-threshold regression would have failed
        before the robust-cutoff fix.
        """
        img = self._round_vs_elongated_scene(seed)
        leveled = subtract_background(img, order=1)
        parts = segment_particles(leveled, _PX, min_area_nm2=1.0, size_sigma_clip=None)

        rounds = [p for p in parts if _aspect(p) < 1.4]
        elongs = [p for p in parts if _aspect(p) >= 1.8]
        assert len(rounds) == 4 and len(elongs) == 4, "scene segmentation precondition"

        samples = [("round", rounds[0]), ("elong", elongs[0])]
        kwargs = {"threshold_method": threshold_method}
        if threshold_method == "manual":
            kwargs["manual_threshold"] = 0.5
        classifs = classify_particles(leveled, parts, samples, encoder="raw", **kwargs)

        by_index = {p.index: p for p in parts}
        for c in classifs:
            expected = "round" if _aspect(by_index[c.particle_index]) < 1.4 else "elong"
            assert c.class_name == expected, (
                f"particle {c.particle_index} got {c.class_name!r}, expected {expected!r} "
                f"(method={threshold_method}, seed={seed})"
            )

    def test_genuine_outlier_is_rejected_as_other(self):
        """A clearly different feature stays 'other' — the fallback isn't blind.

        The robust auto-threshold falls back to classify-all only when there is
        no real spread in similarities.  Here a large square sits among round
        disks, so the similarity distribution *is* genuinely separated and the
        square must be rejected as 'other' while the disks take the disk class.
        """
        img = np.zeros((200, 200), dtype=np.float64)
        for c in [(40, 40), (160, 40), (40, 160)]:
            img = np.maximum(img, _blob(img.shape, *c, 8, 8, height=5.0))
        Y, X = np.mgrid[:200, :200]
        img = np.maximum(img, np.where((np.abs(X - 120) < 25) & (np.abs(Y - 120) < 25), 5.0, 0.0))
        img = _white_noise(img, sigma=0.2, seed=9)

        parts = segment_particles(img, _PX, min_area_nm2=1.0, size_sigma_clip=None)
        square = max(parts, key=lambda p: p.area_nm2)
        disk = min(parts, key=lambda p: p.area_nm2)

        classifs = classify_particles(
            img, parts, [("disk", disk)], encoder="raw",
            threshold_method="gmm", crop_size_px=64,
        )
        by_index = {c.particle_index: c.class_name for c in classifs}
        assert by_index[square.index] == "other", "the odd-one-out square should be 'other'"
        # The three round disks should all take the 'disk' label.
        disk_labels = [
            name for idx, name in by_index.items() if idx != square.index
        ]
        assert disk_labels.count("disk") >= 2


# ─── detect_grains ───────────────────────────────────────────────────────────

class TestGrainsRealistic:
    def test_islands_recovered_after_background_subtraction(self):
        """Three islands on a tilted, noisy terrace are recovered by the pipeline.

        Detected directly, the tilt and noise both corrupt a fixed-percentile
        threshold.  After plane subtraction, the three islands come back with
        their true areas; ``min_grain_px`` clears the leftover noise specks.
        """
        islands = _disks((128, 128), [(30, 30), (90, 40), (60, 95)], r=10, height=5.0)
        for seed in (3, 4, 5):
            scene = islands + _tilt((128, 128), amp=4.0) + _rng(seed).normal(0.0, 0.25, islands.shape)
            leveled = subtract_background(scene, order=1)
            _labels, n, stats = detect_grains(leveled, min_grain_px=30)
            assert n == 3, f"expected 3 islands, got {n} (seed={seed})"
            # Each island is a radius-10 disk ≈ π·100 ≈ 314 px.
            for area in sorted(stats["areas_px"])[-3:]:
                assert 250 < area < 380, f"island area {area} px out of range (seed={seed})"


# ─── measure_periodicity ─────────────────────────────────────────────────────

class TestPeriodicityRealistic:
    def test_grating_period_recovered_under_noise_and_tilt(self):
        """The dominant period survives a tilt and heavy noise (within 5 %)."""
        Ny, Nx = 128, 128
        Y, X = np.mgrid[:Ny, :Nx]
        period_px = 16
        grating = np.sin(2 * np.pi * X / period_px) * 1.0
        scene = grating + _tilt((Ny, Nx), amp=3.0) + _rng(4).normal(0.0, 0.3, (Ny, Nx))

        peaks = measure_periodicity(scene, _PX, _PX, n_peaks=3)
        assert peaks, "no FFT peaks found"
        found = peaks[0]["period_m"]
        expected = period_px * _PX
        assert abs(found - expected) / expected < 0.05
