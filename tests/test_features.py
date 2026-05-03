"""Tests for probeflow.analysis.features — particle segmentation, counting, classification."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("sklearn")

from probeflow.analysis.features import (
    Classification,
    Detection,
    Particle,
    classify_particles,
    count_features,
    segment_particles,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _disk(shape, cx, cy, r, height=1.0):
    """Return a 2-D array with a filled disk of `height` on a flat 0 background."""
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    m = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    a = np.zeros(shape, dtype=np.float64)
    a[m] = height
    return a


def _disks_grid(shape, centers, r, height=1.0):
    a = np.zeros(shape, dtype=np.float64)
    for (cx, cy) in centers:
        a = np.maximum(a, _disk(shape, cx, cy, r, height))
    return a


# ─── segment_particles ──────────────────────────────────────────────────────

class TestSegmentParticles:
    def test_counts_grid_of_disks(self):
        # 4x4 grid of disks, radius 4 px, pitch 16 px → 16 disks
        centers = [(16 + 16 * i, 16 + 16 * j) for i in range(4) for j in range(4)]
        arr = _disks_grid((96, 96), centers, r=4, height=5.0)
        particles = segment_particles(
            arr, pixel_size_m=1e-9,
            min_area_nm2=1.0,
            size_sigma_clip=None,
        )
        assert len(particles) == 16

    def test_particle_has_physical_units(self):
        arr = _disk((64, 64), 32, 32, r=5, height=1.0)
        parts = segment_particles(
            arr, pixel_size_m=2e-10,   # 0.2 nm / pixel
            min_area_nm2=0.1,
        )
        assert len(parts) == 1
        p = parts[0]
        # Disk radius=5 px, pixel=0.2 nm → area ≈ π·(5·0.2)² ≈ π nm² ≈ 3.14
        assert 2.0 < p.area_nm2 < 5.0

    def test_min_area_filter(self):
        # Two disks of very different sizes → filter should drop the small one.
        arr = _disks_grid((64, 64), [(16, 16)], r=2, height=1.0)
        arr = np.maximum(arr, _disk((64, 64), 48, 48, r=8, height=1.0))
        parts = segment_particles(
            arr, pixel_size_m=1e-9,
            min_area_nm2=20.0,
            size_sigma_clip=None,
        )
        assert len(parts) == 1
        assert parts[0].area_nm2 > 20.0

    def test_max_area_filter(self):
        arr = np.ones((32, 32), dtype=np.float64)     # one giant particle
        arr = np.maximum(arr, _disk((32, 32), 16, 16, r=4, height=5.0))
        parts = segment_particles(
            arr, pixel_size_m=1e-9,
            min_area_nm2=0.5,
            max_area_nm2=100.0,
            size_sigma_clip=None,
        )
        # The whole-image particle is >> 100 nm², so it gets filtered out.
        assert all(p.area_nm2 <= 100.0 for p in parts)

    def test_invert_detects_depressions(self):
        arr = np.full((64, 64), 10.0, dtype=np.float64)
        arr = np.where(_disk((64, 64), 32, 32, r=6, height=1.0) > 0, 0.0, arr)
        parts = segment_particles(
            arr, pixel_size_m=1e-9,
            min_area_nm2=1.0,
            invert=True,
            size_sigma_clip=None,
        )
        assert len(parts) == 1

    def test_empty_plane_no_particles(self):
        arr = np.zeros((32, 32), dtype=np.float64)
        parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.1)
        assert parts == []

    def test_to_dict_roundtrip(self):
        arr = _disk((64, 64), 32, 32, r=5, height=1.0)
        parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.1)
        d = parts[0].to_dict()
        assert "area_nm2" in d
        assert "centroid_x_m" in d

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            segment_particles(np.zeros((4, 4, 4)), pixel_size_m=1e-9)

    def test_rejects_bad_pixel_size(self):
        with pytest.raises(ValueError):
            segment_particles(np.zeros((8, 8)), pixel_size_m=0.0)

    def test_manual_threshold(self):
        arr = _disk((64, 64), 32, 32, r=6, height=2.0)
        parts = segment_particles(
            arr, pixel_size_m=1e-9,
            threshold="manual", manual_value=128.0,
            min_area_nm2=0.5,
        )
        assert len(parts) == 1


# ─── count_features ──────────────────────────────────────────────────────────

class TestCountFeatures:
    def test_counts_square_lattice(self):
        # 8×8 grid of gaussian blobs (mimics atoms)
        centers = [(12 + 12 * i, 12 + 12 * j) for i in range(7) for j in range(7)]
        shape = (96, 96)
        Y, X = np.mgrid[:shape[0], :shape[1]]
        arr = np.zeros(shape, dtype=np.float64)
        for cx, cy in centers:
            arr += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 1.5 ** 2))

        # Template = one clean gaussian
        t = np.exp(-((X[:15, :15] - 7) ** 2 + (Y[:15, :15] - 7) ** 2) / (2 * 1.5 ** 2))
        dets = count_features(
            arr, t, pixel_size_m=1e-10,
            min_correlation=0.5,
            min_distance_m=0.5e-9,  # 5 px
        )
        # Should recover ~49 detections (some edge losses allowed).
        assert 40 <= len(dets) <= 55

    def test_detections_have_positions(self):
        arr = _disks_grid((64, 64), [(16, 16), (48, 48)], r=3, height=5.0)
        Y, X = np.mgrid[:64, :64]
        tmpl = _disk((12, 12), 6, 6, r=3, height=5.0)
        dets = count_features(
            arr, tmpl, pixel_size_m=1e-9,
            min_correlation=0.4,
            min_distance_m=5e-9,
        )
        assert len(dets) >= 2
        # Positions round-trip to physical units.
        for d in dets:
            assert d.x_m >= 0 and d.y_m >= 0
            assert d.correlation >= 0.4

    def test_rejects_oversize_template(self):
        arr = np.zeros((32, 32), dtype=np.float64)
        tmpl = np.zeros((64, 64), dtype=np.float64)
        with pytest.raises(ValueError):
            count_features(arr, tmpl, pixel_size_m=1e-9)

    def test_empty_on_low_correlation(self):
        arr = np.zeros((32, 32), dtype=np.float64)
        tmpl = _disk((8, 8), 4, 4, r=2, height=1.0)
        dets = count_features(
            arr, tmpl, pixel_size_m=1e-9,
            min_correlation=0.99,
        )
        assert dets == []

    def test_to_dict(self):
        arr = _disk((64, 64), 32, 32, r=3, height=5.0)
        tmpl = _disk((12, 12), 6, 6, r=3, height=5.0)
        dets = count_features(arr, tmpl, pixel_size_m=1e-9, min_correlation=0.5)
        if dets:
            d = dets[0].to_dict()
            assert "x_m" in d and "correlation" in d


# ─── classify_particles ─────────────────────────────────────────────────────

class TestClassifyParticles:
    def test_two_classes_separate_cleanly(self):
        # Two kinds of "molecules": small bright disks vs larger fainter disks.
        shape = (96, 96)
        centers_a = [(16 + 32 * i, 16) for i in range(3)]   # small/bright
        centers_b = [(16 + 32 * i, 64) for i in range(3)]   # larger/fainter
        arr = np.zeros(shape, dtype=np.float64)
        for cx, cy in centers_a:
            arr = np.maximum(arr, _disk(shape, cx, cy, r=3, height=5.0))
        for cx, cy in centers_b:
            arr = np.maximum(arr, _disk(shape, cx, cy, r=6, height=1.0))

        parts = segment_particles(
            arr, pixel_size_m=1e-9,
            min_area_nm2=2.0,
            size_sigma_clip=None,
        )
        assert len(parts) >= 6

        # Pick one sample from each class (by centroid y coordinate).
        a_sample = next(p for p in parts if p.centroid_y_m < 32e-9)
        b_sample = next(p for p in parts if p.centroid_y_m > 50e-9)
        samples = [("small", a_sample), ("large", b_sample)]

        classifs = classify_particles(
            arr, parts, samples,
            encoder="raw",
            threshold_method="distribution",
        )
        assert len(classifs) == len(parts)
        names = {c.class_name for c in classifs}
        assert {"small", "large"} <= names or "other" in names

    def test_no_samples_all_other(self):
        arr = _disk((64, 64), 32, 32, r=5, height=1.0)
        parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.1)
        classifs = classify_particles(arr, parts, samples=[])
        assert all(c.class_name == "other" for c in classifs)

    def test_empty_particles_empty_result(self):
        arr = np.zeros((32, 32), dtype=np.float64)
        classifs = classify_particles(arr, [], samples=[])
        assert classifs == []

    def test_pca_encoder(self):
        arr = _disks_grid((96, 96),
                          [(16 + 24 * i, 16 + 24 * j) for i in range(3) for j in range(3)],
                          r=4, height=1.0)
        parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5,
                                  size_sigma_clip=None)
        assert len(parts) >= 2
        samples = [("blob", parts[0])]
        classifs = classify_particles(
            arr, parts, samples, encoder="pca_kmeans",
            threshold_method="gmm",
        )
        assert len(classifs) == len(parts)
