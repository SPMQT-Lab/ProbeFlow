"""Tests for probeflow.io.readers.nanonis_sxm.read_sxm — the high-level reader.

The low-level sxm_io primitives are covered by test_sxm_io.py.  These tests
exercise the public entry point that callers (CLI, GUI, batch) actually use:
correct Scan structure, physical metadata, error paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_NANONIS_SCAN = Path(__file__).resolve().parents[1] / "test_data" / "sxm_moire_10nm.sxm"

pytestmark = pytest.mark.skipif(
    not _NANONIS_SCAN.exists(),
    reason=f"sample SXM not found: {_NANONIS_SCAN}",
)


@pytest.fixture(scope="module")
def scan():
    from probeflow.io.readers.nanonis_sxm import read_sxm
    return read_sxm(_NANONIS_SCAN)


# ── Basic Scan structure ──────────────────────────────────────────────────────

class TestReadSxmStructure:
    def test_returns_scan_object(self, scan):
        from probeflow.core.scan_model import Scan
        assert isinstance(scan, Scan)

    def test_has_at_least_one_plane(self, scan):
        assert len(scan.planes) >= 1

    def test_planes_are_2d_float64(self, scan):
        for plane in scan.planes:
            assert plane.ndim == 2
            assert plane.dtype == np.float64

    def test_plane_names_and_units_match_plane_count(self, scan):
        assert len(scan.plane_names) == len(scan.planes)
        assert len(scan.plane_units) == len(scan.planes)

    def test_plane_synthetic_matches_plane_count(self, scan):
        assert len(scan.plane_synthetic) == len(scan.planes)

    def test_source_format_is_sxm(self, scan):
        assert scan.source_format == "sxm"

    def test_source_path_is_absolute(self, scan):
        assert scan.source_path.is_absolute()

    def test_all_planes_same_shape(self, scan):
        shapes = [p.shape for p in scan.planes]
        assert len(set(shapes)) == 1, f"Planes have inconsistent shapes: {shapes}"

    def test_planes_contain_finite_values(self, scan):
        for plane in scan.planes:
            assert np.isfinite(plane).any(), "Plane is all non-finite"


# ── Physical metadata ─────────────────────────────────────────────────────────

class TestReadSxmPhysical:
    def test_scan_range_is_positive(self, scan):
        w_m, h_m = scan.scan_range_m
        assert w_m > 0 and h_m > 0

    def test_scan_range_is_finite(self, scan):
        assert all(np.isfinite(v) for v in scan.scan_range_m)

    def test_header_is_dict(self, scan):
        assert isinstance(scan.header, dict)

    def test_header_has_nanonis_keys(self, scan):
        assert "NANONIS_VERSION" in scan.header or "SCAN_PIXELS" in scan.header


# ── Error handling ────────────────────────────────────────────────────────────

class TestReadSxmErrors:
    def test_missing_scanit_end_raises_value_error(self, tmp_path):
        from probeflow.io.readers.nanonis_sxm import read_sxm
        bad = tmp_path / "truncated.sxm"
        bad.write_bytes(b":NANONIS_VERSION:\n2\n:SCAN_PIXELS:\n4 4\n")
        with pytest.raises(ValueError):
            read_sxm(bad)

    def test_nonexistent_file_raises(self, tmp_path):
        from probeflow.io.readers.nanonis_sxm import read_sxm
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            read_sxm(tmp_path / "does_not_exist.sxm")
