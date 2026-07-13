"""Tests for probeflow.io.readers.nanonis_sxm.read_sxm — the high-level reader.

The low-level sxm_io primitives are covered by test_sxm_io.py.  These tests
exercise the public entry point that callers (CLI, GUI, batch) actually use:
correct Scan structure, physical metadata, error paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_NANONIS_SCAN = Path(__file__).resolve().parents[1] / "test_data" / "nanonis.sxm"

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
    def test_reader_returns_scan_with_consistent_planes_and_source_metadata(self, scan):
        from probeflow.core.scan_model import Scan
        assert isinstance(scan, Scan)
        assert len(scan.planes) >= 1
        for plane in scan.planes:
            assert plane.ndim == 2
            assert plane.dtype == np.float64
        assert len(scan.plane_names) == len(scan.planes)
        assert len(scan.plane_units) == len(scan.planes)
        assert len(scan.plane_synthetic) == len(scan.planes)
        assert scan.source_format == "sxm"
        assert scan.source_path.is_absolute()
        shapes = [p.shape for p in scan.planes]
        assert len(set(shapes)) == 1, f"Planes have inconsistent shapes: {shapes}"
        for plane in scan.planes:
            assert np.isfinite(plane).any(), "Plane is all non-finite"


# ── Physical metadata ─────────────────────────────────────────────────────────

class TestReadSxmPhysical:
    def test_physical_metadata_has_finite_positive_scan_range_and_nanonis_header(self, scan):
        w_m, h_m = scan.scan_range_m
        assert w_m > 0 and h_m > 0
        assert all(np.isfinite(v) for v in scan.scan_range_m)
        assert isinstance(scan.header, dict)
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
