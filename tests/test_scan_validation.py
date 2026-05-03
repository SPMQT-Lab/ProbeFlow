"""Tests for probeflow.core.validation.validate_scan()."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.core.scan_model import Scan
from probeflow.core.scan_loader import load_scan
from probeflow.core.validation import validate_scan


TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"
_CREATEC_STEP    = TESTDATA / "createc_scan_step_20nm.dat"
_CREATEC_TERRACE = TESTDATA / "createc_scan_terrace_109nm.dat"
_NANONIS_SXM     = TESTDATA / "sxm_moire_10nm.sxm"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _minimal_scan(**overrides) -> Scan:
    """Return a valid minimal Scan; keyword overrides replace individual fields."""
    defaults = dict(
        planes=[np.zeros((4, 4))],
        plane_names=["Z forward"],
        plane_units=["m"],
        plane_synthetic=[False],
        header={},
        scan_range_m=(1e-9, 1e-9),
        source_path=Path("fake.dat"),
        source_format="dat",
    )
    defaults.update(overrides)
    return Scan(**defaults)


# ── Test A: real Createc fixtures validate ────────────────────────────────────

class TestRealCreatecFixtures:
    @pytest.mark.parametrize("path,expected_shape", [
        (_CREATEC_STEP,    (330, 511)),
        (_CREATEC_TERRACE, (512, 511)),
    ])
    def test_validates_without_error(self, path, expected_shape):
        scan = load_scan(path)
        validate_scan(scan)  # must not raise

    @pytest.mark.parametrize("path,expected_shape", [
        (_CREATEC_STEP,    (330, 511)),
        (_CREATEC_TERRACE, (512, 511)),
    ])
    def test_post_fix_dimensions(self, path, expected_shape):
        scan = load_scan(path)
        assert scan.planes[0].shape == expected_shape, (
            f"{path.name}: expected shape {expected_shape}, "
            f"got {scan.planes[0].shape}"
        )


# ── Test B: Createc header agrees with loaded width ───────────────────────────

class TestCreatecHeaderConsistency:
    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_nx_header_matches_array_width(self, path):
        scan = load_scan(path)
        Ny, Nx = scan.planes[0].shape
        nx_hdr = int(float(scan.header["Num.X"]))
        assert Nx == nx_hdr

    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_ny_header_matches_array_height(self, path):
        scan = load_scan(path)
        Ny, Nx = scan.planes[0].shape
        ny_hdr = int(float(scan.header["Num.Y"]))
        assert Ny == ny_hdr


# ── Test C: mismatched plane shapes ──────────────────────────────────────────

class TestMismatchedShapes:
    def test_raises_on_different_plane_shapes(self):
        scan = _minimal_scan(
            planes=[np.zeros((4, 4)), np.zeros((4, 5))],
            plane_names=["Z forward", "Z backward"],
            plane_units=["m", "m"],
            plane_synthetic=[False, False],
        )
        with pytest.raises(ValueError, match="shape"):
            validate_scan(scan)


# ── Test D: empty plane list ─────────────────────────────────────────────────

class TestEmptyPlanes:
    def test_raises_on_empty_planes(self):
        scan = _minimal_scan(planes=[], plane_names=[], plane_units=[], plane_synthetic=[])
        with pytest.raises(ValueError, match="empty"):
            validate_scan(scan)


# ── Test E: all-NaN plane ────────────────────────────────────────────────────

class TestAllNaNPlane:
    def test_raises_on_all_nan_plane(self):
        arr = np.full((4, 4), np.nan)
        scan = _minimal_scan(planes=[arr])
        with pytest.raises(ValueError, match="finite"):
            validate_scan(scan)

    def test_partial_nan_is_accepted(self):
        arr = np.zeros((4, 4))
        arr[0, 0] = np.nan
        scan = _minimal_scan(planes=[arr])
        validate_scan(scan)  # must not raise


# ── Test F: Nanonis fixture validates ─────────────────────────────────────────

class TestNanonisFixture:
    def test_validates_without_error(self):
        scan = load_scan(_NANONIS_SXM)
        validate_scan(scan)  # must not raise


# ── Additional edge-case coverage ────────────────────────────────────────────

class TestMetadataLengthMismatches:
    def test_raises_on_short_plane_names(self):
        scan = _minimal_scan(
            planes=[np.zeros((4, 4)), np.zeros((4, 4))],
            plane_names=["Z forward"],        # too short
            plane_units=["m", "m"],
            plane_synthetic=[False, False],
        )
        with pytest.raises(ValueError, match="plane_names"):
            validate_scan(scan)

    def test_raises_on_short_plane_units(self):
        scan = _minimal_scan(
            planes=[np.zeros((4, 4)), np.zeros((4, 4))],
            plane_names=["Z forward", "Z backward"],
            plane_units=["m"],               # too short
            plane_synthetic=[False, False],
        )
        with pytest.raises(ValueError, match="plane_units"):
            validate_scan(scan)


class TestNameAndUnitSanity:
    def test_raises_on_empty_plane_name(self):
        scan = _minimal_scan(plane_names=[""])
        with pytest.raises(ValueError, match="non-empty string"):
            validate_scan(scan)

    def test_raises_on_none_unit(self):
        scan = _minimal_scan(plane_units=[None])
        with pytest.raises(ValueError, match="plane_units"):
            validate_scan(scan)

    def test_empty_string_unit_accepted(self):
        scan = _minimal_scan(plane_units=[""])
        validate_scan(scan)  # empty string is explicitly allowed


class TestSourceFormat:
    def test_raises_on_unknown_format(self):
        scan = _minimal_scan(source_format="hdf5")
        with pytest.raises(ValueError, match="source_format"):
            validate_scan(scan)
