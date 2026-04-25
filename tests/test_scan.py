"""Tests for probeflow.scan — the vendor-agnostic Scan abstraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow import load_scan
from probeflow.dat_sxm import convert_dat_to_sxm
from probeflow.scan import Scan


TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"
_CREATEC_4CH = TESTDATA / "createc_scan_terrace_109nm.dat"
_NANONIS_SXM = TESTDATA / "sxm_moire_10nm.sxm"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sxm(tmp_path, first_sample_dat, cushion_dir) -> Path:
    """Produce a fresh .sxm from a bundled .dat so the round-trip is covered."""
    out_dir = tmp_path / "sxm_src"
    convert_dat_to_sxm(first_sample_dat, out_dir, cushion_dir)
    sxm_files = sorted(out_dir.glob("*.sxm"))
    assert sxm_files, "convert_dat_to_sxm produced no .sxm output"
    return sxm_files[0]


# ─── Dispatch ────────────────────────────────────────────────────────────────

class TestLoadScanDispatch:
    def test_sxm_suffix_dispatches(self, sample_sxm):
        scan = load_scan(sample_sxm)
        assert scan.source_format == "sxm"
        assert scan.source_path == sample_sxm

    def test_dat_suffix_dispatches(self, first_sample_dat):
        scan = load_scan(first_sample_dat)
        assert scan.source_format == "dat"

    def test_unknown_suffix_raises(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported"):
            load_scan(p)


# ─── Common Scan contract ───────────────────────────────────────────────────

class TestScanContract:
    def test_sxm_produces_valid_scan(self, sample_sxm):
        scan = load_scan(sample_sxm)
        assert isinstance(scan, Scan)
        assert scan.n_planes >= 1
        Nx, Ny = scan.dims
        assert Nx > 0 and Ny > 0
        for plane in scan.planes:
            assert plane.dtype == np.float64
            assert plane.shape == (Ny, Nx)

    def test_dat_produces_valid_scan(self, first_sample_dat):
        scan = load_scan(first_sample_dat)
        assert isinstance(scan, Scan)
        assert scan.n_planes == 4  # always canonical 4 planes
        Nx, Ny = scan.dims
        assert Nx > 0 and Ny > 0
        for plane in scan.planes:
            assert plane.dtype == np.float64
            assert plane.shape == (Ny, Nx)

    def test_dat_units_are_physical(self, first_sample_dat):
        scan = load_scan(first_sample_dat)
        assert scan.plane_units == ["m", "m", "A", "A"]
        # Z values should be on the nanometre/picometre scale.
        z_plane = scan.planes[0]
        finite = z_plane[np.isfinite(z_plane)]
        assert finite.size > 0
        # Anything within ±1 mm is "physically plausible" for an STM scan.
        assert float(np.max(np.abs(finite))) < 1e-3

    def test_dat_scan_range_positive(self, first_sample_dat):
        scan = load_scan(first_sample_dat)
        w_m, h_m = scan.scan_range_m
        assert w_m > 0 and h_m > 0

    def test_two_channel_dat_flags_synthetic(self, sample_dat_files):
        # At least one of the two bundled samples is a 2-channel file; that
        # one should have synthetic backward planes flagged.
        had_synthetic = False
        for dat in sample_dat_files:
            scan = load_scan(dat)
            if any(scan.plane_synthetic):
                had_synthetic = True
                # Synthetic planes are always the backward ones (indices 1, 3)
                assert scan.plane_synthetic[1] == scan.plane_synthetic[3]
                assert scan.plane_synthetic[0] is False
                assert scan.plane_synthetic[2] is False
        # The 2-channel file in the bundled sample set must be picked up.
        assert had_synthetic, \
            "Expected at least one sample .dat to be 2-channel (synthetic bwd)"


# ─── save_sxm round-trips ────────────────────────────────────────────────────

class TestSaveSxm:
    def test_sxm_source_roundtrip(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)
        assert out.exists()

        reloaded = load_scan(out)
        assert reloaded.n_planes == scan.n_planes
        for a, b in zip(scan.planes, reloaded.planes):
            finite = np.isfinite(a) & np.isfinite(b)
            # float32 storage round-trip: bit-exact on finite values.
            assert np.allclose(a[finite], b[finite], atol=0, rtol=0)

    def test_dat_source_produces_readable_sxm(self, first_sample_dat, tmp_path):
        scan = load_scan(first_sample_dat)
        out = tmp_path / "from_dat.sxm"
        scan.save_sxm(out)
        assert out.exists()

        # Loaded back as an .sxm it should be a valid Scan with matching dims.
        reloaded = load_scan(out)
        assert reloaded.source_format == "sxm"
        assert reloaded.dims == scan.dims

    def test_dat_processing_flows_through_to_sxm(
        self, first_sample_dat, tmp_path
    ):
        """A processed plane from a .dat-sourced Scan must be present in the
        resulting .sxm (after orientation/casting), up to float32 precision."""
        scan = load_scan(first_sample_dat)
        # Shift Z forward by a distinctive constant — this must survive.
        bump = 1.234e-9  # 1.234 nm
        scan.planes[0] = scan.planes[0] + bump

        out = tmp_path / "bumped.sxm"
        scan.save_sxm(out)

        reloaded = load_scan(out)
        # Plane 0 of the reloaded scan should equal original_plane0 + bump
        # (up to float32 rounding).
        original = load_scan(first_sample_dat).planes[0]
        diff = reloaded.planes[0] - original
        finite = np.isfinite(diff)
        assert np.allclose(diff[finite], bump, atol=1e-14, rtol=1e-3)


# ─── save_png ────────────────────────────────────────────────────────────────

class TestSavePng:
    def test_sxm_sourced_png_writes(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        out = tmp_path / "from_sxm.png"
        scan.save_png(out, plane_idx=0, colormap="gray")
        assert out.exists() and out.stat().st_size > 0

    def test_dat_sourced_png_writes(self, first_sample_dat, tmp_path):
        scan = load_scan(first_sample_dat)
        out = tmp_path / "from_dat.png"
        scan.save_png(out, plane_idx=0, colormap="gray")
        assert out.exists() and out.stat().st_size > 0

    def test_plane_idx_out_of_range_raises(self, first_sample_dat, tmp_path):
        scan = load_scan(first_sample_dat)
        with pytest.raises(ValueError):
            scan.save_png(tmp_path / "x.png", plane_idx=99)


# ─── processing_history field ────────────────────────────────────────────────

class TestProcessingHistory:
    def _make_scan(self, **kwargs):
        return Scan(
            planes=[np.zeros((4, 4))],
            plane_names=["Z forward"],
            plane_units=["m"],
            plane_synthetic=[False],
            header={},
            scan_range_m=(1e-8, 1e-8),
            source_path=Path("/fake/file.sxm"),
            source_format="sxm",
            **kwargs,
        )

    def test_default_is_empty_list(self):
        scan = self._make_scan()
        assert scan.processing_history == []

    def test_instances_do_not_share_history(self):
        a = self._make_scan()
        b = self._make_scan()
        a.processing_history.append({"op": "plane_bg"})
        assert b.processing_history == []

    def test_accepts_list_of_dicts(self):
        entry = {"op": "align_rows", "method": "median"}
        scan = self._make_scan(processing_history=[entry])
        assert scan.processing_history == [entry]


# ─── Backward-plane orientation ──────────────────────────────────────────────

def _fwd_bwd_corrs(scan: Scan) -> tuple[float, float]:
    """Return (corr_direct, corr_flipped) for the Z fwd/bwd pair."""
    fwd = scan.planes[0].ravel()
    bwd = scan.planes[1].ravel()
    corr_direct  = float(np.corrcoef(fwd, bwd)[0, 1])
    corr_flipped = float(np.corrcoef(fwd, np.fliplr(scan.planes[1]).ravel())[0, 1])
    return corr_direct, corr_flipped


class TestBackwardOrientation:
    """Regression suite for Createc backward-plane orientation.

    The core invariant: after reading, Z bwd must be in the same display
    orientation as Z fwd (same surface, same left-right order).  If the plane
    were mirrored, corr(fwd, fliplr(bwd)) would exceed corr(fwd, bwd).
    """

    def test_createc_backward_not_mirrored(self):
        if not _CREATEC_4CH.exists():
            pytest.skip(f"missing fixture: {_CREATEC_4CH.name}")
        scan = load_scan(_CREATEC_4CH)
        assert not any(scan.plane_synthetic), "expected a 4-channel file"
        direct, flipped = _fwd_bwd_corrs(scan)
        assert direct > flipped, (
            f"Z bwd appears mirrored: corr(fwd,bwd)={direct:.3f} "
            f"< corr(fwd,fliplr(bwd))={flipped:.3f}"
        )

    def test_createc_backward_high_correlation(self):
        if not _CREATEC_4CH.exists():
            pytest.skip(f"missing fixture: {_CREATEC_4CH.name}")
        scan = load_scan(_CREATEC_4CH)
        direct, _ = _fwd_bwd_corrs(scan)
        assert direct > 0.85, (
            f"Z fwd/bwd correlation too low ({direct:.3f}); "
            "expected both to show the same surface"
        )

    def test_nanonis_sxm_backward_not_mirrored(self):
        if not _NANONIS_SXM.exists():
            pytest.skip(f"missing fixture: {_NANONIS_SXM.name}")
        scan = load_scan(_NANONIS_SXM)
        if scan.n_planes < 2:
            pytest.skip("sxm fixture has only one plane")
        direct, flipped = _fwd_bwd_corrs(scan)
        assert direct > flipped, (
            f"Nanonis Z bwd appears mirrored: corr(fwd,bwd)={direct:.3f} "
            f"< corr(fwd,fliplr(bwd))={flipped:.3f}"
        )

    def test_dat_to_sxm_roundtrip_preserves_orientation(self, sample_dat_files, tmp_path):
        """Z fwd/bwd orientation must be identical after save_sxm() + load_scan()."""
        four_ch = [f for f in sample_dat_files
                   if not any(load_scan(f).plane_synthetic)]
        if not four_ch:
            pytest.skip("no 4-channel .dat found in sample_dat_files")
        scan = load_scan(four_ch[0])
        direct_dat, _ = _fwd_bwd_corrs(scan)

        out = tmp_path / "rt.sxm"
        scan.save_sxm(out)
        scan_rt = load_scan(out)
        direct_sxm, _ = _fwd_bwd_corrs(scan_rt)

        assert abs(direct_dat - direct_sxm) < 0.02, (
            f"Round-trip changed Z fwd/bwd correlation: "
            f".dat={direct_dat:.4f}, .sxm={direct_sxm:.4f}"
        )
