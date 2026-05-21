"""Tests for CLI scan commands and shared processing_ops helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from probeflow.cli.commands.scan import _cmd_info, _cmd_sxm2png
from probeflow.cli.processing_ops import (
    _ensure_output_available,
    _load_plane_for_analysis,
    _pixel_sizes_m_from_scan,
)
from probeflow.core.scan_model import Scan

TEST_DATA = Path(__file__).parent.parent / "test_data"
SAMPLE_DAT = TEST_DATA / "createc_scan_close_100nm.dat"
SAMPLE_SXM = TEST_DATA / "sxm_moire_10nm.sxm"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"test_data not available: {path.name}")


# ─── _ensure_output_available ────────────────────────────────────────────────

class TestEnsureOutputAvailable:
    def test_missing_path_passes(self, tmp_path):
        out = tmp_path / "does_not_exist.png"
        # Should not raise.
        _ensure_output_available(out, force=False)

    def test_existing_path_without_force_raises(self, tmp_path):
        out = tmp_path / "exists.png"
        out.write_bytes(b"sentinel")
        with pytest.raises(ValueError, match="already exists"):
            _ensure_output_available(out, force=False)

    def test_existing_path_with_force_passes(self, tmp_path):
        out = tmp_path / "exists.png"
        out.write_bytes(b"sentinel")
        # Should not raise when force=True.
        _ensure_output_available(out, force=True)


# ─── _pixel_sizes_m_from_scan ────────────────────────────────────────────────

class TestPixelSizesFromScan:
    def _make_scan(self, scan_range_m, shape):
        return Scan(
            planes=[np.zeros(shape)],
            plane_names=["Z forward"],
            plane_units=["m"],
            plane_synthetic=[False],
            header={},
            scan_range_m=scan_range_m,
            source_path=Path("/fake/file.sxm"),
            source_format="sxm",
        )

    def test_returns_positive_floats(self):
        scan = self._make_scan((50e-9, 50e-9), (50, 50))
        dx, dy = _pixel_sizes_m_from_scan(scan)
        assert isinstance(dx, float)
        assert isinstance(dy, float)
        assert dx > 0
        assert dy > 0

    def test_per_pixel_geometry(self):
        # shape=(Ny=60, Nx=100); scan_range_m=(w_m=100e-9, h_m=60e-9)
        # → dx = 100e-9 / 100 = 1e-9; dy = 60e-9 / 60 = 1e-9
        scan = self._make_scan((100e-9, 60e-9), (60, 100))
        dx, dy = _pixel_sizes_m_from_scan(scan)
        assert dx == pytest.approx(1e-9)
        assert dy == pytest.approx(1e-9)


# ─── _load_plane_for_analysis ────────────────────────────────────────────────

class TestLoadPlaneForAnalysis:
    def test_returns_float64_ndarray_for_valid_dat(self):
        _require(SAMPLE_DAT)
        args = argparse.Namespace(input=SAMPLE_DAT, plane=0, verbose=False)
        arr = _load_plane_for_analysis(args)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2
        assert arr.dtype == np.float64

    def test_returns_none_for_missing_file(self, tmp_path):
        missing = tmp_path / "no_such_file.dat"
        args = argparse.Namespace(input=missing, plane=0, verbose=False)
        result = _load_plane_for_analysis(args)
        assert result is None


# ─── _cmd_info ───────────────────────────────────────────────────────────────

def _info_args(path, *, json_out=False, verbose=False):
    return argparse.Namespace(input=path, json=json_out, verbose=verbose)


class TestCmdInfo:
    def test_returns_0_for_valid_dat(self, capsys):
        _require(SAMPLE_DAT)
        rc = _cmd_info(_info_args(SAMPLE_DAT))
        assert rc == 0

    def test_returns_1_for_missing_file(self, tmp_path):
        missing = tmp_path / "no_such_file.dat"
        rc = _cmd_info(_info_args(missing))
        assert rc == 1

    def test_json_output_is_parseable(self, capsys):
        _require(SAMPLE_DAT)
        rc = _cmd_info(_info_args(SAMPLE_DAT, json_out=True))
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["file"] == str(SAMPLE_DAT)
        assert payload["Nx"] > 0
        assert payload["Ny"] > 0
        assert payload["n_planes"] >= 1


# ─── _cmd_sxm2png ────────────────────────────────────────────────────────────

def _sxm2png_args(input_path, output_path, *, force=False):
    return argparse.Namespace(
        input=input_path,
        output=output_path,
        plane=0,
        colormap="gray",
        clip_low=1.0,
        clip_high=99.0,
        no_scalebar=True,
        scalebar_unit="nm",
        scalebar_pos="bottom-right",
        verbose=False,
        force=force,
    )


class TestCmdSxm2Png:
    def test_writes_png_for_valid_sxm(self, tmp_path):
        _require(SAMPLE_SXM)
        out_png = tmp_path / "out.png"
        rc = _cmd_sxm2png(_sxm2png_args(SAMPLE_SXM, out_png))
        assert rc == 0
        assert out_png.exists()
        assert out_png.stat().st_size > 0
