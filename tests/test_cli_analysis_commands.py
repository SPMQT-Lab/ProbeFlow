"""Tests for CLI analysis commands and dat→sxm conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from probeflow.cli.commands.analysis import (
    _cmd_autoclip,
    _cmd_fft_spectrum,
    _cmd_grains,
    _cmd_histogram,
    _cmd_particles,
    _cmd_periodicity,
)
from probeflow.cli.commands.conversion import _cmd_dat2sxm

TEST_DATA = Path(__file__).parent.parent / "test_data"
SAMPLE_DAT = TEST_DATA / "createc_scan_close_100nm.dat"
ATOMIC_DAT = TEST_DATA / "createc_scan_atomic_11nm.dat"
SAMPLE_SXM = TEST_DATA / "sxm_moire_10nm.sxm"


@pytest.fixture(autouse=True)
def require_test_data():
    missing = [p for p in (SAMPLE_DAT, ATOMIC_DAT, SAMPLE_SXM) if not p.exists()]
    if missing:
        pytest.skip(f"test_data not available: {missing}")


# ─── _cmd_dat2sxm ────────────────────────────────────────────────────────────


class TestDat2Sxm:
    def test_writes_sxm_to_output_dir(self, tmp_path):
        out_dir = tmp_path / "sxm_out"
        out_dir.mkdir()
        ns = argparse.Namespace(rest=[
            "--input-dir", str(SAMPLE_DAT),
            "--output-dir", str(out_dir),
        ])
        rc = _cmd_dat2sxm(ns)
        assert rc == 0
        outputs = list(out_dir.glob("*.sxm"))
        assert len(outputs) == 1
        assert outputs[0].stat().st_size > 0
        assert outputs[0].stem == SAMPLE_DAT.stem

    def test_handles_forwarded_dash_dash(self, tmp_path):
        out_dir = tmp_path / "sxm_out"
        out_dir.mkdir()
        # Leading '--' is stripped by _cmd_dat2sxm before forwarding.
        ns = argparse.Namespace(rest=[
            "--",
            "--input-dir", str(SAMPLE_DAT),
            "--output-dir", str(out_dir),
        ])
        rc = _cmd_dat2sxm(ns)
        assert rc == 0
        assert (out_dir / f"{SAMPLE_DAT.stem}.sxm").exists()

    def test_non_dat_input_raises(self, tmp_path):
        bad = tmp_path / "not_a_dat.txt"
        bad.write_text("hello")
        ns = argparse.Namespace(rest=[
            "--input-dir", str(bad),
            "--output-dir", str(tmp_path),
        ])
        with pytest.raises(ValueError, match="Expected a .dat file or directory"):
            _cmd_dat2sxm(ns)


# ─── _cmd_grains ─────────────────────────────────────────────────────────────


class TestGrains:
    def _make_args(self, **overrides):
        defaults = dict(
            input=ATOMIC_DAT,
            plane=0,
            threshold=50.0,
            below=False,
            min_px=5,
            save_mask=None,
            json=False,
            verbose=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_exit_zero_and_prints_count(self, capsys):
        rc = _cmd_grains(self._make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "Grains detected:" in out

    def test_json_output_has_n_grains_key(self, capsys):
        rc = _cmd_grains(self._make_args(json=True))
        assert rc == 0
        out = capsys.readouterr().out
        # Extract the JSON block (after the "Grains detected:" line)
        json_start = out.index("{")
        payload = json.loads(out[json_start:])
        assert "n_grains" in payload
        assert isinstance(payload["n_grains"], int)

    def test_missing_plane_returns_one(self):
        rc = _cmd_grains(self._make_args(plane=999))
        assert rc == 1

    def test_save_mask_writes_file(self, tmp_path):
        mask_path = tmp_path / "mask.png"
        rc = _cmd_grains(self._make_args(save_mask=mask_path))
        assert rc == 0
        assert mask_path.exists()
        assert mask_path.stat().st_size > 0


# ─── _cmd_autoclip ───────────────────────────────────────────────────────────


class TestAutoclip:
    def _make_args(self, **overrides):
        defaults = dict(input=ATOMIC_DAT, plane=0, json=False, verbose=False)
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_exit_zero(self, capsys):
        rc = _cmd_autoclip(self._make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "clip_low" in out
        assert "clip_high" in out

    def test_json_output_parseable(self, capsys):
        rc = _cmd_autoclip(self._make_args(json=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert "clip_low" in payload
        assert "clip_high" in payload
        assert payload["clip_high"] >= payload["clip_low"]

    def test_missing_plane_returns_one(self):
        rc = _cmd_autoclip(self._make_args(plane=999))
        assert rc == 1


# ─── _cmd_periodicity ────────────────────────────────────────────────────────


class TestPeriodicity:
    def _make_args(self, **overrides):
        defaults = dict(
            input=ATOMIC_DAT,
            plane=0,
            n_peaks=3,
            json=False,
            verbose=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_exit_zero(self, capsys):
        rc = _cmd_periodicity(self._make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "period=" in out
        assert "nm" in out

    def test_json_output_lists_peaks(self, capsys):
        rc = _cmd_periodicity(self._make_args(json=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert len(payload) >= 1
        first = payload[0]
        assert "period_m" in first
        assert "angle_deg" in first

    def test_missing_plane_returns_one(self):
        rc = _cmd_periodicity(self._make_args(plane=999))
        assert rc == 1


# ─── _cmd_histogram ──────────────────────────────────────────────────────────


class TestHistogram:
    def _make_args(self, **overrides):
        defaults = dict(
            input=SAMPLE_DAT,
            output=None,
            plane=0,
            bins=32,
            roi_rect=None,
            roi_polygon=None,
            roi=None,
            sidecar=None,
            verbose=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_stdout(self, capsys):
        rc = _cmd_histogram(self._make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "bin_centre" in out
        # Expect ``bins`` data lines + header line.
        data_lines = [ln for ln in out.strip().splitlines() if not ln.startswith("#")]
        assert len(data_lines) == 32

    def test_csv_output_written(self, tmp_path):
        out = tmp_path / "hist.csv"
        rc = _cmd_histogram(self._make_args(output=out))
        assert rc == 0
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "bin_edge_low" in text
        # Header line + 32 data lines.
        assert len([ln for ln in text.strip().splitlines() if ln]) == 33

    def test_missing_plane_returns_one(self):
        rc = _cmd_histogram(self._make_args(plane=999))
        assert rc == 1


# ─── _cmd_fft_spectrum ───────────────────────────────────────────────────────


class TestFftSpectrum:
    def _make_args(self, **overrides):
        defaults = dict(
            input=SAMPLE_SXM,
            output=None,
            plane=0,
            window="hann",
            window_param=0.25,
            log_scale=True,
            roi_rect=None,
            roi_polygon=None,
            roi=None,
            sidecar=None,
            colormap="gray",
            verbose=False,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_stdout(self, capsys):
        rc = _cmd_fft_spectrum(self._make_args())
        assert rc == 0
        out = capsys.readouterr().out
        assert "magnitude" in out

    def test_png_output_written(self, tmp_path):
        out = tmp_path / "fft.png"
        rc = _cmd_fft_spectrum(self._make_args(output=out))
        assert rc == 0
        assert out.exists()
        assert out.stat().st_size > 0

    def test_missing_plane_returns_one(self):
        rc = _cmd_fft_spectrum(self._make_args(plane=999))
        assert rc == 1


# ─── _cmd_particles (incl. algorithmic step-edge exclusion) ──────────────────


class TestParticles:
    def _make_args(self, **overrides):
        defaults = dict(
            input=SAMPLE_SXM, output=None, plane=0, threshold="otsu",
            manual_value=None, invert=False, min_area=0.5, max_area=None,
            sigma_clip=2.0, no_sigma_clip=False, clip_low=1.0, clip_high=99.0,
            limit=20, json=False, verbose=False,
            exclude_step_edges=False, step_angle=20.0, step_molecule_size=1.0,
            step_margin=0.3, step_min_height=0.0, step_max_overlap=0.25,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smoke_exit_zero(self, capsys):
        pytest.importorskip("cv2")
        assert _cmd_particles(self._make_args()) == 0
        assert "particle" in capsys.readouterr().out.lower()

    def test_exclude_step_edges_runs(self, capsys):
        """The --exclude-step-edges path computes a band and still exits 0."""
        pytest.importorskip("cv2")
        rc = _cmd_particles(self._make_args(exclude_step_edges=True, verbose=True))
        assert rc == 0

    def test_json_output_written(self, tmp_path):
        pytest.importorskip("cv2")
        out = tmp_path / "particles.json"
        rc = _cmd_particles(self._make_args(output=out, exclude_step_edges=True))
        assert rc == 0
        assert out.exists() and out.stat().st_size > 0
        payload = json.loads(out.read_text())
        assert payload["meta"]["kind"] == "particles"
