"""Tests for Createc .dat -> NumPy bundle export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from probeflow.cli.commands.conversion import _cmd_dat2npy
from probeflow.io.converters.createc_dat_to_npy import (
    export_createc_dat_npy,
    main as dat_to_npy_main,
)
from probeflow.io.readers.createc_dat import (
    has_canonical_stm_four_channel_layout,
    read_createc_dat_report,
)
from probeflow.io.readers.createc_scan import createc_public_planes_from_report
from probeflow.core.scan_loader import load_scan


def _find_dat(sample_dat_files, *, channel_count: int):
    for path in sample_dat_files:
        report = read_createc_dat_report(path, include_raw=False)
        if report.detected_channel_count != channel_count:
            continue
        if channel_count == 4 and not has_canonical_stm_four_channel_layout(report):
            continue
        return path
    pytest.skip(f"no suitable {channel_count}-channel Createc .dat fixture available")


def _bundle_dir(root: Path, stem: str, basis: str) -> Path:
    return root / f"{stem}_{basis}_npy"


def _data_dir(bundle_dir: Path) -> Path:
    return bundle_dir / "npy"


class TestCreatecDatToNpy:
    def test_raw_bundle_writes_public_order_arrays_and_header(self, sample_dat_files, tmp_path):
        dat = _find_dat(sample_dat_files, channel_count=4)
        report = read_createc_dat_report(dat)
        bundle_dir = export_createc_dat_npy(dat, tmp_path, basis="raw")

        assert bundle_dir == _bundle_dir(tmp_path, dat.stem, "raw")
        assert bundle_dir.exists()
        assert _data_dir(bundle_dir).exists()

        header_path = bundle_dir / "hdr.txt"
        assert header_path.exists()
        header = header_path.read_text(encoding="utf-8")
        first_key, first_val = next(iter(report.header.items()))
        assert f"{first_key}: {first_val}" in header
        assert "basis:" not in header
        assert "RAW means decoded Createc numerical arrays" not in header

        files = sorted(p.name for p in _data_dir(bundle_dir).glob("*.npy"))
        assert files == [f"{dat.stem}_{i}.npy" for i in range(1, report.detected_channel_count + 1)]

        expected_native = [report.decoded_channels_dac[i] for i in range(report.detected_channel_count)]
        expected_planes = createc_public_planes_from_report(report, expected_native)
        for idx, expected in enumerate(expected_planes, 1):
            arr = np.load(_data_dir(bundle_dir) / f"{dat.stem}_{idx}.npy", allow_pickle=False)
            assert arr.dtype == np.float32
            assert arr.shape == expected.array.shape
            np.testing.assert_array_equal(arr, expected.array)

        sidecar = bundle_dir.with_suffix(".probeflow.json")
        assert sidecar.exists()
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_data["basis"] == "raw"
        assert sidecar_data["artifact_type"] == "createc_dat_npy_bundle"
        assert len(sidecar_data["planes"]) == report.detected_channel_count
        assert sidecar_data["planes"][0]["saved_unit"] == "DAC"

    def test_physical_bundle_matches_load_scan_and_preserves_naming(self, sample_dat_files, tmp_path):
        dat = _find_dat(sample_dat_files, channel_count=4)
        scan = load_scan(dat)
        bundle_dir = export_createc_dat_npy(dat, tmp_path, basis="physical")

        assert bundle_dir == _bundle_dir(tmp_path, dat.stem, "physical")
        header = (bundle_dir / "hdr.txt").read_text(encoding="utf-8")
        first_key, first_val = next(iter(scan.header.items()))
        assert f"{first_key}: {first_val}" in header
        assert "basis:" not in header

        expected_files = [f"{dat.stem}_{i}.npy" for i in range(1, scan.n_planes + 1)]
        assert sorted(p.name for p in _data_dir(bundle_dir).glob("*.npy")) == expected_files

        for idx, expected in enumerate(scan.planes, 1):
            arr = np.load(_data_dir(bundle_dir) / f"{dat.stem}_{idx}.npy", allow_pickle=False)
            assert arr.dtype == np.float64
            np.testing.assert_array_equal(arr, expected)

    def test_legacy_two_channel_dat_exports_synthetic_backwards(self, sample_dat_files, tmp_path):
        dat = _find_dat(sample_dat_files, channel_count=2)
        scan = load_scan(dat)
        bundle_dir = export_createc_dat_npy(dat, tmp_path, basis="physical")

        header = (bundle_dir / "hdr.txt").read_text(encoding="utf-8")
        first_key, first_val = next(iter(scan.header.items()))
        assert f"{first_key}: {first_val}" in header
        assert "basis:" not in header

        assert sorted(p.name for p in _data_dir(bundle_dir).glob("*.npy")) == [
            f"{dat.stem}_{i}.npy" for i in range(1, 5)
        ]
        np.testing.assert_array_equal(
            np.load(_data_dir(bundle_dir) / f"{dat.stem}_2.npy", allow_pickle=False),
            scan.planes[1],
        )
        np.testing.assert_array_equal(
            np.load(_data_dir(bundle_dir) / f"{dat.stem}_4.npy", allow_pickle=False),
            scan.planes[3],
        )

    def test_no_overwrite_without_force(self, sample_dat_files, tmp_path):
        dat = _find_dat(sample_dat_files, channel_count=4)
        export_createc_dat_npy(dat, tmp_path, basis="raw")
        with pytest.raises(FileExistsError):
            export_createc_dat_npy(dat, tmp_path, basis="raw")

    def test_bundle_main_smoke(self, sample_dat_files, tmp_path):
        dat = _find_dat(sample_dat_files, channel_count=4)
        rc = dat_to_npy_main(src=dat, out_root=tmp_path, basis="raw", force=True, verbose=False)
        assert rc == 0
        assert _bundle_dir(tmp_path, dat.stem, "raw").exists()

    def test_cli_forwarder_smoke(self, sample_dat_files, tmp_path):
        dat = _find_dat(sample_dat_files, channel_count=4)
        args = argparse.Namespace(
            rest=[
                "--input-dir", str(dat),
                "--output-dir", str(tmp_path),
                "--basis", "physical",
                "--force",
            ]
        )
        rc = _cmd_dat2npy(args)
        assert rc == 0
        assert _bundle_dir(tmp_path, dat.stem, "physical").exists()
