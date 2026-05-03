"""Tests for probeflow.core.indexing — ProbeFlowItem and index_folder()."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from probeflow.core.indexing import ProbeFlowItem, index_folder


TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"
_CREATEC_STEP    = TESTDATA / "createc_scan_step_20nm.dat"
_CREATEC_TERRACE = TESTDATA / "createc_scan_terrace_109nm.dat"
_CREATEC_QPLUS_10CH = TESTDATA / "createc_scan_qplus_10ch_afm.dat"
_NANONIS_SXM     = TESTDATA / "sxm_moire_10nm.sxm"
_CREATEC_VERT    = TESTDATA / "createc_ivt_telegraph_300mv_a.VERT"
_CREATEC_DIDZ_VERT = TESTDATA / "createc_vert_didz_image_state.VERT"
_NANONIS_SPEC    = TESTDATA / "nanonis_sts_15mv.dat"

_CREATEC_SCAN_SHAPES = {(64, 63), (256, 255), (330, 511), (512, 511), (1024, 1023)}


# ── Test A: Createc scan fixtures ─────────────────────────────────────────────

class TestCreatecScans:
    def test_createc_scans_present(self):
        items = index_folder(TESTDATA)
        dat_items = [it for it in items if it.source_format == "createc_dat"]
        assert len(dat_items) == 13

    def test_item_type_is_scan(self):
        items = index_folder(TESTDATA)
        for it in items:
            if it.source_format == "createc_dat":
                assert it.item_type == "scan"

    def test_shapes_match_fixtures(self):
        items = index_folder(TESTDATA)
        shapes = {it.shape for it in items if it.source_format == "createc_dat"}
        assert shapes == _CREATEC_SCAN_SHAPES

    def test_no_load_error(self):
        items = index_folder(TESTDATA)
        for it in items:
            if it.source_format == "createc_dat":
                assert it.load_error is None

    def test_channels_tuple(self):
        items = index_folder(TESTDATA)
        for it in items:
            if it.source_format == "createc_dat":
                assert isinstance(it.channels, tuple)
                assert len(it.channels) >= 4

    def test_scan_range_positive(self):
        items = index_folder(TESTDATA)
        for it in items:
            if it.source_format == "createc_dat":
                assert it.scan_range is not None
                w, h = it.scan_range
                assert w > 0 and h > 0

    def test_experiment_metadata_present(self):
        items = index_folder(TESTDATA)
        dat_items = [it for it in items if it.source_format == "createc_dat"]
        assert dat_items
        assert all("experiment_metadata" in it.metadata for it in dat_items)
        assert any(
            it.metadata["experiment_metadata"]["acquisition_mode"] == "stm"
            for it in dat_items
        )

    def test_qplus_10_channel_fixture_indexed(self):
        items = index_folder(TESTDATA)
        item = next(it for it in items if it.path == _CREATEC_QPLUS_10CH)
        assert len(item.channels) == 10
        assert item.metadata["experiment_metadata"]["acquisition_mode"] == "afm"


# ── Test B: Nanonis SXM fixture ───────────────────────────────────────────────

class TestNanonisScan:
    def test_sxm_present(self):
        items = index_folder(TESTDATA)
        sxm_items = [it for it in items if it.source_format == "nanonis_sxm"]
        assert len(sxm_items) == 1

    def test_item_type_is_scan(self):
        items = index_folder(TESTDATA)
        sxm_items = [it for it in items if it.source_format == "nanonis_sxm"]
        assert sxm_items[0].item_type == "scan"

    def test_no_load_error(self):
        items = index_folder(TESTDATA)
        sxm_items = [it for it in items if it.source_format == "nanonis_sxm"]
        assert sxm_items[0].load_error is None

    def test_shape_is_set(self):
        items = index_folder(TESTDATA)
        sxm_items = [it for it in items if it.source_format == "nanonis_sxm"]
        assert sxm_items[0].shape is not None


# ── Test C: unrelated files are ignored ───────────────────────────────────────

class TestIgnoreUnrelated:
    def test_text_and_png_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        # a .dat file that isn't a recognised SPM format
        (tmp_path / "random.dat").write_text("col1\tcol2\n1\t2\n")
        items = index_folder(tmp_path)
        assert items == []

    def test_hidden_files_ignored(self, tmp_path):
        (tmp_path / ".DS_Store").write_bytes(b"\x00" * 32)
        items = index_folder(tmp_path)
        assert items == []


# ── Test D: recursive behaviour ───────────────────────────────────────────────

class TestRecursion:
    def test_non_recursive_misses_subfolders(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        shutil.copy(_CREATEC_STEP, sub / _CREATEC_STEP.name)

        items_flat = index_folder(tmp_path, recursive=False)
        assert all(it.path.parent == tmp_path for it in items_flat)
        assert len(items_flat) == 0  # nothing in root

    def test_recursive_finds_subfolders(self, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        shutil.copy(_CREATEC_STEP, sub / _CREATEC_STEP.name)

        items_rec = index_folder(tmp_path, recursive=True)
        names = {it.path.name for it in items_rec}
        assert _CREATEC_STEP.name in names

    def test_output_dir_skipped_in_recursive(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        shutil.copy(_CREATEC_STEP, out / _CREATEC_STEP.name)

        items = index_folder(tmp_path, recursive=True)
        assert all(it.path.parent != out for it in items)


# ── Test E: metadata errors handled gracefully ────────────────────────────────

class TestErrorHandling:
    def _write_bad_dat(self, path: Path) -> None:
        """Write a file with Createc image magic bytes but garbage content."""
        path.write_bytes(b"[Paramco32]\nNum.X=4\nNum.Y=4\nDATA\x00\x01\x02\x03")

    def test_include_errors_true_returns_item_with_error(self, tmp_path):
        bad = tmp_path / "bad.dat"
        self._write_bad_dat(bad)
        items = index_folder(tmp_path, include_errors=True)
        assert len(items) == 1
        assert items[0].load_error is not None

    def test_include_errors_false_skips_bad_file(self, tmp_path):
        bad = tmp_path / "bad.dat"
        self._write_bad_dat(bad)
        items = index_folder(tmp_path, include_errors=False)
        assert items == []

    def test_good_files_unaffected_by_bad_file(self, tmp_path):
        bad = tmp_path / "bad.dat"
        self._write_bad_dat(bad)
        shutil.copy(_NANONIS_SXM, tmp_path / _NANONIS_SXM.name)
        items = index_folder(tmp_path, include_errors=True)
        good = [it for it in items if it.load_error is None]
        assert len(good) == 1
        assert good[0].source_format == "nanonis_sxm"


# ── Test F: path validation ───────────────────────────────────────────────────

class TestPathValidation:
    def test_nonexistent_folder_raises(self, tmp_path):
        with pytest.raises(ValueError, match="exist"):
            index_folder(tmp_path / "no_such_dir")

    def test_file_path_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="directory"):
            index_folder(f)


# ── Test G: ProbeFlowItem contract ───────────────────────────────────────────

class TestProbeFlowItemContract:
    def test_is_frozen(self):
        items = index_folder(TESTDATA)
        item = items[0]
        with pytest.raises((AttributeError, TypeError)):
            item.shape = (1, 1)  # type: ignore[misc]

    def test_display_name_is_stem(self):
        items = index_folder(TESTDATA)
        for it in items:
            assert it.display_name == it.path.stem

    def test_mtime_and_size_populated(self):
        items = index_folder(TESTDATA)
        for it in items:
            assert it.mtime_ns is not None
            assert it.size_bytes is not None
            assert it.size_bytes > 0

    def test_spectrum_items_present(self):
        items = index_folder(TESTDATA)
        spectra = [it for it in items if it.item_type == "spectrum"]
        assert len(spectra) >= 3  # at least the three .VERT files

    def test_spectrum_measurement_metadata_present(self):
        items = index_folder(TESTDATA)
        spectra = [it for it in items if it.source_format == "createc_vert"]
        assert spectra
        assert all("measurement_family" in it.metadata for it in spectra)

    def test_didz_vert_fixture_indexed(self):
        items = index_folder(TESTDATA)
        item = next(it for it in items if it.path == _CREATEC_DIDZ_VERT)
        assert item.metadata["measurement_family"] == "iz"
        assert item.metadata["derivative_label"] == "dI/dz"
        assert item.metadata["height_channel"] == "Z feedback"
        assert "Z feedback" in item.channels
        assert "Raw column 9" not in item.channels
        assert item.metadata["channel_roles"]["Z feedback"] == [
            "z_feedback",
            "height_counts",
        ]
        assert any(
            ch["source_name"] == "Raw column 9"
            and ch["key"] == "Z feedback"
            for ch in item.metadata["channel_info"]
        )

    def test_spectrum_indexing_does_not_call_full_reader(self, tmp_path, monkeypatch):
        shutil.copy(_CREATEC_VERT, tmp_path / _CREATEC_VERT.name)
        shutil.copy(_NANONIS_SPEC, tmp_path / _NANONIS_SPEC.name)

        def fail_full_reader(*_args, **_kwargs):
            raise AssertionError("index_folder should use read_spec_metadata")

        monkeypatch.setattr("probeflow.io.spectroscopy.read_spec_file", fail_full_reader)
        items = index_folder(tmp_path)
        spectra = [it for it in items if it.item_type == "spectrum"]
        assert len(spectra) == 2
        assert all(it.load_error is None for it in spectra)
        assert {it.source_format for it in spectra} == {
            "createc_vert",
            "nanonis_dat_spectrum",
        }
