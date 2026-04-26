"""Tests for Phase 3d-4: GUI folder discovery via index_folder().

These tests cover the pure filtering helpers and the SxmFile/VertFile
conversion layer.  They do not require Qt or a running GUI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from probeflow.indexing import (
    ProbeFlowItem,
    image_browser_items,
    split_indexed_items,
)
from probeflow.gui import _scan_items_to_sxm, _spec_items_to_vert, SxmFile, VertFile


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_item(
    name: str = "test.dat",
    item_type: str = "scan",
    source_format: str = "createc_dat",
    shape=None,
    load_error: str | None = None,
    bias: float | None = None,
    setpoint: float | None = None,
    scan_range=None,
    metadata: dict | None = None,
) -> ProbeFlowItem:
    return ProbeFlowItem(
        path=Path(name),
        display_name=Path(name).stem,
        source_format=source_format,
        item_type=item_type,
        shape=shape,
        scan_range=scan_range,
        bias=bias,
        setpoint=setpoint,
        load_error=load_error,
        metadata=metadata or {},
    )


SAMPLE_ITEMS = [
    _make_item("step.dat",    item_type="scan",     source_format="createc_dat", shape=(330, 511)),
    _make_item("moire.sxm",   item_type="scan",     source_format="nanonis_sxm", shape=(160, 160)),
    _make_item("spec.VERT",   item_type="spectrum", source_format="createc_vert"),
    _make_item("spec.dat",    item_type="spectrum", source_format="nanonis_dat_spectrum"),
    _make_item("broken.dat",  item_type="scan",     source_format="createc_dat", load_error="bad zlib"),
    _make_item("unknown.txt", item_type="unknown",  source_format="unknown"),
]


# ── Test A: image_browser_items returns only non-errored scans ────────────────

class TestImageBrowserItems:
    def test_returns_only_scans(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert all(it.item_type == "scan" for it in result)

    def test_excludes_spectra(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert all(it.source_format not in {"createc_vert", "nanonis_dat_spectrum"}
                   for it in result)

    def test_excludes_errored(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert all(it.load_error is None for it in result)

    def test_count(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert len(result) == 2  # step.dat + moire.sxm

    def test_empty_input(self):
        assert image_browser_items([]) == []


# ── Test B: split_indexed_items separates scans, spectra, errors ──────────────

class TestSplitIndexedItems:
    def test_scans_contain_only_scans(self):
        scans, _, _ = split_indexed_items(SAMPLE_ITEMS)
        assert all(it.item_type == "scan" for it in scans)

    def test_spectra_contain_only_spectra(self):
        _, spectra, _ = split_indexed_items(SAMPLE_ITEMS)
        assert all(it.item_type == "spectrum" for it in spectra)

    def test_errors_contain_errored_items(self):
        _, _, errors = split_indexed_items(SAMPLE_ITEMS)
        assert all(it.load_error for it in errors)

    def test_spectra_not_in_scan_list(self):
        scans, spectra, _ = split_indexed_items(SAMPLE_ITEMS)
        scan_paths = {it.path for it in scans}
        spec_paths = {it.path for it in spectra}
        assert scan_paths.isdisjoint(spec_paths)

    def test_counts(self):
        scans, spectra, errors = split_indexed_items(SAMPLE_ITEMS)
        assert len(scans)   == 2  # step.dat + moire.sxm
        assert len(spectra) == 2  # spec.VERT + spec.dat
        assert len(errors)  == 1  # broken.dat


# ── Test C: errored items excluded from scan list ─────────────────────────────

class TestErrorItemsExcluded:
    def test_errored_scan_not_in_browser(self):
        result = image_browser_items(SAMPLE_ITEMS)
        names = {it.path.name for it in result}
        assert "broken.dat" not in names

    def test_only_errored_items_have_load_error(self):
        _, _, errors = split_indexed_items(SAMPLE_ITEMS)
        assert all(e.load_error is not None for e in errors)
        scans, spectra, _ = split_indexed_items(SAMPLE_ITEMS)
        assert all(e.load_error is None for e in scans + spectra)


# ── Test D: _scan_items_to_sxm conversion ────────────────────────────────────

class TestScanItemsToSxm:
    def test_produces_sxm_file_objects(self):
        result = _scan_items_to_sxm(SAMPLE_ITEMS)
        assert all(isinstance(e, SxmFile) for e in result)

    def test_shape_mapped_correctly(self):
        item = _make_item("a.dat", item_type="scan", source_format="createc_dat",
                          shape=(330, 511))
        result = _scan_items_to_sxm([item])
        assert result[0].Nx == 511
        assert result[0].Ny == 330

    def test_bias_converted_to_mv(self):
        item = _make_item("a.dat", item_type="scan", source_format="createc_dat",
                          shape=(4, 4), bias=0.05)   # 50 mV in V
        result = _scan_items_to_sxm([item])
        assert abs(result[0].bias_mv - 50.0) < 1e-9

    def test_setpoint_converted_to_pa(self):
        item = _make_item("a.dat", item_type="scan", source_format="createc_dat",
                          shape=(4, 4), setpoint=4.4e-10)  # 440 pA
        result = _scan_items_to_sxm([item])
        assert abs(result[0].current_pa - 440.0) < 1e-6

    def test_scan_range_converted_to_nm(self):
        item = _make_item("a.sxm", item_type="scan", source_format="nanonis_sxm",
                          shape=(4, 4), scan_range=(10e-9, 10e-9))
        result = _scan_items_to_sxm([item])
        assert abs(result[0].scan_nm - 10.0) < 1e-9

    def test_source_format_mapped(self):
        dat = _make_item("a.dat", item_type="scan", source_format="createc_dat", shape=(4, 4))
        sxm = _make_item("b.sxm", item_type="scan", source_format="nanonis_sxm", shape=(4, 4))
        results = _scan_items_to_sxm([dat, sxm])
        assert results[0].source_format == "dat"
        assert results[1].source_format == "sxm"

    def test_errored_item_gets_stub_entry(self):
        item = _make_item("bad.dat", item_type="scan", source_format="createc_dat",
                          load_error="zlib error")
        result = _scan_items_to_sxm([item])
        assert len(result) == 1
        assert result[0].Nx == 512  # default

    def test_spectra_excluded(self):
        result = _scan_items_to_sxm(SAMPLE_ITEMS)
        names = {e.path.name for e in result}
        assert "spec.VERT" not in names
        assert "spec.dat"  not in names

    def test_stem_deduplication(self):
        dup1 = _make_item("scan.sxm", item_type="scan", source_format="nanonis_sxm", shape=(4, 4))
        dup2 = _make_item("scan.dat", item_type="scan", source_format="createc_dat", shape=(4, 4))
        # Same stem "scan" — second should be dropped.
        result = _scan_items_to_sxm([dup1, dup2])
        assert len(result) == 1


# ── Test E: _spec_items_to_vert conversion ────────────────────────────────────

class TestSpecItemsToVert:
    def test_produces_vert_file_objects(self):
        result = _spec_items_to_vert(SAMPLE_ITEMS)
        assert all(isinstance(e, VertFile) for e in result)

    def test_sweep_type_populated(self):
        item = _make_item("s.VERT", item_type="spectrum", source_format="createc_vert",
                          metadata={"sweep_type": "bias_sweep", "n_points": 1000})
        result = _spec_items_to_vert([item])
        assert result[0].sweep_type == "bias_sweep"

    def test_n_points_populated(self):
        item = _make_item("s.VERT", item_type="spectrum", source_format="createc_vert",
                          metadata={"sweep_type": "bias_sweep", "n_points": 1024})
        result = _spec_items_to_vert([item])
        assert result[0].n_points == 1024

    def test_scan_items_excluded(self):
        result = _spec_items_to_vert(SAMPLE_ITEMS)
        names = {e.path.name for e in result}
        assert "step.dat" not in names
        assert "moire.sxm" not in names

    def test_errored_spec_gets_stub(self):
        item = _make_item("bad.VERT", item_type="spectrum", source_format="createc_vert",
                          load_error="parse error")
        result = _spec_items_to_vert([item])
        assert len(result) == 1
        assert result[0].sweep_type == "unknown"
        assert result[0].n_points == 0


# ── Test F: round-trip through real fixtures ──────────────────────────────────

class TestRealFixtureRoundTrip:
    TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"

    def test_createc_scans_appear_in_sxm_list(self):
        from probeflow.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        sxm_list = _scan_items_to_sxm(items)
        names = {e.path.name for e in sxm_list}
        assert "createc_scan_step_20nm.dat"    in names
        assert "createc_scan_terrace_109nm.dat" in names

    def test_nanonis_sxm_appears_in_sxm_list(self):
        from probeflow.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        sxm_list = _scan_items_to_sxm(items)
        names = {e.path.name for e in sxm_list}
        assert "sxm_moire_10nm.sxm" in names

    def test_spectra_appear_in_vert_list(self):
        from probeflow.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        vert_list = _spec_items_to_vert(items)
        assert len(vert_list) >= 3

    def test_no_errors_on_real_fixtures(self):
        from probeflow.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        errors = [it for it in items if it.load_error]
        assert errors == []


class TestSpecViewerLifetime:
    TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"

    def test_static_unit_controls_survive_load_cleanup(self):
        try:
            import shiboken6
            from PySide6.QtWidgets import QApplication
            from probeflow.gui import SpecViewerDialog, THEMES
        except Exception as exc:
            pytest.skip(f"Qt unavailable: {exc}")

        app = QApplication.instance()
        if app is None:
            try:
                app = QApplication([])
            except Exception as exc:
                pytest.skip(f"QApplication unavailable: {exc}")

        spec_path = self.TESTDATA / "createc_ivt_telegraph_300mv_a.VERT"
        entry = VertFile(path=spec_path, stem=spec_path.stem)
        dlg = SpecViewerDialog(entry, THEMES["light"])

        # Process any queued deleteLater calls from _load(). The static unit
        # group contains QComboBoxes; deleting it during channel refresh caused
        # a native Qt crash when closing the dialog.
        app.processEvents()
        assert shiboken6.isValid(dlg._z_unit_cb)
        assert shiboken6.isValid(dlg._i_unit_cb)
        assert shiboken6.isValid(dlg._v_unit_cb)

        dlg.close()
        dlg.deleteLater()
        app.processEvents()
