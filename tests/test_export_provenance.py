"""Tests for ExportProvenance, DisplayRangeState.to_dict(), and export paths.

Covers spec tasks 8A–8F:
  A. ExportProvenance.to_dict() is JSON-serialisable
  B. from_scan_export() extracts source metadata
  C. DisplayRangeState.to_dict() preserves mode and limits
  D. JSON export includes provenance block
  E. PNG sidecar provenance written alongside PNG
  F. No visual regression in PNG export
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from probeflow.processing.display_state import DisplayRangeState
from probeflow.provenance.export import (
    ExportProvenance,
    background_processing_warnings,
    build_scan_export_provenance,
    png_display_state,
    processing_state_hash,
)
from probeflow.processing.state import ProcessingState, ProcessingStep
from probeflow.io.writers.json import write_json


# ── Synthetic scan helper ─────────────────────────────────────────────────────

def _make_scan(
    shape=(32, 32),
    n_planes=2,
    source_format="dat",
    scan_range_m=(1e-7, 1e-7),
    source_path=None,
):
    """Return a minimal Scan-like object for testing."""
    from probeflow.core.scan_model import Scan
    rng = np.random.default_rng(42)
    planes = [rng.standard_normal(shape) for _ in range(n_planes)]
    plane_names = ["Z fwd", "Z bwd"][:n_planes]
    plane_units = ["m", "m"][:n_planes]
    return Scan(
        planes=planes,
        plane_names=plane_names,
        plane_units=plane_units,
        plane_synthetic=[False] * n_planes,
        header={},
        scan_range_m=scan_range_m,
        source_path=Path(source_path or "/fake/scan.dat"),
        source_format=source_format,
    )


# ── A: ExportProvenance.to_dict() is JSON-serialisable ───────────────────────

class TestToDict:
    def _minimal(self) -> ExportProvenance:
        return ExportProvenance(
            source_file="/data/scan.dat",
            source_format="dat",
            item_type="scan",
            channel_name="Z fwd",
            channel_index=0,
            array_shape=(128, 128),
            scan_range_m=(1.09e-7, 1.09e-7),
            units="m",
            processing_state={"steps": []},
            display_state={"mode": "percentile", "low_pct": 1.0, "high_pct": 99.0,
                           "vmin": None, "vmax": None},
            probeflow_version="0.0.0b0",
            export_timestamp="2026-04-25T07:15:00Z",
        )

    def test_json_serialisable(self):
        prov = self._minimal()
        serialised = json.dumps(prov.to_dict())
        assert isinstance(serialised, str)
        assert len(serialised) > 0

    def test_roundtrip_keys_present(self):
        prov = self._minimal()
        d = prov.to_dict()
        for key in ("source_file", "source_format", "item_type", "channel_name",
                    "channel_index", "array_shape", "scan_range_m", "units",
                    "processing_state", "display_state",
                    "probeflow_version", "export_timestamp", "export_kind",
                    "source_id", "channel_id", "processing_state_hash",
                    "artifact_id", "warnings"):
            assert key in d, f"Key '{key}' missing from to_dict()"

    def test_array_shape_serialised_as_list(self):
        prov = self._minimal()
        d = prov.to_dict()
        assert isinstance(d["array_shape"], list)
        assert d["array_shape"] == [128, 128]

    def test_scan_range_m_serialised_as_list(self):
        prov = self._minimal()
        d = prov.to_dict()
        assert isinstance(d["scan_range_m"], list)
        assert len(d["scan_range_m"]) == 2

    def test_none_fields_serialise_as_null(self):
        prov = ExportProvenance(
            source_file=None, source_format=None, item_type="scan",
            channel_name=None, channel_index=None, array_shape=None,
            scan_range_m=None, units=None,
            processing_state={"steps": []}, display_state={},
            probeflow_version=None,
            export_timestamp="2026-04-25T00:00:00Z",
        )
        serialised = json.dumps(prov.to_dict())
        parsed = json.loads(serialised)
        assert parsed["source_file"] is None
        assert parsed["array_shape"] is None
        assert parsed["scan_range_m"] is None

    def test_timestamp_iso8601_format(self):
        prov = self._minimal()
        ts = prov.to_dict()["export_timestamp"]
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts), (
            f"Timestamp '{ts}' is not ISO-8601 UTC"
        )


# ── B: from_scan_export() extracts source metadata ───────────────────────────

class TestFromScanExport:
    def test_source_file_is_string(self):
        scan = _make_scan(source_path="/data/test.dat")
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.source_file == "/data/test.dat"

    def test_source_format_extracted(self):
        scan = _make_scan(source_format="dat")
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.source_format == "dat"

    def test_channel_index_stored(self):
        scan = _make_scan(n_planes=2)
        prov = ExportProvenance.from_scan_export(scan, channel_index=1)
        assert prov.channel_index == 1

    def test_channel_name_from_scan(self):
        scan = _make_scan(n_planes=2)
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.channel_name == "Z fwd"

    def test_channel_name_explicit_overrides_scan(self):
        scan = _make_scan(n_planes=2)
        prov = ExportProvenance.from_scan_export(
            scan, channel_index=0, channel_name="Custom"
        )
        assert prov.channel_name == "Custom"

    def test_array_shape_matches_plane(self):
        scan = _make_scan(shape=(64, 48))
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.array_shape == (64, 48)

    def test_scan_range_extracted(self):
        scan = _make_scan(scan_range_m=(2e-7, 3e-7))
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.scan_range_m == pytest.approx((2e-7, 3e-7))

    def test_units_extracted(self):
        scan = _make_scan(n_planes=1)
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.units == "m"

    def test_processing_state_empty_by_default(self):
        scan = _make_scan()
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.processing_state == {"steps": []}

    def test_processing_state_from_object(self):
        scan = _make_scan()
        ps = ProcessingState(steps=[ProcessingStep("align_rows", {"method": "median"})])
        prov = ExportProvenance.from_scan_export(
            scan, channel_index=0, processing_state=ps
        )
        assert prov.processing_state == ps.to_dict()

    def test_display_state_empty_by_default(self):
        scan = _make_scan()
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.display_state == {}

    def test_display_state_from_object(self):
        scan = _make_scan()
        drs = DisplayRangeState()
        drs.set_manual(1.0, 5.0)
        prov = ExportProvenance.from_scan_export(
            scan, channel_index=0, display_state=drs
        )
        assert prov.display_state["mode"] == "manual"
        assert prov.display_state["vmin"] == pytest.approx(1.0)

    def test_item_type_default_scan(self):
        scan = _make_scan()
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.item_type == "scan"

    def test_item_type_custom(self):
        scan = _make_scan()
        prov = ExportProvenance.from_scan_export(
            scan, channel_index=0, item_type="thumbnail"
        )
        assert prov.item_type == "thumbnail"

    def test_probeflow_version_present_or_none(self):
        scan = _make_scan()
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        # May be a string or None depending on installation state
        assert prov.probeflow_version is None or isinstance(prov.probeflow_version, str)

    def test_export_timestamp_iso8601(self):
        scan = _make_scan()
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
                        prov.export_timestamp)

    def test_defensive_missing_source_path(self):
        scan = _make_scan()
        scan = scan.__class__(
            planes=scan.planes, plane_names=scan.plane_names,
            plane_units=scan.plane_units, plane_synthetic=scan.plane_synthetic,
            header={}, scan_range_m=scan.scan_range_m,
            source_path=None, source_format=scan.source_format,
        )
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        assert prov.source_file is None

    def test_result_is_json_serialisable(self):
        scan = _make_scan()
        ps = ProcessingState()
        drs = DisplayRangeState()
        prov = ExportProvenance.from_scan_export(
            scan, channel_index=0, processing_state=ps, display_state=drs
        )
        json.dumps(prov.to_dict())

    def test_processing_state_hash_is_stable(self):
        state_a = {"steps": [{"op": "plane_bg", "params": {"order": 1, "fit_rect": (1, 2, 3, 4)}}]}
        state_b = {"steps": [{"params": {"fit_rect": (1, 2, 3, 4), "order": 1}, "op": "plane_bg"}]}
        assert processing_state_hash(state_a) == processing_state_hash(state_b)

    def test_shared_builder_adds_ids_and_output_context(self, tmp_path):
        scan = _make_scan(source_path="/data/scan.dat")
        ps = ProcessingState(steps=[ProcessingStep("plane_bg", {"order": 1})])
        drs = DisplayRangeState()
        out = tmp_path / "prepared.png"

        prov = build_scan_export_provenance(
            scan,
            channel_index=0,
            processing_state=ps,
            display_state=drs,
            export_kind="prepared_png",
            output_path=out,
        )
        data = prov.to_dict()
        assert data["export_kind"] == "prepared_png"
        assert data["output_path"] == str(out)
        assert data["source_id"]
        assert data["channel_id"]
        assert data["artifact_id"]
        assert data["processing_state_hash"] == processing_state_hash(ps.to_dict())

    def test_background_warning_absent_when_background_recorded(self):
        state = {"steps": [{"op": "stm_line_bg", "params": {"mode": "step_tolerant"}}]}
        assert background_processing_warnings(state) == ()

    def test_background_warning_present_without_background(self):
        state = {"steps": [{"op": "smooth", "params": {"sigma_px": 1.0}}]}
        warnings = background_processing_warnings(state)
        assert len(warnings) == 1
        assert "background" in warnings[0]

    def test_png_display_state_adds_visual_export_options(self):
        display = png_display_state(
            clip_low=2.0,
            clip_high=98.0,
            colormap="plasma",
            add_scalebar=False,
            scalebar_unit="nm",
            scalebar_pos="bottom-left",
        )
        assert display["mode"] == "percentile"
        assert display["low_pct"] == 2.0
        assert display["high_pct"] == 98.0
        assert display["colormap"] == "plasma"
        assert display["add_scalebar"] is False
        assert display["scalebar_pos"] == "bottom-left"


# ── C: DisplayRangeState.to_dict() preserves mode and limits ─────────────────

class TestDisplayRangeStateToDict:
    def test_percentile_mode_keys(self):
        drs = DisplayRangeState()
        d = drs.to_dict()
        assert d["mode"] == "percentile"
        assert "low_pct" in d
        assert "high_pct" in d
        assert "vmin" in d
        assert "vmax" in d

    def test_percentile_mode_vmin_vmax_none(self):
        drs = DisplayRangeState()
        d = drs.to_dict()
        assert d["vmin"] is None
        assert d["vmax"] is None

    def test_percentile_mode_stores_percentiles(self):
        drs = DisplayRangeState()
        drs.set_percentile(2.0, 98.0)
        d = drs.to_dict()
        assert d["mode"] == "percentile"
        assert d["low_pct"] == pytest.approx(2.0)
        assert d["high_pct"] == pytest.approx(98.0)

    def test_manual_mode_stores_vmin_vmax(self):
        drs = DisplayRangeState()
        drs.set_manual(-3.57e-10, -1.32e-10)
        d = drs.to_dict()
        assert d["mode"] == "manual"
        assert d["vmin"] == pytest.approx(-3.57e-10)
        assert d["vmax"] == pytest.approx(-1.32e-10)

    def test_manual_mode_vmin_vmax_not_none(self):
        drs = DisplayRangeState()
        drs.set_manual(0.0, 1.0)
        d = drs.to_dict()
        assert d["vmin"] is not None
        assert d["vmax"] is not None

    def test_percentile_mode_is_json_serialisable(self):
        drs = DisplayRangeState()
        json.dumps(drs.to_dict())

    def test_manual_mode_is_json_serialisable(self):
        drs = DisplayRangeState()
        drs.set_manual(1e-10, 5e-10)
        json.dumps(drs.to_dict())

    def test_reset_returns_to_percentile(self):
        drs = DisplayRangeState()
        drs.set_manual(1.0, 5.0)
        drs.reset()
        d = drs.to_dict()
        assert d["mode"] == "percentile"
        assert d["vmin"] is None
        assert d["vmax"] is None


# ── D: JSON export includes provenance block ──────────────────────────────────

class TestJsonExportProvenance:
    def _make_prov(self):
        scan = _make_scan()
        ps = ProcessingState(steps=[ProcessingStep("align_rows", {"method": "median"})])
        drs = DisplayRangeState()
        drs.set_manual(1e-10, 5e-10)
        return ExportProvenance.from_scan_export(
            scan, channel_index=0, processing_state=ps, display_state=drs
        )

    def test_export_provenance_key_present(self, tmp_path):
        prov = self._make_prov()
        out = tmp_path / "out.json"
        write_json(out, [], kind="particles", provenance=prov)
        data = json.loads(out.read_text())
        assert "export_provenance" in data

    def test_processing_state_key_present(self, tmp_path):
        prov = self._make_prov()
        out = tmp_path / "out.json"
        write_json(out, [], kind="particles", provenance=prov)
        data = json.loads(out.read_text())
        assert "processing_state" in data

    def test_display_state_key_present(self, tmp_path):
        prov = self._make_prov()
        out = tmp_path / "out.json"
        write_json(out, [], kind="particles", provenance=prov)
        data = json.loads(out.read_text())
        assert "display_state" in data

    def test_processing_state_matches_canonical(self, tmp_path):
        scan = _make_scan()
        ps = ProcessingState(steps=[ProcessingStep("smooth", {"sigma_px": 2.0})])
        prov = ExportProvenance.from_scan_export(
            scan, channel_index=0, processing_state=ps
        )
        out = tmp_path / "out.json"
        write_json(out, [], kind="particles", provenance=prov)
        data = json.loads(out.read_text())
        assert data["processing_state"] == ps.to_dict()

    def test_display_state_matches_canonical(self, tmp_path):
        scan = _make_scan()
        drs = DisplayRangeState()
        drs.set_manual(-1e-9, 2e-9)
        prov = ExportProvenance.from_scan_export(
            scan, channel_index=0, display_state=drs
        )
        out = tmp_path / "out.json"
        write_json(out, [], kind="particles", provenance=prov)
        data = json.loads(out.read_text())
        assert data["display_state"]["mode"] == "manual"
        assert data["display_state"]["vmin"] == pytest.approx(-1e-9)

    def test_no_provenance_still_writes_meta(self, tmp_path):
        out = tmp_path / "no_prov.json"
        write_json(out, [], kind="particles")
        data = json.loads(out.read_text())
        assert "meta" in data
        assert "export_provenance" not in data

    def test_items_key_present(self, tmp_path):
        prov = self._make_prov()
        out = tmp_path / "out.json"
        write_json(out, [], kind="particles", provenance=prov)
        data = json.loads(out.read_text())
        assert "items" in data


# ── E: PNG sidecar provenance ─────────────────────────────────────────────────

def _gray_lut():
    return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)


class TestPngSidecar:
    def _make_prov(self):
        scan = _make_scan()
        drs = DisplayRangeState()
        return ExportProvenance.from_scan_export(scan, channel_index=0, display_state=drs)

    def _export(self, arr, out, prov=None):
        from probeflow.processing import export_png
        export_png(arr, out, "gray", 1.0, 99.0,
                   lut_fn=lambda _: _gray_lut(),
                   scan_range_m=(0.0, 0.0),
                   add_scalebar=False,
                   provenance=prov)

    def test_sidecar_created_when_provenance_given(self, tmp_path):
        arr = np.random.default_rng(0).standard_normal((32, 32))
        out = tmp_path / "test.png"
        self._export(arr, out, prov=self._make_prov())
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        assert sidecar.exists(), "Sidecar .provenance.json not created"

    def test_sidecar_is_valid_json(self, tmp_path):
        arr = np.random.default_rng(0).standard_normal((32, 32))
        out = tmp_path / "test.png"
        self._export(arr, out, prov=self._make_prov())
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text())
        assert isinstance(data, dict)

    def test_sidecar_contains_expected_fields(self, tmp_path):
        arr = np.random.default_rng(0).standard_normal((32, 32))
        out = tmp_path / "test.png"
        self._export(arr, out, prov=self._make_prov())
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text())
        for key in ("source_file", "source_format", "item_type",
                    "processing_state", "display_state", "export_timestamp"):
            assert key in data, f"Key '{key}' missing from sidecar"

    def test_no_sidecar_without_provenance(self, tmp_path):
        arr = np.random.default_rng(0).standard_normal((32, 32))
        out = tmp_path / "test.png"
        self._export(arr, out)
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        assert not sidecar.exists(), "Sidecar should not be created without provenance"

    def test_png_still_written_when_provenance_given(self, tmp_path):
        arr = np.random.default_rng(0).standard_normal((32, 32))
        out = tmp_path / "test.png"
        self._export(arr, out, prov=self._make_prov())
        assert out.exists() and out.stat().st_size > 0


# ── F: No visual regression in PNG export ────────────────────────────────────

class TestPngNoRegression:
    def test_png_output_unchanged_without_provenance(self, tmp_path):
        from probeflow.processing import export_png
        rng = np.random.default_rng(999)
        arr = rng.standard_normal((64, 64))
        lut_fn = lambda _: _gray_lut()

        out_no_prov = tmp_path / "no_prov.png"
        out_with_prov = tmp_path / "with_prov.png"
        prov = ExportProvenance(
            source_file=None, source_format=None, item_type="scan",
            channel_name=None, channel_index=0, array_shape=(64, 64),
            scan_range_m=None, units=None,
            processing_state={"steps": []}, display_state={},
            probeflow_version=None,
            export_timestamp="2026-04-25T00:00:00Z",
        )

        kwargs = dict(lut_fn=lut_fn, scan_range_m=(0.0, 0.0), add_scalebar=False)
        export_png(arr, out_no_prov, "gray", 1.0, 99.0, **kwargs)
        export_png(arr, out_with_prov, "gray", 1.0, 99.0, **kwargs, provenance=prov)

        # Both PNGs must exist and have identical content
        assert out_no_prov.read_bytes() == out_with_prov.read_bytes(), (
            "Passing provenance must not change the PNG pixel content"
        )

    def test_write_png_passes_provenance_to_export_png(self, tmp_path):
        """write_png with provenance= writes a sidecar alongside the PNG."""
        from probeflow.io.writers.png import write_png
        scan = _make_scan()
        out = tmp_path / "scan.png"
        prov = ExportProvenance.from_scan_export(scan, channel_index=0)
        write_png(scan, out, plane_idx=0, provenance=prov)
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        assert out.exists(), "PNG not written"
        assert sidecar.exists(), "Sidecar not written via write_png"

    def test_write_png_builds_standard_provenance_by_default(self, tmp_path):
        from probeflow.io.writers.png import write_png

        scan = _make_scan()
        scan.processing_history = [
            {"op": "align_rows", "params": {"method": "median"}, "timestamp": "T"},
        ]
        out = tmp_path / "scan.png"
        write_png(
            scan,
            out,
            plane_idx=0,
            colormap="plasma",
            clip_low=2.0,
            clip_high=98.0,
            add_scalebar=False,
        )

        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["export_kind"] == "png"
        assert data["processing_state"]["steps"][0]["op"] == "align_rows"
        assert data["display_state"]["colormap"] == "plasma"
        assert data["display_state"]["add_scalebar"] is False
        assert data["artifact_id"]

    def test_prepared_png_export_writes_handoff_sidecar_warning(self, tmp_path):
        from probeflow.provenance.prepared_export import write_prepared_png

        scan = _make_scan()
        out = tmp_path / "aisurf_input.png"
        write_prepared_png(
            scan,
            out,
            processing_state=ProcessingState(steps=[
                ProcessingStep("smooth", {"sigma_px": 1.0}),
            ]),
            add_scalebar=False,
        )

        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["export_kind"] == "prepared_png"
        assert data["warnings"]
        assert data["processing_state"]["steps"][0]["op"] == "smooth"

    def test_prepared_png_export_omits_warning_after_background(self, tmp_path):
        from probeflow.provenance.prepared_export import write_prepared_png

        scan = _make_scan()
        out = tmp_path / "aisurf_input.png"
        write_prepared_png(
            scan,
            out,
            processing_state=ProcessingState(steps=[
                ProcessingStep("plane_bg", {"order": 1}),
            ]),
            add_scalebar=False,
        )

        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["export_kind"] == "prepared_png"
        assert data["warnings"] == []
        assert data["processing_state_hash"]

    def test_cli_sxm2png_writes_standard_sidecar(self, tmp_path):
        from probeflow.cli import main as cli_main

        src = (
            Path(__file__).resolve().parents[1]
            / "anonymised_testdata"
            / "sxm_moire_10nm.sxm"
        )
        out = tmp_path / "moire.png"
        rc = cli_main(["sxm2png", str(src), "-o", str(out), "--no-scalebar"])

        assert rc == 0
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["export_kind"] == "cli_sxm2png"
        assert data["processing_state"] == {"steps": []}
        assert data["display_state"]["mode"] == "percentile"
        assert data["display_state"]["colormap"] == "gray"
        assert data["display_state"]["add_scalebar"] is False
        assert data["source_id"]

    def test_cli_pipeline_png_records_processing_sidecar(self, tmp_path):
        from probeflow.cli import main as cli_main

        src = (
            Path(__file__).resolve().parents[1]
            / "anonymised_testdata"
            / "sxm_moire_10nm.sxm"
        )
        out = tmp_path / "moire_processed.png"
        rc = cli_main([
            "pipeline", str(src),
            "--steps", "align-rows:median", "plane-bg:1",
            "--png", "-o", str(out),
            "--no-scalebar",
        ])

        assert rc == 0
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        ops = [step["op"] for step in data["processing_state"]["steps"]]
        assert data["export_kind"] == "cli_png"
        assert ops == ["align_rows", "plane_bg"]
        assert data["processing_state_hash"]

    def test_cli_prepare_png_warns_without_background_processing(self, tmp_path):
        from probeflow.cli import main as cli_main

        src = (
            Path(__file__).resolve().parents[1]
            / "anonymised_testdata"
            / "sxm_moire_10nm.sxm"
        )
        out = tmp_path / "aisurf_raw.png"
        rc = cli_main(["prepare-png", str(src), str(out)])

        assert rc == 0
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["export_kind"] == "prepared_png"
        assert data["processing_state"] == {"steps": []}
        assert data["warnings"]
        assert data["display_state"]["add_scalebar"] is False

    def test_cli_prepare_png_records_background_processing(self, tmp_path):
        from probeflow.cli import main as cli_main

        src = (
            Path(__file__).resolve().parents[1]
            / "anonymised_testdata"
            / "sxm_moire_10nm.sxm"
        )
        out = tmp_path / "aisurf_prepared.png"
        rc = cli_main([
            "prepare-png", str(src), str(out),
            "--steps", "align-rows:median", "plane-bg:1",
            "--colormap", "plasma",
        ])

        assert rc == 0
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        ops = [step["op"] for step in data["processing_state"]["steps"]]
        assert data["export_kind"] == "prepared_png"
        assert ops == ["align_rows", "plane_bg"]
        assert data["warnings"] == []
        assert data["display_state"]["colormap"] == "plasma"
