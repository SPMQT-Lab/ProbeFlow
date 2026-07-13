from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from probeflow.core.scan_model import Scan
from probeflow.core.roi import ROI, ROISet
from probeflow.processing.state import ProcessingState, ProcessingStep as StateStep
from probeflow.provenance import (
    ProcessingHistory,
    SourceRecord,
    processing_history_from_scan,
)


def _scan(tmp_path: Path) -> Scan:
    return Scan(
        planes=[np.zeros((8, 8), dtype=np.float64)],
        plane_names=["FT"],
        plane_units=["m"],
        plane_synthetic=[False],
        header={"BIAS": "1.0"},
        scan_range_m=(10e-9, 10e-9),
        source_path=tmp_path / "A250326.105721.dat",
        source_format="dat",
    )


def test_source_record_creation():
    record = SourceRecord(
        source_filename="scan.dat",
        source_path="/data/scan.dat",
        source_file_type="Createc .dat",
        channel="FT",
        loader_name="Createc .dat reader",
        loader_version="1.0",
        metadata={"scan_range_m": [1e-8, 1e-8]},
        file_hash="abc123",
    )

    data = record.to_dict()

    assert data["source_filename"] == "scan.dat"
    assert data["source_path"] == "scan.dat"
    assert data["source_file_type"] == "Createc .dat"
    assert data["channel"] == "FT"
    assert data["file_hash"] == "abc123"


def test_export_serialization_redacts_local_paths_and_identity_fields(tmp_path):
    from probeflow.core.source_identity import build_source_identity, sanitize_export_data

    source = tmp_path / "scan.dat"
    source.write_bytes(b"fixture")
    identity = build_source_identity(
        source,
        source_format="dat",
        item_type="scan",
    )
    assert identity["source_path"] == "scan.dat"

    safe = sanitize_export_data({
        "source_path": "/Users/alice/private/scan.dat",
        "export_path": r"C:\Users\bob\Desktop\figure.png",
        "processing_state": {
            "steps": [{"params": {"source_path": "/home/carol/operand.sxm"}}],
        },
        "source_header": {
            "Username / Username": "alice",
            "ActDriveDir": r"V:\Data\alice\project",
            "LastMemoFile": r"C:\Users\alice\memo.txt",
        },
    })

    assert safe["source_path"] == "scan.dat"
    assert safe["export_path"] == "figure.png"
    assert safe["processing_state"]["steps"][0]["params"]["source_path"] == "operand.sxm"
    assert safe["source_header"]["Username / Username"] == ""
    assert safe["source_header"]["ActDriveDir"] == "project"
    assert safe["source_header"]["LastMemoFile"] == "memo.txt"
    serialized = json.dumps(safe)
    assert "/Users/" not in serialized
    assert "/home/" not in serialized
    assert "alice" not in serialized


def test_processing_step_append_records_state_ids(tmp_path):
    history = ProcessingHistory(
        SourceRecord(
            source_filename="scan.sxm",
            source_path="/data/scan.sxm",
            source_file_type="Nanonis .sxm",
            channel="Z",
            loader_name="Nanonis .sxm reader",
            loader_version="1.0",
        )
    )
    before = history.current_state_id

    step = history.append_step(
        operation_id="align_rows",
        operation_name="Row alignment",
        operation_version="1.0",
        parameters={"method": "median"},
        timestamp="2026-05-10T00:00:00Z",
    )

    assert step.input_state_id == before
    assert step.output_state_id == history.current_state_id
    assert step.operation_id == "align_rows"
    assert step.parameters == {"method": "median"}


def test_processing_history_json_roundtrip(tmp_path):
    scan = _scan(tmp_path)
    state = ProcessingState([
        StateStep("align_rows", {"method": "median"}),
        StateStep("smooth", {"sigma_px": 2.0}),
    ])
    history = processing_history_from_scan(scan, channel_index=0, processing_state=state)

    restored = ProcessingHistory.from_json(history.to_json())

    assert restored.source_record.source_filename == "A250326.105721.dat"
    assert [step.operation_id for step in restored.steps] == [
        "file_load",
        "align_rows",
        "smooth",
    ]
    assert restored.steps[1].parameters == {"method": "median"}


def test_processing_history_from_scan_with_empty_steps_dict(tmp_path):
    scan = _scan(tmp_path)
    scan.processing_state = None
    history = processing_history_from_scan(
        scan, channel_index=0, processing_state={"steps": []}
    )
    assert [step.operation_id for step in history.steps] == ["file_load"]
    assert history.current_state_id is not None


def test_provenance_step_canonical_name_and_legacy_alias():
    """Regression for review arch-backend #12 — the provenance package
    no longer collides with ``probeflow.processing.state.ProcessingStep``
    by accident.  ``ProvenanceStep`` is the canonical name; the legacy
    ``ProcessingStep`` re-export is preserved as an alias for backward
    compatibility."""
    from probeflow import provenance
    from probeflow.provenance.records import ProcessingStep, ProvenanceStep
    # Both names refer to the exact same class
    assert ProcessingStep is ProvenanceStep
    assert provenance.ProcessingStep is provenance.ProvenanceStep
    # Both are exported via __all__
    assert "ProvenanceStep" in provenance.__all__
    assert "ProcessingStep" in provenance.__all__


def test_step_summary_roi_branch(tmp_path):
    from probeflow.provenance.records import ProcessingHistory, SourceRecord

    history = ProcessingHistory(
        SourceRecord(
            source_filename="scan.dat",
            source_path="/data/scan.dat",
            source_file_type="Createc .dat",
            channel="FT",
            loader_name="Createc .dat reader",
            loader_version="1.0",
        )
    )
    history.append_step(
        operation_id="roi",
        operation_name="ROI-scoped Gaussian blur/smoothing",
        parameters={"step": {"op": "smooth", "params": {"sigma_px": 1.5}}, "roi": "bg"},
    )

    summary = history.short_summary()
    assert "ROI-scoped" in summary
    assert "bg" in summary
    assert "{'op'" not in summary


def test_step_summary_mask_branch_and_scope_semantics():
    from probeflow.provenance.records import ProcessingHistory, SourceRecord

    history = ProcessingHistory(
        SourceRecord(
            source_filename="scan.dat",
            source_path="/data/scan.dat",
            source_file_type="Createc .dat",
            channel="FT",
            loader_name="Createc .dat reader",
            loader_version="1.0",
        )
    )
    history.append_step(
        operation_id="mask",
        operation_name="Mask-scoped Gaussian blur/smoothing",
        parameters={
            "step": {"op": "smooth", "params": {"sigma_px": 1.5}},
            "mask": "edges",
            "scope_semantics": "full_image_compute_masked_paste",
        },
    )
    summary = history.short_summary()
    assert "Mask-scoped" in summary
    assert "edges" in summary
    assert "computed full-image, applied inside scope" in summary


def test_step_summary_region_scope_reads_as_region_not_roi():
    from probeflow.provenance.records import ProcessingHistory, SourceRecord

    history = ProcessingHistory(
        SourceRecord(
            source_filename="scan.dat",
            source_path="/data/scan.dat",
            source_file_type="Createc .dat",
            channel="FT",
            loader_name="reader",
            loader_version="1.0",
        )
    )
    history.append_step(
        operation_id="roi",
        operation_name="Region-scoped Gaussian blur/smoothing",
        parameters={
            "step": {"op": "smooth", "params": {"sigma_px": 1.5}},
            "scope_kind": "region",
            "scope_semantics": "full_image_compute_masked_paste",
            "frozen_geometry": {"kind": "rectangle",
                                "geometry": {"x": 0, "y": 0, "width": 4, "height": 4}},
        },
    )
    summary = history.short_summary()
    assert "Region-scoped" in summary
    assert "ROI-scoped" not in summary
    assert "frozen geometry" in summary


def test_build_export_record_round_trips_masks(tmp_path):
    from probeflow.core.mask import ImageMask, MaskSet
    from probeflow.provenance.records import ProcessingHistory, SourceRecord, build_export_record

    history = ProcessingHistory(
        SourceRecord(
            source_filename="scan.dat",
            source_path="/data/scan.dat",
            source_file_type="Createc .dat",
            channel="FT",
            loader_name="reader",
            loader_version="1.0",
        )
    )
    ms = MaskSet(image_id="img")
    ms.add(ImageMask.new(np.ones((4, 4), bool), name="m1"))

    record = build_export_record(
        history,
        export_path=tmp_path / "out.png",
        export_format="png",
        masks=ms.to_dict(),
    )
    assert record.masks is not None
    data = json.loads(record.to_json(default=str))
    assert data["masks"]["image_id"] == "img"
    # Round-trips back through from_dict.
    from probeflow.provenance.records import ExportRecord
    assert ExportRecord.from_dict(data).masks["image_id"] == "img"


def test_png_export_writes_probeflow_sidecar_with_history(tmp_path):
    from probeflow.io.writers.png import write_png

    scan = _scan(tmp_path)
    scan.processing_history = [
        {"op": "align_rows", "params": {"method": "median"}, "timestamp": "T"},
        {"op": "smooth", "params": {"sigma_px": 1.5}, "timestamp": "T"},
    ]
    out = tmp_path / "image.png"

    write_png(scan, out, add_scalebar=False)

    sidecar = out.with_suffix(".probeflow.json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    ops = [
        step["operation_id"]
        for step in data["processing_history"]["steps"]
    ]
    assert data["source_info"]["source_filename"] == "A250326.105721.dat"
    assert "align_rows" in ops
    assert "smooth" in ops
    assert "export_png" in ops
    assert data["processing_history"]["steps"][1]["parameters"] == {
        "method": "median",
    }
    assert data["display_settings"]["add_scalebar"] is False
    assert "not raw data" in data["warning"]


def test_probeflow_sidecar_preserves_rois_for_lookup(tmp_path):
    from probeflow.io.roi_sidecar import load_roi_set_sidecar
    from probeflow.io.writers.png import write_png
    from probeflow.provenance.export import build_scan_export_provenance

    scan = _scan(tmp_path)
    roi = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
                  name="terrace")
    roi_set = ROISet(image_id=str(scan.source_path))
    roi_set.add(roi)
    out = tmp_path / "image.png"
    prov = build_scan_export_provenance(
        scan,
        channel_index=0,
        roi_set=roi_set,
        export_kind="viewer_png",
        output_path=out,
    )

    write_png(scan, out, provenance=prov, add_scalebar=False)

    data = json.loads(out.with_suffix(".probeflow.json").read_text(encoding="utf-8"))
    assert data["rois"]["image_id"] == scan.source_path.name
    loaded, used = load_roi_set_sidecar(out)
    assert used == out.with_suffix(".probeflow.json")
    assert loaded.get_by_name("terrace").id == roi.id


def test_provenance_sidecar_write_failure_preserves_existing_file(
    monkeypatch,
    tmp_path,
):
    import probeflow.provenance.export as export_mod
    from probeflow.provenance.export import (
        build_scan_export_provenance,
        write_provenance_sidecars,
    )

    scan = _scan(tmp_path)
    out = tmp_path / "image.png"
    sidecar = out.with_suffix(".probeflow.json")
    sidecar.write_text("sidecar sentinel", encoding="utf-8")
    prov = build_scan_export_provenance(
        scan,
        channel_index=0,
        export_kind="png",
        output_path=out,
    )

    def fail_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(export_mod._json, "dump", fail_dump)

    with pytest.raises(OSError, match="disk full"):
        write_provenance_sidecars(
            out,
            prov,
            legacy=False,
            probeflow=True,
            export_format="png",
            overwrite=True,
        )

    assert sidecar.read_text(encoding="utf-8") == "sidecar sentinel"
    assert not list(tmp_path.glob("*.probeflow.json.tmp*.json"))


def test_viewer_png_export_reports_provenance_build_failure(monkeypatch, tmp_path):
    from probeflow.gui.viewer.png_export import save_viewer_png

    scan = _scan(tmp_path)

    class DisplayRange:
        def resolve(self, arr):
            return float(np.nanmin(arr)), float(np.nanmax(arr))

    def fail_provenance(*args, **kwargs):
        raise RuntimeError("boom")

    def fail_if_exported(*args, **kwargs):
        raise AssertionError("PNG export should stop when provenance fails")

    monkeypatch.setattr("probeflow.core.scan_loader.load_scan", lambda path: scan)
    monkeypatch.setattr(
        "probeflow.provenance.export.build_scan_export_provenance",
        fail_provenance,
    )
    monkeypatch.setattr("probeflow.processing.export_png", fail_if_exported)

    msg = save_viewer_png(
        np.ones((4, 4), dtype=float),
        str(tmp_path / "viewer.png"),
        scan.source_path,
        "gray",
        1.0,
        99.0,
        DisplayRange(),
        {},
        None,
        0,
        "FT",
    )

    assert msg.startswith("Export error: provenance could not be built:")


def test_png_export_embeds_human_summary(tmp_path):
    from PIL import Image
    from probeflow.io.writers.png import write_png

    scan = _scan(tmp_path)
    scan.processing_history = [
        {"op": "align_rows", "params": {"method": "median"}, "timestamp": "T"},
    ]
    out = tmp_path / "image.png"

    write_png(scan, out, add_scalebar=False)

    img = Image.open(out)
    assert "ProbeFlow provenance" in img.info
    assert "not raw data" in img.info["ProbeFlow provenance"]
    assert "Original file: A250326.105721.dat" in img.info["ProbeFlow provenance"]


def test_dat_to_sxm_conversion_writes_conversion_warning(first_sample_dat, cushion_dir, tmp_path):
    from probeflow.io.converters.createc_dat_to_sxm import convert_dat_to_sxm
    from probeflow.io.sxm_io import parse_sxm_header

    convert_dat_to_sxm(first_sample_dat, tmp_path, cushion_dir)

    out = tmp_path / f"{first_sample_dat.stem}.sxm"
    sidecar = out.with_suffix(".probeflow.json")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    warnings = "\n".join(data["warnings"])
    ops = [
        step["operation_id"]
        for step in data["processing_history"]["steps"]
    ]
    comment = parse_sxm_header(out).get("COMMENT", "")

    assert "converted from Createc .dat to Nanonis-compatible .sxm" in warnings
    assert "compatibility export" in warnings
    assert "Converter version:" in warnings
    assert "dat_to_sxm" in ops
    assert "converted from Createc .dat to Nanonis-compatible .sxm" in comment
