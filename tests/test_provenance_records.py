from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probeflow.core.scan_model import Scan
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
    assert data["source_file_type"] == "Createc .dat"
    assert data["channel"] == "FT"
    assert data["file_hash"] == "abc123"


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
