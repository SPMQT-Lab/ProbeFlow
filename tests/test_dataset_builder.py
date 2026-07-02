from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from probeflow.cli.parser import _build_parser
from probeflow.core.mask import ImageMask
from probeflow.dataset_builder.annotations import save_mask_annotation
from probeflow.dataset_builder.export import export_dataset
from probeflow.dataset_builder.loading import load_scan_plane
from probeflow.dataset_builder.models import DatasetExportSpec, DatasetTaskConfig
from probeflow.dataset_builder.proposals import generate_proposal
from probeflow.dataset_builder.queue import build_queue


def _sample_scan(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[1] / "test_data" / "sample_input" / "sxm" / "A250320.191933.sxm"
    dst = tmp_path / src.name
    shutil.copyfile(src, dst)
    return dst


def test_dataset_builder_review_state_preserves_existing_probeflow_sidecar(tmp_path):
    scan_path = _sample_scan(tmp_path)
    sidecar = scan_path.parent / f"{scan_path.stem}.probeflow.json"
    sidecar.write_text(
        json.dumps({"record_type": "existing_probe_flow_payload", "keep": True}),
        encoding="utf-8",
    )
    _scan, arr, _px_x, _px_y = load_scan_plane(scan_path, 0)
    mask_data = np.zeros(arr.shape, dtype=bool)
    mask_data[0, 0] = True
    config = DatasetTaskConfig(plane_index=0)

    save_mask_annotation(
        scan_path,
        ImageMask.new(mask_data, method="manual", name=config.label_name),
        config=config,
        status="accepted",
    )

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["keep"] is True
    assert payload["dataset_builder"]["records"]["step_edge_mask:plane0:step_edge"]["status"] == "accepted"

    queue = build_queue(tmp_path, plane_index=0)
    assert len(queue) == 1
    assert queue[0].status == "accepted"
    assert queue[0].has_mask_sidecar is True


def test_dataset_builder_step_edge_proposal_matches_plane_shape(tmp_path):
    scan_path = _sample_scan(tmp_path)
    scan, arr, px_x, px_y = load_scan_plane(scan_path, 0)
    config = DatasetTaskConfig(
        plane_index=0,
        proposal_params={"molecule_size_nm": 1.0, "margin_nm": 0.1},
    )

    proposal = generate_proposal(
        arr,
        px_x_m=px_x,
        px_y_m=px_y,
        config=config,
        source_channel=scan.plane_names[0],
    )

    assert proposal.label_type == "mask"
    assert proposal.mask is not None
    assert proposal.mask.shape == arr.shape
    assert proposal.parameters["source_channel"] == scan.plane_names[0]


def test_dataset_builder_export_writes_manifest_and_artifacts(tmp_path):
    scan_path = _sample_scan(tmp_path)
    _scan, arr, _px_x, _px_y = load_scan_plane(scan_path, 0)
    mask_data = np.zeros(arr.shape, dtype=bool)
    mask_data[:2, :3] = True
    config = DatasetTaskConfig(plane_index=0, annotator="tester")
    save_mask_annotation(
        scan_path,
        ImageMask.new(mask_data, method="manual", name=config.label_name),
        config=config,
        status="accepted",
        notes="unit test",
    )

    out = tmp_path / "dataset"
    summary = export_dataset(
        DatasetExportSpec(
            source=tmp_path,
            output_dir=out,
            plane_index=0,
            overwrite=False,
        )
    )

    assert summary["n_exported"] == 1
    rows = list(csv.DictReader((out / "manifest.csv").open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    row = rows[0]
    assert row["review_status"] == "accepted"
    assert row["annotator"] == "tester"
    assert (out / row["array_path"]).exists()
    assert (out / row["preview_path"]).exists()
    assert (out / row["provenance_path"]).exists()
    mask_paths = json.loads(row["mask_paths_json"])
    assert (out / mask_paths["step_edge"]).exists()
    assert (out / mask_paths["step_edge_png"]).exists()


def test_dataset_cli_parser_exposes_dataset_subcommands():
    parser = _build_parser()

    args = parser.parse_args([
        "dataset",
        "export",
        "scans",
        "out",
        "--status",
        "accepted",
        "uncertain",
    ])

    assert args.command == "dataset"
    assert args.dataset_command == "export"
    assert args.status == ["accepted", "uncertain"]

