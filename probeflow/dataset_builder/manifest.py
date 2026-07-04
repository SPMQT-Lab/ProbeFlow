"""Manifest writing for Dataset Builder exports."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = (
    "dataset_schema_version",
    "sample_id",
    "source_path",
    "source_format",
    "source_hash",
    "scan_stem",
    "plane_index",
    "channel_name",
    "channel_unit",
    "array_shape",
    "scan_range_m",
    "pixel_size_x_m",
    "pixel_size_y_m",
    "task",
    "label_type",
    "label_names",
    "review_status",
    "annotator",
    "reviewed_at",
    "proposal_method",
    "proposal_parameters_json",
    "task_data_json",
    "processing_state_hash",
    "array_path",
    "preview_path",
    "mask_paths_json",
    "seed_path",
    "quickseg_result_path",
    "roi_path",
    "objects_path",
    "provenance_path",
    "native_mask_sidecar",
    "native_roi_sidecar",
    "native_probeflow_sidecar",
    "notes",
)


def write_manifest(output_dir: str | Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    out = Path(output_dir)
    csv_path = out / "manifest.csv"
    json_path = out / "manifest.json"
    out.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=out,
        prefix="manifest.csv.tmp",
        suffix=".csv",
    )
    try:
        with open(tmp_fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})
        Path(tmp_path).replace(csv_path)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=out,
        prefix="manifest.json.tmp",
        suffix=".json",
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump({"schema_version": 1, "rows": rows}, fh, indent=2, default=str)
        Path(tmp_path).replace(json_path)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return csv_path, json_path
