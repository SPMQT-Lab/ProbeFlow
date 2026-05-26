"""Tests for the ProbeFlow-backed preview analysis facade."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probeflow.analysis.preview import PreviewAnalysisParams, preview_record, run_preview
from probeflow.core.scan_loader import load_scan


FIXTURE = Path(__file__).resolve().parents[1] / "test_data" / "createc_scan_preview_120nm.dat"


def test_run_preview_on_real_dat_returns_feature_rows_and_json_safe_record() -> None:
    result = run_preview(FIXTURE, PreviewAnalysisParams())

    assert result.source_path == FIXTURE
    assert result.source_format == "dat"
    assert result.raw_plane.shape == result.background_corrected_plane.shape
    assert result.background_image.shape == result.raw_plane.shape
    assert len(result.feature_rows) >= 1
    assert len(result.overlay_coordinates) == len(result.feature_rows)
    assert result.analysis_status in {"success", "empty"}

    record = preview_record(result)
    json.dumps(record)
    assert record["source_format"] == "dat"
    assert record["scan_metadata"]["source_format"] == "createc_dat"
    assert record["feature_rows"]


def test_run_preview_accepts_loaded_scan_and_points_only_mode() -> None:
    scan = load_scan(FIXTURE)
    result = run_preview(
        scan,
        PreviewAnalysisParams(feature_mode="points_only", min_distance_px=4.0),
    )

    assert result.source_path == FIXTURE
    assert result.scan_shape == scan.planes[0].shape
    assert all(row.source == "points" for row in result.feature_rows)


@dataclass
class _FakeScan:
    planes: list[np.ndarray]
    plane_names: list[str]
    plane_units: list[str]
    scan_range_m: tuple[float, float]
    source_path: Path
    source_format: str
    header: dict[str, object]
    experiment_metadata: dict[str, object]


def test_segmentation_only_returns_empty_on_flat_scan() -> None:
    scan = _FakeScan(
        planes=[np.zeros((32, 32), dtype=np.float64)],
        plane_names=["Z forward"],
        plane_units=["m"],
        scan_range_m=(32e-9, 32e-9),
        source_path=Path("flat.dat"),
        source_format="dat",
        header={},
        experiment_metadata={},
    )

    result = run_preview(scan, PreviewAnalysisParams(feature_mode="segmentation_only"))
    assert result.feature_rows == ()
    assert any("segmentation returned no features" in w for w in result.warnings)
