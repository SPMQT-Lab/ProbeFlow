"""Tests for exporting Particle Statistics results to CSV / JSON."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np

from probeflow.analysis.adstat_adapter import compare_point_source_view_spec
from probeflow.measurements.adstat_export import (
    export_result_csvs,
    export_result_json,
    panel_curve_csv_text,
    verdict_rows_csv_text,
)


def _spec():
    rng = np.random.default_rng(0)
    pts_nm = rng.uniform(5.0, 95.0, size=(40, 2))
    points_m = pts_nm * 1e-9
    pixel_nm = 100.0 / 256.0
    points_px = pts_nm / pixel_nm
    source = SimpleNamespace(
        label="export test",
        source_type="synthetic",
        points_px=points_px,
        points_m=points_m,
        metadata={},
    )
    scan = SimpleNamespace(scan_range_m=(100e-9, 100e-9), dims=(256, 256))
    return compare_point_source_view_spec(
        source, scan=scan, image_shape=(256, 256), n_simulations=5, random_seed=0
    )


def test_export_csvs_writes_curve_and_verdict_files(tmp_path):
    spec = _spec()
    written = export_result_csvs(spec, tmp_path, base="run A")
    names = {p.name for p in written}
    # Curve statistics each get a file; the verdict table gets one too.
    assert any("pair_correlation_g_r" in n for n in names)
    assert any("nearest_neighbor_distribution" in n for n in names)
    assert any(n.endswith("_verdicts.csv") for n in names)
    # base label is slugged (space -> underscore)
    assert all(n.startswith("run_A_") for n in names)


def test_curve_csv_has_distance_and_observed_columns(tmp_path):
    spec = _spec()
    gr = next(p for p in spec.panels if getattr(p, "statistic", "") == "pair_correlation_g_r")
    text = panel_curve_csv_text(gr)
    assert text is not None
    rows = list(csv.reader(text.splitlines()))
    header = rows[0]
    assert header[0]  # x label present (distance axis)
    assert "observed" in header
    assert "model_low" in header and "model_high" in header
    assert len(rows) > 2  # header + several data rows
    # data rows are numeric
    float(rows[1][0])
    float(rows[1][1])


def test_verdict_csv_round_trips(tmp_path):
    spec = _spec()
    text = verdict_rows_csv_text(spec)
    assert text is not None
    rows = list(csv.reader(text.splitlines()))
    assert rows[0][0] == "model"
    assert len(rows) > 1


def test_export_json_snapshot(tmp_path):
    spec = _spec()
    out = export_result_json(spec, tmp_path / "result.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "panels" in payload
    assert "verdict_rows" in payload
    assert isinstance(payload["panels"], list) and payload["panels"]


def test_curve_csv_skips_non_curve_panels():
    # A real-space scatter panel (2-D points, no 1-D x curve) is not a curve.
    panel = SimpleNamespace(
        statistic="real_space", x_label="x (nm)", x=None, observed=np.zeros((5, 2))
    )
    assert panel_curve_csv_text(panel) is None
