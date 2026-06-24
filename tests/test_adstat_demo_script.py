"""Tests for the user-facing AdStat synthetic demo script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("adstat")


def _load_demo_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "adstat_demo.py"
    spec = importlib.util.spec_from_file_location("probeflow_adstat_demo_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_adstat_demo_script_generates_teaching_artifacts(tmp_path) -> None:
    module = _load_demo_module()

    exit_code = module.main([
        "--output-dir",
        str(tmp_path),
        "--points",
        "12",
        "--n-simulations",
        "4",
        "--seed",
        "123",
    ])

    assert exit_code == 0
    points_csv = tmp_path / "synthetic_points.csv"
    result_json = tmp_path / "adstat_result_view_spec.json"
    preview_png = tmp_path / "synthetic_points_preview.png"
    assert points_csv.exists()
    assert result_json.exists()
    assert preview_png.exists()
    assert len(points_csv.read_text(encoding="utf-8").splitlines()) == 13

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["demo"]["seed"] == 123
    panels = payload["result_view_spec"]["panels"]
    assert panels[0]["kind"] == "realspace"
    assert payload["result_view_spec"]["verdict_rows"]
