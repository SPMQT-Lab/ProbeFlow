import json
import shutil
from pathlib import Path

from probeflow.core.indexing import index_folder
from probeflow.gui.models import SxmFile
from probeflow.io.scanflow_acquisition import (
    load_scanflow_scan_sidecar,
    load_scanflow_session_manifest,
    scanflow_sidecar_path,
)


TESTDATA = Path(__file__).resolve().parents[1] / "test_data"
CREATEC_SCAN = TESTDATA / "createc_scan_step_20nm.dat"


def _scanflow_payload(raw_name: str) -> dict:
    return {
        "schema": "scanflow.acquisition.v1",
        "record_type": "scanflow_scan",
        "raw_file": {"path": raw_name, "source_format": "createc_dat"},
        "session": {"session_id": "session-1", "routine": "mosaic"},
        "step": {"index": 1, "kind": "scan", "label": "tile"},
        "scan_parameters": {"bias_V": 0.1, "setpoint_A": 5e-11},
        "position": {"scan_offset_nm": [1.0, 2.0]},
        "motion": {"ok": True, "reason": "unit test"},
        "drift": {"enabled": False},
        "quality": {},
        "safety": {"enabled": True, "aborted": False},
    }


def test_scanflow_sidecar_attaches_to_indexed_scan_metadata(tmp_path):
    scan = tmp_path / CREATEC_SCAN.name
    shutil.copyfile(CREATEC_SCAN, scan)
    sidecar = scanflow_sidecar_path(scan)
    sidecar.write_text(json.dumps(_scanflow_payload(scan.name)), encoding="utf-8")

    items = index_folder(tmp_path)

    item = next(it for it in items if it.path.name == scan.name)
    metadata = item.metadata["scanflow_acquisition"]
    assert metadata["schema"] == "scanflow.acquisition.v1"
    assert metadata["session"]["session_id"] == "session-1"
    assert metadata["position"]["scan_offset_nm"] == [1.0, 2.0]


def test_scanflow_sidecar_survives_gui_model_conversion(tmp_path):
    scan = tmp_path / CREATEC_SCAN.name
    shutil.copyfile(CREATEC_SCAN, scan)
    sidecar = scanflow_sidecar_path(scan)
    sidecar.write_text(json.dumps(_scanflow_payload(scan.name)), encoding="utf-8")

    items = index_folder(tmp_path)
    item = next(it for it in items if it.path.name == scan.name)
    gui_entry = SxmFile.from_index_item(item)

    assert gui_entry.scanflow_acquisition["session"]["session_id"] == "session-1"
    assert gui_entry.scanflow_acquisition["step"]["label"] == "tile"


def test_scanflow_loaders_do_not_import_scanflow(tmp_path):
    scan = tmp_path / "scan.dat"
    sidecar = scanflow_sidecar_path(scan)
    sidecar.write_text(json.dumps(_scanflow_payload(scan.name)), encoding="utf-8")
    manifest = tmp_path / "scanflow_session.json"
    manifest.write_text(
        json.dumps({
            "schema": "scanflow.session.v1",
            "record_type": "scanflow_session",
            "session_id": "session-1",
            "recipe": {"name": "Test", "routine": "survey"},
            "scans": [{"dat_path": "scan.dat", "sidecar_path": sidecar.name}],
        }),
        encoding="utf-8",
    )

    assert load_scanflow_scan_sidecar(scan)["raw_file"]["path"] == "scan.dat"
    assert load_scanflow_session_manifest(tmp_path)["session_id"] == "session-1"
