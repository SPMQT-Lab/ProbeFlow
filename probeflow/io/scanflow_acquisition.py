"""Readers for neutral ScanFlow acquisition sidecars.

This module deliberately does not import ScanFlow. The sidecar is a stable JSON
contract between acquisition and analysis, not a Python package dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCAN_SIDECAR_SCHEMA = "scanflow.acquisition.v1"
SESSION_SCHEMA = "scanflow.session.v1"


def scanflow_sidecar_path(scan_path: Path | str) -> Path:
    return Path(scan_path).with_suffix(".scanflow.json")


def session_manifest_path(folder: Path | str) -> Path:
    return Path(folder) / "scanflow_session.json"


def load_scanflow_scan_sidecar(
    path: Path | str,
    *,
    missing_ok: bool = True,
) -> dict[str, Any] | None:
    """Load a sibling ``*.scanflow.json`` sidecar for a scan file."""
    p = Path(path)
    sidecar = p if p.name.endswith(".scanflow.json") else scanflow_sidecar_path(p)
    if not sidecar.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(sidecar)
    payload = _load_json_object(sidecar)
    schema = payload.get("schema")
    if schema != SCAN_SIDECAR_SCHEMA:
        raise ValueError(f"not a ScanFlow scan sidecar: {sidecar} schema={schema!r}")
    return payload


def load_scanflow_session_manifest(path: Path | str) -> dict[str, Any]:
    """Load ``scanflow_session.json`` without importing ScanFlow."""
    p = Path(path)
    if p.is_dir():
        p = session_manifest_path(p)
    payload = _load_json_object(p)
    schema = payload.get("schema")
    if schema != SESSION_SCHEMA:
        raise ValueError(f"not a ScanFlow session manifest: {p} schema={schema!r}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload
