"""Dataset Builder state stored in native ProbeFlow sidecars."""

from __future__ import annotations

import datetime as _dt
import json
import tempfile
from pathlib import Path
from typing import Any

from probeflow.dataset_builder.models import ReviewRecord, record_key

STATE_SCHEMA_VERSION = 1
STATE_KEY = "dataset_builder"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_state_sidecar_path(scan_path: str | Path) -> Path:
    # REVIEW(2026-07-06, high): `<stem>.probeflow.json` next to the scan is
    # the SAME path a provenance sidecar gets when the user exports
    # `<stem>.csv`/`.gwy` beside the data (write_provenance_sidecars →
    # out_path.with_suffix(".probeflow.json")). A non-forced export then dies
    # with a confusing FileExistsError, and `--force`/overwrite=True replaces
    # this file WHOLESALE — wiping every dataset_builder review record for
    # the scan. Rename the state file (e.g. `<stem>.dataset.json`) while the
    # format is young; keep a one-shot migration read from the old name.
    path = Path(scan_path)
    return path.parent / f"{path.stem}.probeflow.json"


def load_state_payload(scan_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = default_state_sidecar_path(scan_path)
    if not path.exists():
        return {}, path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read ProbeFlow sidecar {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"ProbeFlow sidecar {path} must contain a JSON object")
    return data, path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".tmp",
        suffix=".json",
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        Path(tmp_path).replace(path)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _state_root(payload: dict[str, Any]) -> dict[str, Any]:
    root = payload.setdefault(STATE_KEY, {})
    if not isinstance(root, dict):
        raise ValueError(f"{STATE_KEY!r} in ProbeFlow sidecar must be a JSON object")
    root.setdefault("schema_version", STATE_SCHEMA_VERSION)
    root.setdefault("records", {})
    if not isinstance(root["records"], dict):
        raise ValueError(f"{STATE_KEY}.records must be a JSON object")
    return root


def load_review_record(
    scan_path: str | Path,
    *,
    task: str,
    plane_index: int,
    label_name: str,
) -> ReviewRecord | None:
    payload, _ = load_state_payload(scan_path)
    root = payload.get(STATE_KEY)
    if not isinstance(root, dict):
        return None
    records = root.get("records")
    if not isinstance(records, dict):
        return None
    data = records.get(record_key(task, plane_index, label_name))
    if not isinstance(data, dict):
        return None
    return ReviewRecord.from_dict(data)


def save_review_record(scan_path: str | Path, record: ReviewRecord) -> Path:
    payload, path = load_state_payload(scan_path)
    root = _state_root(payload)
    records = root["records"]
    records[record.key] = record.to_dict()
    _write_json_atomic(path, payload)
    return path


# REVIEW(2026-07-06, low): dead code — nothing calls mark_exported, so the
# "exported" status and exported_at timestamps in REVIEW_STATUSES never
# happen. Either wire it into export_dataset (per exported row) or drop the
# status from the lifecycle; half-wired it will confuse the queue UI later.
def mark_exported(scan_path: str | Path, record: ReviewRecord) -> ReviewRecord:
    exported = ReviewRecord(
        source_path=record.source_path,
        plane_index=record.plane_index,
        task=record.task,
        label_type=record.label_type,
        label_name=record.label_name,
        status="exported",
        annotator=record.annotator,
        notes=record.notes,
        proposal_method=record.proposal_method,
        proposal_parameters=record.proposal_parameters,
        task_data=record.task_data,
        updated_at=utc_now(),
        exported_at=utc_now(),
    )
    save_review_record(scan_path, exported)
    return exported
