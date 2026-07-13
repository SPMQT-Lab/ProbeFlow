"""Compatibility helpers between ProcessingState and legacy history entries."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from probeflow.processing.state import ProcessingState, ProcessingStep


def processing_history_entries_from_state(
    state: ProcessingState,
    *,
    timestamp: str | None = None,
    timestamps: list[str | None] | tuple[str | None, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return legacy ``Scan.processing_history`` entries for a state."""
    if timestamps is None:
        ts = timestamp or datetime.now().isoformat()
        timestamps = [ts] * len(state.steps)

    entries: list[dict[str, Any]] = []
    for idx, step in enumerate(state.steps):
        entry = {
            "op": step.op,
            "params": deepcopy(step.params),
        }
        ts = timestamps[idx] if idx < len(timestamps) else None
        if ts is not None:
            entry["timestamp"] = ts
        entries.append(entry)
    return entries


# Provenance bookkeeping steps that are not data-processing operations; a
# replayable ProcessingState must not contain them (mirrors the
# ``_has_data_processing`` classification in provenance/records.py).
_NON_DATA_OPS = {"file_load", "dat_to_sxm"}


def processing_state_from_history(
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> ProcessingState:
    """Convert history entries into a canonical ``ProcessingState``.

    Accepts both legacy ``Scan.processing_history`` entries
    (``{"op", "params"}``) and provenance-record steps
    (``{"operation_id", "parameters"}`` — the ``ProcessingHistory`` step
    format stored in ``.probeflow.json`` sidecars).  Record bookkeeping steps
    (``file_load``, ``export_*``, ``dat_to_sxm``) are skipped so the result
    replays cleanly through ``apply_processing_state``.  Before this accepted
    the record format, feeding it ``.probeflow.json`` steps silently produced
    an *empty* state — a replay would reproduce the raw image while claiming
    to be processed.
    """
    state, _timestamps = processing_state_and_timestamps_from_history(history)
    return state


def processing_state_and_timestamps_from_history(
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> tuple[ProcessingState, list[str | None]]:
    """Normalize replayable steps and their timestamps in the same pass."""
    steps: list[ProcessingStep] = []
    timestamps: list[str | None] = []
    for entry in history or ():
        if not isinstance(entry, dict):
            continue
        op = entry.get("op") or entry.get("operation_id")
        if not op:
            continue
        op = str(op)
        if op in _NON_DATA_OPS or op.startswith("export_"):
            continue
        if "op" not in entry and "operation_id" in entry:
            params = dict(entry.get("parameters") or {})
        else:
            params = entry.get("params")
            if params is None:
                params = {
                    key: value
                    for key, value in entry.items()
                    if key not in {"op", "timestamp"}
                }
            params = dict(params)
        steps.append(ProcessingStep(op, deepcopy(params)))
        timestamp = entry.get("timestamp")
        timestamps.append(str(timestamp) if timestamp is not None else None)
    return ProcessingState(steps=steps), timestamps


def processing_state_dict_from_history(
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> dict[str, Any]:
    """Return ProcessingState JSON for legacy history entries."""
    return processing_state_from_history(history).to_dict()
