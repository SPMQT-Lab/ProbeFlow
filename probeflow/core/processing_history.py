"""Compatibility between ProcessingState and legacy history dictionaries."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from probeflow.core.processing_state import ProcessingState, ProcessingStep


def processing_history_entries_from_state(
    state: ProcessingState,
    *,
    timestamp: str | None = None,
    timestamps: list[str | None] | tuple[str | None, ...] | None = None,
) -> list[dict[str, Any]]:
    """Return legacy ``Scan.processing_history`` entries for a state."""
    if timestamps is None:
        value = timestamp or datetime.now().isoformat()
        timestamps = [value] * len(state.steps)

    entries: list[dict[str, Any]] = []
    for index, step in enumerate(state.steps):
        entry = {
            "op": step.op,
            "params": deepcopy(step.params),
        }
        step_timestamp = timestamps[index] if index < len(timestamps) else None
        if step_timestamp is not None:
            entry["timestamp"] = step_timestamp
        entries.append(entry)
    return entries


# Provenance bookkeeping steps are not replayable numerical operations.
_NON_DATA_OPS = frozenset({"file_load", "dat_to_sxm"})


def processing_state_from_history(
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> ProcessingState:
    """Convert legacy or provenance history entries into ProcessingState."""
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


__all__ = [
    "processing_history_entries_from_state",
    "processing_state_and_timestamps_from_history",
    "processing_state_dict_from_history",
    "processing_state_from_history",
]
