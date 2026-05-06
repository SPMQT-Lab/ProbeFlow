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


def processing_state_from_history(
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> ProcessingState:
    """Convert legacy history entries into canonical ``ProcessingState``."""
    steps: list[ProcessingStep] = []
    for entry in history or ():
        if not isinstance(entry, dict):
            continue
        op = entry.get("op")
        if not op:
            continue
        params = entry.get("params")
        if params is None:
            params = {
                key: value
                for key, value in entry.items()
                if key not in {"op", "timestamp"}
            }
        steps.append(ProcessingStep(str(op), deepcopy(dict(params))))
    return ProcessingState(steps=steps)


def processing_state_dict_from_history(
    history: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> dict[str, Any]:
    """Return ProcessingState JSON for legacy history entries."""
    return processing_state_from_history(history).to_dict()
