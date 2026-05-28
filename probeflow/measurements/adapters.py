"""Adapters between legacy and canonical measurement records."""

from __future__ import annotations

from typing import Any

from probeflow.measurements.models import MeasurementResult


def legacy_measurement_to_result(
    legacy_result: Any,
    measurement_id: str,
) -> MeasurementResult:
    """Convert a duck-typed legacy measurement result into the canonical model.

    The legacy ``probeflow.analysis.measurements.MeasurementResult`` dataclass
    was removed in arch-backend #1 (2026-05-28).  This adapter is kept as a
    safety net for any future dialog that emits an object with the legacy
    duck-typed attributes (``values``, ``units``, ``kind``, ``summary``,
    ``roi_id``, ``source``, ``channel``, ``context``, ``notes``).  All
    in-tree dialogs now emit the canonical
    :class:`probeflow.measurements.models.MeasurementResult` directly.
    """
    units = dict(getattr(legacy_result, "units", {}) or {})
    values = dict(getattr(legacy_result, "values", {}) or {})
    kind = str(getattr(legacy_result, "kind", "") or "")

    if kind == "roi_stats":
        if "mean" in values and "mean_height" not in values:
            values["mean_height"] = values.pop("mean")
        if "mean" in units and "mean_height" not in units:
            units["mean_height"] = units.pop("mean")

    summary = str(getattr(legacy_result, "summary", "") or "")
    roi_id = getattr(legacy_result, "roi_id", None)
    context = {
        k: v for k, v in {
            "summary": summary,
            "roi_id": roi_id,
        }.items() if v
    }
    context.update(dict(getattr(legacy_result, "context", {}) or {}))

    source_label = str(getattr(legacy_result, "source", "") or "")
    source_path = context.get("source_path") or source_label or None

    return MeasurementResult(
        measurement_id=measurement_id,
        kind=kind,
        source_label=source_label,
        source_path=source_path,
        channel=getattr(legacy_result, "channel", None),
        x_unit=_x_unit(kind, units),
        y_unit=None,
        z_unit=_z_unit(units),
        values=values,
        context=context,
        notes=str(getattr(legacy_result, "notes", "") or ""),
    )


def _x_unit(kind: str, units: dict[str, str]) -> str | None:
    if kind == "angle":
        return units.get("angle_deg") or "deg"
    return (
        units.get("length_m")
        or units.get("length")
        or units.get("period_m")
        or next(iter(units.values()), None)
    )


def _z_unit(units: dict[str, str]) -> str | None:
    return (
        units.get("mean_height")
        or units.get("height_m")
        or units.get("mean_height_m")
    )
