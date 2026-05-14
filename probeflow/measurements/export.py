"""Export helpers for measurement results and detected feature points."""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from probeflow.measurements.models import FeaturePoint, MeasurementResult, Scalar


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("probeflow")
    except Exception:
        return None


def measurement_to_flat_dict(result: MeasurementResult) -> dict[str, Scalar]:
    """Flatten a measurement into table-friendly scalar columns."""
    row: dict[str, Scalar] = {
        "measurement_id": result.measurement_id,
        "kind": result.kind,
        "source_label": result.source_label,
        "source_path": result.source_path,
        "channel": result.channel,
        "x_unit": result.x_unit,
        "y_unit": result.y_unit,
        "z_unit": result.z_unit,
        "notes": result.notes,
    }
    for key in sorted(result.values):
        row[f"value.{key}"] = _clean_scalar(result.values[key])
    for key in sorted(result.context):
        row[f"context.{key}"] = _clean_scalar(result.context[key])
    return row


def measurements_to_tsv(results: Iterable[MeasurementResult]) -> str:
    """Return tab-separated measurement text suitable for spreadsheets."""
    return _measurements_to_delimited_text(results, delimiter="\t")


def measurements_to_csv_text(results: Iterable[MeasurementResult]) -> str:
    """Return CSV text for measurement results."""
    return _measurements_to_delimited_text(results, delimiter=",")


def measurements_to_csv(results: Iterable[MeasurementResult], path: str | Path) -> None:
    """Write measurement results as CSV."""
    Path(path).write_text(measurements_to_csv_text(results), encoding="utf-8")


def measurements_to_json_text(results: Iterable[MeasurementResult]) -> str:
    """Return nested JSON text for measurement results."""
    payload = {
        "export_type": "probeflow_measurements",
        "schema_version": "1",
        "created_at": _utc_now(),
        "probeflow_version": _get_version(),
        "measurements": [_measurement_to_json_dict(result) for result in results],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def measurements_to_json(results: Iterable[MeasurementResult], path: str | Path) -> None:
    """Write measurement results as JSON."""
    Path(path).write_text(measurements_to_json_text(results), encoding="utf-8")


def feature_points_to_csv_text(
    points: Iterable[FeaturePoint],
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    """Return CSV text for detected feature points."""
    rows = list(points)
    if not rows:
        return ""
    metadata = metadata or {}
    x_unit = str(metadata.get("x_unit") or "")
    y_unit = str(metadata.get("y_unit") or "")
    z_unit = str(metadata.get("z_unit") or "")
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "point_id",
        "x_px",
        "y_px",
        "x_phys",
        "y_phys",
        "z_value",
        "x_unit",
        "y_unit",
        "z_unit",
        "channel",
        "source_label",
        "roi_id",
    ])
    for point in rows:
        writer.writerow([
            point.point_id,
            f"{float(point.x_px):.10g}",
            f"{float(point.y_px):.10g}",
            f"{float(point.x_phys):.10g}",
            f"{float(point.y_phys):.10g}",
            f"{float(point.z_value):.10g}",
            x_unit,
            y_unit,
            z_unit,
            point.channel,
            point.source_label,
            point.roi_id or "",
        ])
    return out.getvalue()


def feature_points_to_json_text(
    points: Iterable[FeaturePoint],
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    """Return JSON text preserving point metadata and coordinates."""
    payload = {
        "export_type": "probeflow_feature_points",
        "metadata": metadata or {},
        "points": [asdict(point) for point in points],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _measurements_to_delimited_text(
    results: Iterable[MeasurementResult],
    *,
    delimiter: str,
) -> str:
    rows = [measurement_to_flat_dict(result) for result in results]
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return out.getvalue()


def _clean_scalar(value: object) -> Scalar:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else str(number)
    return str(value)


def _measurement_to_json_dict(result: MeasurementResult) -> dict[str, object]:
    row = asdict(result)
    row["values"] = {
        key: _clean_scalar(value)
        for key, value in result.values.items()
    }
    row["context"] = {
        key: _clean_scalar(value)
        for key, value in result.context.items()
    }
    return row
