"""Serialise basic point-pattern statistics to CSV / JSON.

The Point-statistics tool computes descriptive quantities for a set of marked
points (count, area, density, nearest-neighbour distances, the pair-correlation
curve). These helpers turn those into files the user can open in a spreadsheet
or load elsewhere — CSV for the scalar summary, JSON for the scalars plus the
g(r) / nearest-neighbour curves.

Dependencies are numpy + stdlib only.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import numpy as np


def _fmt_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def point_stats_csv_text(scalars: list[tuple[str, Any, str]]) -> str:
    """CSV of scalar quantities as ``quantity,value,unit`` rows.

    ``scalars`` is an ordered list of ``(label, value, unit)`` triples; ``None``
    values render as an empty cell so a missing (e.g. uncalibrated) quantity is
    still represented.
    """
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["quantity", "value", "unit"])
    for label, value, unit in scalars:
        writer.writerow([str(label), _fmt_csv(value), str(unit)])
    return out.getvalue()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def point_stats_json_text(
    scalars: list[tuple[str, Any, str]],
    curves: dict[str, dict[str, Any]] | None = None,
) -> str:
    """JSON with the scalar summary plus optional named curves.

    ``curves`` maps a curve name to a dict of columns; numpy arrays are
    converted to lists so the object is JSON-serialisable.
    """
    obj: dict[str, Any] = {
        "statistics": [
            {
                "quantity": str(label),
                "value": _jsonable(value),
                "unit": str(unit),
            }
            for label, value, unit in scalars
        ]
    }
    if curves:
        obj["curves"] = {
            name: {
                key: _jsonable(val)
                for key, val in columns.items()
            }
            for name, columns in curves.items()
        }
    return json.dumps(obj, indent=2)
