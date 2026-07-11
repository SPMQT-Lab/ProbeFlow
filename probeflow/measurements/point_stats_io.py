"""Serialise basic point-pattern statistics to CSV / JSON.

The Point-statistics tool computes descriptive quantities for a set of marked
points (count, area, density, nearest-neighbour distances, the pair-correlation
curve). These helpers turn those into files the user can open in a spreadsheet
or load elsewhere — CSV for the scalar summary, JSON for the scalars plus the
g(r) / nearest-neighbour curves.

Dependencies are numpy + stdlib only.
"""

from __future__ import annotations

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
    lines = ["quantity,value,unit"]
    for label, value, unit in scalars:
        safe = str(label).replace(",", ";")
        lines.append(f"{safe},{_fmt_csv(value)},{unit}")
    return "\n".join(lines) + "\n"


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
            {"quantity": label, "value": value, "unit": unit}
            for label, value, unit in scalars
        ]
    }
    if curves:
        obj["curves"] = {
            name: {
                key: (val.tolist() if isinstance(val, np.ndarray) else val)
                for key, val in columns.items()
            }
            for name, columns in curves.items()
        }
    return json.dumps(obj, indent=2)
