"""Shared measurement result model for ProbeFlow analysis tools.

All new measurement tools should produce a MeasurementResult. The summary
field is a short human-readable string suitable for a table row. Detailed
values are stored in the values dict with units in the units dict.

This module is GUI-free and must remain importable from worker threads.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MeasurementResult:
    """A compact, exportable measurement record produced by any analysis tool."""

    id: str
    kind: str
    source: str
    channel: str
    roi_id: str | None
    summary: str
    values: dict[str, float | int | str | None]
    units: dict[str, str] = field(default_factory=dict)
    context: dict[str, float | int | str | None] = field(default_factory=dict)
    notes: str = ""


def results_to_csv(results: list[MeasurementResult]) -> str:
    """Serialize a list of MeasurementResults to a CSV string.

    One row per result. Value columns are flattened with the key as column
    name; duplicate keys across results get a ``None`` fill in rows that
    lack them.
    """
    if not results:
        return ""
    all_value_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.values:
            if k not in seen:
                all_value_keys.append(k)
                seen.add(k)

    out = io.StringIO()
    writer = csv.writer(out)
    header = ["id", "kind", "source", "channel", "roi_id", "summary", "notes"]
    header += all_value_keys
    header += [f"{k}_unit" for k in all_value_keys if k in {k for r in results for k in r.units}]
    writer.writerow(header)

    unit_keys = [k for k in all_value_keys if any(k in r.units for r in results)]
    for r in results:
        row: list[Any] = [
            r.id, r.kind, r.source, r.channel,
            r.roi_id or "", r.summary, r.notes,
        ]
        row += [r.values.get(k, "") for k in all_value_keys]
        row += [r.units.get(k, "") for k in unit_keys]
        writer.writerow(row)
    return out.getvalue()


def result_to_text(result: MeasurementResult) -> str:
    """Return a plain-text summary of one result for clipboard copy."""
    lines: list[str] = [f"{result.kind}  —  {result.source}"]
    if result.channel:
        lines.append(f"Channel: {result.channel}")
    if result.roi_id:
        lines.append(f"ROI: {result.roi_id}")
    lines.append(result.summary)
    for k, v in result.values.items():
        unit = result.units.get(k, "")
        unit_str = f" {unit}" if unit else ""
        lines.append(f"  {k} = {v}{unit_str}")
    if result.notes:
        lines.append(f"Notes: {result.notes}")
    return "\n".join(lines)
