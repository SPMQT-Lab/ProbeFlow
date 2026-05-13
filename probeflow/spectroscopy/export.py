"""Export helpers for displayed spectroscopy data."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from probeflow.spectroscopy.models import DisplayedSpectrum


def displayed_spectra_to_csv_text(spectra: Iterable[DisplayedSpectrum]) -> str:
    """Return long/tidy CSV text for the displayed data."""
    traces = list(spectra)
    if not traces:
        return ""

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["# export_type", "probeflow_displayed_spectra"])
    writer.writerow(["# trace_count", str(len(traces))])
    for idx, spec in enumerate(traces):
        writer.writerow([f"# trace_{idx}_source_file", spec.source_file])
        writer.writerow([f"# trace_{idx}_spectrum_id", spec.spectrum_id])
        writer.writerow([f"# trace_{idx}_x_channel", spec.x_channel])
        writer.writerow([f"# trace_{idx}_y_channel", spec.y_channel])
        writer.writerow([f"# trace_{idx}_display_y_label", spec.y_label])
        writer.writerow([f"# trace_{idx}_display_y_unit", spec.y_unit])
        writer.writerow([f"# trace_{idx}_options", json.dumps(asdict(spec.options), sort_keys=True)])
        writer.writerow([f"# trace_{idx}_excluded_indices", json.dumps(spec.excluded_indices)])
    writer.writerow([
        "source_file",
        "spectrum_id",
        "trace_label",
        "x_channel",
        "x_value",
        "x_unit",
        "y_channel",
        "y_display",
        "y_unit",
    ])
    for spec in traces:
        for x_val, y_val in zip(spec.x_display, spec.y_display):
            writer.writerow([
                spec.source_file,
                spec.spectrum_id,
                spec.label,
                spec.x_channel,
                f"{float(x_val):.10g}",
                spec.x_unit,
                spec.y_channel,
                f"{float(y_val):.10g}",
                spec.y_unit,
            ])
    return out.getvalue()


def displayed_spectra_to_json_text(spectra: Iterable[DisplayedSpectrum]) -> str:
    """Return JSON text containing metadata and displayed arrays."""
    traces = []
    for spec in spectra:
        traces.append({
            "source_file": spec.source_file,
            "spectrum_id": spec.spectrum_id,
            "label": spec.label,
            "x_channel": spec.x_channel,
            "y_channel": spec.y_channel,
            "x_label": spec.x_label,
            "y_label": spec.y_label,
            "x_unit": spec.x_unit,
            "y_unit": spec.y_unit,
            "display_options": asdict(spec.options),
            "metadata": spec.metadata,
            "excluded_indices": spec.excluded_indices,
            "x": [float(v) for v in spec.x_display],
            "y": [float(v) for v in spec.y_display],
        })
    return json.dumps(
        {"export_type": "probeflow_displayed_spectra", "traces": traces},
        indent=2,
        sort_keys=True,
    )


def displayed_spectra_to_txt_text(spectra: Iterable[DisplayedSpectrum]) -> str:
    """Return a simple human-readable text export."""
    traces = list(spectra)
    if not traces:
        return ""
    lines = ["# export_type: probeflow_displayed_spectra", f"# trace_count: {len(traces)}"]
    for spec in traces:
        lines.append("")
        lines.append(f"# source_file: {spec.source_file}")
        lines.append(f"# spectrum_id: {spec.spectrum_id}")
        lines.append(f"# x_channel: {spec.x_channel}")
        lines.append(f"# y_channel: {spec.y_channel}")
        lines.append(f"# display_y_label: {spec.y_label}")
        lines.append(f"# display_y_unit: {spec.y_unit}")
        lines.append(f"# options: {json.dumps(asdict(spec.options), sort_keys=True)}")
        lines.append(f"# excluded_indices: {json.dumps(spec.excluded_indices)}")
        lines.append(f"{spec.x_label}\t{spec.y_label}")
        for x_val, y_val in zip(spec.x_display, spec.y_display):
            lines.append(f"{float(x_val):.10g}\t{float(y_val):.10g}")
    return "\n".join(lines) + "\n"


def displayed_spectra_to_clipboard_text(spectra: Iterable[DisplayedSpectrum]) -> str:
    """Return clipboard text for displayed spectra."""
    return displayed_spectra_to_csv_text(spectra)


def export_displayed_spectra_csv(spectra: Iterable[DisplayedSpectrum], path: str | Path) -> None:
    Path(path).write_text(displayed_spectra_to_csv_text(spectra), encoding="utf-8")


def export_displayed_spectra_json(spectra: Iterable[DisplayedSpectrum], path: str | Path) -> None:
    Path(path).write_text(displayed_spectra_to_json_text(spectra), encoding="utf-8")


def export_displayed_spectra_txt(spectra: Iterable[DisplayedSpectrum], path: str | Path) -> None:
    Path(path).write_text(displayed_spectra_to_txt_text(spectra), encoding="utf-8")
