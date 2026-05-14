"""MeasurementResult adapters for spectroscopy measurements."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from probeflow.measurements.models import MeasurementResult, Scalar
from probeflow.spectroscopy.measurement import SpectrumDeltaMeasurement
from probeflow.spectroscopy.models import DisplayedSpectrum


def spectrum_delta_to_result(
    measurement: SpectrumDeltaMeasurement,
    *,
    measurement_id: str,
    trace: DisplayedSpectrum | None = None,
    notes: str = "",
) -> MeasurementResult:
    """Convert a displayed-trace delta measurement to a generic result."""
    p1 = measurement.point1
    source_label = p1.trace_name
    source_path = p1.source_file
    context: dict[str, Scalar] = {
        "data_basis": "displayed_trace",
        "point1_index": p1.index,
        "point2_index": measurement.point2.index,
    }
    if trace is not None:
        source_label = f"{Path(trace.source_file).name}:{trace.y_channel}"
        source_path = trace.source_file
        context.update(_display_context(trace))

    return MeasurementResult(
        measurement_id=measurement_id,
        kind="spectrum_delta",
        source_label=source_label,
        source_path=source_path,
        channel=p1.y_channel,
        x_unit=p1.x_unit,
        y_unit=p1.y_unit,
        values={
            "x1": p1.x,
            "y1": p1.y,
            "x2": measurement.point2.x,
            "y2": measurement.point2.y,
            "dx": measurement.dx,
            "dy": measurement.dy,
            "slope": measurement.slope,
        },
        context=context,
        notes=notes,
    )


def _display_context(trace: DisplayedSpectrum) -> dict[str, Scalar]:
    options = asdict(trace.options)
    context: dict[str, Scalar] = {
        "display_label": trace.label,
        "spectrum_id": trace.spectrum_id,
        "source_file": trace.source_file,
        "x_channel": trace.x_channel,
        "y_channel": trace.y_channel,
        "smoothing": str(options.get("smoothing_mode", "none")),
        "smoothing_window": options.get("smoothing_points"),
        "savgol_polyorder": options.get("savgol_polyorder"),
        "derivative": "on" if options.get("derivative") else "off",
        "normalization": str(options.get("normalize_mode", "none")),
        "normalization_channel": options.get("normalize_channel"),
        "outlier_mask": str(options.get("outlier_mode", "none")),
        "outlier_threshold": options.get("outlier_threshold"),
        "offset": options.get("vertical_offset"),
    }
    return context
