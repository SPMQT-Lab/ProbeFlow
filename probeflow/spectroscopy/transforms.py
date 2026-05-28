"""Pure display transforms for spectroscopy traces."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from probeflow.spectroscopy._kernels import (
    numeric_derivative as _numeric_derivative,
)
from probeflow.spectroscopy.models import (
    DisplayedSpectrum,
    SpectrumDisplayOptions,
    SpectrumTrace,
)
from probeflow.spectroscopy.normalization import (
    apply_normalization,
    normalization_description,
)
from probeflow.spectroscopy.outliers import apply_outlier_mask
from probeflow.spectroscopy.smoothing import apply_smoothing


def numerical_derivative(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return copied x and dy/dx for strictly monotonic x."""
    x_arr = np.asarray(x, dtype=np.float64).copy()
    y_arr = _numeric_derivative(x_arr, np.asarray(y, dtype=np.float64))
    return x_arr, y_arr


def apply_vertical_offset(y: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Return a copy of ``y`` shifted by ``offset``."""
    return np.asarray(y, dtype=np.float64).copy() + float(offset)


def make_displayed_spectrum(
    trace: SpectrumTrace,
    options: SpectrumDisplayOptions | None = None,
    *,
    channel_lookup: Mapping[str, np.ndarray] | None = None,
) -> DisplayedSpectrum:
    """Build a non-mutating displayed view.

    Operation order is:
    raw x/y copy -> smoothing -> derivative -> normalization -> outlier mask
    -> vertical offset -> displayed/exported trace.
    """
    opts = options or SpectrumDisplayOptions()
    x = np.asarray(trace.x_raw, dtype=np.float64).copy()
    y = np.asarray(trace.y_raw, dtype=np.float64).copy()
    y_unit = trace.y_unit
    y_label = trace.y_label

    y = apply_smoothing(
        y,
        mode=opts.smoothing_mode,
        points=opts.smoothing_points,
        polyorder=opts.savgol_polyorder,
    )

    if opts.derivative:
        if (opts.normalize_mode or "none").strip().lower() == "channel":
            raise ValueError("channel normalization cannot be combined with numerical derivative")
        x, y = numerical_derivative(x, y)
        y_unit = f"{y_unit}/{trace.x_unit}".rstrip("/") if y_unit else ""
        if trace.x_unit == "V" and _looks_like_current(trace.y_channel, y_label):
            y_label = "numerical dI/dV"
        else:
            y_label = f"numerical d({y_label})/d{trace.x_unit or 'x'}"

    y = apply_normalization(
        y,
        mode=opts.normalize_mode,
        constant=opts.normalize_constant,
        setpoint=_metadata_float(trace.metadata, "setpoint_a", "setpoint"),
        channel=opts.normalize_channel,
        channel_lookup=channel_lookup,
    )
    if (opts.normalize_mode or "none").lower() not in {"none", "off"}:
        y_label = f"{y_label} ({normalization_description(opts)})"
        y_unit = "relative"

    x_display, y_display, keep = apply_outlier_mask(
        x,
        y,
        mode=opts.outlier_mode,
        threshold=opts.outlier_threshold,
    )
    y_display = apply_vertical_offset(y_display, opts.vertical_offset)

    raw_points = int(np.asarray(trace.y_raw).size)
    metadata = dict(trace.metadata)
    metadata.update({
        "raw_points": raw_points,
        "displayed_points": int(y_display.size),
        "excluded_indices": np.flatnonzero(~keep).astype(int).tolist(),
        "display_pipeline": _display_pipeline(opts),
    })

    return DisplayedSpectrum(
        source_file=trace.source_file,
        spectrum_id=trace.spectrum_id,
        label=f"{trace.spectrum_id} {y_label}".strip(),
        x_channel=trace.x_channel,
        y_channel=trace.y_channel,
        x_display=x_display,
        y_display=y_display,
        mask=keep,
        options=replace(opts),
        metadata=metadata,
        x_label=trace.x_label,
        y_label=y_label,
        x_unit=trace.x_unit,
        y_unit=y_unit,
    )


def _display_pipeline(opts: SpectrumDisplayOptions) -> list[str]:
    return [
        "copy raw x/y arrays",
        f"smoothing={opts.smoothing_mode}",
        f"derivative={'on' if opts.derivative else 'off'}",
        f"normalization={opts.normalize_mode}",
        f"outlier_mask={opts.outlier_mode}",
        f"vertical_offset={opts.vertical_offset:g}",
    ]


def _metadata_float(metadata: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _looks_like_current(channel: str, label: str) -> bool:
    text = f"{channel} {label}".lower()
    return "current" in text or channel in {"I", "i"}
