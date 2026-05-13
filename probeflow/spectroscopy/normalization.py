"""Pure normalization helpers for spectroscopy display traces."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from probeflow.spectroscopy.models import SpectrumDisplayOptions

NORMALIZATION_LABELS = [
    "Off",
    "Setpoint",
    "Constant",
    "Max abs",
    "Channel",
]
NORMALIZATION_LABEL_TO_MODE = {
    "Off": "none",
    "Setpoint": "setpoint",
    "Constant": "constant",
    "Max abs": "max_abs",
    "Channel": "channel",
}


def normalize_mode(label: str) -> str:
    """Return the display-pipeline mode key for a GUI normalization label."""
    return NORMALIZATION_LABEL_TO_MODE.get(label, "none")


def normalization_formula_text(
    *,
    derivative: bool,
    mode_label: str,
    constant: float,
    channel: str,
    offset: float,
) -> str:
    """Return a compact visible formula for the current display normalization."""
    base = "dy/dx" if derivative else "y"
    mode = normalize_mode(mode_label)
    if mode == "none":
        expr = base
    elif mode == "setpoint":
        expr = f"{base} / setpoint"
    elif mode == "constant":
        expr = f"{base} / {constant:.6g}"
    elif mode == "max_abs":
        expr = f"{base} / max(abs({base}))"
    elif mode == "channel":
        expr = f"{base} / {channel or 'channel'}"
    else:
        expr = f"{base} / {mode}"
    if offset != 0.0:
        expr = f"{expr} + {offset:.6g}"
    return expr


def normalization_description(opts: SpectrumDisplayOptions) -> str:
    """Return label text describing an applied normalization mode."""
    mode = (opts.normalize_mode or "none").strip().lower()
    if mode == "setpoint":
        return "divided by setpoint"
    if mode == "constant":
        return f"divided by constant {opts.normalize_constant:g}"
    if mode == "channel":
        return f"divided by channel {opts.normalize_channel}"
    if mode in {"max_abs", "maxabs", "max"}:
        return "divided by max |y|"
    return f"normalized by {mode}"


def apply_normalization(
    y: np.ndarray,
    *,
    mode: str = "none",
    constant: float | None = None,
    setpoint: float | None = None,
    channel: str | None = None,
    channel_lookup: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Return a normalized copy of ``y``."""
    arr = np.asarray(y, dtype=np.float64).copy()
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off"}:
        return arr

    finite = arr[np.isfinite(arr)]

    if mode == "setpoint":
        try:
            setpoint_value = float(setpoint) if setpoint is not None else np.nan
        except (TypeError, ValueError):
            setpoint_value = np.nan
        if not np.isfinite(setpoint_value) or setpoint_value == 0.0:
            raise ValueError("setpoint normalization requires a finite non-zero setpoint")
        return arr / setpoint_value

    if mode == "constant":
        if constant is None or not np.isfinite(constant) or float(constant) == 0.0:
            raise ValueError("constant normalization requires a finite non-zero constant")
        return arr / float(constant)

    if mode in {"max_abs", "maxabs", "max"}:
        if finite.size == 0:
            raise ValueError("max-absolute normalization requires at least one finite value")
        scale = float(np.nanmax(np.abs(finite)))
        if scale == 0.0:
            raise ValueError("max-absolute normalization requires a non-zero maximum")
        return arr / scale

    if mode == "channel":
        if not channel:
            raise ValueError("channel normalization requires a denominator channel")
        if channel_lookup is None or channel not in channel_lookup:
            raise ValueError(f"normalization channel not available: {channel}")
        denom = np.asarray(channel_lookup[channel], dtype=np.float64)
        if denom.shape != arr.shape:
            raise ValueError("normalization channel must match the selected trace length")
        if np.any(~np.isfinite(denom)) or np.any(denom == 0):
            raise ValueError("normalization channel contains invalid or zero values")
        return arr / denom

    raise ValueError(f"Unknown normalization mode: {mode!r}")
