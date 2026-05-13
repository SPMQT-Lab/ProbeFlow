"""Pure display transforms for spectroscopy traces."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from probeflow.processing.spectroscopy import (
    numeric_derivative as _numeric_derivative,
    smooth_spectrum,
)
from probeflow.spectroscopy.models import (
    DisplayedSpectrum,
    SpectrumDisplayOptions,
    SpectrumTrace,
)


def apply_smoothing(
    y: np.ndarray,
    *,
    mode: str = "none",
    points: int | None = None,
    polyorder: int = 2,
) -> np.ndarray:
    """Return a smoothed copy of ``y``."""
    arr = np.asarray(y, dtype=np.float64).copy()
    mode = (mode or "none").strip().lower()
    if mode in {"none", "off"}:
        return arr

    n = arr.size
    pts = int(points if points is not None else 7)
    if pts <= 0:
        raise ValueError("smoothing points must be positive")
    if pts > n:
        raise ValueError("smoothing points must not exceed the number of data points")

    if mode in {"gaussian", "gauss"}:
        sigma = max(pts / 2.355, 0.1)
        return smooth_spectrum(arr, method="gaussian", sigma=sigma)

    if mode in {"savgol", "savitzky-golay", "savitzky_golay"}:
        if pts % 2 == 0:
            raise ValueError("Savitzky-Golay window length must be odd")
        if pts <= polyorder:
            raise ValueError("Savitzky-Golay window length must exceed polynomial order")
        return smooth_spectrum(
            arr,
            method="savgol",
            window_length=pts,
            polyorder=int(polyorder),
        )

    raise ValueError(f"Unknown smoothing mode: {mode!r}")


def numerical_derivative(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return copied x and dy/dx for strictly monotonic x."""
    x_arr = np.asarray(x, dtype=np.float64).copy()
    y_arr = _numeric_derivative(x_arr, np.asarray(y, dtype=np.float64))
    return x_arr, y_arr


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


def apply_outlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "none",
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y with outliers omitted plus a keep-mask over the input arrays."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have matching shapes")

    mode = (mode or "none").strip().lower()
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mode in {"none", "off"}:
        return x_arr[keep].copy(), y_arr[keep].copy(), keep

    limit = float(threshold if threshold is not None else 6.0)
    if limit <= 0:
        raise ValueError("outlier threshold must be positive")

    if mode == "mad":
        scores = _robust_scores(y_arr)
        keep &= scores <= limit
    elif mode in {"jump", "derivative_jump", "derivative-jump"}:
        jumps = np.diff(y_arr)
        bad = np.zeros(y_arr.shape, dtype=bool)
        abs_jumps = np.abs(jumps)
        finite_jumps = abs_jumps[np.isfinite(abs_jumps)]
        typical = float(np.nanpercentile(finite_jumps, 25)) if finite_jumps.size else 0.0
        jump_limit = limit * typical if typical > 0 else 0.0
        for i in range(1, y_arr.size - 1):
            left = y_arr[i] - y_arr[i - 1]
            right = y_arr[i + 1] - y_arr[i]
            if (
                np.isfinite(left)
                and np.isfinite(right)
                and np.sign(left) != np.sign(right)
                and min(abs(left), abs(right)) > jump_limit
            ):
                bad[i] = True
        if not np.any(bad):
            jump_scores = _robust_scores(jumps)
            bad_jumps = jump_scores > limit
            bad[:-1] |= bad_jumps
            bad[1:] |= bad_jumps
        keep &= ~bad
    else:
        raise ValueError(f"Unknown outlier mode: {mode!r}")

    return x_arr[keep].copy(), y_arr[keep].copy(), keep


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
            y_label = "dI/dV"
        else:
            y_label = f"d({y_label})/d{trace.x_unit or 'x'}"

    y = apply_normalization(
        y,
        mode=opts.normalize_mode,
        constant=opts.normalize_constant,
        setpoint=_metadata_float(trace.metadata, "setpoint_a", "setpoint"),
        channel=opts.normalize_channel,
        channel_lookup=channel_lookup,
    )
    if (opts.normalize_mode or "none").lower() not in {"none", "off"}:
        y_label = f"{y_label} ({_normalization_description(opts)})"
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


def _robust_scores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    scores = np.zeros(arr.shape, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        scores[:] = np.inf
        return scores

    med = float(np.nanmedian(arr[finite]))
    dev = np.abs(arr - med)
    mad = float(np.nanmedian(dev[finite]))
    if mad > 0:
        scores = dev / mad
        scores[~finite] = np.inf
        return scores

    scores[:] = 0.0
    scores[(dev > 0) | ~finite] = np.inf
    return scores


def _normalization_description(opts: SpectrumDisplayOptions) -> str:
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
