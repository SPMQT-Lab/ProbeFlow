"""ROI-triggered analysis actions extracted from ImageViewerDialog."""

from __future__ import annotations

from typing import Callable

import numpy as np


def show_roi_fft(roi, display_arr: np.ndarray, parent=None) -> None:
    """Compute FFT magnitude inside *roi* and report the output shape."""
    from PySide6.QtWidgets import QMessageBox
    try:
        from probeflow.processing.image import fft_magnitude
        mag, _qx, _qy = fft_magnitude(display_arr, roi=roi)
        QMessageBox.information(
            parent, "FFT",
            f"FFT computed for ROI '{roi.name}'.\n"
            f"Output shape: {mag.shape[0]} × {mag.shape[1]}",
        )
    except Exception as exc:
        QMessageBox.warning(parent, "FFT error", str(exc))


def show_roi_histogram(
    roi,
    display_arr: np.ndarray,
    channel_unit_fn: Callable[[], tuple[float, str, str]],
    parent=None,
) -> None:
    """Show a basic stats summary for pixels inside *roi*."""
    from PySide6.QtWidgets import QMessageBox

    mask = roi.to_mask(display_arr.shape[:2])
    vals = display_arr[mask]
    if len(vals) == 0:
        QMessageBox.information(parent, "Histogram", "No pixels in ROI.")
        return

    scale, unit, _ = channel_unit_fn()
    unit_str = f" {unit}" if unit else ""
    QMessageBox.information(
        parent, f"Histogram: {roi.name}",
        f"Pixels: {len(vals)}\n"
        f"Min:  {float(vals.min()) * scale:.4g}{unit_str}\n"
        f"Max:  {float(vals.max()) * scale:.4g}{unit_str}\n"
        f"Mean: {float(vals.mean()) * scale:.4g}{unit_str}",
    )


def plot_roi_line_profile(
    roi,
    display_arr: np.ndarray,
    pixel_size_xy_m: tuple[float, float],
    channel_unit_fn: Callable[[], tuple[float, str, str]],
    line_profile_panel,
    theme: dict,
) -> dict | None:
    """Compute the line profile for *roi*, render it, and return its live metrics.

    The returned dict (``length``, ``x_unit``, ``length_px``, ``height_diff``,
    ``z_unit``) lets callers show a live length/height readout. Returns ``None`` if
    the profile could not be computed.
    """
    import math

    from probeflow.analysis.spec_plot import choose_display_unit

    try:
        px_x, px_y = pixel_size_xy_m
        from probeflow.processing.image import line_profile
        width_px = float(roi.geometry.get("width", 1)) if roi.geometry else 1.0
        s_m, values = line_profile(
            display_arr, roi=roi,
            pixel_size_x_m=px_x, pixel_size_y_m=px_y,
            width_px=max(1.0, width_px),
        )
        scale, unit, name = channel_unit_fn()
        x_scale, x_unit = choose_display_unit("m", s_m)
        y_vals = values * scale
        line_profile_panel.setVisible(True)
        line_profile_panel.plot_profile(
            s_m * x_scale,
            y_vals,
            x_label=f"Distance [{x_unit}]",
            y_label=f"{name} [{unit}]" if unit else name,
            theme=theme,
        )
        if hasattr(line_profile_panel, "set_source_label"):
            line_profile_panel.set_source_label(
                f"Line ROI: {roi.name} ({roi.id[:8]})",
                theme=theme,
            )
        # Live metrics — match the saved line_profile_measurement semantics:
        # length = arc-length span, height_diff = peak-to-valley over the profile.
        geom = roi.geometry or {}
        length = float(s_m[-1] - s_m[0]) * x_scale if s_m.size else 0.0
        length_px = math.hypot(
            float(geom.get("x2", 0.0)) - float(geom.get("x1", 0.0)),
            float(geom.get("y2", 0.0)) - float(geom.get("y1", 0.0)),
        )
        finite = y_vals[np.isfinite(y_vals)]
        height_diff = (
            float(np.max(finite) - np.min(finite)) if finite.size else None
        )
        return {
            "length": length,
            "x_unit": x_unit,
            "length_px": length_px,
            "height_diff": height_diff,
            "z_unit": unit,
        }
    except Exception as exc:
        line_profile_panel.show_empty(str(exc), theme=theme)
        return None
