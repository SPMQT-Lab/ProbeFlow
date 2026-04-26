"""Bridge between GUI processing-state dict and the canonical ProcessingState.

No Qt imports — this module can be tested without a running Qt event loop.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probeflow.scan import Scan

# Keys in the GUI processing state dict that correspond to numeric data
# transforms (as opposed to display-only settings like grain overlays,
# colourmap, or clip percentiles).
NUMERIC_PROC_KEYS: tuple[str, ...] = (
    "remove_bad_lines",
    "align_rows",
    "bg_order",
    "bg_step_tolerance",
    "background_fit_rect",
    "stm_line_bg",
    "facet_level",
    "smooth_sigma",
    "highpass_sigma",
    "edge_method",
    "fft_mode",
    "fft_soft_border",
    "periodic_notches",
    "periodic_notch_radius",
    "patch_interpolate_rect",
    "patch_interpolate_iterations",
    "linear_undistort",
    "set_zero_xy",
    "processing_scope",
    "roi_rect",
)


def processing_state_from_gui(gui_state: dict) -> "ProcessingState":
    """Convert a GUI processing dict into a canonical :class:`ProcessingState`.

    The GUI dict uses keys such as ``"remove_bad_lines"``, ``"bg_order"``, etc.
    Display-only keys (``colormap``, ``clip_low``, ``clip_high``,
    ``grain_threshold``, ``grain_above``) are silently ignored.

    Operation order matches the existing GUI application order.
    """
    from probeflow.processing_state import ProcessingState, ProcessingStep

    steps = []

    roi_scope = gui_state.get("processing_scope") == "roi"
    roi_eligible = {
        "smooth",
        "gaussian_high_pass",
        "edge_detect",
        "fourier_filter",
        "fft_soft_border",
    }
    roi_rect = None
    if roi_scope:
        try:
            roi_rect = tuple(int(v) for v in gui_state.get("roi_rect", ()))
            if len(roi_rect) != 4:
                roi_rect = None
        except (TypeError, ValueError):
            roi_rect = None

    def _append_step(step: ProcessingStep):
        if roi_rect is not None and step.op in roi_eligible:
            steps.append(ProcessingStep("roi", {
                "rect": roi_rect,
                "step": {"op": step.op, "params": dict(step.params)},
            }))
        else:
            steps.append(step)

    if gui_state.get("remove_bad_lines"):
        _append_step(ProcessingStep("remove_bad_lines", {"threshold_mad": 5.0}))

    align = gui_state.get("align_rows")
    if align:
        _append_step(ProcessingStep("align_rows", {"method": str(align)}))

    bg_order = gui_state.get("bg_order")
    if bg_order is not None:
        params = {
            "order": int(bg_order),
            "step_tolerance": bool(gui_state.get("bg_step_tolerance", False)),
        }
        fit_rect = gui_state.get("background_fit_rect")
        if fit_rect is not None:
            try:
                fit_rect_tuple = tuple(int(v) for v in fit_rect)
                if len(fit_rect_tuple) == 4:
                    params["fit_rect"] = fit_rect_tuple
            except (TypeError, ValueError):
                pass
        _append_step(ProcessingStep("plane_bg", params))

    if gui_state.get("stm_line_bg") == "step_tolerant":
        _append_step(ProcessingStep("stm_line_bg", {"mode": "step_tolerant"}))

    if gui_state.get("facet_level"):
        _append_step(ProcessingStep("facet_level", {"threshold_deg": 3.0}))

    smooth_sigma = gui_state.get("smooth_sigma")
    if smooth_sigma:
        _append_step(ProcessingStep("smooth", {"sigma_px": float(smooth_sigma)}))

    highpass_sigma = gui_state.get("highpass_sigma")
    if highpass_sigma:
        _append_step(ProcessingStep("gaussian_high_pass", {
            "sigma_px": float(highpass_sigma),
        }))

    edge_method = gui_state.get("edge_method")
    if edge_method:
        _append_step(ProcessingStep("edge_detect", {
            "method": str(edge_method),
            "sigma":  float(gui_state.get("edge_sigma",  1.0)),
            "sigma2": float(gui_state.get("edge_sigma2", 2.0)),
        }))

    patch_rect = gui_state.get("patch_interpolate_rect")
    if patch_rect is not None:
        try:
            patch_rect_tuple = tuple(int(v) for v in patch_rect)
        except (TypeError, ValueError):
            patch_rect_tuple = ()
        if len(patch_rect_tuple) == 4:
            _append_step(ProcessingStep("patch_interpolate", {
                "rect": patch_rect_tuple,
                "iterations": int(gui_state.get("patch_interpolate_iterations", 200)),
            }))

    fft_mode = gui_state.get("fft_mode")
    if fft_mode is not None:
        _append_step(ProcessingStep("fourier_filter", {
            "mode":   str(fft_mode),
            "cutoff": float(gui_state.get("fft_cutoff", 0.10)),
            "window": str(gui_state.get("fft_window",   "hanning")),
        }))

    if gui_state.get("fft_soft_border"):
        _append_step(ProcessingStep("fft_soft_border", {
            "mode":        str(gui_state.get("fft_soft_mode",        "low_pass")),
            "cutoff":      float(gui_state.get("fft_soft_cutoff",      0.10)),
            "border_frac": float(gui_state.get("fft_soft_border_frac", 0.12)),
        }))

    notches = gui_state.get("periodic_notches")
    if notches:
        peaks = []
        for peak in notches:
            try:
                peaks.append((int(peak[0]), int(peak[1])))
            except (TypeError, ValueError, IndexError):
                continue
        if peaks:
            _append_step(ProcessingStep("periodic_notch_filter", {
                "peaks": peaks,
                "radius_px": float(gui_state.get("periodic_notch_radius", 3.0)),
            }))

    if gui_state.get("linear_undistort"):
        shear_x = float(gui_state.get("undistort_shear_x", 0.0))
        scale_y = float(gui_state.get("undistort_scale_y", 1.0))
        if shear_x != 0.0 or scale_y != 1.0:
            _append_step(ProcessingStep("linear_undistort", {
                "shear_x": shear_x,
                "scale_y": scale_y,
            }))

    set_zero = gui_state.get("set_zero_xy")
    if set_zero is not None:
        try:
            x_px, y_px = int(set_zero[0]), int(set_zero[1])
            _append_step(ProcessingStep("set_zero_point", {
                "x_px":  x_px,
                "y_px":  y_px,
                "patch": int(gui_state.get("set_zero_patch", 1)),
            }))
        except (TypeError, ValueError, IndexError):
            pass

    return ProcessingState(steps=steps)


def apply_processing_state_to_scan(
    scan: "Scan",
    proc_state: dict,
    *,
    plane_idx: int = 0,
) -> "Scan":
    """Apply GUI processing state to a Scan before export.

    Converts *proc_state* to a canonical :class:`ProcessingState`, applies it
    via :func:`~probeflow.processing_state.apply_processing_state`, and
    records each step in ``scan.processing_history``.

    Updates ``scan.planes[plane_idx]`` in place and returns *scan*.
    Display-only settings (grain overlay, colormap, clip percentiles) are ignored.
    """
    from probeflow.processing_state import apply_processing_state

    if plane_idx < 0 or plane_idx >= len(scan.planes):
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{len(scan.planes)} plane(s)"
        )

    state    = processing_state_from_gui(proc_state)
    now      = datetime.now().isoformat()
    a        = apply_processing_state(scan.planes[plane_idx], state)

    scan.planes[plane_idx] = a

    for step in state.steps:
        scan.processing_history.append({
            "op":        step.op,
            "params":    dict(step.params),
            "timestamp": now,
        })

    return scan
