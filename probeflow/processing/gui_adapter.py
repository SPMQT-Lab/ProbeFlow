"""Bridge GUI processing state into canonical processing operations.

No Qt imports — this module can be tested without a running Qt event loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from probeflow.core.scan_model import Scan
    from probeflow.processing.state import ProcessingState

# Keys in the GUI processing state dict that correspond to numeric data
# transforms (as opposed to display-only settings like grain overlays,
# colourmap, or clip percentiles).
NUMERIC_PROC_KEYS: tuple[str, ...] = (
    "remove_bad_lines",
    "remove_bad_lines_threshold",
    "remove_bad_lines_polarity",
    "remove_bad_lines_min_segment_length_px",
    "remove_bad_lines_max_adjacent_bad_lines",
    "align_rows",
    "plane_bg",
    "stm_background",
    "smooth_sigma",
    "highpass_sigma",
    "edge_method",
    "fft_mode",
    "fft_soft_border",
    "periodic_notches",
    "periodic_notch_radius",
    "linear_undistort",
    "set_zero_xy",
    "set_zero_plane_points",
    "processing_scope",
    "processing_roi_id",
    "roi_id",
    "geometric_ops",
)


def processing_state_from_gui(gui_state: dict) -> "ProcessingState":
    """Convert a GUI processing dict into a canonical :class:`ProcessingState`.

    The GUI dict uses keys such as ``"remove_bad_lines"``, ``"align_rows"``, etc.
    Display-only keys (``colormap``, ``clip_low``, ``clip_high``,
    ``grain_threshold``, ``grain_above``) are silently ignored.

    Operation order matches the existing GUI application order.
    """
    from probeflow.processing.state import ProcessingState, ProcessingStep

    steps = []

    roi_scope = gui_state.get("processing_scope") == "roi"
    roi_eligible = {
        "smooth",
        "gaussian_high_pass",
        "edge_detect",
        "fourier_filter",
        "fft_soft_border",
    }
    roi_id = None
    if roi_scope:
        roi_id = gui_state.get("processing_roi_id") or gui_state.get("roi_id")
        if roi_id is not None:
            roi_id = str(roi_id)

    skipped_roi_scope_warning = False

    def _append_step(step: ProcessingStep):
        nonlocal skipped_roi_scope_warning
        if step.op in roi_eligible and roi_scope:
            params = {"step": {"op": step.op, "params": dict(step.params)}}
            if roi_id is not None:
                params["roi_id"] = roi_id
            else:
                if not skipped_roi_scope_warning:
                    warnings.warn(
                        "ROI-scoped processing was requested but no supported area "
                        "ROI was provided; ROI-aware local filter step(s) skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                    skipped_roi_scope_warning = True
                return
            steps.append(ProcessingStep("roi", params))
        else:
            steps.append(step)

    bad_lines_method = gui_state.get("remove_bad_lines")
    if bad_lines_method:
        # Legacy boolean True maps to the original MAD method.
        if bad_lines_method is True or bad_lines_method == "True":
            bad_lines_method = "mad"
        threshold = gui_state.get(
            "remove_bad_lines_threshold",
            gui_state.get("threshold_mad", 5.0),
        )
        _append_step(ProcessingStep("remove_bad_lines", {
            "threshold_mad": float(threshold),
            "method": str(bad_lines_method),
            "polarity": str(gui_state.get("remove_bad_lines_polarity", "bright")),
            "min_segment_length_px": int(
                gui_state.get("remove_bad_lines_min_segment_length_px", 2)
            ),
            "max_adjacent_bad_lines": int(
                gui_state.get("remove_bad_lines_max_adjacent_bad_lines", 1)
            ),
        }))

    align = gui_state.get("align_rows")
    if align:
        _append_step(ProcessingStep("align_rows", {"method": str(align)}))

    plane_bg = gui_state.get("plane_bg")
    if isinstance(plane_bg, dict):
        _append_step(ProcessingStep("plane_bg", {
            "order": int(plane_bg.get("order", 1)),
        }))

    stm_bg = gui_state.get("stm_background")
    if isinstance(stm_bg, dict):
        params = {
            "fit_region": str(stm_bg.get("fit_region", "whole_image")),
            "line_statistic": str(stm_bg.get("line_statistic", "median")),
            "model": str(stm_bg.get("model", "linear")),
            "linear_x_first": bool(stm_bg.get("linear_x_first", False)),
            "preserve_level": str(stm_bg.get("preserve_level", "median")),
        }
        if stm_bg.get("blur_length") is not None:
            params["blur_length"] = float(stm_bg["blur_length"])
        if stm_bg.get("jump_threshold") is not None:
            params["jump_threshold"] = float(stm_bg["jump_threshold"])
        if stm_bg.get("fit_roi_id") is not None:
            params["fit_roi_id"] = str(stm_bg["fit_roi_id"])
            params["applied_to"] = "whole_image"
        _append_step(ProcessingStep("stm_background", params))

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

    zero_plane = gui_state.get("set_zero_plane_points")
    if zero_plane is not None:
        points = []
        for point in zero_plane:
            try:
                points.append((int(point[0]), int(point[1])))
            except (TypeError, ValueError, IndexError):
                continue
        if len(points) >= 3:
            _append_step(ProcessingStep("set_zero_plane", {
                "points_px": points[:3],
                "patch": int(gui_state.get("set_zero_patch", 1)),
            }))

    for op_spec in gui_state.get("geometric_ops") or []:
        try:
            if isinstance(op_spec, str):
                op_name = op_spec
                op_params: dict = {}
            else:
                op_name = str(op_spec["op"])
                op_params = dict(op_spec.get("params", {}))
        except (TypeError, KeyError):
            continue
        if op_name in ("flip_horizontal", "flip_vertical",
                       "rotate_90_cw", "rotate_180", "rotate_270_cw"):
            _append_step(ProcessingStep(op_name, {}))
        elif op_name == "rotate_arbitrary":
            _append_step(ProcessingStep("rotate_arbitrary", {
                "angle_degrees": float(op_params.get("angle_degrees", 0.0)),
                "order": int(op_params.get("order", 1)),
            }))

    return ProcessingState(steps=steps)


def gui_state_has_numeric_processing(gui_state: dict | None) -> bool:
    """Return whether a GUI dict emits at least one canonical processing step."""
    return bool(processing_state_from_gui(gui_state or {}).steps)


def apply_processing_state_to_scan(
    scan: "Scan",
    proc_state: dict,
    *,
    plane_idx: int = 0,
) -> "Scan":
    """Apply GUI processing state to a Scan before export.

    Converts *proc_state* to a canonical :class:`ProcessingState`, applies it
    via :func:`~probeflow.processing.state.apply_processing_state`, and
    records each step on ``scan.processing_state``.

    Updates ``scan.planes[plane_idx]`` in place and returns *scan*.
    Display-only settings (grain overlay, colormap, clip percentiles) are ignored.
    """
    from probeflow.processing.state import apply_processing_state

    if plane_idx < 0 or plane_idx >= len(scan.planes):
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{len(scan.planes)} plane(s)"
        )

    state    = processing_state_from_gui(proc_state)
    a        = apply_processing_state(scan.planes[plane_idx], state)

    scan.planes[plane_idx] = a
    scan.record_processing_state(state)

    return scan
