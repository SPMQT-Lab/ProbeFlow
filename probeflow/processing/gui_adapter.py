"""Bridge GUI processing state into canonical processing operations.

No Qt imports — this module can be tested without a running Qt event loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from probeflow.core.op_vocab import SIMPLE_GEOMETRIC_OPS

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
    "arithmetic_ops",
)

# Value-filter ops routed through the FFT viewer's generic apply channel
# (apply_correction_fn → ``geometric_ops``).  Their params are already in
# canonical form for ``apply_processing_state``, so they pass straight through.
# Without this, the geometric_ops dispatch below would silently drop them and
# "Apply" would do nothing.
_FILTER_OPS_PASSTHROUGH: frozenset[str] = frozenset({
    "mains_pickup_suppression",
    "inverse_fft_filter",
})


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

    def _warn_skipped_step(step_name: str, reason: str) -> None:
        warnings.warn(
            f"Skipped GUI processing step {step_name!r}: {reason}",
            UserWarning,
            stacklevel=2,
        )

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

    # Review image-proc #3: row alignment must not precede background fitting.
    # Aligning rows first removes row medians/means that the background model
    # needs in order to fit the original surface, especially when features
    # occupy only some scan lines.
    align = gui_state.get("align_rows")
    if align:
        _append_step(ProcessingStep("align_rows", {"method": str(align)}))

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
        invalid_peaks = 0
        for peak in notches:
            try:
                peaks.append((int(peak[0]), int(peak[1])))
            except (TypeError, ValueError, IndexError):
                invalid_peaks += 1
                continue
        if peaks:
            _append_step(ProcessingStep("periodic_notch_filter", {
                "peaks": peaks,
                "radius_px": float(gui_state.get("periodic_notch_radius", 3.0)),
            }))
        elif invalid_peaks:
            _warn_skipped_step("periodic_notch_filter", "no valid notch coordinates")

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
        except (TypeError, ValueError, IndexError) as exc:
            _warn_skipped_step("set_zero_point", str(exc) or "invalid coordinates")

    zero_plane = gui_state.get("set_zero_plane_points")
    if zero_plane is not None:
        points = []
        invalid_points = 0
        for point in zero_plane:
            try:
                points.append((int(point[0]), int(point[1])))
            except (TypeError, ValueError, IndexError):
                invalid_points += 1
                continue
        if len(points) >= 3:
            _append_step(ProcessingStep("set_zero_plane", {
                "points_px": points[:3],
                "patch": int(gui_state.get("set_zero_patch", 1)),
            }))
        else:
            detail = "requires at least three valid points"
            if invalid_points:
                detail += f"; ignored {invalid_points} invalid point(s)"
            _warn_skipped_step("set_zero_plane", detail)

    # Quantize bit-depth steps are deferred to the very end of the
    # pipeline (review image-proc #2, fixed 2026-05-28).  Running
    # quantize before arithmetic_ops broke the "256 distinct levels"
    # guarantee: a subsequent constant add/subtract shifted every
    # pixel by a non-integer-multiple of the quantum, so the saved
    # 8-bit/16-bit image silently contained more than 2**bits distinct
    # values.  Collect any quantize specs from geometric_ops and emit
    # them after arithmetic_ops below.
    deferred_quantize_specs: list[dict] = []

    for op_spec in gui_state.get("geometric_ops") or []:
        if op_spec is None:
            continue
        try:
            if isinstance(op_spec, str):
                op_name = op_spec
                op_params: dict = {}
            else:
                op_name = str(op_spec["op"])
                op_params = dict(op_spec.get("params", {}))
        except (TypeError, KeyError) as exc:
            _warn_skipped_step("geometric_ops", str(exc) or "invalid operation spec")
            continue
        if op_name in SIMPLE_GEOMETRIC_OPS:
            _append_step(ProcessingStep(op_name, {}))
        elif op_name == "affine_lattice_correction":
            import numpy as _np
            raw_matrix = op_params.get("matrix")
            if raw_matrix is None:
                _warn_skipped_step(
                    "affine_lattice_correction",
                    "missing correction matrix",
                )
                continue
            try:
                matrix = _np.asarray(raw_matrix, dtype=float).tolist()
            except (TypeError, ValueError) as exc:
                _warn_skipped_step(
                    "affine_lattice_correction",
                    str(exc) or "invalid correction matrix",
                )
                continue
            params: dict = {
                "matrix": matrix,
                "expand_canvas": bool(op_params.get("expand_canvas", True)),
                "interpolation": str(op_params.get("interpolation", "bilinear")),
                "fill_mode": str(op_params.get("fill_mode", "nan")),
            }
            if op_params.get("fill_value") is not None:
                params["fill_value"] = float(op_params["fill_value"])
            if op_params.get("full_matrix") is not None:
                try:
                    params["full_matrix"] = _np.asarray(
                        op_params["full_matrix"], dtype=float
                    ).tolist()
                except (TypeError, ValueError):
                    pass
            if "preserve_orientation" in op_params:
                params["preserve_orientation"] = bool(
                    op_params["preserve_orientation"]
                )
            for key in (
                "polar_rotation_deg",
                "ideal_a_nm",
                "ideal_b_nm",
                "ideal_angle_deg",
            ):
                if op_params.get(key) is not None:
                    try:
                        params[key] = float(op_params[key])
                    except (TypeError, ValueError):
                        pass
            for key in ("measured_a_nm", "measured_b_nm"):
                if op_params.get(key) is not None:
                    try:
                        params[key] = [float(v) for v in op_params[key]]
                    except (TypeError, ValueError):
                        pass
            if isinstance(op_params.get("known_structure"), dict):
                known = op_params["known_structure"]
                try:
                    params["known_structure"] = {
                        "name": str(known.get("name", "")),
                        "symmetry": str(known.get("symmetry", "")),
                        "a_nm": float(known.get("a_nm", 0.0)),
                        "b_nm": float(known.get("b_nm", 0.0)),
                        "angle_deg": float(known.get("angle_deg", 0.0)),
                        "unit": str(known.get("unit", "")),
                    }
                except (TypeError, ValueError):
                    pass
            _append_step(ProcessingStep("affine_lattice_correction", params))
        elif op_name == "rotate_arbitrary":
            _append_step(ProcessingStep("rotate_arbitrary", {
                "angle_degrees": float(op_params.get("angle_degrees", 0.0)),
                "order": int(op_params.get("order", 1)),
            }))
        elif op_name == "shear":
            _append_step(ProcessingStep("shear", {
                "shear_x": float(op_params.get("shear_x", 0.0)),
                "shear_y": float(op_params.get("shear_y", 0.0)),
                "interpolation": str(op_params.get("interpolation", "bilinear")),
            }))
        elif op_name == "scale_image":
            _append_step(ProcessingStep("scale_image", {
                "new_height": int(op_params["new_height"]),
                "new_width": int(op_params["new_width"]),
                "order": int(op_params.get("order", 1)),
            }))
        elif op_name == "image_threshold":
            thr_params: dict = {"mode": str(op_params.get("mode", "clip"))}
            if op_params.get("lower") is not None:
                thr_params["lower"] = float(op_params["lower"])
            if op_params.get("upper") is not None:
                thr_params["upper"] = float(op_params["upper"])
            _append_step(ProcessingStep("image_threshold", thr_params))
        elif op_name == "quantize_bit_depth":
            # Defer to the end (after arithmetic_ops).  See note above.
            q_params: dict = {"bits": int(op_params.get("bits", 8))}
            if op_params.get("vmin") is not None:
                q_params["vmin"] = float(op_params["vmin"])
            if op_params.get("vmax") is not None:
                q_params["vmax"] = float(op_params["vmax"])
            deferred_quantize_specs.append(q_params)
        elif op_name in _FILTER_OPS_PASSTHROUGH:
            # FFT value-filters (mains pickup, inverse-FFT reconstruction): the
            # stored params are already canonical for apply_processing_state.
            _append_step(ProcessingStep(op_name, op_params))
        else:
            _warn_skipped_step("geometric_ops", f"unhandled op {op_name!r}")

    for op_spec in gui_state.get("arithmetic_ops") or []:
        try:
            if isinstance(op_spec, dict) and op_spec.get("op") == "roi":
                roi_params = dict(op_spec.get("params", {}))
                nested = dict(roi_params.get("step", {}))
                if nested.get("op") != "arithmetic":
                    continue
                params = dict(nested.get("params", {}))
                roi_id_for_step = roi_params.get("roi_id")
            else:
                params = dict(op_spec.get("params", {}))
                roi_id_for_step = op_spec.get("roi_id")
        except (AttributeError, TypeError, ValueError) as exc:
            _warn_skipped_step("arithmetic", str(exc) or "invalid arithmetic spec")
            continue

        step = ProcessingStep("arithmetic", params)
        if roi_id_for_step is None:
            steps.append(step)
        else:
            steps.append(ProcessingStep("roi", {
                "roi_id": str(roi_id_for_step),
                "step": {"op": "arithmetic", "params": dict(params)},
            }))

    # Emit deferred quantize_bit_depth steps last, so any arithmetic
    # offset applied above lands BEFORE quantization and the saved
    # image actually contains 2**bits distinct values
    # (review image-proc #2).
    for q_params in deferred_quantize_specs:
        steps.append(ProcessingStep("quantize_bit_depth", q_params))

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
    via :func:`~probeflow.processing.state.apply_processing_state_with_calibration`,
    and records each step on ``scan.processing_state``.  When a shape-changing
    step grew the canvas (``rotate_arbitrary``, ``shear``, or
    ``affine_lattice_correction`` with canvas expansion), ``scan.scan_range_m``
    is updated to match the new array shape so PNG scale bars, FFT k-axes,
    and feature pixel→nm conversions stay correct (review image-proc #4).

    Updates ``scan.planes[plane_idx]`` in place and returns *scan*.
    Display-only settings (grain overlay, colormap, clip percentiles) are ignored.
    """
    from probeflow.processing.state import apply_processing_state_with_calibration

    if plane_idx < 0 or plane_idx >= len(scan.planes):
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{len(scan.planes)} plane(s)"
        )

    state = processing_state_from_gui(proc_state)
    raw_range: tuple[float, float] | None = None
    try:
        w_m, h_m = float(scan.scan_range_m[0]), float(scan.scan_range_m[1])
        if w_m > 0 and h_m > 0:
            raw_range = (w_m, h_m)
    except (TypeError, ValueError, IndexError, AttributeError):
        pass
    a, new_range = apply_processing_state_with_calibration(
        scan.planes[plane_idx], state, scan_range_m=raw_range,
    )

    scan.planes[plane_idx] = a
    if new_range is not None:
        scan.scan_range_m = new_range
    scan.record_processing_state(state)

    return scan
