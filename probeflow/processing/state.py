"""ProcessingState — canonical representation of numerical processing choices.

This module separates *data-transforming* operations (bad-line removal, row
alignment, background subtraction, smoothing, FFT, edge detection) from
display-only settings (colormap, clip percentiles, vmin/vmax, grain overlay).

Typical call order
------------------
    state = ProcessingState(steps=[
        ProcessingStep("remove_bad_lines"),
        ProcessingStep("align_rows", {"method": "median"}),
        ProcessingStep("plane_bg", {"order": 1}),
    ])
    processed_arr = apply_processing_state(raw_arr, state)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from probeflow.core.roi import AREA_ROI_KINDS
# ProcessingState / ProcessingStep moved to probeflow.core.processing_state
# (review arch-backend #14).  Re-exported here so the historical
# ``from probeflow.processing.state import ProcessingState`` import path
# continues to work.
from probeflow.core.processing_state import (  # noqa: F401
    _ROI_ELIGIBLE_OPS,
    _SUPPORTED_OPS,
    ProcessingState,
    ProcessingStep,
)


def apply_operation_with_optional_roi(
    image: np.ndarray,
    operation,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply ``operation`` globally, or copy its result back only inside ``roi_mask``."""
    source = np.asarray(image, dtype=np.float64).copy()
    if roi_mask is None:
        return np.asarray(operation(source), dtype=np.float64)

    mask = np.asarray(roi_mask, dtype=bool)
    if mask.shape != source.shape:
        raise ValueError(
            f"roi_mask shape {mask.shape} does not match image shape {source.shape}"
        )
    if not mask.any():
        raise ValueError("roi_mask is empty")

    processed_full = np.asarray(operation(source.copy()), dtype=np.float64)
    if processed_full.shape != source.shape:
        raise ValueError(
            "ROI-scoped operation must return the same shape as the input image"
        )
    result = source.copy()
    result[mask] = processed_full[mask]
    return result

# ── ROI reference validation ─────────────────────────────────────────────────

def _roi_refs_from_expr(expr: Any, param_name: str) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if not isinstance(expr, dict):
        return refs
    if "ref" in expr:
        refs.append({"param": f"{param_name}.ref", "value": str(expr["ref"])})
    if "invert" in expr:
        refs.append({"param": f"{param_name}.invert", "value": str(expr["invert"])})
    if "combine" in expr:
        for value in expr.get("combine") or ():
            refs.append({"param": f"{param_name}.combine", "value": str(value)})
    return refs


def roi_references_from_state(state: "ProcessingState") -> list[dict[str, Any]]:
    """Return ROI ids/names referenced by a processing state.

    The returned dictionaries are intentionally plain JSON-like values so GUI,
    CLI, and tests can report stale ROI references without importing Qt or
    depending on a concrete ROISet implementation.
    """
    refs: list[dict[str, Any]] = []
    for step_index, step in enumerate(state.steps):
        params = step.params or {}
        if step.op == "roi" and params.get("roi_id") is not None:
            refs.append({
                "step_index": step_index,
                "op": step.op,
                "param": "roi_id",
                "value": str(params["roi_id"]),
            })
        if step.op == "plane_bg":
            for param in ("fit_roi_id", "apply_roi_id", "exclude_roi_id"):
                if params.get(param) is not None:
                    refs.append({
                        "step_index": step_index,
                        "op": step.op,
                        "param": param,
                        "value": str(params[param]),
                    })
            for param in ("fit_roi_expr", "apply_roi_expr", "exclude_roi_expr"):
                for ref in _roi_refs_from_expr(params.get(param), param):
                    ref.update({"step_index": step_index, "op": step.op})
                    refs.append(ref)
        if step.op == "stm_background" and params.get("fit_roi_id") is not None:
            refs.append({
                "step_index": step_index,
                "op": step.op,
                "param": "fit_roi_id",
                "value": str(params["fit_roi_id"]),
            })
    return refs


def missing_roi_references(
    state: "ProcessingState",
    roi_set: "Any | None",
) -> list[dict[str, Any]]:
    """Return ROI references in *state* that are not present in *roi_set*.

    Lookup accepts both UUIDs and names because historical CLI/provenance paths
    allowed either.  This validator is deliberately non-mutating; callers decide
    whether a missing reference should warn, block rendering, or abort export.
    """
    refs = roi_references_from_state(state)
    if not refs:
        return []
    if roi_set is None:
        return refs

    missing: list[dict[str, Any]] = []
    for ref in refs:
        value = str(ref["value"])
        found = roi_set.get(value)
        if found is None and hasattr(roi_set, "get_by_name"):
            found = roi_set.get_by_name(value)
        if found is None:
            missing.append(ref)
    return missing


def assert_roi_references_resolved(
    state: "ProcessingState",
    roi_set: "Any | None",
) -> None:
    """Raise ``ValueError`` if any ROI reference in *state* cannot be resolved.

    Use this in export paths where silent substitution is not acceptable.
    Interactive display paths should use :func:`missing_roi_references` and
    show a warning instead.
    """
    missing = missing_roi_references(state, roi_set)
    if missing:
        refs = ", ".join(
            f"{m['param']}={m['value']!r}" for m in missing[:3]
        )
        if len(missing) > 3:
            refs += f", +{len(missing) - 3} more"
        raise ValueError(
            f"Processing state references {len(missing)} ROI(s) that are not "
            f"present in the current ROI set: {refs}. Export aborted."
        )


# ── ROI expression resolver ───────────────────────────────────────────────────

def resolve_roi_expr(
    expr: "dict[str, Any] | str | None",
    roi_set: "Any | None",
    image_shape: "tuple[int, int]",
) -> "Any | None":
    """Resolve a ROI expression from a processing step param to a concrete ROI.

    Supported expression forms
    --------------------------
    ``None``
        No ROI — returns ``None``.
    ``{"ref": "<name_or_id>"}``
        Look up a ROI by name or UUID in *roi_set*.
    ``{"invert": "<name_or_id>"}``
        Compute the Shapely invert of the named ROI within *image_shape*.
    ``{"combine": ["<id_or_name>", ...], "mode": "<union|intersection|...>"}``
        Combine the named ROIs using Shapely geometry algebra.

    Raises
    ------
    KeyError
        If a referenced ROI name / UUID is not found in *roi_set*.
    ValueError
        If the expression is malformed or *roi_set* is ``None`` for an
        expression that needs it.
    """
    if expr is None:
        return None
    if not isinstance(expr, dict):
        raise ValueError(
            f"ROI expression must be a dict or None, got {type(expr).__name__!r}: {expr!r}"
        )
    if roi_set is None:
        raise ValueError(
            "ROI expression requires a roi_set but none was provided to "
            "apply_processing_state."
        )

    def _lookup(name_or_id: str) -> "Any":
        roi = roi_set.get(name_or_id) or roi_set.get_by_name(name_or_id)
        if roi is None:
            raise KeyError(
                f"ROI {name_or_id!r} not found in roi_set "
                f"(available: {', '.join(r.name for r in roi_set.rois) or '(none)'})"
            )
        return roi

    if "ref" in expr:
        return _lookup(str(expr["ref"]))

    if "invert" in expr:
        from probeflow.core.roi import invert as _invert
        return _invert(_lookup(str(expr["invert"])), image_shape)

    if "combine" in expr:
        from probeflow.core.roi import combine as _combine
        rois = [_lookup(str(rid)) for rid in expr["combine"]]
        mode = str(expr.get("mode", "union"))
        return _combine(rois, mode)

    raise ValueError(
        f"Unrecognised ROI expression keys: {list(expr.keys())!r}. "
        "Expected 'ref', 'invert', or 'combine'."
    )


def _resolve_bg_roi_param(
    params: "dict[str, Any]",
    prefix: str,
    image_shape: "tuple[int, int]",
    roi_set: "Any | None",
) -> "Any | None":
    """Resolve fit_roi / apply_roi / exclude_roi from a plane_bg step params dict."""
    roi_id = params.get(f"{prefix}_id")
    roi_expr = params.get(f"{prefix}_expr")

    if roi_id is not None and roi_expr is not None:
        raise ValueError(
            f"Cannot specify both {prefix}_id and {prefix}_expr in the same step."
        )

    if roi_id is not None:
        if roi_set is None:
            import warnings
            warnings.warn(
                f"plane_bg step has {prefix}_id={roi_id!r} but no roi_set was passed "
                "to apply_processing_state — ROI parameter ignored.",
                UserWarning,
                stacklevel=4,
            )
            return None
        roi = roi_set.get(str(roi_id)) or roi_set.get_by_name(str(roi_id))
        if roi is None:
            import warnings
            warnings.warn(
                f"plane_bg: {prefix}_id={roi_id!r} not found in roi_set — "
                "ROI parameter ignored.",
                UserWarning,
                stacklevel=4,
            )
            return None
        return roi

    if roi_expr is not None:
        return resolve_roi_expr(roi_expr, roi_set, image_shape)

    return None


def _resolve_mask_roi_param(
    params: "dict[str, Any]",
    prefix: str,
    image_shape: "tuple[int, int]",
    roi_set: "Any | None",
) -> "np.ndarray | None":
    roi_id = params.get(f"{prefix}_roi_id")
    if roi_id is None:
        return None
    if roi_set is None:
        import warnings
        warnings.warn(
            f"stm_background step has {prefix}_roi_id={roi_id!r} but no roi_set "
            "was passed to apply_processing_state — ROI fit mask ignored.",
            UserWarning,
            stacklevel=4,
        )
        return None
    roi = roi_set.get(str(roi_id)) or roi_set.get_by_name(str(roi_id))
    if roi is None:
        import warnings
        warnings.warn(
            f"stm_background: {prefix}_roi_id={roi_id!r} not found in roi_set — "
            "ROI fit mask ignored.",
            UserWarning,
            stacklevel=4,
        )
        return None
    if getattr(roi, "kind", None) not in AREA_ROI_KINDS:
        import warnings
        warnings.warn(
            f"stm_background: {prefix}_roi_id={roi_id!r} is not an area ROI — "
            "ROI fit mask ignored.",
            UserWarning,
            stacklevel=4,
        )
        return None
    return roi.to_mask(image_shape)


def _load_arithmetic_operand_image(params: "dict[str, Any]") -> np.ndarray:
    """Load the raw plane referenced by an image-arithmetic step."""
    source_path = params.get("source_path")
    if not source_path:
        raise ValueError("Image arithmetic step is missing source_path.")
    try:
        plane_idx = int(params.get("plane_idx", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Image arithmetic step has an invalid plane_idx.") from exc

    from probeflow.core.scan_loader import load_scan

    try:
        scan = load_scan(Path(str(source_path)))
    except Exception as exc:
        raise ValueError(
            f"Could not load image arithmetic operand {source_path!r}: {exc}"
        ) from exc

    if plane_idx < 0 or plane_idx >= len(scan.planes):
        raise ValueError(
            f"Image arithmetic plane_idx={plane_idx} is out of range for "
            f"{source_path!r} ({len(scan.planes)} plane(s))."
        )
    return np.asarray(scan.planes[plane_idx], dtype=np.float64)


# ── Canonical apply function ──────────────────────────────────────────────────

def apply_processing_state(
    arr: np.ndarray,
    state: "ProcessingState",
    roi_set: "Any | None" = None,
    *,
    pixel_size_x_m: float | None = None,
    pixel_size_y_m: float | None = None,
    _depth: int = 0,
) -> np.ndarray:
    """Apply *state* steps in order to *arr*.

    Parameters
    ----------
    arr:
        Input 2-D numeric array (will not be mutated).
    state:
        Processing steps to apply.
    roi_set:
        Optional :class:`probeflow.core.roi.ROISet`.  ``roi`` steps reference
        ROIs by ``roi_id`` and resolve them from this set at execution time. If
        the ID is not found, the step is skipped with a warning.
    pixel_size_x_m, pixel_size_y_m:
        Physical pixel size in metres for each axis.  When provided, these
        are forwarded to operations whose semantics depend on physical units
        — currently :func:`subtract_background` (with ``step_tolerance=True``)
        and :func:`facet_level` (both interpret ``step_threshold_deg`` /
        ``threshold_deg`` as a real surface slope).  Before this kwarg was
        added (review image-proc #1, 2026-05-28) every GUI/CLI invocation
        silently fell back to the kernels' 1.0 m/pixel default, so a
        ``step_threshold_deg=3°`` (tan ≈ 0.052) was interpreted as
        ``0.052 data-units per pixel`` instead of a real surface slope —
        the step-tolerant background fit degraded to non-step-tolerant
        and was biased by step edges on every stepped-surface workflow.
        Pass calibrated values whenever the scan has them.

    Returns
    -------
    np.ndarray of float64, same shape as *arr*.

    Raises
    ------
    ValueError
        If a step contains an unrecognised operation name, or if ROI-in-ROI
        nesting exceeds depth 2.
    """
    # Always return a fresh float64 copy so raw Scan planes are never mutated.
    a = arr.astype(np.float64, copy=True)

    if not state.steps:
        return a

    import probeflow.processing as _proc

    for step in state.steps:
        p = step.params
        if step.op == "remove_bad_lines":
            a = _proc.remove_bad_lines(
                a,
                threshold_mad=float(p.get("threshold_mad", 5.0)),
                method=str(p.get("method", "mad")),
                polarity=str(p.get("polarity", "bright")),
                min_segment_length_px=int(p.get("min_segment_length_px", 2)),
                max_adjacent_bad_lines=int(p.get("max_adjacent_bad_lines", 1)),
            )
        elif step.op == "align_rows":
            a = _proc.align_rows(a, method=p.get("method", "median"))
        elif step.op == "plane_bg":
            # Resolve new ROI expression parameters (fit_roi, apply_roi, exclude_roi)
            fit_roi = _resolve_bg_roi_param(p, "fit_roi", a.shape, roi_set)
            apply_roi = _resolve_bg_roi_param(p, "apply_roi", a.shape, roi_set)
            exclude_roi = _resolve_bg_roi_param(p, "exclude_roi", a.shape, roi_set)
            # Calibration kwargs (review image-proc #1): forward to the
            # kernel so step_threshold_deg is a real surface slope, not
            # a data-units-per-pixel ratio.  When the caller didn't
            # supply pixel sizes the kernel defaults (1.0 m/pixel) are
            # preserved for backward compatibility.
            cal_kwargs: dict[str, float] = {}
            if pixel_size_x_m is not None:
                cal_kwargs["pixel_size_x_m"] = float(pixel_size_x_m)
            if pixel_size_y_m is not None:
                cal_kwargs["pixel_size_y_m"] = float(pixel_size_y_m)
            a = _proc.subtract_background(
                a,
                order=int(p.get("order", 1)),
                fit_roi=fit_roi,
                apply_roi=apply_roi,
                exclude_roi=exclude_roi,
                step_tolerance=bool(p.get("step_tolerance", False)),
                fit_rect=p.get("fit_rect"),
                **cal_kwargs,
            )
        elif step.op == "stm_line_bg":
            a = _proc.stm_line_background(
                a,
                mode=str(p.get("mode", "step_tolerant")),
            )
        elif step.op == "stm_background":
            fit_mask = _resolve_mask_roi_param(p, "fit", a.shape, roi_set)
            a = _proc.apply_stm_background(
                a,
                _proc.STMBackgroundParams(
                    fit_region=str(p.get("fit_region", "whole_image")),
                    line_statistic=str(p.get("line_statistic", "median")),
                    model=str(p.get("model", "linear")),
                    linear_x_first=bool(p.get("linear_x_first", False)),
                    blur_length=p.get("blur_length"),
                    jump_threshold=p.get("jump_threshold"),
                    preserve_level=str(p.get("preserve_level", "median")),
                ),
                mask=fit_mask,
            )
        elif step.op == "facet_level":
            # Review image-proc #1: forward calibration so threshold_deg
            # is a real surface slope.
            cal_kwargs: dict[str, float] = {}
            if pixel_size_x_m is not None:
                cal_kwargs["pixel_size_x_m"] = float(pixel_size_x_m)
            if pixel_size_y_m is not None:
                cal_kwargs["pixel_size_y_m"] = float(pixel_size_y_m)
            a = _proc.facet_level(
                a,
                threshold_deg=float(p.get("threshold_deg", 3.0)),
                **cal_kwargs,
            )
        elif step.op == "smooth":
            a = _proc.gaussian_smooth(a, sigma_px=float(p.get("sigma_px", 1.0)))
        elif step.op == "gaussian_high_pass":
            a = _proc.gaussian_high_pass(
                a,
                sigma_px=float(p.get("sigma_px", 8.0)),
            )
        elif step.op == "edge_detect":
            a = _proc.edge_detect(
                a,
                method=p.get("method", "laplacian"),
                sigma=float(p.get("sigma", 1.0)),
                sigma2=float(p.get("sigma2", 2.0)),
            )
        elif step.op == "fourier_filter":
            a = _proc.fourier_filter(
                a,
                mode=p.get("mode", "low_pass"),
                cutoff=float(p.get("cutoff", 0.10)),
                window=str(p.get("window", "hanning")),
            )
        elif step.op == "fft_soft_border":
            a = _proc.fft_soft_border(
                a,
                mode=str(p.get("mode", "low_pass")),
                cutoff=float(p.get("cutoff", 0.10)),
                border_frac=float(p.get("border_frac", 0.12)),
            )
        elif step.op == "periodic_notch_filter":
            a = _proc.periodic_notch_filter(
                a,
                p.get("peaks", ()),
                radius_px=float(p.get("radius_px", 3.0)),
            )
        elif step.op == "linear_undistort":
            a = _proc.linear_undistort(
                a,
                shear_x=float(p.get("shear_x", 0.0)),
                scale_y=float(p.get("scale_y", 1.0)),
            )
        elif step.op == "affine_lattice_correction":
            matrix = np.asarray(p["matrix"], dtype=np.float64)
            a = _proc.affine_lattice_correction(
                a,
                matrix,
                expand_canvas=bool(p.get("expand_canvas", True)),
                interpolation=str(p.get("interpolation", "bilinear")),
                fill_mode=str(p.get("fill_mode", "nan")),
                fill_value=float(p["fill_value"]) if p.get("fill_value") is not None else None,
            )
        elif step.op == "arithmetic":
            operand_type = str(p.get("operand_type", "constant"))
            operand_image = None
            if operand_type == "image":
                operand_image = _load_arithmetic_operand_image(p)
            elif operand_type == "generated":
                operand_image = _proc.generate_arithmetic_pattern(
                    a.shape,
                    str(p.get("pattern", "checkerboard")),
                    float(p.get("amplitude_si", 0.0)),
                    period_px=int(p.get("period_px", 16)),
                    seed=int(p.get("seed", 1)),
                )
            a = _proc.apply_arithmetic(
                a,
                operation=str(p.get("operation", "add")),
                operand_type=operand_type,
                value_si=p.get("value_si"),
                factor=p.get("factor"),
                operand_image=operand_image,
            )
        elif step.op == "set_zero_point":
            a = _proc.set_zero_point(
                a,
                int(p.get("y_px", 0)),
                int(p.get("x_px", 0)),
                patch=int(p.get("patch", 1)),
            )
        elif step.op == "set_zero_plane":
            a = _proc.set_zero_plane(
                a,
                p.get("points_px", ()),
                patch=int(p.get("patch", 1)),
            )
        elif step.op == "roi":
            if _depth >= 2:
                raise ValueError(
                    "ROI-in-ROI nesting exceeded maximum depth of 2. "
                    "Nested 'roi' steps inside 'roi' steps are not allowed."
                )
            try:
                nested = ProcessingStep.from_dict(p.get("step", {}))
            except (KeyError, TypeError, ValueError):
                continue
            if nested.op not in _ROI_ELIGIBLE_OPS:
                continue
            roi_id = p.get("roi_id")
            if roi_id is None:
                continue
            if roi_set is None:
                import warnings
                warnings.warn(
                    f"roi step references roi_id={roi_id!r} but no roi_set "
                    "was passed to apply_processing_state — step skipped.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            roi_ref = str(roi_id)
            roi_obj = roi_set.get(roi_ref)
            if roi_obj is None and hasattr(roi_set, "get_by_name"):
                roi_obj = roi_set.get_by_name(roi_ref)
            if roi_obj is None:
                import warnings
                warnings.warn(
                    f"roi_id={roi_id!r} not found in roi_set — step skipped.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            mask = roi_obj.to_mask(a.shape)
            bounds = roi_obj.bounds(a.shape)
            if mask is None or bounds is None or not mask.any():
                continue
            a = apply_operation_with_optional_roi(
                a,
                lambda image, nested=nested: apply_processing_state(
                    image,
                    ProcessingState(steps=[nested]),
                    roi_set,
                    pixel_size_x_m=pixel_size_x_m,
                    pixel_size_y_m=pixel_size_y_m,
                    _depth=_depth + 1,
                ),
                mask,
            )
        elif step.op in ("flip_horizontal", "flip_vertical",
                         "rotate_90_cw", "rotate_180", "rotate_270_cw"):
            fn = getattr(_proc, step.op)
            a = fn(a)
        elif step.op == "rotate_arbitrary":
            roi_steps = [s for s in state.steps if s.op == "roi"]
            if roi_steps:
                import warnings
                warnings.warn(
                    "rotate_arbitrary invalidates existing ROI geometry. "
                    "ROI steps in the processing state have been skipped.",
                    UserWarning,
                    stacklevel=2,
                )
            a = _proc.rotate_arbitrary(
                a,
                angle_degrees=float(p.get("angle_degrees", 0.0)),
                order=int(p.get("order", 1)),
            )
        elif step.op == "shear":
            a = _proc.shear(
                a,
                shear_x=float(p.get("shear_x", 0.0)),
                shear_y=float(p.get("shear_y", 0.0)),
                interpolation=str(p.get("interpolation", "bilinear")),
            )
        elif step.op == "scale_image":
            a = _proc.scale_image(
                a,
                int(p["new_height"]),
                int(p["new_width"]),
                order=int(p.get("order", 1)),
            )
        elif step.op == "image_threshold":
            lower = float(p["lower"]) if p.get("lower") is not None else None
            upper = float(p["upper"]) if p.get("upper") is not None else None
            a = _proc.threshold_image(
                a,
                lower=lower,
                upper=upper,
                mode=str(p.get("mode", "clip")),
            )
        elif step.op == "quantize_bit_depth":
            # Review physics #3 / numerical #3 / image-proc #7 cluster:
            # forward explicit vmin/vmax from the step params when set
            # (the GUI captures these for cross-scan reproducibility),
            # else the kernel falls back to a robust percentile band.
            q_kwargs: dict[str, float] = {}
            if p.get("vmin") is not None:
                q_kwargs["vmin"] = float(p["vmin"])
            if p.get("vmax") is not None:
                q_kwargs["vmax"] = float(p["vmax"])
            a = _proc.quantize_bit_depth(a, bits=int(p["bits"]), **q_kwargs)
        else:
            raise ValueError(
                f"Unknown processing operation {step.op!r}. "
                f"Supported: {sorted(_SUPPORTED_OPS)}"
            )

    return a


# ── Shape-changing ops: scan_range_m bookkeeping ──────────────────────────────

# Per-op rules for updating scan_range_m when a step changes the array shape.
# The two conventions:
#   * ``preserve_pixel_size``: rotate_arbitrary / shear / affine_lattice_correction
#     with expand_canvas=True grow the bounding canvas while keeping each pixel's
#     physical size unchanged.  scan_range_m must grow proportionally with the
#     new shape so that ``pixel_size = scan_range_m / shape`` stays correct.
#   * ``preserve_extent``: scale_image resamples to a new pixel density while
#     the physical extent is preserved.  scan_range_m stays fixed; pixel_size
#     scales inversely with the new shape.
_SHAPE_CHANGING_PIXEL_SIZE_PRESERVING: frozenset[str] = frozenset({
    "rotate_arbitrary",
    "shear",
    "affine_lattice_correction",
})

_SHAPE_CHANGING_EXTENT_PRESERVING: frozenset[str] = frozenset({
    "scale_image",
})


def _update_scan_range_for_op(
    op: str,
    scan_range_m: tuple[float, float] | None,
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
) -> tuple[float, float] | None:
    """Return updated ``scan_range_m`` after a step changed the array shape.

    See :data:`_SHAPE_CHANGING_PIXEL_SIZE_PRESERVING` and
    :data:`_SHAPE_CHANGING_EXTENT_PRESERVING` for the per-op semantics.
    Returns ``None`` if the input was ``None``; otherwise always returns a
    ``(float, float)`` tuple.
    """
    if scan_range_m is None:
        return None
    old_h, old_w = old_shape
    new_h, new_w = new_shape
    if op in _SHAPE_CHANGING_PIXEL_SIZE_PRESERVING:
        if old_w <= 0 or old_h <= 0:
            return (float(scan_range_m[0]), float(scan_range_m[1]))
        w_m = float(scan_range_m[0]) * new_w / old_w
        h_m = float(scan_range_m[1]) * new_h / old_h
        return (w_m, h_m)
    # scale_image, or any other op that changed shape unexpectedly: leave the
    # physical extent untouched (the safer default — pixel_size adjusts via
    # the new shape).
    return (float(scan_range_m[0]), float(scan_range_m[1]))


def apply_processing_state_with_calibration(
    arr: np.ndarray,
    state: "ProcessingState",
    roi_set: "Any | None" = None,
    *,
    scan_range_m: tuple[float, float] | None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Apply *state* to *arr* and return the post-processing scan_range.

    Threads pixel calibration through the pipeline two ways:

    1. For each step, the *current* pixel size (``scan_range_m / shape``) is
       forwarded as ``pixel_size_x_m`` / ``pixel_size_y_m`` to
       :func:`apply_processing_state`, so step-tolerance and facet-level ops
       interpret their slope thresholds against real surface geometry
       (review image-proc #1).
    2. When a step changes the array shape (``rotate_arbitrary``, ``shear``,
       ``affine_lattice_correction``, ``scale_image``), ``scan_range_m`` is
       updated per :func:`_update_scan_range_for_op` so that downstream
       consumers — PNG scale bars, FFT k-axes, feature pixel→nm conversions,
       saved provenance — see a calibration consistent with the new array
       shape (review image-proc #4).

    Parameters
    ----------
    arr
        Input 2-D array.
    state, roi_set
        Forwarded to :func:`apply_processing_state`.
    scan_range_m
        ``(width_m, height_m)`` for *arr* as captured at the source.  Pass the
        scan's current ``scan_range_m`` here.  May be ``None`` if calibration
        is unknown — pixel-size kwargs are then omitted from the kernel calls
        and the returned ``new_scan_range_m`` will also be ``None``.

    Returns
    -------
    (new_arr, new_scan_range_m)
        ``new_arr`` is the processed array (float64).  ``new_scan_range_m``
        is the updated ``(width_m, height_m)`` reflecting any shape-changing
        steps, or ``None`` if the input was ``None``.
    """
    current_range = (
        (float(scan_range_m[0]), float(scan_range_m[1]))
        if scan_range_m is not None else None
    )
    a = arr.astype(np.float64, copy=True)
    if not state.steps:
        return a, current_range

    for step in state.steps:
        h_in, w_in = a.shape
        psx = psy = None
        if current_range is not None and w_in > 0 and h_in > 0:
            psx = current_range[0] / w_in
            psy = current_range[1] / h_in
        a = apply_processing_state(
            a,
            ProcessingState(steps=[step]),
            roi_set=roi_set,
            pixel_size_x_m=psx,
            pixel_size_y_m=psy,
        )
        h_out, w_out = a.shape
        if (h_out, w_out) != (h_in, w_in):
            current_range = _update_scan_range_for_op(
                step.op, current_range, (h_in, w_in), (h_out, w_out),
            )
    return a, current_range


# ── Geometric transform + ROISet update ───────────────────────────────────────

def apply_geometric_op_to_scan(
    scan: "Any",
    operation: str,
    params: dict | None = None,
    roi_set: "Any | None" = None,
) -> tuple["Any", "Any | None"]:
    """Apply a geometric transform to all planes of *scan*, updating *roi_set*.

    Parameters
    ----------
    scan
        A Scan object with a ``.planes`` list of 2-D arrays.
    operation
        One of: ``"flip_horizontal"``, ``"flip_vertical"``,
        ``"rotate_90_cw"`` (alias ``"rot90_cw"``),
        ``"rotate_180"`` (alias ``"rot180"``),
        ``"rotate_270_cw"`` (alias ``"rot270_cw"``),
        ``"rotate_arbitrary"``.

        Both vocabularies are accepted: the long form
        (``rotate_90_cw`` etc.) matches the kernel functions in
        ``probeflow.processing.geometry`` and the names emitted by
        ``processing_state_from_gui``; the short form (``rot90_cw``)
        matches the canonical names used by
        :class:`~probeflow.core.roi.ROI.transform`.  Review
        arch-backend #9 (2026-05-28) tightened this so passing either
        vocabulary now works uniformly through this function (previously
        only the short form was dispatched per-plane while the
        scan-range swap below already accepted both).
    params
        Extra parameters for the operation.  For ``rotate_arbitrary``:
        ``{"angle_degrees": float, "order": int}``.  For lossless ops: {}.
    roi_set
        Optional :class:`~probeflow.core.roi.ROISet`.  If provided, ROI
        coordinates are transformed in-place and invalidated ROIs are removed
        with a warning.

    Returns
    -------
    (scan, roi_set)
        *scan* is mutated in place (planes updated); *roi_set* is also mutated
        in place (invalidated ROIs removed) if provided.
    """
    import warnings as _warnings
    import probeflow.processing as _proc

    params = params or {}
    if not scan.planes:
        return scan, roi_set

    image_shape = scan.planes[0].shape  # (Ny, Nx)

    # Normalize the operation to the short form used by ROI.transform
    # so the rest of this function (including the LOSSLESS frozenset,
    # the per-plane dispatch, and the scan-range swap) only has to
    # know one vocabulary.  Aliases mirror those in
    # probeflow.core.roi.ROISet.transform_all.
    _LONG_TO_SHORT = {
        "rotate_90_cw": "rot90_cw",
        "rotate_180": "rot180",
        "rotate_270_cw": "rot270_cw",
    }
    canonical_op = _LONG_TO_SHORT.get(operation, operation)

    _LOSSLESS = frozenset({
        "flip_horizontal", "flip_vertical",
        "rot90_cw", "rot180", "rot270_cw",
    })

    for i, plane in enumerate(scan.planes):
        if canonical_op == "flip_horizontal":
            scan.planes[i] = _proc.flip_horizontal(plane)
        elif canonical_op == "flip_vertical":
            scan.planes[i] = _proc.flip_vertical(plane)
        elif canonical_op == "rot90_cw":
            scan.planes[i] = _proc.rotate_90_cw(plane)
        elif canonical_op == "rot180":
            scan.planes[i] = _proc.rotate_180(plane)
        elif canonical_op == "rot270_cw":
            scan.planes[i] = _proc.rotate_270_cw(plane)
        elif canonical_op == "rotate_arbitrary":
            scan.planes[i] = _proc.rotate_arbitrary(
                plane,
                angle_degrees=float(params.get("angle_degrees", 0.0)),
                order=int(params.get("order", 1)),
            )
        else:
            raise ValueError(
                f"apply_geometric_op_to_scan: unknown operation {operation!r}"
            )

    # Swap scan_range_m for operations that transpose width and height.
    # 90° and 270° rotations exchange the physical X and Y extents;
    # flips and 180° rotation leave the aspect ratio unchanged.
    if canonical_op in ("rot90_cw", "rot270_cw"):
        w, h = scan.scan_range_m
        scan.scan_range_m = (h, w)
    elif canonical_op == "flip_horizontal" and hasattr(scan, "plane_names"):
        scan.plane_names = [
            _swap_forward_backward_label(name)
            for name in getattr(scan, "plane_names", [])
        ]

    if roi_set is not None:
        # transform_all accepts both vocabularies via its own alias map.
        invalidated = roi_set.transform_all(operation, params, image_shape)
        if canonical_op in _LOSSLESS and invalidated:
            raise RuntimeError(
                f"Internal error: lossless operation {operation!r} invalidated "
                f"{len(invalidated)} ROI(s). This is a bug in ROI.transform()."
            )
        if canonical_op == "rotate_arbitrary" and invalidated:
            roi_set.rois = [r for r in roi_set.rois if r.id not in set(invalidated)]
            if getattr(roi_set, "active_roi_id", None) in set(invalidated):
                roi_set.active_roi_id = None
            _warnings.warn(
                f"rotate_arbitrary invalidated {len(invalidated)} ROI(s). "
                "They have been removed from the ROISet.",
                UserWarning,
                stacklevel=2,
            )

    return scan, roi_set


def _swap_forward_backward_label(label: str) -> str:
    """Swap scan-direction words in a plane label after horizontal mirroring."""
    import re

    placeholders = {
        "forward": "__PROBEFLOW_DIR_FORWARD__",
        "backward": "__PROBEFLOW_DIR_BACKWARD__",
        "fwd": "__PROBEFLOW_DIR_FWD__",
        "bwd": "__PROBEFLOW_DIR_BWD__",
    }
    replacements = {
        "__PROBEFLOW_DIR_FORWARD__": "backward",
        "__PROBEFLOW_DIR_BACKWARD__": "forward",
        "__PROBEFLOW_DIR_FWD__": "bwd",
        "__PROBEFLOW_DIR_BWD__": "fwd",
    }

    out = str(label)
    for word, placeholder in placeholders.items():
        out = re.sub(
            rf"\b{word}\b",
            lambda _match, placeholder=placeholder: placeholder,
            out,
            flags=re.IGNORECASE,
        )
    for placeholder, replacement in replacements.items():
        out = out.replace(placeholder, replacement)
    return out
