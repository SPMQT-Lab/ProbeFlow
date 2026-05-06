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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Supported operations (must match probeflow.processing function names) ─────

_SUPPORTED_OPS: frozenset[str] = frozenset({
    "remove_bad_lines",
    "align_rows",
    "plane_bg",
    "stm_line_bg",
    "facet_level",
    "smooth",
    "gaussian_high_pass",
    "edge_detect",
    "fourier_filter",
    "fft_soft_border",
    "periodic_notch_filter",
    "patch_interpolate",
    "linear_undistort",
    "set_zero_point",
    "set_zero_plane",
    "roi",
    "flip_horizontal",
    "flip_vertical",
    "rotate_90_cw",
    "rotate_180",
    "rotate_270_cw",
    "rotate_arbitrary",
})

_ROI_ELIGIBLE_OPS: frozenset[str] = frozenset({
    "smooth",
    "gaussian_high_pass",
    "edge_detect",
    "fourier_filter",
    "fft_soft_border",
})


def roi_geometry_mask(
    shape: tuple[int, int],
    geometry: dict[str, Any] | None,
) -> np.ndarray | None:
    """Return a boolean mask for a rectangle/ellipse/polygon ROI geometry."""
    if not geometry:
        return None
    from probeflow.core.roi import roi_from_legacy_geometry_dict
    roi = roi_from_legacy_geometry_dict(shape, geometry)
    if roi is None:
        return None
    return roi.to_mask(shape)


def _rect_from_geometry(shape: tuple[int, int], geometry: dict[str, Any]):
    for key in ("rect_px", "bounds_px", "rect"):
        rect = geometry.get(key)
        if rect is not None:
            try:
                if len(rect) == 4:
                    return rect
            except TypeError:
                pass
    bounds_frac = geometry.get("bounds_frac")
    if bounds_frac is None:
        return ()
    try:
        x0f, y0f, x1f, y1f = [float(v) for v in bounds_frac]
    except (TypeError, ValueError):
        return ()
    Ny, Nx = shape
    return (
        int(round(min(x0f, x1f) * (Nx - 1))),
        int(round(min(y0f, y1f) * (Ny - 1))),
        int(round(max(x0f, x1f) * (Nx - 1))),
        int(round(max(y0f, y1f) * (Ny - 1))),
    )


def roi_geometry_bounds(
    shape: tuple[int, int],
    geometry: dict[str, Any] | None,
) -> tuple[int, int, int, int] | None:
    """Return inclusive pixel bounds for an area ROI geometry."""

    mask = roi_geometry_mask(shape, geometry)
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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


def _points_from_geometry(
    shape: tuple[int, int],
    geometry: dict[str, Any],
) -> list[tuple[float, float]]:
    raw = geometry.get("points_px")
    if raw is None:
        raw = geometry.get("points")
    if raw is None and geometry.get("points_frac") is not None:
        Ny, Nx = shape
        points = []
        for item in geometry.get("points_frac", ()):
            try:
                points.append((
                    float(item[0]) * (Nx - 1),
                    float(item[1]) * (Ny - 1),
                ))
            except (TypeError, ValueError, IndexError):
                continue
        raw = points
    if raw is None:
        raw = ()
    points: list[tuple[float, float]] = []
    for item in raw:
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return points


def _clamped_rect(
    shape: tuple[int, int],
    rect,
) -> tuple[int, int, int, int]:
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in rect]
    except (TypeError, ValueError):
        raise ValueError("bad rect")
    Ny, Nx = shape
    x0 = max(0, min(Nx - 1, x0))
    x1 = max(0, min(Nx - 1, x1))
    y0 = max(0, min(Ny - 1, y0))
    y1 = max(0, min(Ny - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty rect")
    return x0, y0, x1, y1


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingStep:
    """One numerical processing operation applied to scan data."""

    op: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingStep":
        return cls(op=str(data["op"]), params=dict(data.get("params", {})))


@dataclass
class ProcessingState:
    """Ordered list of numerical processing steps.

    Represents operations that change the numerical image data.
    Does not include display-only settings such as colormap, vmin/vmax,
    percentile clipping, histogram state, or overlays.
    """

    steps: list[ProcessingStep] = field(default_factory=list)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict.

        Example output::

            {
              "steps": [
                {"op": "align_rows", "params": {"method": "median"}},
                {"op": "plane_bg",   "params": {"order": 1}}
              ]
            }
        """
        return {
            "steps": [
                {"op": step.op, "params": deepcopy(step.params)}
                for step in self.steps
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingState":
        """Deserialise from the dict produced by :meth:`to_dict`."""
        steps = []
        for item in data.get("steps", []):
            steps.append(ProcessingStep(
                op=str(item["op"]),
                params=deepcopy(dict(item.get("params", {}))),
            ))
        return cls(steps=steps)


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


# ── Canonical apply function ──────────────────────────────────────────────────

def apply_processing_state(
    arr: np.ndarray,
    state: "ProcessingState",
    roi_set: "Any | None" = None,
) -> np.ndarray:
    """Apply *state* steps in order to *arr*.

    Parameters
    ----------
    arr:
        Input 2-D numeric array (will not be mutated).
    state:
        Processing steps to apply.
    roi_set:
        Optional :class:`probeflow.core.roi.ROISet`.  When a ``roi`` step
        references an ROI by ``roi_id``, the ROI is looked up in this set at
        execution time.  If the ID is not found, the step is skipped with a
        warning.  Inline geometry (``rect`` / ``geometry`` params) still works
        without a ``roi_set``.

    Returns
    -------
    np.ndarray of float64, same shape as *arr*.

    Raises
    ------
    ValueError
        If a step contains an unrecognised operation name.
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
            )
        elif step.op == "align_rows":
            a = _proc.align_rows(a, method=p.get("method", "median"))
        elif step.op == "plane_bg":
            fit_geometry = p.get("fit_geometry")
            fit_mask_legacy = roi_geometry_mask(a.shape, fit_geometry) if fit_geometry else None
            # Resolve new ROI expression parameters (fit_roi, apply_roi, exclude_roi)
            fit_roi = _resolve_bg_roi_param(p, "fit_roi", a.shape, roi_set)
            apply_roi = _resolve_bg_roi_param(p, "apply_roi", a.shape, roi_set)
            exclude_roi = _resolve_bg_roi_param(p, "exclude_roi", a.shape, roi_set)
            a = _proc.subtract_background(
                a,
                order=int(p.get("order", 1)),
                fit_roi=fit_roi,
                apply_roi=apply_roi,
                exclude_roi=exclude_roi,
                step_tolerance=bool(p.get("step_tolerance", False)),
                fit_rect=p.get("fit_rect"),
                fit_mask=fit_mask_legacy,
            )
        elif step.op == "stm_line_bg":
            a = _proc.stm_line_background(
                a,
                mode=str(p.get("mode", "step_tolerant")),
            )
        elif step.op == "facet_level":
            a = _proc.facet_level(
                a,
                threshold_deg=float(p.get("threshold_deg", 3.0)),
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
        elif step.op == "patch_interpolate":
            geometry = p.get("geometry")
            if geometry:
                mask = roi_geometry_mask(a.shape, geometry)
            else:
                try:
                    x0, y0, x1, y1 = _clamped_rect(a.shape, p.get("rect", ()))
                except ValueError:
                    continue
                mask = np.zeros(a.shape, dtype=bool)
                mask[y0:y1 + 1, x0:x1 + 1] = True
            if mask is None or not mask.any():
                continue
            a = _proc.patch_interpolate(
                a,
                mask,
                method=str(p.get("method", "line_fit")),
                rim_px=int(p.get("rim_px", 20)),
                iterations=int(p.get("iterations", 200)),
            )
        elif step.op == "linear_undistort":
            a = _proc.linear_undistort(
                a,
                shear_x=float(p.get("shear_x", 0.0)),
                scale_y=float(p.get("scale_y", 1.0)),
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
            try:
                nested = ProcessingStep.from_dict(p.get("step", {}))
            except (KeyError, TypeError, ValueError):
                continue
            if nested.op not in _ROI_ELIGIBLE_OPS:
                continue
            # Three ways to specify the region:
            # 1. roi_id  — look up by UUID in the provided roi_set
            # 2. geometry — legacy dict format
            # 3. rect    — (x0, y0, x1, y1) pixel rect
            roi_id = p.get("roi_id")
            if roi_id is not None:
                if roi_set is None:
                    import warnings
                    warnings.warn(
                        f"roi step references roi_id={roi_id!r} but no roi_set "
                        "was passed to apply_processing_state — step skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                roi_obj = roi_set.get(roi_id)
                if roi_obj is None:
                    import warnings
                    warnings.warn(
                        f"roi_id={roi_id!r} not found in roi_set — step skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                mask = roi_obj.to_mask(a.shape)
                r0, r1, c0, c1 = roi_obj.bounds(a.shape)
                bounds = (c0, r0, c1, r1)  # (x0, y0, x1, y1)
            else:
                geometry = p.get("geometry")
                if geometry:
                    mask = roi_geometry_mask(a.shape, geometry)
                    bounds = roi_geometry_bounds(a.shape, geometry)
                else:
                    try:
                        x0, y0, x1, y1 = _clamped_rect(a.shape, p.get("rect", ()))
                    except ValueError:
                        continue
                    mask = np.zeros(a.shape, dtype=bool)
                    mask[y0:y1 + 1, x0:x1 + 1] = True
                    bounds = (x0, y0, x1, y1)
            if mask is None or bounds is None or not mask.any():
                continue
            a = apply_operation_with_optional_roi(
                a,
                lambda image, nested=nested: apply_processing_state(
                    image,
                    ProcessingState(steps=[nested]),
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
        else:
            raise ValueError(
                f"Unknown processing operation {step.op!r}. "
                f"Supported: {sorted(_SUPPORTED_OPS)}"
            )

    return a


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
        One of: "flip_horizontal", "flip_vertical", "rot90_cw", "rot180",
        "rot270_cw", "rotate_arbitrary".
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

    _LOSSLESS = frozenset({
        "flip_horizontal", "flip_vertical",
        "rot90_cw", "rot180", "rot270_cw",
    })

    for i, plane in enumerate(scan.planes):
        if operation == "flip_horizontal":
            scan.planes[i] = _proc.flip_horizontal(plane)
        elif operation == "flip_vertical":
            scan.planes[i] = _proc.flip_vertical(plane)
        elif operation == "rot90_cw":
            scan.planes[i] = _proc.rotate_90_cw(plane)
        elif operation == "rot180":
            scan.planes[i] = _proc.rotate_180(plane)
        elif operation == "rot270_cw":
            scan.planes[i] = _proc.rotate_270_cw(plane)
        elif operation == "rotate_arbitrary":
            scan.planes[i] = _proc.rotate_arbitrary(
                plane,
                angle_degrees=float(params.get("angle_degrees", 0.0)),
                order=int(params.get("order", 1)),
            )
        else:
            raise ValueError(f"apply_geometric_op_to_scan: unknown operation {operation!r}")

    if roi_set is not None:
        invalidated = roi_set.transform_all(operation, params, image_shape)
        if operation in _LOSSLESS and invalidated:
            raise RuntimeError(
                f"Internal error: lossless operation {operation!r} invalidated "
                f"{len(invalidated)} ROI(s). This is a bug in ROI.transform()."
            )
        if operation == "rotate_arbitrary" and invalidated:
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
