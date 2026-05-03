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
    try:
        kind = str(geometry.get("kind", ""))
    except AttributeError:
        return None
    if kind == "rectangle":
        rect = _rect_from_geometry(shape, geometry)
        try:
            x0, y0, x1, y1 = _clamped_rect(shape, rect)
        except ValueError:
            return None
        mask = np.zeros(shape, dtype=bool)
        mask[y0:y1 + 1, x0:x1 + 1] = True
        return mask
    if kind == "ellipse":
        rect = _rect_from_geometry(shape, geometry)
        try:
            x0, y0, x1, y1 = _clamped_rect(shape, rect)
        except ValueError:
            return None
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max(0.5, (x1 - x0 + 1) / 2.0)
        ry = max(0.5, (y1 - y0 + 1) / 2.0)
        return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    if kind == "polygon":
        points = _points_from_geometry(shape, geometry)
        if len(points) < 3:
            return None
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        x = xx.astype(float) + 0.5
        y = yy.astype(float) + 0.5
        inside = np.zeros(shape, dtype=bool)
        xj, yj = points[-1]
        for xi, yi in points:
            crosses = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            )
            inside ^= crosses
            xj, yj = xi, yi
        return inside
    return None


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
                {"op": step.op, "params": dict(step.params)}
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
                params=dict(item.get("params", {})),
            ))
        return cls(steps=steps)


# ── Canonical apply function ──────────────────────────────────────────────────

def apply_processing_state(arr: np.ndarray, state: ProcessingState) -> np.ndarray:
    """Apply *state* steps in order to *arr*.

    Parameters
    ----------
    arr:
        Input 2-D numeric array (will not be mutated).
    state:
        Processing steps to apply.

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
            fit_mask = roi_geometry_mask(a.shape, fit_geometry) if fit_geometry else None
            a = _proc.subtract_background(
                a,
                order=int(p.get("order", 1)),
                step_tolerance=bool(p.get("step_tolerance", False)),
                fit_rect=p.get("fit_rect"),
                fit_mask=fit_mask,
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
            x0, y0, x1, y1 = bounds
            crop = a[y0:y1 + 1, x0:x1 + 1]
            processed = apply_processing_state(crop, ProcessingState(steps=[nested]))
            local_mask = mask[y0:y1 + 1, x0:x1 + 1]
            a = a.copy()
            target = a[y0:y1 + 1, x0:x1 + 1]
            target[local_mask] = processed[local_mask]
        else:
            raise ValueError(
                f"Unknown processing operation {step.op!r}. "
                f"Supported: {sorted(_SUPPORTED_OPS)}"
            )

    return a
