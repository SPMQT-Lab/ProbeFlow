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
    "roi",
})

_ROI_ELIGIBLE_OPS: frozenset[str] = frozenset({
    "smooth",
    "gaussian_high_pass",
    "edge_detect",
    "fourier_filter",
    "fft_soft_border",
})


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
            a = _proc.remove_bad_lines(a)
        elif step.op == "align_rows":
            a = _proc.align_rows(a, method=p.get("method", "median"))
        elif step.op == "plane_bg":
            a = _proc.subtract_background(
                a,
                order=int(p.get("order", 1)),
                step_tolerance=bool(p.get("step_tolerance", False)),
                fit_rect=p.get("fit_rect"),
            )
        elif step.op == "stm_line_bg":
            a = _proc.stm_line_background(
                a,
                mode=str(p.get("mode", "step_tolerant")),
            )
        elif step.op == "facet_level":
            a = _proc.facet_level(a)
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
            rect = p.get("rect", ())
            try:
                x0, y0, x1, y1 = [int(v) for v in rect]
            except (TypeError, ValueError):
                continue
            Ny, Nx = a.shape
            x0 = max(0, min(Nx - 1, x0))
            x1 = max(0, min(Nx - 1, x1))
            y0 = max(0, min(Ny - 1, y0))
            y1 = max(0, min(Ny - 1, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros(a.shape, dtype=bool)
            mask[y0:y1 + 1, x0:x1 + 1] = True
            a = _proc.patch_interpolate(
                a,
                mask,
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
        elif step.op == "roi":
            try:
                nested = ProcessingStep.from_dict(p.get("step", {}))
            except (KeyError, TypeError, ValueError):
                continue
            if nested.op not in _ROI_ELIGIBLE_OPS:
                continue
            rect = p.get("rect", ())
            try:
                x0, y0, x1, y1 = [int(v) for v in rect]
            except (TypeError, ValueError):
                continue
            Ny, Nx = a.shape
            x0 = max(0, min(Nx - 1, x0))
            x1 = max(0, min(Nx - 1, x1))
            y0 = max(0, min(Ny - 1, y0))
            y1 = max(0, min(Ny - 1, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = a[y0:y1 + 1, x0:x1 + 1]
            processed = apply_processing_state(crop, ProcessingState(steps=[nested]))
            a = a.copy()
            a[y0:y1 + 1, x0:x1 + 1] = processed
        else:
            raise ValueError(
                f"Unknown processing operation {step.op!r}. "
                f"Supported: {sorted(_SUPPORTED_OPS)}"
            )

    return a
