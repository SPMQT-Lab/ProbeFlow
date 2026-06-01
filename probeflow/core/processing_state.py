"""Canonical ``ProcessingState`` dataclass — domain model only.

Moved out of :mod:`probeflow.processing.state` to break a lazy-import cycle:
:class:`probeflow.core.scan_model.Scan` needs ``ProcessingState`` for its
``processing_state`` attribute but the rest of ``probeflow.processing.state``
(``apply_processing_state``, ROI helpers, operand resolvers, …) depends on
the rest of ``probeflow.processing`` and would create an import-time cycle
if ``core`` imported it at module load time.

``probeflow.processing.state`` re-exports the symbols defined here so the
historical ``from probeflow.processing.state import ProcessingState`` import
path stays stable (review arch-backend #14).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


# ── Supported operations (must match probeflow.processing function names) ─────

_SUPPORTED_OPS: frozenset[str] = frozenset({
    "remove_bad_lines",
    "align_rows",
    "plane_bg",
    "stm_line_bg",
    "stm_background",
    "facet_level",
    "smooth",
    "gaussian_high_pass",
    "edge_detect",
    "fourier_filter",
    "fft_soft_border",
    "periodic_notch_filter",
    "mains_pickup_suppression",
    "inverse_fft_filter",
    "linear_undistort",
    "affine_lattice_correction",
    "arithmetic",
    "set_zero_point",
    "set_zero_plane",
    "roi",
    "flip_horizontal",
    "flip_vertical",
    "rotate_90_cw",
    "rotate_180",
    "rotate_270_cw",
    "rotate_arbitrary",
    "shear",
    "scale_image",
    "image_threshold",
    "quantize_bit_depth",
})

_ROI_ELIGIBLE_OPS: frozenset[str] = frozenset({
    "smooth",
    "gaussian_high_pass",
    "edge_detect",
    "fourier_filter",
    "fft_soft_border",
    "arithmetic",
})


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingStep:
    """One numerical processing operation applied to scan data."""

    op: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in _SUPPORTED_OPS:
            raise ValueError(
                f"Unknown processing operation {self.op!r}. "
                f"Supported operations: {sorted(_SUPPORTED_OPS)}"
            )

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
    probeflow_version: str | None = field(default=None, compare=False, repr=False)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict.

        Example output::

            {
              "probeflow_version": "1.2.3",
              "steps": [
                {"op": "align_rows", "params": {"method": "median"}},
                {"op": "plane_bg",   "params": {"order": 1}}
              ]
            }
        """
        try:
            from probeflow import __version__ as _pf_version
        except ImportError:
            _pf_version = None
        return {
            "probeflow_version": _pf_version,
            "steps": [
                {"op": step.op, "params": deepcopy(step.params)}
                for step in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingState":
        """Deserialise from the dict produced by :meth:`to_dict`."""
        import warnings
        stored_version = data.get("probeflow_version")
        if stored_version is not None:
            try:
                from probeflow import __version__ as _pf_version
                stored_major = int(str(stored_version).split(".")[0])
                current_major = int(str(_pf_version).split(".")[0])
                if stored_major != current_major:
                    warnings.warn(
                        f"Processing state was saved with probeflow {stored_version!r} "
                        f"but current version is {_pf_version!r}. "
                        "Results may differ.",
                        UserWarning,
                        stacklevel=2,
                    )
            except (ValueError, AttributeError, ImportError):
                pass
        steps = []
        for item in data.get("steps", []):
            steps.append(ProcessingStep(
                op=str(item["op"]),
                params=deepcopy(dict(item.get("params", {}))),
            ))
        return cls(steps=steps, probeflow_version=stored_version)


__all__ = [
    "ProcessingState",
    "ProcessingStep",
    "_ROI_ELIGIBLE_OPS",
    "_SUPPORTED_OPS",
]
