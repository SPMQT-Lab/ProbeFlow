"""Pure data model for loaded STM image scans.

This module deliberately contains no top-level reader, writer, validation, GUI,
or CLI imports. Keep it that way: readers, writers, validation, metadata, CLI,
and GUI can all depend on the :class:`Scan` model without creating import
cycles. The ``save_*`` convenience methods retain lazy writer imports only for
public API compatibility.

The public compatibility import remains ``from probeflow.core.scan_model import Scan``;
new internal code that only needs the dataclass should prefer importing from
``probeflow.core.scan_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from probeflow.core.processing_state import ProcessingState


PLANE_CANON_NAMES: tuple[str, ...] = (
    "Z forward", "Z backward", "Current forward", "Current backward",
)
PLANE_CANON_UNITS: tuple[str, ...] = ("m", "m", "A", "A")



@dataclass(init=False)
class Scan:
    """A parsed STM topography scan with all planes in display orientation.

    Attributes
    ----------
    planes
        List of 2-D float64 arrays. Canonical STM scans use SI units and the
        public order Z forward, Z backward, Current forward, Current backward.
        Selected-channel or auxiliary layouts preserve native channel order
        with units recorded in ``plane_units``. Each array is oriented for
        display: origin at the top-left and scan direction left-to-right.
    plane_names, plane_units
        Parallel lists describing each plane.
    plane_synthetic
        True when a plane was synthesised, e.g. backward mirrored from forward
        because the original file only had forward channels.
    header
        The raw vendor header dict.
    scan_range_m
        Physical ``(total_width_m, total_height_m)`` — the complete scan range in metres.
        Pixel-to-metre conversion: ``pixel_size = scan_range_m / image_shape``.
    source_path
        Absolute path to the file we loaded from.
    source_format
        ``"sxm"`` | ``"dat"`` | ``"sm4"`` identifies the reader that produced
        this Scan.
    warnings
        Reader or conversion warnings that should stay attached to the loaded
        scan for provenance and GUI surfacing.
    """

    planes: list[np.ndarray]
    plane_names: list[str]
    plane_units: list[str]
    plane_synthetic: list[bool]
    header: dict
    scan_range_m: tuple[float, float]
    source_path: Path
    source_format: str
    experiment_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    _processing_state: Any = field(default=None, init=False, repr=False)
    _processing_history_timestamps: list[str | None] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __init__(
        self,
        planes: list[np.ndarray],
        plane_names: list[str],
        plane_units: list[str],
        plane_synthetic: list[bool],
        header: dict,
        scan_range_m: tuple[float, float],
        source_path: Path,
        source_format: str,
        processing_state: Any | None = None,
        processing_history: list[dict] | None = None,
        experiment_metadata: dict[str, Any] | None = None,
        warnings: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.planes = planes
        self.plane_names = plane_names
        self.plane_units = plane_units
        self.plane_synthetic = plane_synthetic
        self.header = header
        self.scan_range_m = scan_range_m
        self.source_path = source_path
        self.source_format = source_format
        self.experiment_metadata = dict(experiment_metadata or {})
        self.warnings = tuple(str(w) for w in warnings or ())
        self._processing_state = None
        self._processing_history_timestamps = []
        if processing_state is not None and processing_history is not None:
            raise ValueError(
                "Pass either processing_state or processing_history, not both."
            )
        self.processing_state = processing_state
        if processing_history is not None:
            self.processing_history = processing_history

    @staticmethod
    def _empty_processing_state():
        return ProcessingState()

    @staticmethod
    def _coerce_processing_state(value):
        if value is None:
            return ProcessingState()
        if isinstance(value, ProcessingState):
            return ProcessingState.from_dict(value.to_dict())
        if isinstance(value, dict):
            return ProcessingState.from_dict(value)
        if hasattr(value, "to_dict"):
            return ProcessingState.from_dict(value.to_dict())
        raise TypeError(f"Unsupported processing_state: {type(value).__name__}")

    @property
    def processing_state(self):
        """Canonical numerical processing state for this scan."""
        if self._processing_state is None:
            self._processing_state = self._empty_processing_state()
        return self._processing_state

    @processing_state.setter
    def processing_state(self, value) -> None:
        state = self._coerce_processing_state(value)
        self._processing_state = state
        self._processing_history_timestamps = [None] * len(state.steps)

    def record_processing_state(
        self,
        state,
        *,
        timestamp: str | None = None,
    ) -> None:
        """Append canonical processing steps to this scan."""
        from datetime import datetime

        state = self._coerce_processing_state(state)
        if not state.steps:
            return
        current = self.processing_state
        self._processing_state = ProcessingState(steps=[
            *current.steps,
            *state.steps,
        ])
        ts = timestamp or datetime.now().isoformat()
        self._processing_history_timestamps.extend([ts] * len(state.steps))

    def _processing_history_entries(self) -> list[dict[str, Any]]:
        from probeflow.processing.history import processing_history_entries_from_state
        return processing_history_entries_from_state(
            self.processing_state,
            timestamps=self._processing_history_timestamps,
        )

    @property
    def processing_history(self) -> list[dict]:
        """History entries derived from ``processing_state``."""
        return list(self._processing_history_entries())

    @processing_history.setter
    def processing_history(self, entries: list[dict] | None) -> None:
        from probeflow.processing.history import processing_state_from_history
        self._processing_state = processing_state_from_history(entries)
        self._processing_history_timestamps = [
            entry.get("timestamp") if isinstance(entry, dict) else None
            for entry in (entries or [])
            if isinstance(entry, dict) and entry.get("op")
        ]

    @property
    def n_planes(self) -> int:
        return len(self.planes)

    @property
    def dims(self) -> tuple[int, int]:
        """Scan dimensions as ``(Nx, Ny)`` (matches Nanonis SCAN_PIXELS)."""
        if not self.planes:
            return (0, 0)
        Ny, Nx = self.planes[0].shape
        return (Nx, Ny)

    def save_sxm(self, out_path, **kwargs) -> None:
        """Write this Scan to a Nanonis ``.sxm`` file."""
        from probeflow.io.writers.sxm import write_sxm
        write_sxm(self, out_path, **kwargs)

    def save_png(
        self,
        out_path,
        plane_idx: int = 0,
        *,
        colormap: str = "gray",
        clip_low: float = 1.0,
        clip_high: float = 99.0,
        add_scalebar: bool = True,
        scalebar_unit: str = "nm",
        scalebar_pos: str = "bottom-right",
        provenance=None,
        overwrite: bool = False,
        overwrite_sidecars: bool = False,
    ) -> None:
        """Render one plane to a colourised PNG with an optional scale bar."""
        from probeflow.io.writers.png import write_png
        write_png(
            self, out_path, plane_idx=plane_idx,
            colormap=colormap, clip_low=clip_low, clip_high=clip_high,
            add_scalebar=add_scalebar,
            scalebar_unit=scalebar_unit, scalebar_pos=scalebar_pos,
            provenance=provenance,
            overwrite=overwrite,
            overwrite_sidecars=overwrite_sidecars,
        )

    def save_pdf(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Render one plane to a publication-ready PDF."""
        from probeflow.io.writers.pdf import write_pdf
        write_pdf(self, out_path, plane_idx=plane_idx, **kwargs)

    def save_csv(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Dump one plane as a 2-D CSV grid."""
        from probeflow.io.writers.csv import write_csv
        write_csv(self, out_path, plane_idx=plane_idx, **kwargs)

    def save_gwy(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Write this Scan to a Gwyddion ``.gwy`` file."""
        from probeflow.io.writers.gwy import write_gwy
        write_gwy(self, out_path, plane_idx=plane_idx, **kwargs)

    def save(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Suffix-driven save: ``.sxm`` / ``.gwy`` / ``.png`` / ``.pdf`` / ``.csv``."""
        from probeflow.io.writers import save_scan
        save_scan(self, out_path, plane_idx=plane_idx, **kwargs)
