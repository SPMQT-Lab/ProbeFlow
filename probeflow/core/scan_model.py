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
from typing import Any, List, Tuple

import numpy as np


PLANE_CANON_NAMES: tuple[str, ...] = (
    "Z forward", "Z backward", "Current forward", "Current backward",
)
PLANE_CANON_UNITS: tuple[str, ...] = ("m", "m", "A", "A")


@dataclass
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
        Physical ``(width_m, height_m)``.
    source_path
        Absolute path to the file we loaded from.
    source_format
        ``"sxm"`` | ``"dat"`` identifies the reader that produced this Scan.
    """

    planes: List[np.ndarray]
    plane_names: List[str]
    plane_units: List[str]
    plane_synthetic: List[bool]
    header: dict
    scan_range_m: Tuple[float, float]
    source_path: Path
    source_format: str
    processing_history: List[dict] = field(default_factory=list)
    experiment_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_planes(self) -> int:
        return len(self.planes)

    @property
    def dims(self) -> Tuple[int, int]:
        """Scan dimensions as ``(Nx, Ny)`` (matches Nanonis SCAN_PIXELS)."""
        if not self.planes:
            return (0, 0)
        Ny, Nx = self.planes[0].shape
        return (Nx, Ny)

    def save_sxm(self, out_path) -> None:
        """Write this Scan to a Nanonis ``.sxm`` file."""
        from probeflow.io.writers.sxm import write_sxm
        write_sxm(self, out_path)

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
    ) -> None:
        """Render one plane to a colourised PNG with an optional scale bar."""
        from probeflow.io.writers.png import write_png
        write_png(
            self, out_path, plane_idx=plane_idx,
            colormap=colormap, clip_low=clip_low, clip_high=clip_high,
            add_scalebar=add_scalebar,
            scalebar_unit=scalebar_unit, scalebar_pos=scalebar_pos,
            provenance=provenance,
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
