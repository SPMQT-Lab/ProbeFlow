"""
A vendor-agnostic representation of an STM topography scan.

The :class:`Scan` dataclass holds image planes in *display orientation*
(origin top-left, forward scan direction left-to-right) and SI physical units,
together with vendor metadata.  It is produced by the readers in
``probeflow.readers`` and consumed by the writers in ``probeflow.writers``.

Point-spectroscopy files (Createc ``.VERT``) are a different shape of data
and live in ``probeflow.spec_io`` — don't shoehorn them into Scan.

Example
-------
>>> from probeflow.scan import load_scan
>>> scan = load_scan("some_file.sxm")        # or "some_file.dat"
>>> scan.n_planes, scan.dims                 # (4, (256, 256))
>>> scan.save_png("out.png", plane_idx=0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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
        List of 2-D float64 arrays in SI units.  By convention plane 0 is
        Z forward, plane 1 is Z backward, plane 2 is Current forward, plane 3
        is Current backward.  Each array is oriented for display: origin at
        the top-left and scan direction left-to-right.
    plane_names, plane_units
        Parallel lists describing each plane.
    plane_synthetic
        True when a plane was synthesised (e.g. backward mirrored from forward
        because the original file only had forward channels).
    header
        The raw vendor header dict (source-specific keys).
    scan_range_m
        Physical ``(width_m, height_m)``.
    source_path
        Absolute path to the file we loaded from.
    source_format
        ``"sxm"`` | ``"dat"`` — identifies the reader that produced this Scan.
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

    # ── Derived properties ──────────────────────────────────────────────────

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

    # ── Convenience writers (thin wrappers around probeflow.writers.*) ──────

    def save_sxm(self, out_path) -> None:
        """Write this Scan to a Nanonis ``.sxm`` file."""
        from probeflow.writers.sxm import write_sxm
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
    ) -> None:
        """Render one plane to a colourised PNG with an optional scale bar."""
        from probeflow.writers.png import write_png
        write_png(
            self, out_path, plane_idx=plane_idx,
            colormap=colormap, clip_low=clip_low, clip_high=clip_high,
            add_scalebar=add_scalebar,
            scalebar_unit=scalebar_unit, scalebar_pos=scalebar_pos,
        )

    def save_pdf(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Render one plane to a publication-ready PDF (matplotlib)."""
        from probeflow.writers.pdf import write_pdf
        write_pdf(self, out_path, plane_idx=plane_idx, **kwargs)

    def save_csv(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Dump one plane as a 2-D CSV grid (with a ``#``-comment header)."""
        from probeflow.writers.csv import write_csv
        write_csv(self, out_path, plane_idx=plane_idx, **kwargs)

    def save(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Suffix-driven save: ``.sxm`` / ``.png`` / ``.pdf`` / ``.csv``."""
        from probeflow.writers import save_scan
        save_scan(self, out_path, plane_idx=plane_idx, **kwargs)


# ── Dispatcher ──────────────────────────────────────────────────────────────

SUPPORTED_SUFFIXES: tuple[str, ...] = (".sxm", ".dat")


def load_scan(path) -> Scan:
    """Load an STM scan file, dispatching on its content signature.

    Supported formats:
      * ``.sxm`` — Nanonis topography
      * ``.dat`` — Createc topography

    Point-spectroscopy files (Createc ``.VERT`` and Nanonis ``.dat`` spec)
    are not scans — use :func:`probeflow.spec_io.read_spec_file` instead.
    """
    from probeflow.file_type import FileType, sniff_file_type

    p = Path(path)
    ft = sniff_file_type(p)

    if ft == FileType.NANONIS_IMAGE:
        from probeflow.readers.sxm import read_sxm
        return read_sxm(p)
    if ft == FileType.CREATEC_IMAGE:
        from probeflow.readers.dat import read_dat
        return read_dat(p)
    if ft == FileType.NANONIS_SPEC:
        raise ValueError(
            f"{p.name}: this is a Nanonis spectroscopy file — "
            "use probeflow.spec_io.read_spec_file to load it."
        )
    if ft == FileType.CREATEC_SPEC:
        raise ValueError(
            f"{p.name}: this is a Createc .VERT spectroscopy file — "
            "use probeflow.spec_io.read_spec_file to load it."
        )

    raise ValueError(
        f"Unsupported or unrecognised scan file: {p}. "
        f"Supported: {', '.join(SUPPORTED_SUFFIXES)}"
    )
