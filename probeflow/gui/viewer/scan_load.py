"""Qt-free scan loading for the viewer.

Accepts plain Python values, returns a plain dataclass — no Qt dependency.
Mirrors the domain block in ImageViewerDialog._load_current_source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from probeflow.provenance.records import ProcessingHistory

_FALLBACK_PLANE_NAMES = ["Z (fwd)", "Z (bwd)", "I (fwd)", "I (bwd)"]
_FALLBACK_PLANE_UNITS = ["m", "m", "A", "A"]


@dataclass
class ViewerScanData:
    """Plain-Python result of loading a scan for the viewer."""
    raw_arr: Optional[np.ndarray]
    plane_names: list[str]
    plane_units: list[str]
    scan_header: dict
    scan_range_m: Optional[tuple]
    scan_shape: Optional[tuple]
    source_format: str
    n_planes: int
    processing_history: Optional["ProcessingHistory"] = field(default=None)
    # Reader warnings (e.g. "payload contains 1 complete plane(s) but the
    # header indicates 4 — file may have been incompletely written"). The
    # readers degrade gracefully on partial files and record why; the viewer
    # must surface this or the user just sees missing channels.
    scan_warnings: list[str] = field(default_factory=list)


def load_scan_for_viewer(path: Path, channel_idx: int) -> ViewerScanData:
    """Load *path* and extract the data the viewer needs to display it.

    ``channel_idx`` is the requested plane index; it is clamped to the
    available range.  On failure returns a ``ViewerScanData`` with
    ``raw_arr=None`` and empty metadata — the caller shows an error state
    without needing to catch exceptions.
    """
    from probeflow.core.scan_loader import load_scan
    from probeflow.provenance.records import processing_history_from_scan

    try:
        scan = load_scan(path)
        idx = max(0, min(channel_idx, scan.n_planes - 1))
        raw_arr = scan.planes[idx] if scan.n_planes > 0 else None
        history = processing_history_from_scan(
            scan,
            channel_index=idx,
            channel_name=(
                scan.plane_names[idx]
                if idx < len(scan.plane_names) else None
            ),
            processing_state={"steps": []},
        )
        return ViewerScanData(
            raw_arr=raw_arr,
            plane_names=list(scan.plane_names),
            plane_units=list(scan.plane_units),
            scan_header=scan.header or {},
            scan_range_m=scan.scan_range_m,
            scan_shape=scan.planes[0].shape if scan.planes else None,
            source_format=scan.source_format,
            n_planes=scan.n_planes,
            processing_history=history,
            scan_warnings=[str(w) for w in (getattr(scan, "warnings", ()) or ())],
        )
    except Exception:
        return ViewerScanData(
            raw_arr=None,
            plane_names=list(_FALLBACK_PLANE_NAMES),
            plane_units=list(_FALLBACK_PLANE_UNITS),
            scan_header={},
            scan_range_m=None,
            scan_shape=None,
            source_format="",
            n_planes=0,
            processing_history=None,
        )


__all__ = ["ViewerScanData", "load_scan_for_viewer"]
