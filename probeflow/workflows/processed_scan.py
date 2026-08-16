"""Prepare a source Scan for processed-image export."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_processed_scan(
    path: Path,
    channel_idx: int,
    processed_plane: np.ndarray | None,
    processing_state,
    *,
    scan_range_m: tuple[float, float] | None = None,
):
    """Load a source scan and attach the values represented by the viewer."""
    from probeflow.core.scan_loader import load_scan
    from probeflow.processing.gui_adapter import processing_state_from_gui

    scan = load_scan(path)
    idx = max(0, min(channel_idx, scan.n_planes - 1))
    if processed_plane is None:
        if scan.n_planes == 0:
            raise ValueError("No image data loaded.")
        plane = scan.planes[idx]
    else:
        plane = processed_plane

    scan.planes[idx] = np.asarray(plane, dtype=np.float64).copy()
    if scan_range_m is not None:
        scan.scan_range_m = (float(scan_range_m[0]), float(scan_range_m[1]))
    state = processing_state_from_gui(processing_state or {})
    if state.steps:
        scan.record_processing_state(state)
    return scan, idx


__all__ = ["load_processed_scan"]
