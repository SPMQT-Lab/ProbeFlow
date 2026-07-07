"""Dataset Builder loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.core.scan_loader import load_scan


def load_scan_plane(path: str | Path, plane_index: int = 0):
    """Load one scan plane and return ``(scan, array, px_x_m, px_y_m)``."""

    scan = load_scan(path)
    if plane_index < 0 or plane_index >= scan.n_planes:
        raise ValueError(
            f"plane_index={plane_index} out of range for {scan.n_planes} plane(s)"
        )
    arr = np.asarray(scan.planes[plane_index], dtype=np.float64)
    ny, nx = arr.shape
    # REVIEW(2026-07-06, med): scan.scan_range_m can be None (formats/paths
    # without calibrated range) — this unpack then raises TypeError and takes
    # down the whole export loop. Guard and fall back to the 1e-10 default.
    width_m, height_m = scan.scan_range_m
    px_x_m = float(width_m) / nx if nx and width_m else 1e-10
    px_y_m = float(height_m) / ny if ny and height_m else 1e-10
    return scan, arr, px_x_m, px_y_m


def plane_sample_id(path: str | Path, plane_index: int) -> str:
    # REVIEW(2026-07-06, high): stem-only IDs collide across subfolders (the
    # queue walks recursively). Two scans named image_001.sxm in different
    # days' folders export to the SAME arrays/<id>.npy — with overwrite=True
    # the second silently replaces the first (the manifest then has two rows
    # pointing at one file); without it the export dies mid-run. Disambiguate
    # with a short hash of the relative path (or the parent folder name).
    p = Path(path)
    return f"{p.stem}_plane{int(plane_index)}"

