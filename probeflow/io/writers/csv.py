"""Export a :class:`probeflow.core.scan_model.Scan` plane as a 2-D CSV grid.

The file holds the raw array values in their physical units, one row per
scan line.  Two header-comment lines are prepended (starting with ``#``) to
record the pixel dimensions, scan range, and units — downstream tools that
strip ``#`` comments (pandas, numpy.loadtxt) ignore them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.io.common import check_overwrite


def write_csv(
    scan,
    out_path,
    plane_idx: int = 0,
    *,
    delimiter: str = ",",
    fmt: str = "%.6e",
) -> None:
    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    if plane_idx < 0 or plane_idx >= scan.n_planes:
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{scan.n_planes} plane(s)"
        )

    arr = scan.planes[plane_idx]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w_m, h_m = scan.scan_range_m
    Ny, Nx = arr.shape
    unit = scan.plane_units[plane_idx] if plane_idx < len(scan.plane_units) else ""
    name = scan.plane_names[plane_idx] if plane_idx < len(scan.plane_names) else f"plane {plane_idx}"

    header = (
        f"plane={name} units={unit} "
        f"Nx={Nx} Ny={Ny} "
        f"width_m={w_m:.6e} height_m={h_m:.6e} "
        f"source={scan.source_path.name}"
    )
    np.savetxt(out_path, arr, fmt=fmt, delimiter=delimiter,
               header=header, comments="# ")
