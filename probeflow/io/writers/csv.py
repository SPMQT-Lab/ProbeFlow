"""Export a :class:`probeflow.core.scan_model.Scan` plane as a 2-D CSV grid.

The file holds the raw array values in their physical units, one row per
scan line.  Two header-comment lines are prepended (starting with ``#``) to
record the pixel dimensions, scan range, and units — downstream tools that
strip ``#`` comments (pandas, numpy.loadtxt) ignore them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.io.common import check_output_available, check_overwrite
from probeflow.provenance.export import (
    build_scan_export_provenance,
    check_provenance_sidecar_collisions,
    write_provenance_sidecars,
)


def write_csv(
    scan,
    out_path,
    plane_idx: int = 0,
    *,
    delimiter: str = ",",
    fmt: str = "%.6e",
    provenance=None,
    overwrite: bool = False,
    overwrite_sidecars: bool = False,
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
    if provenance is None:
        provenance = build_scan_export_provenance(
            scan,
            channel_index=plane_idx,
            display_state={"delimiter": delimiter, "fmt": fmt},
            export_kind="csv",
            output_path=out_path,
        )
    check_provenance_sidecar_collisions(
        out_path,
        legacy=hasattr(provenance, "to_dict"),
        overwrite=overwrite_sidecars,
    )
    check_output_available(out_path, overwrite=overwrite)

    w_m, h_m = scan.scan_range_m
    Ny, Nx = arr.shape
    unit = scan.plane_units[plane_idx] if plane_idx < len(scan.plane_units) else ""
    name = scan.plane_names[plane_idx] if plane_idx < len(scan.plane_names) else f"plane {plane_idx}"
    source_name = scan.source_path.name if scan.source_path is not None else "unknown"

    header = (
        f"plane={name} units={unit} "
        f"Nx={Nx} Ny={Ny} "
        f"width_m={w_m:.6e} height_m={h_m:.6e} "
        f"source={source_name}"
    )
    np.savetxt(out_path, arr, fmt=fmt, delimiter=delimiter,
               header=header, comments="# ")
    write_provenance_sidecars(
        out_path,
        provenance,
        export_format="csv",
        overwrite=overwrite_sidecars,
    )
