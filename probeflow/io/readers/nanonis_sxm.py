"""Reader for Nanonis ``.sxm`` files — returns a :class:`probeflow.core.scan_model.Scan`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.core.scan_model import Scan
from probeflow.io.sxm_io import (
    parse_sxm_header,
    read_all_sxm_planes,
    sxm_payload_plane_count,
    sxm_plane_metadata,
    sxm_scan_range,
)


def read_sxm(path) -> Scan:
    """Load a Nanonis ``.sxm`` into a Scan (display-oriented, SI units)."""
    path = Path(path)
    hdr, planes = read_all_sxm_planes(path, orient=True)
    if not planes:
        raise ValueError(f"{path}: no data planes could be read")

    names, units = sxm_plane_metadata(hdr, len(planes))
    synthetic = [False] * len(planes)
    scan_range_m = sxm_scan_range(hdr)

    return Scan(
        planes=[p.astype(np.float64) for p in planes],
        plane_names=names,
        plane_units=units,
        plane_synthetic=synthetic,
        header=dict(hdr),
        scan_range_m=scan_range_m,
        source_path=path,
        source_format="sxm",
    )


def read_sxm_metadata(path):
    """Return :class:`~probeflow.core.metadata.ScanMetadata` for a Nanonis ``.sxm``."""
    from probeflow.core.metadata import metadata_from_sxm_header

    path = Path(path)
    hdr = parse_sxm_header(path)
    n_planes = sxm_payload_plane_count(path, hdr)
    if n_planes <= 0:
        raise ValueError(f"{path}: no data planes could be read")
    return metadata_from_sxm_header(path, hdr, n_planes)
