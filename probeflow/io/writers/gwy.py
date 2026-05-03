"""Write a :class:`probeflow.core.scan_model.Scan` to a Gwyddion ``.gwy`` file.

This uses the optional ``gwyfile`` package to serialise a top-level
``GwyContainer`` with one ``GwyDataField`` for the requested scan plane.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from probeflow.io.common import check_overwrite
from probeflow.provenance.export import build_scan_export_provenance
from probeflow.core.scan_model import Scan


def _import_gwyfile():
    """Import the optional Gwyddion serialization helpers."""
    try:
        from gwyfile.objects import GwyContainer, GwyDataField
    except ImportError as exc:
        raise ImportError(
            "Writing .gwy files requires the optional 'gwyfile' package. "
            "Install it with `pip install gwyfile`."
        ) from exc
    return GwyContainer, GwyDataField


def _plane_meta(
    GwyContainer,
    scan: Scan,
    plane_idx: int,
    plane_name: str,
    plane_unit: str,
    visible: bool,
    out_path=None,
):
    """Build a small metadata container for the exported plane."""
    prov = build_scan_export_provenance(
        scan,
        channel_index=plane_idx,
        channel_name=plane_name,
        export_kind="gwy",
        output_path=out_path,
    )
    meta = GwyContainer()
    meta["ProbeFlow source path"] = (
        str(scan.source_path) if scan.source_path is not None else ""
    )
    meta["ProbeFlow source format"] = str(scan.source_format)
    meta["ProbeFlow plane index"] = int(plane_idx)
    meta["ProbeFlow plane name"] = str(plane_name)
    meta["ProbeFlow plane unit"] = str(plane_unit)
    meta["ProbeFlow visible on open"] = bool(visible)
    meta["ProbeFlow scan width (m)"] = float(scan.scan_range_m[0])
    meta["ProbeFlow scan height (m)"] = float(scan.scan_range_m[1])
    meta["ProbeFlow num planes"] = int(scan.n_planes)
    meta["ProbeFlow export provenance"] = json.dumps(prov.to_dict(), sort_keys=True)
    meta["ProbeFlow processing state"] = json.dumps(
        prov.processing_state,
        sort_keys=True,
    )
    meta["ProbeFlow processing state hash"] = str(prov.processing_state_hash)
    meta["ProbeFlow processing steps"] = int(
        len(prov.processing_state.get("steps", []))
    )
    meta["ProbeFlow source id"] = str(prov.source_id or "")
    meta["ProbeFlow channel id"] = str(prov.channel_id or "")
    meta["ProbeFlow artifact id"] = str(prov.artifact_id or "")
    return meta


def write_gwy(
    scan: Scan,
    out_path,
    plane_idx: int = 0,
    *,
    include_meta: bool = True,
) -> None:
    """Write one plane of *scan* to a Gwyddion ``.gwy`` file.

    Parameters
    ----------
    scan:
        The ProbeFlow scan object to export.
    out_path:
        Destination path.
    plane_idx:
        Plane to mark visible when Gwyddion opens the file.
    include_meta:
        If true, add a small metadata container under ``/0/meta``.
    """
    out_path = Path(out_path)
    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    if plane_idx < 0 or plane_idx >= scan.n_planes:
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{scan.n_planes} plane(s)"
        )

    GwyContainer, GwyDataField = _import_gwyfile()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    container = GwyContainer()
    container.filename = str(out_path)

    width_m, height_m = scan.scan_range_m
    width_m = float(width_m)
    height_m = float(height_m)

    plane_name = (
        scan.plane_names[plane_idx]
        if plane_idx < len(scan.plane_names)
        else f"Plane {plane_idx}"
    )
    plane_unit = (
        scan.plane_units[plane_idx]
        if plane_idx < len(scan.plane_units)
        else ""
    )
    data = np.asarray(scan.planes[plane_idx], dtype=np.float64)
    field = GwyDataField(
        data,
        xreal=width_m,
        yreal=height_m,
        xoff=0.0,
        yoff=0.0,
        si_unit_xy="m",
        si_unit_z=plane_unit or None,
    )
    container["/0/data"] = field
    container["/0/data/title"] = str(plane_name)
    container["/0/data/visible"] = True
    if include_meta:
        container["/0/meta"] = _plane_meta(
            GwyContainer,
            scan,
            plane_idx,
            plane_name,
            plane_unit,
            True,
            out_path,
        )

    container.tofile(str(out_path))
