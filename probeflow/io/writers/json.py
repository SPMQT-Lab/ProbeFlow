"""JSON writer for analysis results (particles, detections, classifications,
lattice). Serialises with full scan provenance so results are reproducible.

Unlike the other writers in this package, the JSON writer does **not** consume
a :class:`Scan` plane — it writes a list of analysis objects (dataclasses) with
a ``to_dict()`` method, plus a ``meta`` block identifying the source scan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


def _encode(obj):
    """Small fallback encoder for numpy scalars / arrays."""
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_json(
    out_path,
    items: Iterable,
    *,
    kind: str,
    scan=None,
    extra_meta: Optional[dict] = None,
    provenance=None,  # ExportProvenance | None
) -> None:
    """Write a list of dataclass-like objects to JSON with scan provenance.

    Parameters
    ----------
    out_path
        Destination ``.json`` file.
    items
        Iterable of objects with a ``to_dict()`` method (Particles, Detections,
        Classifications, or LatticeResult).
    kind
        Short identifier stored under ``meta.kind`` — e.g. ``"particles"``,
        ``"detections"``, ``"classifications"``, ``"lattice"``.
    scan
        Optional :class:`probeflow.core.scan_model.Scan` whose identity is recorded in
        the ``meta`` block.
    extra_meta
        Additional metadata merged into the ``meta`` block.
    provenance
        Optional :class:`probeflow.provenance.export.ExportProvenance`.
        When provided, ``export_provenance``, ``processing_state``, and
        ``display_state`` are added as top-level keys.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict = {"kind": kind}
    if scan is not None:
        w_m, h_m = scan.scan_range_m
        Nx, Ny = scan.dims
        meta.update({
            "source_path": str(scan.source_path),
            "source_format": scan.source_format,
            "scan_range_m": [float(w_m), float(h_m)],
            "pixels": [int(Nx), int(Ny)],
            "pixel_size_x_m": float(w_m) / Nx if Nx else None,
            "pixel_size_y_m": float(h_m) / Ny if Ny else None,
            "plane_names": list(scan.plane_names),
            "plane_units": list(scan.plane_units),
        })
        # Include processing history when no ExportProvenance is provided
        # (ExportProvenance carries it at the top level via processing_state).
        if scan.processing_history and provenance is None:
            meta["processing_state"] = {
                "steps": [
                    {"op": h["op"], "params": h["params"]}
                    for h in scan.processing_history
                ]
            }
    if extra_meta:
        meta.update(extra_meta)

    payload: dict = {"meta": meta}

    if provenance is not None:
        payload["export_provenance"] = provenance.to_dict()
        payload["processing_state"]  = provenance.processing_state
        payload["display_state"]     = provenance.display_state

    payload["items"] = [
        it.to_dict() if hasattr(it, "to_dict") else dict(it) for it in items
    ]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_encode)
