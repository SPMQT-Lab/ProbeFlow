"""Validation for loaded :class:`probeflow.core.scan_model.Scan` objects.

Call :func:`validate_scan` immediately after loading to catch internal
inconsistencies before they silently corrupt downstream rendering or writing.
"""

from __future__ import annotations

import numpy as np

from probeflow.core.scan_model import Scan

_SUPPORTED_FORMATS = {"dat", "sxm"}

_CREATEC_PLANE_NAMES = [
    "Z forward",
    "Z backward",
    "Current forward",
    "Current backward",
]


def validate_scan(scan: Scan) -> None:
    """Raise ``ValueError`` if *scan* is internally inconsistent.

    Checks are non-destructive: the scan is never modified.
    """
    _validate_common(scan)
    if scan.source_format == "dat":
        _validate_createc(scan)
    elif scan.source_format == "sxm":
        _validate_nanonis(scan)


# ── Common checks ────────────────────────────────────────────────────────────

def _err(scan: Scan, msg: str) -> ValueError:
    path = getattr(scan, "source_path", None)
    fmt = getattr(scan, "source_format", "unknown")
    prefix = f"{path} [{fmt}]" if path else f"[{fmt}]"
    return ValueError(f"{prefix}: {msg}")


def _validate_common(scan: Scan) -> None:
    if not scan.planes:
        raise _err(scan, "planes list is empty")

    n = len(scan.planes)

    for i, plane in enumerate(scan.planes):
        if not isinstance(plane, np.ndarray):
            raise _err(scan, f"planes[{i}] is not a NumPy array (got {type(plane).__name__})")
        if plane.ndim != 2:
            raise _err(scan, f"planes[{i}] is {plane.ndim}-D; expected 2-D")
        if plane.shape[0] == 0 or plane.shape[1] == 0:
            raise _err(scan, f"planes[{i}] has zero dimension: shape={plane.shape}")

    shape0 = scan.planes[0].shape
    for i, plane in enumerate(scan.planes[1:], start=1):
        if plane.shape != shape0:
            raise _err(
                scan,
                f"planes[{i}] shape {plane.shape} != planes[0] shape {shape0}",
            )

    if len(scan.plane_names) != n:
        raise _err(
            scan,
            f"plane_names length {len(scan.plane_names)} != planes length {n}",
        )
    if len(scan.plane_units) != n:
        raise _err(
            scan,
            f"plane_units length {len(scan.plane_units)} != planes length {n}",
        )
    if len(scan.plane_synthetic) != n:
        raise _err(
            scan,
            f"plane_synthetic length {len(scan.plane_synthetic)} != planes length {n}",
        )

    for i, name in enumerate(scan.plane_names):
        if not isinstance(name, str) or not name:
            raise _err(scan, f"plane_names[{i}] is not a non-empty string: {name!r}")

    for i, unit in enumerate(scan.plane_units):
        if not isinstance(unit, str):
            raise _err(scan, f"plane_units[{i}] is not a string: {unit!r}")

    for i, plane in enumerate(scan.planes):
        if not np.any(np.isfinite(plane)):
            raise _err(scan, f"planes[{i}] contains no finite values (all-NaN or all-inf)")

    if not scan.source_format or scan.source_format not in _SUPPORTED_FORMATS:
        raise _err(
            scan,
            f"source_format {scan.source_format!r} is not one of {sorted(_SUPPORTED_FORMATS)}",
        )


# ── Format-specific checks ───────────────────────────────────────────────────

def _validate_createc(scan: Scan) -> None:
    Ny, Nx = scan.planes[0].shape

    nx_hdr = scan.header.get("Num.X")
    if nx_hdr is not None:
        try:
            nx_expected = int(float(nx_hdr))
        except (ValueError, TypeError):
            nx_expected = None
        if nx_expected is not None and Nx != nx_expected:
            raise _err(
                scan,
                f"array width {Nx} != Num.X header {nx_expected} "
                "(first-column strip may not have updated the header)",
            )

    ny_hdr = scan.header.get("Num.Y")
    if ny_hdr is not None:
        try:
            ny_expected = int(float(ny_hdr))
        except (ValueError, TypeError):
            ny_expected = None
        if ny_expected is not None and Ny != ny_expected:
            raise _err(
                scan,
                f"array height {Ny} != Num.Y header {ny_expected}",
            )

    if any(scan.plane_synthetic):
        if scan.plane_names != _CREATEC_PLANE_NAMES:
            raise _err(
                scan,
                f"synthetic Createc scan has unexpected plane names: "
                f"{scan.plane_names} (expected {_CREATEC_PLANE_NAMES})",
            )


def _validate_nanonis(scan: Scan) -> None:
    sr = getattr(scan, "scan_range_m", None)
    if sr is None:
        raise _err(scan, "scan_range_m is missing")
    try:
        w, h = sr
    except (TypeError, ValueError):
        raise _err(scan, f"scan_range_m cannot be unpacked as (width, height): {sr!r}")
    if not (w > 0 and h > 0):
        raise _err(scan, f"scan_range_m must be positive; got {sr}")
