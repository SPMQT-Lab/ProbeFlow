"""Scan loading dispatcher and public compatibility exports.

``Scan`` itself lives in :mod:`probeflow.core.scan_model` so low-level readers,
writers, validation, metadata, CLI, and GUI can depend on the pure model
without depending on this dispatcher. Keep this module focused on loading and
backward-compatible imports.
"""

from __future__ import annotations

from probeflow.core.scan_model import PLANE_CANON_NAMES, PLANE_CANON_UNITS, Scan


SUPPORTED_SUFFIXES: tuple[str, ...] = (".sxm", ".dat")


def _validate(scan: Scan) -> None:
    from probeflow.core.validation import validate_scan
    validate_scan(scan)


def load_scan(path) -> Scan:
    """Load an STM scan file, dispatching on its content signature.

    Supported formats:
      * ``.sxm`` - Nanonis topography
      * ``.dat`` - Createc topography

    Point-spectroscopy files (Createc ``.VERT`` and Nanonis ``.dat`` spec)
    are not scans - use :func:`probeflow.io.spectroscopy.read_spec_file` instead.
    """
    from probeflow.core.loaders import identify_scan_file

    sig = identify_scan_file(path)

    if sig.source_format == "sxm":
        from probeflow.io.readers.nanonis_sxm import read_sxm
        scan = read_sxm(sig.path)
        _validate(scan)
        return scan
    if sig.source_format == "dat":
        from probeflow.io.readers.createc_scan import read_dat
        scan = read_dat(sig.path)
        _validate(scan)
        return scan

    raise ValueError(
        f"Unsupported or unrecognised scan file: {sig.path}. "
        f"Supported: {', '.join(SUPPORTED_SUFFIXES)}"
    )


__all__ = [
    "PLANE_CANON_NAMES",
    "PLANE_CANON_UNITS",
    "SUPPORTED_SUFFIXES",
    "Scan",
    "load_scan",
]
