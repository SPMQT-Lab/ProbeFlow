"""Explicit staged loader identification helpers.

ProbeFlow's loading contract is intentionally small:

``sniff -> read_metadata -> read_full``

The low-level content sniffing lives in :mod:`probeflow.io.file_type`.  This
module adds a slightly higher-level identification step that resolves a path
into a concrete supported scan or spectroscopy source format before metadata
or full-data readers are dispatched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from probeflow.io.file_type import FileType, sniff_file_type


@dataclass(frozen=True)
class LoadSignature:
    """Resolved file identity for a supported ProbeFlow loader path."""

    path: Path
    file_type: FileType
    item_type: str
    source_format: str


def identify_scan_file(path) -> LoadSignature:
    """Resolve *path* as a supported scan file or raise a contextual error."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not p.is_file():
        raise ValueError(f"{p}: expected a file path for scan loading")
    ft = sniff_file_type(p)
    suffix = p.suffix.lower()

    if ft == FileType.NANONIS_IMAGE:
        return LoadSignature(p, ft, "scan", "sxm")
    if ft == FileType.CREATEC_IMAGE:
        return LoadSignature(p, ft, "scan", "dat")
    # ``.sxm`` is unambiguous, so let malformed headers fail in the reader's
    # metadata/full-load stages rather than at the sniff stage.
    if ft == FileType.UNKNOWN and suffix == ".sxm":
        return LoadSignature(p, FileType.NANONIS_IMAGE, "scan", "sxm")
    if ft == FileType.NANONIS_SPEC:
        raise ValueError(
            f"{p.name}: identified as spectroscopy during scan sniff stage; "
            "use probeflow.io.spectroscopy.read_spec_file or read_spec_metadata."
        )
    if ft == FileType.CREATEC_SPEC:
        raise ValueError(
            f"{p.name}: identified as Createc .VERT spectroscopy during "
            "scan sniff stage; use probeflow.io.spectroscopy.read_spec_file or "
            "read_spec_metadata."
        )
    raise ValueError(
        f"Unsupported or unrecognised scan file: {p.name}. "
        "Sniff stage could not identify a supported scan file."
    )


def identify_spectrum_file(path) -> LoadSignature:
    """Resolve *path* as a supported spectroscopy file or raise an error."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not p.is_file():
        raise ValueError(f"{p}: expected a file path for spectroscopy loading")
    ft = sniff_file_type(p)
    suffix = p.suffix.lower()

    if ft == FileType.NANONIS_SPEC:
        return LoadSignature(p, ft, "spectrum", "nanonis_dat_spectrum")
    if ft == FileType.CREATEC_SPEC:
        return LoadSignature(p, ft, "spectrum", "createc_vert")
    # ``.VERT`` is unambiguous, so allow malformed files through to the
    # metadata/full parser where DATA/header validation already lives.
    if ft == FileType.UNKNOWN and suffix == ".vert":
        return LoadSignature(p, FileType.CREATEC_SPEC, "spectrum", "createc_vert")
    if ft == FileType.NANONIS_IMAGE:
        raise ValueError(
            f"{p.name}: identified as Nanonis scan image during spectroscopy "
            "sniff stage; use probeflow.core.scan_loader.load_scan or read_scan_metadata."
        )
    if ft == FileType.CREATEC_IMAGE:
        raise ValueError(
            f"{p.name}: identified as Createc scan image during spectroscopy "
            "sniff stage; use probeflow.core.scan_loader.load_scan or read_scan_metadata."
        )
    raise ValueError(
        f"{p.name}: sniff stage could not identify a supported spectroscopy file."
    )
