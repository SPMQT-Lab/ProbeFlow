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

from probeflow.core.file_type import FileType, sniff_file_type
from probeflow.core.formats.builtins import BUILTIN_FORMATS
from probeflow.core.formats.model import FormatDefinition


@dataclass(frozen=True)
class LoadSignature:
    """Resolved file identity for a supported ProbeFlow loader path."""

    path: Path
    file_type: FileType
    item_type: str
    source_format: str
    format_id: str | None = None


def _signature(path: Path, definition: FormatDefinition) -> LoadSignature:
    return LoadSignature(
        path=path,
        file_type=definition.file_type,
        item_type=definition.kind,
        source_format=definition.load_identifier,
        format_id=definition.format_id,
    )


def identify_scan_file(path) -> LoadSignature:
    """Resolve *path* as a supported scan file or raise a contextual error."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if not p.is_file():
        raise ValueError(f"{p}: expected a file path for scan loading")
    p = p.resolve()
    ft = sniff_file_type(p)
    suffix = p.suffix.lower()

    definition = BUILTIN_FORMATS.by_file_type(ft)
    if definition is not None and definition.kind == "scan":
        return _signature(p, definition)
    # ``.sxm`` is unambiguous, so let malformed headers fail in the reader's
    # metadata/full-load stages rather than at the sniff stage.
    if ft == FileType.UNKNOWN:
        fallback = next(
            (
                candidate
                for candidate in BUILTIN_FORMATS.for_suffix(suffix, kind="scan")
                if candidate.allow_suffix_fallback
            ),
            None,
        )
        if fallback is not None:
            return _signature(p, fallback)
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
    p = p.resolve()
    ft = sniff_file_type(p)
    suffix = p.suffix.lower()

    definition = BUILTIN_FORMATS.by_file_type(ft)
    if definition is not None and definition.kind == "spectrum":
        return _signature(p, definition)
    # ``.VERT`` is unambiguous, so allow malformed files through to the
    # metadata/full parser where DATA/header validation already lives.
    if ft == FileType.UNKNOWN:
        fallback = next(
            (
                candidate
                for candidate in BUILTIN_FORMATS.for_suffix(suffix, kind="spectrum")
                if candidate.allow_suffix_fallback
            ),
            None,
        )
        if fallback is not None:
            return _signature(p, fallback)
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
    if ft == FileType.RHK_SM4_IMAGE:
        raise ValueError(
            f"{p.name}: identified as RHK SM4 image during spectroscopy "
            "sniff stage; use probeflow.core.scan_loader.load_scan or read_scan_metadata."
        )
    raise ValueError(
        f"{p.name}: sniff stage could not identify a supported spectroscopy file."
    )
