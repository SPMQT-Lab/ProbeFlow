"""Content-sniffing dispatcher for probe-microscopy files.

File extensions overlap (``.dat`` is used by both Createc topography and
Nanonis spectroscopy), so we identify files by a short content signature
instead of just the suffix.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


# Read at most this many bytes from the start of each file while sniffing.
_SNIFF_BYTES = 8192


class FileType(Enum):
    CREATEC_IMAGE = "createc_image"
    CREATEC_SPEC = "createc_spec"
    NANONIS_IMAGE = "nanonis_image"
    NANONIS_SPEC = "nanonis_spec"
    UNKNOWN = "unknown"


def sniff_file_type(path) -> FileType:
    """Identify a file by its content signature, not its suffix.

    Reads the first ~8 KB of the file and matches against known vendor
    signatures.  Never raises: a missing, unreadable, or unrecognised
    file returns :data:`FileType.UNKNOWN`.
    """
    try:
        p = Path(path)
        with p.open("rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except (OSError, ValueError):
        return FileType.UNKNOWN

    if not head:
        return FileType.UNKNOWN

    # Nanonis spec (.dat): pure ASCII, starts with "Experiment\t".
    # Check this BEFORE Createc image because both may share a .dat suffix.
    if head.startswith(b"Experiment\t"):
        return FileType.NANONIS_SPEC

    # Createc spec (.VERT): starts with [ParVERT30].
    if head.startswith(b"[ParVERT30]"):
        return FileType.CREATEC_SPEC

    # Createc image (.dat): starts with [Paramco32].
    if head.startswith(b"[Paramco32]"):
        return FileType.CREATEC_IMAGE

    # Nanonis image (.sxm): header contains ":NANONIS_VERSION:".
    if b":NANONIS_VERSION:" in head:
        return FileType.NANONIS_IMAGE

    # Fallback: a file with a DATA marker followed by binary bytes is a
    # Createc image whose magic header we didn't recognise.
    if _has_binary_data_block(head):
        return FileType.CREATEC_IMAGE

    return FileType.UNKNOWN


def _has_binary_data_block(head: bytes) -> bool:
    """Return True if ``head`` contains a ``DATA`` marker followed by
    non-ASCII bytes within the sniffed window."""
    idx = head.find(b"DATA")
    if idx < 0:
        return False
    # Look at the bytes after the DATA marker (skip over CRLF/newline) and
    # see if any byte is non-ASCII (>= 0x80) — this is the binary payload.
    tail = head[idx + 4:]
    # Skip immediate EOL bytes
    tail = tail.lstrip(b"\r\n")
    if not tail:
        return False
    for b in tail:
        if b >= 0x80:
            return True
    return False
