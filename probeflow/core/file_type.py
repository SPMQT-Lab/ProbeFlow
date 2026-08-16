"""Content-sniffing dispatcher for probe-microscopy files.

File extensions overlap (``.dat`` is used by both Createc topography and
Nanonis spectroscopy), so we identify files by a short content signature
instead of just the suffix.

This module lives in ``core`` because format dispatch is a loading contract
concern, not an IO implementation detail — ``core.loaders`` and
``core.indexing`` both need it without pulling in writer/reader dependencies.
"""

from __future__ import annotations

from pathlib import Path

from probeflow.core.formats.builtins import BUILTIN_FORMATS
from probeflow.core.formats.detection import (
    SNIFF_BYTES,
    has_binary_data_block,
    is_rhk_sm4,
)
from probeflow.core.formats.model import FileType


# Read at most this many bytes from the start of each file while sniffing.
_SNIFF_BYTES = SNIFF_BYTES

# Only attempt content sniffing for files with these suffixes.
# This prevents false positives on source/test files that happen to contain
# scanner magic strings as Python literals (e.g. b":NANONIS_VERSION:").
_SNIFF_SUFFIXES = BUILTIN_FORMATS.supported_suffixes


def has_supported_suffix(path) -> bool:
    """Return whether *path* can contain a currently supported probe file."""
    try:
        return Path(path).suffix.lower() in _SNIFF_SUFFIXES
    except (TypeError, ValueError):
        return False


def sniff_file_type(path) -> FileType:
    """Identify a file by its content signature, not its suffix.

    Reads the first ~8 KB of the file and matches against known vendor
    signatures.  Never raises: a missing, unreadable, or unrecognised
    file returns :data:`FileType.UNKNOWN`.

    As a fast pre-filter, files whose suffix is not in :data:`_SNIFF_SUFFIXES`
    are rejected immediately without reading any bytes.  This prevents false
    positives on source or test files that happen to contain scanner magic
    strings as Python byte literals.
    """
    try:
        p = Path(path)
        if not has_supported_suffix(p):
            return FileType.UNKNOWN
        with p.open("rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except (OSError, ValueError):
        return FileType.UNKNOWN

    if not head:
        return FileType.UNKNOWN

    definition = BUILTIN_FORMATS.match_header(head)
    return definition.file_type if definition is not None else FileType.UNKNOWN


def _has_binary_data_block(head: bytes) -> bool:
    """Return True if ``head`` contains a ``DATA`` marker followed by a zlib
    compressed stream.

    Createc image payloads are always zlib-deflate compressed, so the first
    byte after the DATA marker (and any trailing EOL) is always ``0x78`` — the
    zlib CMF byte for deflate with a 32 KB window.  Checking for this specific
    byte prevents false positives on text or markdown files that happen to
    contain the word "DATA" followed by a stray non-ASCII character.
    """
    return has_binary_data_block(head)


__all__ = [
    "FileType",
    "has_supported_suffix",
    "is_rhk_sm4",
    "sniff_file_type",
]
