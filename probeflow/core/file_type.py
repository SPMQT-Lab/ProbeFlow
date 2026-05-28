"""Content-sniffing dispatcher for probe-microscopy files.

File extensions overlap (``.dat`` is used by both Createc topography and
Nanonis spectroscopy), so we identify files by a short content signature
instead of just the suffix.

This module lives in ``core`` because format dispatch is a loading contract
concern, not an IO implementation detail — ``core.loaders`` and
``core.indexing`` both need it without pulling in writer/reader dependencies.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


# Read at most this many bytes from the start of each file while sniffing.
_SNIFF_BYTES = 8192

# Only attempt content sniffing for files with these suffixes.
# This prevents false positives on source/test files that happen to contain
# scanner magic strings as Python literals (e.g. b":NANONIS_VERSION:").
_SNIFF_SUFFIXES: frozenset[str] = frozenset({
    ".sxm",   # Nanonis image
    ".dat",   # Createc image or Nanonis spec
    ".sm4",   # RHK SM4
    ".vert",  # Createc spec
})


class FileType(Enum):
    CREATEC_IMAGE = "createc_image"
    CREATEC_SPEC = "createc_spec"
    NANONIS_IMAGE = "nanonis_image"
    NANONIS_SPEC = "nanonis_spec"
    RHK_SM4_IMAGE = "rhk_sm4_image"
    UNKNOWN = "unknown"


# RHK SM4: "STiMage 005." encoded as UTF-16-LE, starting at byte offset 2.
_SM4_MAGIC = bytes([
    0x53, 0x00, 0x54, 0x00, 0x69, 0x00, 0x4D, 0x00,
    0x61, 0x00, 0x67, 0x00, 0x65, 0x00, 0x20, 0x00,
    0x30, 0x00, 0x30, 0x00, 0x35, 0x00, 0x2E, 0x00,
])
_SM4_MAGIC_OFFSET = 2


def is_rhk_sm4(head: bytes) -> bool:
    """Return ``True`` when *head* contains the RHK SM4 magic at byte offset 2."""
    end = _SM4_MAGIC_OFFSET + len(_SM4_MAGIC)
    return len(head) >= end and head[_SM4_MAGIC_OFFSET:end] == _SM4_MAGIC


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
        if p.suffix.lower() not in _SNIFF_SUFFIXES:
            return FileType.UNKNOWN
        with p.open("rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except (OSError, ValueError):
        return FileType.UNKNOWN

    if not head:
        return FileType.UNKNOWN

    if is_rhk_sm4(head):
        return FileType.RHK_SM4_IMAGE

    # Nanonis spec (.dat): pure ASCII, starts with "Experiment\t".
    # Check this BEFORE Createc image because both may share a .dat suffix.
    if head.startswith(b"Experiment\t"):
        return FileType.NANONIS_SPEC

    # Createc spec (.VERT): starts with [ParVERT30] or [ParVERT32].
    if head.startswith((b"[ParVERT30]", b"[ParVERT32]")):
        return FileType.CREATEC_SPEC

    # Createc image (.dat): starts with [Paramco32].
    if head.startswith(b"[Paramco32]"):
        return FileType.CREATEC_IMAGE

    # Nanonis image (.sxm): header contains ":NANONIS_VERSION:".
    if b":NANONIS_VERSION:" in head:
        return FileType.NANONIS_IMAGE

    # Fallback: Createc images with older or non-standard parameter blocks.
    # We require the file to begin with '[' (characteristic of Createc INI
    # headers such as [Paramco30]) to avoid false positives on Nanonis
    # spectroscopy .dat files that happen to contain "DATA" followed by a
    # byte that is 0x78 (ASCII 'x').
    if head.startswith(b"[") and _has_binary_data_block(head):
        return FileType.CREATEC_IMAGE

    return FileType.UNKNOWN


def _has_binary_data_block(head: bytes) -> bool:
    """Return True if ``head`` contains a ``DATA`` marker followed by a zlib
    compressed stream.

    Createc image payloads are always zlib-deflate compressed, so the first
    byte after the DATA marker (and any trailing EOL) is always ``0x78`` — the
    zlib CMF byte for deflate with a 32 KB window.  Checking for this specific
    byte prevents false positives on text or markdown files that happen to
    contain the word "DATA" followed by a stray non-ASCII character.
    """
    idx = head.find(b"DATA")
    if idx < 0:
        return False
    tail = head[idx + 4:].lstrip(b"\r\n")
    # zlib stream always begins with 0x78 (deflate, window size 32 KB)
    return len(tail) >= 2 and tail[0] == 0x78
