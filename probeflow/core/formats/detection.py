"""Content-signature predicates for built-in microscope file structures."""

from __future__ import annotations


SNIFF_BYTES = 8192

# "STiMage 005." encoded as UTF-16-LE at byte offset 2.
_SM4_MAGIC = bytes(
    [
        0x53, 0x00, 0x54, 0x00, 0x69, 0x00, 0x4D, 0x00,
        0x61, 0x00, 0x67, 0x00, 0x65, 0x00, 0x20, 0x00,
        0x30, 0x00, 0x30, 0x00, 0x35, 0x00, 0x2E, 0x00,
    ]
)
_SM4_MAGIC_OFFSET = 2


def is_rhk_sm4(head: bytes) -> bool:
    end = _SM4_MAGIC_OFFSET + len(_SM4_MAGIC)
    return len(head) >= end and head[_SM4_MAGIC_OFFSET:end] == _SM4_MAGIC


def is_nanonis_spec(head: bytes) -> bool:
    return head.startswith(b"Experiment\t")


def is_createc_spec(head: bytes) -> bool:
    return head.startswith((b"[ParVERT30]", b"[ParVERT32]"))


def is_createc_image(head: bytes) -> bool:
    return head.startswith(b"[Paramco32]")


def is_nanonis_image(head: bytes) -> bool:
    return b":NANONIS_VERSION:" in head


def is_legacy_createc_image(head: bytes) -> bool:
    return head.startswith(b"[") and has_binary_data_block(head)


def has_binary_data_block(head: bytes) -> bool:
    """Return whether a DATA marker is followed by a zlib deflate stream."""
    index = head.find(b"DATA")
    if index < 0:
        return False
    tail = head[index + 4:].lstrip(b"\r\n")
    return len(tail) >= 2 and tail[0] == 0x78
