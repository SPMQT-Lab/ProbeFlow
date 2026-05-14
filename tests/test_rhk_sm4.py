"""Tests for the RHK SM4 image reader and scan integration."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from probeflow.core.indexing import index_folder
from probeflow.core.scan_loader import load_scan
from probeflow.gui.models import scan_image_folder
from probeflow.gui.rendering import render_scan_image
from probeflow.io.file_type import FileType, sniff_file_type
from probeflow.io.readers.rhk_sm4 import (
    GUID_SIZE,
    MAGIC_OFFSET,
    MAGIC_TOTAL_SIZE,
    OBJECT_SIZE,
    PAGE_HEADER_SIZE,
    RHK_DATA_IMAGE,
    RHK_OBJECT_PAGE_DATA,
    RHK_OBJECT_PAGE_HEADER,
    RHK_OBJECT_PAGE_INDEX_ARRAY,
    RHK_OBJECT_STRING_DATA,
    SM4_MAGIC,
    is_rhk_sm4,
    parse_object_table,
    parse_page_header,
    read_rhk_sm4,
)
from probeflow.processing.image import subtract_background


def _pack_object(typ: int, offset: int, size: int) -> bytes:
    return struct.pack("<III", typ, offset, size)


def _page_header(
    *,
    x_size: int = 4,
    y_size: int = 3,
    page_type: int = 1,
    scan_dir: int = 0,
    z_scale: float = 0.5,
    z_offset: float = 1.0,
) -> bytes:
    data = bytearray(PAGE_HEADER_SIZE)
    pos = 0
    for value in (PAGE_HEADER_SIZE, 13, page_type, 0, 0):
        struct.pack_into("<I", data, pos, value)
        pos += 4
    for value in (0, 0):
        struct.pack_into("<i", data, pos, value)
        pos += 4
    for value in (x_size, y_size, 0, scan_dir, 0, x_size * y_size * 4):
        struct.pack_into("<I", data, pos, value)
        pos += 4
    for value in (0, x_size * y_size - 1):
        struct.pack_into("<i", data, pos, value)
        pos += 4
    for value in (
        1.0, 2.0, z_scale, 1e-9,
        0.0, 0.0, z_offset, 0.0,
        0.1, 2e-10, 15.0,
    ):
        struct.pack_into("<d", data, pos, value)
        pos += 8
    for value in (0, x_size, y_size, 0):
        struct.pack_into("<I", data, pos, value)
        pos += 4
    return bytes(data)


def _strings(*values: str) -> bytes:
    return ("\x00".join(values)).encode("utf-16-le")


def _append(buf: bytearray, payload: bytes) -> int:
    offset = len(buf)
    buf.extend(payload)
    return offset


def _synthetic_sm4(path: Path) -> Path:
    buf = bytearray(MAGIC_TOTAL_SIZE + 12 + OBJECT_SIZE)
    buf[MAGIC_OFFSET:MAGIC_OFFSET + len(SM4_MAGIC)] = SM4_MAGIC
    struct.pack_into("<III", buf, MAGIC_TOTAL_SIZE, 2, 1, OBJECT_SIZE)

    page_index_array_offset = len(buf)
    buf.extend(b"\x00" * (2 * (GUID_SIZE + 4 * 4)))
    struct.pack_into(
        "<III",
        buf,
        MAGIC_TOTAL_SIZE + 12,
        RHK_OBJECT_PAGE_INDEX_ARRAY,
        page_index_array_offset,
        2 * (GUID_SIZE + 4 * 4),
    )

    header = _page_header()
    raw = np.arange(12, dtype="<i4").tobytes()
    strings = _strings(
        "Topo page", "system", "session", "user", "path",
        "2026-05-14", "12:00:00", "m", "m", "m",
    )
    header_offset = _append(buf, header)
    data_offset = _append(buf, raw)
    strings_offset = _append(buf, strings)

    page0_offset = len(buf)
    buf.extend(b"0" * GUID_SIZE)
    buf.extend(struct.pack("<IIII", RHK_DATA_IMAGE, 0, 3, 4))
    buf.extend(_pack_object(RHK_OBJECT_PAGE_HEADER, header_offset, len(header)))
    buf.extend(_pack_object(RHK_OBJECT_PAGE_DATA, data_offset, len(raw)))
    buf.extend(_pack_object(RHK_OBJECT_STRING_DATA, strings_offset, len(strings)))

    page1_offset = len(buf)
    buf.extend(b"1" * GUID_SIZE)
    buf.extend(struct.pack("<IIII", 1, 0, 0, 4))

    struct.pack_into("<IIII", buf, page_index_array_offset + GUID_SIZE, page0_offset, RHK_DATA_IMAGE, 0, 0)
    second_entry = page_index_array_offset + GUID_SIZE + 4 * 4
    struct.pack_into("<IIII", buf, second_entry + GUID_SIZE, page1_offset, 1, 0, 0)

    path.write_bytes(buf)
    return path


def test_magic_detection_from_bytes():
    data = b"\x00\x00" + SM4_MAGIC + b"\x00" * 32
    assert is_rhk_sm4(data)


def test_sniff_identifies_rhk_sm4(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.SM4")
    assert sniff_file_type(path) is FileType.RHK_SM4_IMAGE


def test_object_table_parses_little_endian_records():
    data = b"\x00" * 16 + _pack_object(3, 64, 170)
    objects = parse_object_table(data + b"\x00" * 256, 16, 1)
    assert objects[0].type == 3
    assert objects[0].offset == 64
    assert objects[0].size == 170


def test_page_header_parser_extracts_core_fields():
    header = _page_header(x_size=8, y_size=6, page_type=2, scan_dir=3)
    parsed = parse_page_header(header)
    assert parsed["x_size"] == 8
    assert parsed["y_size"] == 6
    assert parsed["page_type"] == 2
    assert parsed["scan_dir"] == 3
    assert parsed["z_scale"] == 0.5
    assert parsed["bias"] == 0.1


def test_read_rhk_sm4_decodes_image_and_skips_non_image(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.sm4")
    sm4 = read_rhk_sm4(path)
    assert sm4.page_count == 2
    assert len(sm4.pages) == 1
    assert any("Skipped non-image RHK page" in note for note in sm4.parser_notes)
    page = sm4.pages[0]
    assert page.page_type_label == "Topography"
    assert page.scan_dir_label == "Right"
    assert page.label == "Topo page"
    assert page.physical_data.shape == (3, 4)
    np.testing.assert_allclose(page.physical_data.ravel()[:3], [1.0, 1.5, 2.0])


def test_load_scan_exposes_sm4_image_pages(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.sm4")
    scan = load_scan(path)
    assert scan.source_format == "sm4"
    assert scan.n_planes == 1
    assert scan.plane_names == ["Topo page [Right]"]
    assert scan.plane_units == ["m"]
    assert scan.scan_range_m[0] == pytest.approx(4e-9)
    assert scan.scan_range_m[1] == pytest.approx(6e-9)


def test_index_and_gui_entry_include_sm4(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.sm4")
    items = index_folder(tmp_path)
    assert len(items) == 1
    assert items[0].source_format == "rhk_sm4"
    assert items[0].shape == (3, 4)
    entries = scan_image_folder(tmp_path)
    assert len(entries) == 1
    assert entries[0].path == path
    assert entries[0].source_format == "sm4"


def test_sm4_image_uses_normal_thumbnail_and_processing_paths(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.sm4")
    scan = load_scan(path)
    corrected = subtract_background(scan.planes[0], order=1)
    assert corrected.shape == scan.planes[0].shape
    img = render_scan_image(scan_path=path, colormap="gray", size=(64, 64))
    assert img is not None
    assert img.size[0] <= 64 and img.size[1] <= 64
