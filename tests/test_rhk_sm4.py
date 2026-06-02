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
from probeflow.core.file_type import FileType, is_rhk_sm4, sniff_file_type
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
    _decode_image_payload,
    _parse_string_data,
    parse_object_table,
    parse_page_header,
    read_rhk_sm4,
    read_sm4,
)
from probeflow.processing.image import subtract_background

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_SM4 = REPO_ROOT / "test_data" / "VT260430_0004.sm4"


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
    """Build a correctly-formatted SM4 page header.

    field_size and string_count are uint16; scale/offset fields are float32.
    PAGE_HEADER_SIZE bytes total (116 bytes).
    """
    data = bytearray(PAGE_HEADER_SIZE)
    pos = 0

    # field_size (u16) + string_count (u16)
    struct.pack_into("<H", data, pos, PAGE_HEADER_SIZE); pos += 2
    struct.pack_into("<H", data, pos, 13); pos += 2  # string_count

    # page_type, data_sub_source, line_type (u32 each)
    for value in (page_type, 0, 0):
        struct.pack_into("<I", data, pos, value); pos += 4

    # x_coord, y_coord (i32 each)
    for value in (0, 0):
        struct.pack_into("<i", data, pos, value); pos += 4

    # x_size, y_size, image_type, scan_dir, group_id, data_size (u32 each)
    for value in (x_size, y_size, 0, scan_dir, 0, x_size * y_size * 4):
        struct.pack_into("<I", data, pos, value); pos += 4

    # min_z_value, max_z_value (i32 each)
    for value in (0, x_size * y_size - 1):
        struct.pack_into("<i", data, pos, value); pos += 4

    # Scale/offset fields are float32 (not float64)
    for value in (
        1.0, 2.0, z_scale, 1e-9,
        0.0, 0.0, z_offset, 0.0,
        0.125, 2e-10, 15.0,        # bias=0.125 is exact in f32
    ):
        struct.pack_into("<f", data, pos, value); pos += 4

    # color_info_count, grid_x_size, grid_y_size, object_count (u32 each)
    for value in (0, x_size, y_size, 0):
        struct.pack_into("<I", data, pos, value); pos += 4

    return bytes(data)


def _strings(*values: str) -> bytes:
    """Encode strings in SM4 format: uint16 char-count + UTF-16LE content."""
    buf = bytearray()
    for s in values:
        encoded = s.encode("utf-16-le")
        buf += len(s).to_bytes(2, "little")
        buf += encoded
    return bytes(buf)


def _append(buf: bytearray, payload: bytes) -> int:
    offset = len(buf)
    buf.extend(payload)
    return offset


def _synthetic_sm4(path: Path, *, z_units: str = "m") -> Path:
    """Write a minimal but correctly-formatted SM4 file to *path*.

    File layout:
      [0..1]   2 zero bytes
      [2..37]  SM4 magic region (MAGIC_TOTAL_SIZE=36 bytes)
      [38..]   file header + file-level object table + page content

    File header is at MAGIC_OFFSET + MAGIC_TOTAL_SIZE = 38 (not 36).
    Two pages: page 0 is a Topography image, page 1 is a non-image page.
    Page index records use the correct stride:
      PAGE_INDEX_ARRAY_SIZE + object_count × OBJECT_SIZE
    """
    buf = bytearray()

    # Bytes 0..1: padding before magic
    buf += b"\x00\x00"

    # Bytes 2..37: magic region (SM4_MAGIC is 24 bytes; pad to 36)
    buf += SM4_MAGIC
    buf += b"\x00" * (MAGIC_TOTAL_SIZE - len(SM4_MAGIC))

    # File header at offset 38 (= MAGIC_OFFSET + MAGIC_TOTAL_SIZE)
    file_header_pos = len(buf)  # == 38
    buf += b"\x00" * 20        # placeholder: page_count, object_count, object_field_size, r1, r2

    # File-level object table (1 entry for PAGE_INDEX_ARRAY)
    file_obj_table_pos = len(buf)
    buf += b"\x00" * OBJECT_SIZE  # placeholder

    # ── Page content ──────────────────────────────────────────────────────────
    header_bytes = _page_header()
    header_pos = _append(buf, header_bytes)

    raw_data = np.arange(12, dtype="<i4").tobytes()
    data_pos = _append(buf, raw_data)

    string_data = _strings(
        "Topo page", "system", "session", "user", "path",
        "2026-05-14", "12:00:00", "m", "m", z_units,
    )
    strings_pos = _append(buf, string_data)

    # ── Page index array ──────────────────────────────────────────────────────
    # Page 0: image page, 3 objects (header, data, strings)
    pia_pos = len(buf)

    buf += b"\x00" * GUID_SIZE  # GUID for page 0
    buf += struct.pack("<IIII", RHK_DATA_IMAGE, 0, 3, 4)  # data_type, source, object_count, minor
    buf += _pack_object(RHK_OBJECT_PAGE_HEADER, header_pos, len(header_bytes))
    buf += _pack_object(RHK_OBJECT_PAGE_DATA, data_pos, len(raw_data))
    buf += _pack_object(RHK_OBJECT_STRING_DATA, strings_pos, len(string_data))

    # Page 1: non-image page, 0 objects (data_type=1, not RHK_DATA_IMAGE)
    buf += b"\x01" * GUID_SIZE  # GUID for page 1
    buf += struct.pack("<IIII", 1, 0, 0, 4)  # data_type=1, source=0, object_count=0, minor=4
    # no objects follow for page 1

    # ── Fix up placeholders ───────────────────────────────────────────────────
    # File header: page_count=2, object_count=1, object_field_size=OBJECT_SIZE, r1=0, r2=0
    struct.pack_into("<IIIII", buf, file_header_pos, 2, 1, OBJECT_SIZE, 0, 0)

    # File-level object table: PAGE_INDEX_ARRAY at absolute offset pia_pos
    # size = PAGE_INDEX_ARRAY_SIZE (size of one page-index record, per SM4 convention)
    from probeflow.io.readers.rhk_sm4 import PAGE_INDEX_ARRAY_SIZE
    struct.pack_into("<III", buf, file_obj_table_pos,
                     RHK_OBJECT_PAGE_INDEX_ARRAY, pia_pos, PAGE_INDEX_ARRAY_SIZE)

    path.write_bytes(buf)
    return path


# ── Detection tests ──────────────────────────────────────────────────────────

def test_magic_detection_from_bytes():
    data = b"\x00\x00" + SM4_MAGIC + b"\x00" * 32
    assert is_rhk_sm4(data)


def test_sniff_identifies_rhk_sm4(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.SM4")
    assert sniff_file_type(path) is FileType.RHK_SM4_IMAGE


# ── Low-level parser tests ───────────────────────────────────────────────────

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
    assert parsed["z_scale"] == pytest.approx(0.5)
    assert parsed["bias"] == pytest.approx(0.125)


def test_decode_image_payload_uses_data_size_for_dtype():
    notes: list[str] = []
    payload = np.array([0, 256, -256, 1024], dtype="<i2").tobytes()

    arr = _decode_image_payload(
        payload,
        x_size=2,
        y_size=2,
        notes=notes,
        page_index=0,
        data_size=2,
    )

    np.testing.assert_array_equal(arr, np.array([[0.0, 256.0], [-256.0, 1024.0]]))
    assert notes == []


def test_decode_image_payload_warns_when_data_size_disagrees_with_payload():
    notes: list[str] = []
    payload = np.arange(4, dtype="<i4").tobytes()

    arr = _decode_image_payload(
        payload,
        x_size=2,
        y_size=2,
        notes=notes,
        page_index=2,
        data_size=8,
    )

    np.testing.assert_array_equal(arr, np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert any("data_size=8" in note for note in notes)


def test_parse_string_data_records_truncated_string_note():
    notes: list[str] = []
    payload = (5).to_bytes(2, "little") + "ab".encode("utf-16-le")

    strings = _parse_string_data(payload, notes=notes, page_index=3)

    assert strings == {}
    assert any("string block truncated" in note for note in notes)


# ── Synthetic-file integration tests ─────────────────────────────────────────

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
    # raw=[0..11] as int32, z_scale=0.5, z_offset=1.0 → [1.0, 1.5, 2.0, ...]
    np.testing.assert_allclose(page.physical_data.ravel()[:3], [1.0, 1.5, 2.0], rtol=1e-5)


def test_load_scan_exposes_sm4_image_pages(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.sm4")
    scan = load_scan(path)
    assert scan.source_format == "sm4"
    assert scan.n_planes == 1
    assert scan.plane_names == ["Topo page [Right]"]
    assert scan.plane_units == ["m"]
    assert scan.scan_range_m[0] == pytest.approx(4e-9, rel=1e-4)
    assert scan.scan_range_m[1] == pytest.approx(6e-9, rel=1e-4)


def test_sm4_scan_conversion_normalises_length_z_units(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.sm4", z_units="nm")
    scan = load_scan(path)

    assert scan.plane_units == ["m"]
    np.testing.assert_allclose(scan.planes[0].ravel()[:3], [1.0e-9, 1.5e-9, 2.0e-9])


def test_sm4_metadata_only_parse_skips_payload_decode(tmp_path):
    path = _synthetic_sm4(tmp_path / "image.sm4")
    sm4 = read_rhk_sm4(path, metadata_only=True)
    assert len(sm4.pages) == 1
    page = sm4.pages[0]
    # Headers/strings are parsed, but image arrays are left empty.
    assert page.x_size == 4 and page.y_size == 3
    assert page.raw_data.size == 0
    assert page.physical_data.size == 0


def test_sm4_metadata_only_matches_full_decode(tmp_path):
    from probeflow.core.metadata import metadata_from_rhk_sm4, metadata_from_scan

    path = _synthetic_sm4(tmp_path / "image.sm4")
    fast = metadata_from_rhk_sm4(read_rhk_sm4(path, metadata_only=True))
    slow = metadata_from_scan(read_sm4(path))
    assert fast == slow


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


# ── Real-file tests (VT260430_0004.sm4) ──────────────────────────────────────

@pytest.mark.skipif(not REAL_SM4.exists(), reason="VT260430_0004.sm4 fixture not present")
class TestRealSM4:
    def test_reads_four_image_pages(self):
        sm4 = read_rhk_sm4(REAL_SM4)
        assert sm4.page_count == 4
        assert len(sm4.pages) == 4

    def test_page_labels_dimensions_units(self):
        sm4 = read_rhk_sm4(REAL_SM4)
        expected = [
            ("Topography", "Left",  "m"),
            ("Topography", "Right", "m"),
            ("Current",    "Left",  "A"),
            ("Current",    "Right", "A"),
        ]
        for page, (label, direction, unit) in zip(sm4.pages, expected):
            assert page.page_type_label == label, f"page {page.page_index}: wrong label"
            assert page.scan_dir_label == direction, f"page {page.page_index}: wrong direction"
            assert page.physical_data.shape == (512, 512), f"page {page.page_index}: wrong shape"
            assert page.z_unit == unit, f"page {page.page_index}: wrong unit"

    def test_topography_data_is_finite_and_nonflat(self):
        sm4 = read_rhk_sm4(REAL_SM4)
        topo = sm4.pages[0].physical_data
        assert np.isfinite(topo).all(), "topography contains non-finite values"
        assert np.nanstd(topo) > 0, "topography is flat (zero std)"

    def test_not_float32_garbage(self):
        """Correct int32 decoding gives nm-scale heights, not 1e-31-scale float32 garbage."""
        sm4 = read_rhk_sm4(REAL_SM4)
        topo = sm4.pages[0].physical_data
        assert np.nanmax(np.abs(topo)) > 1e-9, (
            f"topography values are too small ({np.nanmax(np.abs(topo)):.3e}); "
            "likely decoded as float32 instead of int32"
        )

    def test_scan_conversion_exposes_four_planes(self):
        scan = read_sm4(REAL_SM4)
        assert len(scan.planes) == 4
        assert scan.planes[0].shape == (512, 512)
        assert "Topography" in scan.plane_names[0]
        assert scan.plane_units[0] == "m"
        assert scan.plane_units[2] == "A"

    def test_thumbnail_not_blank(self):
        img = render_scan_image(scan_path=REAL_SM4, colormap="gray", size=(128, 128))
        assert img is not None
        arr = np.array(img.convert("L"))
        assert arr.std() > 0, "thumbnail has zero contrast (blank)"
