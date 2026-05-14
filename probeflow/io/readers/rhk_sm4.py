"""Reader for RHK Technology ``.sm4`` image pages.

RHK SM4 files are object/page-indexed binary containers, not simple
header-plus-array files. This module keeps the parser isolated from GUI code
and exposes two layers:

* :func:`read_rhk_sm4` returns the parsed container with per-page metadata.
* :func:`read_sm4` converts image pages into ProbeFlow's common ``Scan`` model.

This first implementation is intentionally image-focused. Non-image pages are
skipped with parser notes; spectroscopy and line data are preserved only as
metadata notes for a later dedicated reader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import struct
import zlib

import numpy as np

from probeflow.core.scan_model import Scan


SM4_MAGIC = bytes([
    0x53, 0x00, 0x54, 0x00, 0x69, 0x00, 0x4D, 0x00,
    0x61, 0x00, 0x67, 0x00, 0x65, 0x00, 0x20, 0x00,
    0x30, 0x00, 0x30, 0x00, 0x35, 0x00, 0x2E, 0x00,
])
MAGIC_OFFSET = 2
MAGIC_TOTAL_SIZE = 36
OBJECT_SIZE = 12
GUID_SIZE = 16
PAGE_INDEX_ARRAY_SIZE = GUID_SIZE + 4 * 4
PAGE_INDEX_SIZE = GUID_SIZE + 4 * 4
PAGE_HEADER_SIZE = 170

RHK_OBJECT_PAGE_INDEX_HEADER = 1
RHK_OBJECT_PAGE_INDEX_ARRAY = 2
RHK_OBJECT_PAGE_HEADER = 3
RHK_OBJECT_PAGE_DATA = 4
RHK_OBJECT_STRING_DATA = 10
RHK_OBJECT_PRM = 13
RHK_OBJECT_PRM_HEADER = 15

RHK_DATA_IMAGE = 0

PAGE_TYPE_LABELS = {
    1: "Topography",
    2: "Current",
    3: "Aux",
    4: "Force",
    5: "Signal",
    6: "FFT",
}
SCAN_DIRECTION_LABELS = {
    0: "Right",
    1: "Left",
    2: "Up",
    3: "Down",
}
SOURCE_LABELS = {
    0: "Raw",
    1: "Processed",
    2: "Calculated",
    3: "Imported",
}
_STRING_KEYS = (
    "label",
    "system_text",
    "session_text",
    "user_text",
    "path",
    "date",
    "time",
    "x_units",
    "y_units",
    "z_units",
    "x_label",
    "y_label",
    "status_channel_text",
)


@dataclass(frozen=True)
class RHKObject:
    """One SM4 object-table entry."""

    type: int
    offset: int
    size: int


@dataclass(frozen=True)
class RHKSM4Page:
    """Decoded image page and metadata from an RHK SM4 file."""

    page_index: int
    data_type: int
    source: int
    page_type: int
    page_type_label: str
    scan_dir: int
    scan_dir_label: str
    x_size: int
    y_size: int
    x_scale: float
    y_scale: float
    z_scale: float
    xy_scale: float
    x_offset: float
    y_offset: float
    z_offset: float
    bias: float
    current: float
    angle: float
    label: str | None
    x_unit: str | None
    y_unit: str | None
    z_unit: str | None
    raw_data: np.ndarray
    physical_data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RHKSM4File:
    """Parsed RHK SM4 container."""

    path: Path
    page_count: int
    pages: list[RHKSM4Page]
    metadata: dict[str, Any]
    parser_notes: list[str]


def is_rhk_sm4(path_or_bytes) -> bool:
    """Return ``True`` when bytes contain the RHK SM4 magic at offset 2."""
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        data = bytes(path_or_bytes)
    else:
        try:
            with Path(path_or_bytes).open("rb") as fh:
                data = fh.read(MAGIC_OFFSET + len(SM4_MAGIC))
        except (OSError, ValueError):
            return False
    end = MAGIC_OFFSET + len(SM4_MAGIC)
    return len(data) >= end and data[MAGIC_OFFSET:end] == SM4_MAGIC


def read_rhk_sm4(path) -> RHKSM4File:
    """Parse an RHK SM4 container and decode supported image pages."""
    path = Path(path)
    data = path.read_bytes()
    if not is_rhk_sm4(data):
        raise ValueError(f"{path}: not an RHK SM4 file")
    if len(data) < MAGIC_TOTAL_SIZE + 12:
        raise ValueError(f"{path}: truncated RHK SM4 file header")

    notes: list[str] = []
    page_count = _u32(data, MAGIC_TOTAL_SIZE)
    object_count = _u32(data, MAGIC_TOTAL_SIZE + 4)
    object_field_size = _u32(data, MAGIC_TOTAL_SIZE + 8)
    if object_field_size not in (0, OBJECT_SIZE):
        notes.append(f"Unexpected file object field size {object_field_size}; using 12 bytes.")
    objects = parse_object_table(
        data,
        MAGIC_TOTAL_SIZE + 12,
        object_count,
        container_start=0,
        notes=notes,
    )

    pages = _parse_pages(data, objects, page_count, notes)
    metadata = {
        "file_object_count": object_count,
        "object_field_size": object_field_size,
        "image_page_count": len(pages),
        "parser_notes": list(notes),
    }
    return RHKSM4File(
        path=path,
        page_count=page_count,
        pages=pages,
        metadata=metadata,
        parser_notes=notes,
    )


def iter_rhk_sm4_images(path) -> Iterable[RHKSM4Page]:
    """Yield decoded image pages from *path*."""
    yield from read_rhk_sm4(path).pages


def read_sm4(path) -> Scan:
    """Load image pages from an RHK ``.sm4`` file as a ProbeFlow ``Scan``."""
    sm4 = read_rhk_sm4(path)
    if not sm4.pages:
        note = "; ".join(sm4.parser_notes) if sm4.parser_notes else "no image pages found"
        raise ValueError(f"{Path(path)}: no supported RHK SM4 image pages ({note})")

    first = sm4.pages[0]
    first_shape = first.physical_data.shape
    pages: list[RHKSM4Page] = []
    for page in sm4.pages:
        if page.physical_data.shape != first_shape:
            sm4.parser_notes.append(
                f"Skipped RHK image page {page.page_index} in Scan conversion: "
                f"shape {page.physical_data.shape} differs from first image shape {first_shape}."
            )
            continue
        pages.append(page)
    planes = [np.asarray(page.physical_data, dtype=np.float64) for page in pages]
    names = [_page_name(page) for page in pages]
    units = [page.z_unit or "" for page in pages]
    scan_range_m = _scan_range_m(first)
    header = {
        "RHK_SM4": True,
        "page_count": sm4.page_count,
        "parser_notes": list(sm4.parser_notes),
        "pages": [_page_metadata(page) for page in pages],
    }
    return Scan(
        planes=planes,
        plane_names=names,
        plane_units=units,
        plane_synthetic=[False] * len(planes),
        header=header,
        scan_range_m=scan_range_m,
        source_path=Path(path),
        source_format="sm4",
    )


def read_sm4_metadata(path):
    """Return :class:`~probeflow.core.metadata.ScanMetadata` for an RHK SM4 file."""
    from probeflow.core.metadata import metadata_from_scan

    return metadata_from_scan(read_sm4(path))


def parse_object_table(
    data: bytes,
    offset: int,
    count: int,
    *,
    container_start: int = 0,
    notes: list[str] | None = None,
) -> list[RHKObject]:
    """Parse ``count`` little-endian SM4 object records."""
    if count < 0 or count > 100000:
        raise ValueError(f"Implausible RHK object count: {count}")
    end = offset + count * OBJECT_SIZE
    if offset < 0 or end > len(data):
        raise ValueError("RHK object table extends outside file")
    objects: list[RHKObject] = []
    for i in range(count):
        typ, rel_offset, size = struct.unpack_from("<III", data, offset + i * OBJECT_SIZE)
        obj_offset = _resolve_offset(data, rel_offset, size, container_start)
        if obj_offset is None:
            if notes is not None:
                notes.append(
                    f"Skipped object {i}: type={typ}, offset={rel_offset}, size={size} outside file."
                )
            continue
        objects.append(RHKObject(int(typ), int(obj_offset), int(size)))
    return objects


def parse_page_header(data: bytes, offset: int = 0) -> dict[str, Any]:
    """Parse the useful fixed fields of an RHK page header.

    The offsets follow the public RHK/Gwyddion structure notes used by the
    SM4 importer. Unknown/reserved tail bytes are intentionally preserved only
    indirectly via parser notes; image import relies on dimensions, scales,
    units, and page labels rather than opaque reserved fields.
    """
    if offset < 0 or offset + PAGE_HEADER_SIZE > len(data):
        raise ValueError("RHK page header extends outside buffer")
    names_u32 = (
        "field_size", "string_count", "page_type", "data_sub_source",
        "line_type",
    )
    values: dict[str, Any] = {}
    pos = offset
    for name in names_u32:
        values[name] = _u32(data, pos)
        pos += 4
    values["x_coord"] = _i32(data, pos); pos += 4
    values["y_coord"] = _i32(data, pos); pos += 4
    for name in (
        "x_size", "y_size", "image_type", "scan_dir", "group_id", "data_size",
    ):
        values[name] = _u32(data, pos)
        pos += 4
    values["min_z_value"] = _i32(data, pos); pos += 4
    values["max_z_value"] = _i32(data, pos); pos += 4
    for name in (
        "x_scale", "y_scale", "z_scale", "xy_scale",
        "x_offset", "y_offset", "z_offset", "period",
        "bias", "current", "angle",
    ):
        values[name] = _f64(data, pos)
        pos += 8
    for name in ("color_info_count", "grid_x_size", "grid_y_size", "object_count"):
        values[name] = _u32(data, pos)
        pos += 4
    return values


def _parse_pages(
    data: bytes,
    file_objects: list[RHKObject],
    declared_page_count: int,
    notes: list[str],
) -> list[RHKSM4Page]:
    pages: list[RHKSM4Page] = []
    page_index_array = _first_object(file_objects, RHK_OBJECT_PAGE_INDEX_ARRAY)
    if page_index_array is not None:
        pages.extend(_parse_page_index_array(data, page_index_array, declared_page_count, notes))
    if not pages:
        page = _parse_page_from_objects(data, file_objects, 0, RHK_DATA_IMAGE, 0, notes)
        if page is not None:
            pages.append(page)
    return pages


def _parse_page_index_array(
    data: bytes,
    obj: RHKObject,
    declared_page_count: int,
    notes: list[str],
) -> list[RHKSM4Page]:
    if obj.size < PAGE_INDEX_ARRAY_SIZE:
        notes.append("RHK page index array is too small.")
        return []
    count = declared_page_count or obj.size // PAGE_INDEX_ARRAY_SIZE
    count = min(count, obj.size // PAGE_INDEX_ARRAY_SIZE)
    pages: list[RHKSM4Page] = []
    seen_offsets: set[int] = set()
    for i in range(count):
        entry_offset = obj.offset + i * PAGE_INDEX_ARRAY_SIZE
        fields = struct.unpack_from("<IIII", data, entry_offset + GUID_SIZE)
        candidates = [f for f in fields if 0 <= f < len(data) and f not in seen_offsets]
        page = None
        for candidate in candidates:
            try:
                page = _parse_page_index(data, candidate, i, notes)
            except Exception as exc:
                notes.append(f"Could not parse RHK page {i} at {candidate}: {exc}")
                page = None
            if page is not None:
                seen_offsets.add(candidate)
                break
        if page is None:
            data_type = fields[1] if len(fields) > 1 else -1
            notes.append(f"Skipped RHK page {i}: no parseable page index entry (fields={fields}, data_type={data_type}).")
        else:
            pages.append(page)
    return pages


def _parse_page_index(
    data: bytes,
    offset: int,
    page_index: int,
    notes: list[str],
) -> RHKSM4Page | None:
    if offset < 0 or offset + PAGE_INDEX_SIZE > len(data):
        raise ValueError("page index extends outside file")
    data_type = _u32(data, offset + GUID_SIZE)
    source = _u32(data, offset + GUID_SIZE + 4)
    object_count = _u32(data, offset + GUID_SIZE + 8)
    minor_version = _u32(data, offset + GUID_SIZE + 12)
    objects = parse_object_table(
        data,
        offset + PAGE_INDEX_SIZE,
        object_count,
        container_start=offset,
        notes=notes,
    )
    if data_type != RHK_DATA_IMAGE:
        notes.append(
            f"Skipped non-image RHK page: index={page_index}, data_type={data_type}, "
            f"source={source}, minor_version={minor_version}."
        )
        return None
    page = _parse_page_from_objects(data, objects, page_index, data_type, source, notes)
    if page is not None:
        page.metadata["minor_version"] = minor_version
    return page


def _parse_page_from_objects(
    data: bytes,
    objects: list[RHKObject],
    page_index: int,
    data_type: int,
    source: int,
    notes: list[str],
) -> RHKSM4Page | None:
    header_obj = _first_object(objects, RHK_OBJECT_PAGE_HEADER)
    data_obj = _first_object(objects, RHK_OBJECT_PAGE_DATA)
    if header_obj is None or data_obj is None:
        notes.append(f"Skipped RHK page {page_index}: missing page header or page data object.")
        return None
    header = parse_page_header(data, header_obj.offset)
    x_size = int(header["x_size"])
    y_size = int(header["y_size"])
    if x_size <= 0 or y_size <= 0 or x_size > 1_000_000 or y_size > 1_000_000:
        notes.append(f"Skipped RHK page {page_index}: implausible dimensions {x_size}x{y_size}.")
        return None
    strings = _parse_string_data(_object_bytes(data, _first_object(objects, RHK_OBJECT_STRING_DATA)))
    raw = _decode_image_payload(
        _object_bytes(data, data_obj),
        x_size=x_size,
        y_size=y_size,
        notes=notes,
        page_index=page_index,
    )
    if raw is None:
        return None
    z_scale = _finite_or(header["z_scale"], 1.0)
    z_offset = _finite_or(header["z_offset"], 0.0)
    physical = raw.astype(np.float64) * z_scale + z_offset
    page_type = int(header["page_type"])
    scan_dir = int(header["scan_dir"])
    label = strings.get("label") or PAGE_TYPE_LABELS.get(page_type)
    meta = dict(header)
    meta.update({
        "strings": strings,
        "source": source,
        "source_label": SOURCE_LABELS.get(source, f"RHK source {source}"),
        "page_type_label": PAGE_TYPE_LABELS.get(page_type, f"RHK page type {page_type}"),
        "scan_dir_label": SCAN_DIRECTION_LABELS.get(scan_dir, f"RHK scan direction {scan_dir}"),
    })
    return RHKSM4Page(
        page_index=page_index,
        data_type=data_type,
        source=source,
        page_type=page_type,
        page_type_label=meta["page_type_label"],
        scan_dir=scan_dir,
        scan_dir_label=meta["scan_dir_label"],
        x_size=x_size,
        y_size=y_size,
        x_scale=float(header["x_scale"]),
        y_scale=float(header["y_scale"]),
        z_scale=float(header["z_scale"]),
        xy_scale=float(header["xy_scale"]),
        x_offset=float(header["x_offset"]),
        y_offset=float(header["y_offset"]),
        z_offset=float(header["z_offset"]),
        bias=float(header["bias"]),
        current=float(header["current"]),
        angle=float(header["angle"]),
        label=label,
        x_unit=strings.get("x_units"),
        y_unit=strings.get("y_units"),
        z_unit=strings.get("z_units"),
        raw_data=raw,
        physical_data=physical,
        metadata=meta,
    )


def _decode_image_payload(
    payload: bytes,
    *,
    x_size: int,
    y_size: int,
    notes: list[str],
    page_index: int,
) -> np.ndarray | None:
    n = x_size * y_size
    expected = n * 4
    buf = payload
    if len(buf) not in (n * 2, expected, n * 8):
        try:
            buf = zlib.decompress(payload)
            notes.append(f"Decompressed RHK page {page_index} image payload with zlib.")
        except zlib.error:
            notes.append(
                f"Skipped RHK page {page_index}: image payload size {len(payload)} "
                f"does not match {x_size}x{y_size} numeric data."
            )
            return None
    if len(buf) == n * 2:
        arr = np.frombuffer(buf, dtype="<i2", count=n).astype(np.float64)
    elif len(buf) == expected:
        arr = np.frombuffer(buf, dtype="<i4", count=n).astype(np.float64)
    elif len(buf) == n * 8:
        arr = np.frombuffer(buf, dtype="<f8", count=n).astype(np.float64)
    else:
        notes.append(
            f"Skipped RHK page {page_index}: decompressed payload size {len(buf)} "
            f"still does not match expected image data."
        )
        return None
    return arr.reshape((y_size, x_size)).copy()


def _parse_string_data(payload: bytes | None) -> dict[str, str]:
    if not payload:
        return {}
    text = ""
    for encoding in ("utf-16-le", "utf-8", "latin-1"):
        try:
            text = payload.decode(encoding, errors="ignore")
            break
        except UnicodeError:
            continue
    parts = [part.strip("\x00 \r\n\t") for part in text.replace("\x00\x00", "\x00").split("\x00")]
    parts = [part for part in parts if part]
    return {key: value for key, value in zip(_STRING_KEYS, parts)}


def _object_bytes(data: bytes, obj: RHKObject | None) -> bytes | None:
    if obj is None:
        return None
    if obj.offset < 0 or obj.offset + obj.size > len(data):
        return None
    return data[obj.offset:obj.offset + obj.size]


def _first_object(objects: list[RHKObject], obj_type: int) -> RHKObject | None:
    return next((obj for obj in objects if obj.type == obj_type), None)


def _resolve_offset(
    data: bytes,
    offset: int,
    size: int,
    container_start: int,
) -> int | None:
    candidates = [offset]
    if container_start:
        candidates = [container_start + offset, offset]
    for candidate in candidates:
        if 0 <= candidate <= len(data) and size >= 0 and candidate + size <= len(data):
            return int(candidate)
    return None


def _page_name(page: RHKSM4Page) -> str:
    label = page.label or page.page_type_label
    return f"{label} [{page.scan_dir_label}]"


def _page_metadata(page: RHKSM4Page) -> dict[str, Any]:
    data = dict(page.metadata)
    data.update({
        "page_index": page.page_index,
        "data_type": page.data_type,
        "source": page.source,
        "page_type": page.page_type,
        "scan_dir": page.scan_dir,
        "label": page.label,
        "x_unit": page.x_unit,
        "y_unit": page.y_unit,
        "z_unit": page.z_unit,
    })
    return data


def _scan_range_m(page: RHKSM4Page) -> tuple[float, float]:
    # RHK SM4 stores per-axis scale fields plus an xy scale. The common
    # convention used here is metres-per-pixel = axis_scale * xy_scale when
    # xy_scale is finite/non-zero, otherwise axis_scale. The raw scale fields
    # are kept in metadata so this can be verified against Gwyddion fixtures.
    xy_scale = _finite_or(page.xy_scale, 1.0)
    if xy_scale == 0.0:
        xy_scale = 1.0
    dx = _finite_or(page.x_scale, 1.0) * xy_scale
    dy = _finite_or(page.y_scale, 1.0) * xy_scale
    return abs(dx) * page.x_size, abs(dy) * page.y_size


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else fallback


def _u32(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", data, offset)[0])


def _i32(data: bytes, offset: int) -> int:
    return int(struct.unpack_from("<i", data, offset)[0])


def _f64(data: bytes, offset: int) -> float:
    return float(struct.unpack_from("<d", data, offset)[0])
