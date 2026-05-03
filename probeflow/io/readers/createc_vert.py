"""Low-level Createc ``.VERT`` spectroscopy decoding and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

from probeflow.io.common import (
    _f,
    find_hdr,
    get_dac_bits,
    i_scale_a_per_dac,
    v_per_dac,
    z_scale_m_per_dac,
)
from probeflow.core.source_identity import build_source_identity

log = logging.getLogger(__name__)

_SPEC_OUTPUT_CHANNELS: dict[str, tuple[str, ...]] = {
    "ParVERT30": (
        "I",
        "dI/dV",
        "d2I/dV2",
        "ADC0",
        "ADC1",
        "ADC2",
        "ADC3",
        "NA01",
        "NA02",
        "NA03",
        "di_q",
        "di2_q",
        "DAC0",
    ),
    "ParVERT32": (
        "I",
        "dI/dV",
        "d2I/dV2",
        "ADC0",
        "ADC1",
        "ADC2",
        "ADC3",
        "NA01",
        "NA02",
        "NA03",
        "di_q",
        "di2_q",
        "DAC0",
    ),
}


@dataclass(frozen=True)
class CreatecSpecChannelInfo:
    """Best-known interpretation for one decoded Createc spectroscopy column."""

    column_index: int
    raw_name: str
    canonical_name: str
    raw_unit: str
    unit: str
    scale_factor: float
    semantic: str
    origin: str


@dataclass(frozen=True)
class CreatecVertDecodeReport:
    """Structured report for a decoded Createc ``.VERT`` spectroscopy file."""

    path: Path
    source: dict[str, Any]
    header: dict[str, str]
    file_version: str
    data_offset: int
    params_line: str
    spec_total_points: int
    spec_pos_x: int
    spec_pos_y: int
    channel_code: int
    output_channel_count_marker: str
    column_names: tuple[str, ...]
    raw_table_shape: tuple[int, int]
    channel_info: tuple[CreatecSpecChannelInfo, ...]
    raw_columns: dict[str, np.ndarray] | None
    bias_min_mv: float | None
    bias_max_mv: float | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def read_createc_vert_report(
    path,
    *,
    include_arrays: bool = True,
) -> CreatecVertDecodeReport:
    """Decode a Createc ``.VERT`` file into a structured report."""

    path = Path(path)
    raw = path.read_bytes()
    data_pos = raw.find(b"DATA")
    if data_pos < 0:
        raise ValueError(f"{path.name}: missing DATA marker")

    header, file_version = _parse_createc_vert_header_and_version(raw[:data_pos])
    data_section = raw[data_pos:]
    params_line, data_text, data_offset = _split_data_section(path, data_pos, data_section)
    spec_total, spec_x, spec_y, channel_code, out_marker = _parse_params_line(
        path, params_line
    )
    base_names = _base_column_names(file_version, out_marker)
    selected = _selected_output_channels(file_version, channel_code)
    expected_names = ("idx", *base_names, *selected)

    if include_arrays:
        arr = _parse_numeric_table(path, data_text)
        row_count, col_count = arr.shape
    else:
        row_count, col_count, bias_min, bias_max = _summarise_numeric_table(
            path, data_text
        )
        arr = None

    if col_count < len(expected_names):
        raise ValueError(
            f"{path.name}: expected at least {len(expected_names)} data columns "
            f"from .VERT layout, got {col_count}"
        )

    warnings: list[str] = []
    column_names = list(expected_names)
    if col_count > len(expected_names):
        for k in range(len(expected_names), col_count):
            name = f"Raw column {k}"
            column_names.append(name)
            warnings.append(f"preserved unrecognised spectroscopy column {k}")

    if row_count != spec_total:
        warnings.append(
            f"params line reports {spec_total} point(s), parsed {row_count} row(s)"
        )

    if include_arrays and arr is not None:
        bias_min = float(np.nanmin(arr[:, 1])) if row_count and col_count > 1 else None
        bias_max = float(np.nanmax(arr[:, 1])) if row_count and col_count > 1 else None
        raw_columns = {
            name: np.asarray(arr[:, i], dtype=np.float64).copy()
            for i, name in enumerate(column_names)
        }
    else:
        raw_columns = None

    bits = get_dac_bits(header)
    # VERT DAC columns use the full +/-10 V span.
    vpd = v_per_dac(bits) * 2
    scale_factors = {
        "V": 1e-3,
        "Z": z_scale_m_per_dac(header, vpd),
        "I": i_scale_a_per_dac(header, vpd, negative=False),
    }
    channel_info, channel_warnings = _channel_info(column_names, scale_factors)
    warnings.extend(channel_warnings)

    return CreatecVertDecodeReport(
        path=path,
        source=build_source_identity(
            path,
            source_format="createc_vert",
            item_type="spectrum",
            data_offset=data_offset,
        ),
        header=header,
        file_version=file_version,
        data_offset=data_offset,
        params_line=params_line,
        spec_total_points=spec_total,
        spec_pos_x=spec_x,
        spec_pos_y=spec_y,
        channel_code=channel_code,
        output_channel_count_marker=out_marker,
        column_names=tuple(column_names),
        raw_table_shape=(row_count, col_count),
        channel_info=channel_info,
        raw_columns=raw_columns,
        bias_min_mv=bias_min,
        bias_max_mv=bias_max,
        warnings=tuple(warnings),
    )


def parse_createc_vert_header(path) -> dict[str, str]:
    """Read only the header of a Createc ``.VERT`` file."""

    path = Path(path)
    chunks: list[bytes] = []
    tail = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            search = tail + chunk
            pos = search.find(b"DATA")
            if pos >= 0:
                prev_len = sum(len(c) for c in chunks)
                chunks.append(chunk)
                blob = b"".join(chunks)
                abs_pos = prev_len - len(tail) + pos
                header, _version = _parse_createc_vert_header_and_version(
                    blob[:abs_pos]
                )
                return header
            chunks.append(chunk)
            tail = chunk[-3:]
    raise ValueError(f"{path.name}: missing DATA marker")


def detect_createc_vert_time_trace(
    hdr: dict[str, str],
    bias_mv: np.ndarray,
    threshold_mv: float,
) -> bool:
    """Return True when a VERT file is a time trace rather than a bias sweep."""

    vpoint_volts: list[float] = []
    for i in range(8):
        t = _f(hdr.get(f"Vpoint{i}.t", "0"), 0.0)
        v = _f(hdr.get(f"Vpoint{i}.V", None), None)
        if t is not None and t > 0 and v is not None:
            vpoint_volts.append(v)

    if len(vpoint_volts) >= 2:
        return (max(vpoint_volts) - min(vpoint_volts)) < threshold_mv

    if bias_mv.size == 0:
        return True
    return float(np.nanmax(bias_mv) - np.nanmin(bias_mv)) < threshold_mv


def _parse_createc_vert_header_and_version(hb: bytes) -> tuple[dict[str, str], str]:
    if any(b > 0x7F for b in hb):
        log.warning(
            "_parse_createc_vert_header: non-ASCII bytes found; decoded as latin-1"
        )

    lines = hb.splitlines()
    version = "ParVERT30"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(b"[") and stripped.endswith(b"]"):
            version = (
                stripped[1:-1].decode("latin-1", errors="replace").strip()
                or version
            )
            break

    hdr: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if b"=" not in line:
            continue
        k, _, v = line.partition(b"=")
        if b"/" in k:
            k = k.split(b"/")[-1]
        key = k.decode("latin-1", errors="replace").strip()
        val = v.decode("latin-1", errors="replace").strip()
        if key:
            hdr[key] = val
    return hdr, version


def _split_data_section(
    path: Path,
    data_pos: int,
    data_section: bytes,
) -> tuple[str, str, int]:
    eol = b"\r\n" if data_section[4:6] == b"\r\n" else b"\n"
    eol_len = len(eol)
    first_eol = data_section.find(eol)
    if first_eol < 0:
        raise ValueError(f"{path.name}: missing line ending after DATA marker")
    params_start = first_eol + eol_len
    second_eol = data_section.find(eol, params_start)
    if second_eol < 0:
        raise ValueError(f"{path.name}: missing spectroscopy params line")
    params_bytes = data_section[params_start:second_eol]
    params_line = params_bytes.decode("latin-1", errors="replace").strip()
    data_offset = data_pos + second_eol + eol_len
    data_text = data_section[second_eol + eol_len :].decode(
        "latin-1", errors="replace"
    )
    return params_line, data_text, data_offset


def _parse_params_line(path: Path, params_line: str) -> tuple[int, int, int, int, str]:
    nums = [int(x) for x in re.findall(r"[-+]?\d+", params_line)]
    if len(nums) < 4:
        raise ValueError(
            f"{path.name}: malformed .VERT spectroscopy params line: {params_line!r}"
        )
    out_marker = f"v{nums[6]}" if len(nums) >= 7 else "v2"
    return nums[0], nums[1], nums[2], nums[3], out_marker


def _base_column_names(file_version: str, out_marker: str) -> tuple[str, ...]:
    if file_version == "ParVERT32" and out_marker == "v3":
        return ("V", "Z", "X")
    return ("V", "Z")


def _selected_output_channels(file_version: str, channel_code: int) -> tuple[str, ...]:
    names = _SPEC_OUTPUT_CHANNELS.get(file_version, _SPEC_OUTPUT_CHANNELS["ParVERT30"])
    selected = [
        name
        for i, name in enumerate(names)
        if bool(channel_code & (1 << i))
    ]
    return tuple(selected)


def _parse_numeric_table(path: Path, data_text: str) -> np.ndarray:
    clean = "\n".join(
        ln.rstrip("\t ") for ln in data_text.splitlines() if ln.strip()
    )
    if not clean:
        raise ValueError(f"{path.name}: no data rows found after DATA marker")
    try:
        arr = np.loadtxt(StringIO(clean), dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path.name}: failed to parse data section - {exc}") from exc
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _summarise_numeric_table(
    path: Path,
    data_text: str,
) -> tuple[int, int, float | None, float | None]:
    n_rows = 0
    col_count: int | None = None
    bias_min: float | None = None
    bias_max: float | None = None

    for line in data_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values = np.fromstring(
                stripped.replace("\t", " "),
                sep=" ",
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ValueError(f"{path.name}: failed to parse data row - {exc}") from exc
        if values.size == 0:
            continue
        if col_count is None:
            col_count = int(values.size)
        elif values.size != col_count:
            raise ValueError(
                f"{path.name}: inconsistent data column count "
                f"({values.size} != {col_count})"
            )
        if values.size > 1:
            bias = float(values[1])
            bias_min = bias if bias_min is None else min(bias_min, bias)
            bias_max = bias if bias_max is None else max(bias_max, bias)
        n_rows += 1

    if col_count is None:
        raise ValueError(f"{path.name}: no data rows found after DATA marker")
    return n_rows, col_count, bias_min, bias_max


def _channel_info(
    column_names: list[str],
    scale_factors: dict[str, float],
) -> tuple[tuple[CreatecSpecChannelInfo, ...], list[str]]:
    warnings: list[str] = []
    infos: list[CreatecSpecChannelInfo] = []
    seen: set[str] = set()

    for idx, raw_name in enumerate(column_names):
        if raw_name == "idx":
            continue

        canonical = _unique_name(raw_name, seen)
        origin = "base" if raw_name in {"V", "Z", "X"} else "bitmask"
        raw_unit = "DAC"
        unit = "unknown"
        scale = 1.0
        semantic = "unknown"

        if raw_name == "V":
            raw_unit = "mV"
            unit = "V"
            scale = scale_factors["V"]
            semantic = "bias"
        elif raw_name == "Z":
            unit = "m"
            scale = scale_factors["Z"]
            semantic = "z"
        elif raw_name == "I":
            unit = "A"
            scale = scale_factors["I"]
            semantic = "current"
        elif raw_name.startswith("ADC") or raw_name == "DAC0":
            semantic = "instrument"
            warnings.append(f"{raw_name} decoded without physical calibration")
        elif raw_name in {"dI/dV", "d2I/dV2", "di_q", "di2_q"}:
            semantic = "lockin"
            warnings.append(f"{raw_name} decoded without physical calibration")
        elif raw_name.startswith("Raw column"):
            origin = "extra"
            warnings.append(f"{raw_name} decoded without physical calibration")

        infos.append(
            CreatecSpecChannelInfo(
                column_index=idx,
                raw_name=raw_name,
                canonical_name=canonical,
                raw_unit=raw_unit,
                unit=unit,
                scale_factor=float(scale),
                semantic=semantic,
                origin=origin,
            )
        )

    return tuple(infos), warnings


def _unique_name(name: str, seen: set[str]) -> str:
    if name not in seen:
        seen.add(name)
        return name
    i = 2
    while f"{name} {i}" in seen:
        i += 1
    unique = f"{name} {i}"
    seen.add(unique)
    return unique
