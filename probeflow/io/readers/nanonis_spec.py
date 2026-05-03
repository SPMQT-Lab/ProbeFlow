"""Reader for Nanonis ``.dat`` point-spectroscopy files.

File format:
    * ASCII, tab-separated, Windows line endings.
    * Header: ``key<TAB>value`` per line (value may be empty).
    * After a ``[DATA]`` line: one header row ``Name (Unit)<TAB>Name (Unit)...``
      followed by numeric rows.

We parse into :class:`probeflow.io.spectroscopy.SpecData` — the same container used
for Createc .VERT files, so downstream code is vendor-agnostic.
"""

from __future__ import annotations

import logging
import re
from io import StringIO
from pathlib import Path
from typing import Optional, Union

import numpy as np

from probeflow.core.source_identity import build_source_identity
from probeflow.io.spectroscopy import (
    SpecChannel,
    SpecData,
    SpecMetadata,
    _add_channel_metadata_overlay,
    infer_spec_channel_roles,
)

log = logging.getLogger(__name__)


# "Name (unit)" — unit may be empty; both parts are trimmed.
_COLUMN_RE = re.compile(r"^(.+?)\s*\(([^)]*)\)\s*$")

# Secondary-channel name patterns worth showing by default alongside Current.
# Matched as "startswith" against the bare channel name (no unit suffix).
_DEFAULT_SECONDARY_PREFIXES = (
    "LI Demod",       # Nanonis lock-in demod channels
    "LockIn",         # LockIn AVG / LockIn
    "OC M1 Freq",     # Oscillation control frequency shift (AFM)
    "Input 6",        # Kelvin probe / bias-dependent auxiliary
)


# Map Nanonis Experiment tag → normalised sweep_type label used elsewhere.
_SWEEP_TYPE_MAP = {
    "bias spectroscopy": "bias_sweep",
    "z spectroscopy":    "z_sweep",
    "history data":      "time_trace",
}


def read_nanonis_spec(path: Union[str, Path]) -> SpecData:
    """Read a Nanonis ``.dat`` spectroscopy file into a :class:`SpecData`."""
    path = Path(path)
    text = path.read_text(encoding="latin-1", errors="replace")
    lines = text.splitlines()

    # Split header from data on the [DATA] marker.
    data_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "[DATA]":
            data_idx = i
            break
    if data_idx is None:
        raise ValueError(f"{path.name}: missing [DATA] marker")

    header_lines = lines[:data_idx]
    if data_idx + 1 >= len(lines):
        raise ValueError(f"{path.name}: no column header after [DATA]")
    col_header = lines[data_idx + 1]
    data_lines = [ln for ln in lines[data_idx + 2:] if ln.strip()]

    # Parse header into a dict (key<TAB>value; trailing tabs stripped).
    hdr: dict[str, str] = {}
    for line in header_lines:
        if "\t" not in line:
            continue
        key, _, rest = line.partition("\t")
        value = rest.split("\t")[0] if rest else ""
        key = key.strip()
        if key:
            hdr[key] = value.strip()

    experiment = hdr.get("Experiment", "").strip().lower()
    sweep_type = _SWEEP_TYPE_MAP.get(experiment, experiment.replace(" ", "_") or "unknown")

    # Parse column header → list of (raw_label, name, unit).
    raw_columns = col_header.split("\t")
    columns: list[tuple[str, str, str]] = []
    for raw in raw_columns:
        raw = raw.strip()
        if not raw:
            continue
        m = _COLUMN_RE.match(raw)
        if m:
            name = m.group(1).strip()
            unit = m.group(2).strip()
        else:
            name, unit = raw, ""
        columns.append((raw, name, unit))

    if not columns:
        raise ValueError(f"{path.name}: empty column header")

    # Parse the numeric block with np.loadtxt (fast, tolerant of trailing tabs).
    clean = "\n".join(ln.rstrip("\t ") for ln in data_lines)
    if not clean:
        raise ValueError(f"{path.name}: no data rows after [DATA]")
    try:
        arr = np.loadtxt(StringIO(clean), dtype=np.float64, delimiter="\t")
        if arr.ndim == 1:
            # Either a single row or a single column — reshape as one row.
            arr = arr.reshape(1, -1)
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: failed to parse data block — {exc}"
        ) from exc

    if arr.shape[1] != len(columns):
        raise ValueError(
            f"{path.name}: column header has {len(columns)} names but data "
            f"block has {arr.shape[1]} columns"
        )

    # Build channels dict keyed by channel NAME (without the (unit) suffix).
    channels: dict[str, np.ndarray] = {}
    y_units: dict[str, str] = {}
    channel_info: dict[str, SpecChannel] = {}
    channel_order: list[str] = []
    seen_names: set[str] = set()
    for i, (raw, name, unit) in enumerate(columns):
        key = _unique_channel_key(name, seen_names)
        if key != name:
            log.warning(
                "%s: duplicate channel name %r — using key %r",
                path.name,
                name,
                key,
            )
        channels[key] = arr[:, i]
        y_units[key] = unit
        channel_info[key] = _spec_channel_from_column(raw, key, name, unit)
        channel_order.append(key)

    # Pick X axis: prefer a "Bias calc" column for bias sweeps, else "Bias",
    # else fall back to the first column.
    x_label = None
    x_unit = None
    x_array = None
    if sweep_type == "bias_sweep":
        x_candidates = ("Bias calc", "Bias")
        for cand in x_candidates:
            if cand in channels:
                x_array = channels[cand]
                x_unit = y_units.get(cand, "V")
                x_label = f"{cand} ({x_unit})" if x_unit else cand
                break
    if x_array is None:
        first_name = channel_order[0]
        x_array = channels[first_name]
        x_unit = y_units.get(first_name, "")
        x_label = f"{first_name} ({x_unit})" if x_unit else first_name

    # Pick default channels to display: a forward Current plus any known
    # secondary measurements (lock-in, freq-shift, Kelvin input).
    default_channels: list[str] = []
    # Prefer explicitly averaged variants over raw first/second/…
    current_prefs = ("Current [AVG]", "Current")
    seen_current: Optional[str] = None
    for pref in current_prefs:
        for name in channel_order:
            if "[bwd]" in name or "[BWD]" in name.upper():
                continue
            if name.startswith(pref):
                seen_current = name
                break
        if seen_current:
            break
    if seen_current:
        default_channels.append(seen_current)

    for name in channel_order:
        if name in default_channels:
            continue
        if "[bwd]" in name or "[BWD]" in name.upper():
            continue
        if any(name.startswith(pref) for pref in _DEFAULT_SECONDARY_PREFIXES):
            default_channels.append(name)

    if not default_channels and channel_order:
        # Always show something — pick the first non-X channel.
        for name in channel_order:
            if name != (x_label or "").split(" (")[0]:
                default_channels.append(name)
                break

    # Tip position from the header.
    pos_x_m = _parse_header_float(hdr, "X (m)")
    pos_y_m = _parse_header_float(hdr, "Y (m)")

    metadata = {
        "filename": path.name,
        "sweep_type": sweep_type,
        "n_points": int(arr.shape[0]),
        "title": hdr.get("Experiment", ""),
        "experiment": hdr.get("Experiment", ""),
        "source": build_source_identity(
            path,
            source_format="nanonis_dat_spectrum",
            item_type="spectrum",
            data_offset=_data_offset_bytes(path),
        ),
    }
    _add_channel_metadata_overlay(
        metadata,
        channel_info,
        channel_order,
        [channel_info[name].source_name for name in channel_order],
    )

    log.info(
        "%s: %s, %d pts, %d channels, pos=(%.3g, %.3g) m",
        path.name, sweep_type, metadata["n_points"], len(channels),
        pos_x_m, pos_y_m,
    )

    return SpecData(
        header=hdr,
        channels=channels,
        x_array=x_array,
        x_label=x_label,
        x_unit=x_unit,
        y_units=y_units,
        position=(pos_x_m, pos_y_m),
        metadata=metadata,
        channel_order=channel_order,
        default_channels=default_channels,
        channel_info=channel_info,
    )


def read_nanonis_spec_metadata(path: Union[str, Path]) -> SpecMetadata:
    """Read Nanonis spectroscopy metadata without loading numeric arrays."""
    path = Path(path)
    hdr, columns, n_points = _read_nanonis_spec_header_summary(path)
    if n_points <= 0:
        raise ValueError(f"{path.name}: no data rows after [DATA]")

    experiment = hdr.get("Experiment", "").strip().lower()
    sweep_type = _SWEEP_TYPE_MAP.get(experiment, experiment.replace(" ", "_") or "unknown")

    channel_info = _channel_info_from_columns(columns)
    channel_order = list(channel_info)
    channel_names = tuple(channel_order)
    units = tuple(channel_info[name].unit for name in channel_order)
    pos_x_m = _parse_header_float(hdr, "X (m)")
    pos_y_m = _parse_header_float(hdr, "Y (m)")
    bias = _parse_optional_header_float(hdr, "Bias>Bias (V)")
    comment = hdr.get("Experiment", "").strip() or None
    acquisition_datetime = (
        hdr.get("Saved Date", "").strip()
        or hdr.get("Date", "").strip()
        or None
    )
    metadata = {
        "filename": path.name,
        "sweep_type": sweep_type,
        "n_points": n_points,
        "title": comment or "",
        "experiment": hdr.get("Experiment", ""),
        "source": build_source_identity(
            path,
            source_format="nanonis_dat_spectrum",
            item_type="spectrum",
            data_offset=_data_offset_bytes(path),
        ),
    }
    _add_channel_metadata_overlay(
        metadata,
        channel_info,
        channel_order,
        [channel_info[name].source_name for name in channel_order],
    )
    return SpecMetadata(
        path=path,
        source_format="nanonis_dat_spectrum",
        channels=channel_names,
        units=units,
        position=(pos_x_m, pos_y_m),
        metadata=metadata,
        bias=bias,
        comment=comment,
        acquisition_datetime=acquisition_datetime,
        raw_header=hdr,
        channel_info=tuple(channel_info[name] for name in channel_order),
    )


def _channel_info_from_columns(
    columns: list[tuple[str, str, str]],
) -> dict[str, SpecChannel]:
    channel_info: dict[str, SpecChannel] = {}
    seen_names: set[str] = set()
    for raw, name, unit in columns:
        key = _unique_channel_key(name, seen_names)
        channel_info[key] = _spec_channel_from_column(raw, key, name, unit)
    return channel_info


def _unique_channel_key(name: str, seen: set[str]) -> str:
    if name not in seen:
        seen.add(name)
        return name
    idx = 2
    while f"{name} {idx}" in seen:
        idx += 1
    key = f"{name} {idx}"
    seen.add(key)
    return key


def _spec_channel_from_column(
    raw: str,
    key: str,
    source_name: str,
    unit: str,
) -> SpecChannel:
    return SpecChannel(
        key=key,
        source_name=source_name,
        source_label=raw,
        unit=unit,
        roles=infer_spec_channel_roles(source_name),
        display_label=key,
    )


def _read_nanonis_spec_header_summary(
    path: Path,
) -> tuple[dict[str, str], list[tuple[str, str, str]], int]:
    """Stream header, column names, and row count from a Nanonis .dat file."""
    hdr: dict[str, str] = {}
    columns: list[tuple[str, str, str]] = []
    n_points = 0
    in_data = False
    have_column_header = False

    with path.open("r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not in_data:
                if stripped == "[DATA]":
                    in_data = True
                    continue
                if "\t" not in line:
                    continue
                key, _, rest = line.partition("\t")
                value = rest.split("\t")[0] if rest else ""
                key = key.strip()
                if key:
                    hdr[key] = value.strip()
                continue

            if not have_column_header:
                columns = _parse_column_header(line)
                if not columns:
                    raise ValueError(f"{path.name}: empty column header")
                have_column_header = True
                continue

            if stripped:
                _parse_numeric_row_summary(path, line, len(columns))
                n_points += 1

    if not in_data:
        raise ValueError(f"{path.name}: missing [DATA] marker")
    if not have_column_header:
        raise ValueError(f"{path.name}: no column header after [DATA]")
    return hdr, columns, n_points


def _parse_numeric_row_summary(
    path: Path,
    line: str,
    expected_columns: int,
) -> None:
    parts = line.rstrip("\r\n").rstrip("\t ").split("\t")
    if len(parts) != expected_columns:
        raise ValueError(
            f"{path.name}: data row has {len(parts)} column(s), "
            f"expected {expected_columns}"
        )
    try:
        for value in parts:
            float(value)
    except ValueError as exc:
        raise ValueError(f"{path.name}: failed to parse data row - {exc}") from exc


def _parse_column_header(col_header: str) -> list[tuple[str, str, str]]:
    columns: list[tuple[str, str, str]] = []
    for raw in col_header.split("\t"):
        raw = raw.strip()
        if not raw:
            continue
        m = _COLUMN_RE.match(raw)
        if m:
            name = m.group(1).strip()
            unit = m.group(2).strip()
        else:
            name, unit = raw, ""
        columns.append((raw, name, unit))
    return columns


def _data_offset_bytes(path: Path) -> int | None:
    """Return byte offset where numeric Nanonis spectroscopy rows begin."""
    with path.open("rb") as fh:
        marker_seen = False
        while True:
            line = fh.readline()
            if not line:
                return None
            if marker_seen:
                return fh.tell()
            if line.strip() == b"[DATA]":
                marker_seen = True


def _parse_header_float(hdr: dict[str, str], key: str) -> float:
    raw = hdr.get(key, "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _parse_optional_header_float(hdr: dict[str, str], key: str) -> Optional[float]:
    raw = hdr.get(key, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
