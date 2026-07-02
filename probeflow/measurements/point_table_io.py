"""GUI-free readers for point-position tables (CSV / JSON) for Particle Statistics.

ProbeFlow can analyse externally produced point collections by importing a file
from disk.  This module sniffs a file to understand its shape, then loads it into
one or more :class:`~probeflow.measurements.feature_sets.FeatureSet` objects that
the Particle Statistics UI already knows how to analyse and pool.

Supported inputs:

* **Generic CSV** — two position columns (with or without a leading
  particle-number / id column), delimiter and header auto-detected, units
  inferred from header names (``x_px`` / ``x_nm`` / ``x_m`` / ``x_phys`` + unit
  columns) or supplied by the caller.
* **ProbeFlow CSV exports** — Feature Finder
  (``index,x_px,y_px,x_nm,y_nm,value``) and the measurements point export
  (``point_id,x_px,y_px,x_phys,y_phys,...,x_unit,...``).
* **ProbeFlow JSON** — Feature Counting ``write_json`` particle/detection files
  (``meta`` + ``items``, with embedded calibration) and a saved
  :class:`FeatureSetStore` (``{"version", "feature_sets": [...]}``).

The CSV path has no third-party dependencies (numpy + stdlib only).  Feature
Counting JSON is routed through the AdStat adapter's
``feature_counting_to_particle_table`` when available, falling back to a direct
field read if AdStat is not installed.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from probeflow.measurements.feature_sets import FeatureSet, FeatureSetStore


# metres per unit; ``px`` is special (needs a pixel size, handled separately)
_UNIT_SCALE_M: dict[str, float] = {
    "m": 1.0,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    "angstrom": 1e-10,
    "pm": 1e-12,
}
ACCEPTED_UNITS = ("px", "nm", "angstrom", "pm", "um", "m")

# Header-name hints, matched after :func:`_normalise_header_name` (lower-cased,
# BOM/whitespace stripped, "x (nm)" / "x [nm]" / "x.nm" collapsed to "x_nm").
_PX_NAMES = {"x_px": "y_px", "xpx": "ypx", "x_pixel": "y_pixel", "col": "row", "px_x": "px_y"}
_NM_NAMES = {"x_nm": "y_nm", "xnm": "ynm"}
_M_NAMES = {"x_m": "y_m", "centroid_x_m": "centroid_y_m", "x_meter": "y_meter"}
_UM_NAMES = {"x_um": "y_um", "xum": "yum", "x_µm": "y_µm"}
_A_NAMES = {"x_a": "y_a", "x_å": "y_å", "x_ang": "y_ang", "x_angstrom": "y_angstrom"}
_PM_NAMES = {"x_pm": "y_pm", "xpm": "ypm"}
_PHYS_NAMES = {"x_phys": "y_phys", "x_physical": "y_physical"}
_BARE_NAMES = {"x": "y", "x_pos": "y_pos", "posx": "posy", "pos_x": "pos_y", "xpos": "ypos"}
# Coordinate-looking names whose unit is genuinely ambiguous. ImageJ Results
# tables call the centroid columns XM/YM, but their unit is whatever the image
# calibration was (pixels, µm, ...): treating them as metres would silently
# import positions wrong by many orders of magnitude.
_AMBIGUOUS_NAMES = {"xm": "ym"}

_ID_HEADERS = {"index", "id", "point_id", "particle", "particle_id", "n", "#", "no", "num"}
_FRAME_HEADERS = ("frame", "frame_id", "slice", "image", "image_id", "stack")
_COMMENT_PREFIXES = ("#", "%", "//")


@dataclass(frozen=True)
class PointTablePreview:
    """What ``sniff_point_table`` learned about a file before a full load."""

    path: str
    kind: str  # generic_csv | probeflow_csv | probeflow_json | feature_set_store_json
    n_points: int
    n_sets: int = 1
    delimiter: str | None = None
    has_header: bool = False
    has_id_column: bool = False
    columns: tuple[str, ...] = ()
    units: str = "unknown"  # px | nm | angstrom | pm | um | m | unknown
    scan_range_m: tuple[float, float] | None = None  # embedded calibration if present
    image_shape: tuple[int, int] | None = None
    bbox_raw: tuple[float, float, float, float] | None = None  # xmin,ymin,xmax,ymax in `units`
    needs_calibration: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Which columns will be read, for a dialog preview to highlight.
    x_col: int | None = None
    y_col: int | None = None
    sample_rows: tuple[tuple[str, ...], ...] = ()  # first data rows, as read
    # Per-axis pixel size derived from paired px+physical columns, when present.
    pixel_size_m: tuple[float, float] | None = None
    frame_column: str | None = None  # multi-image column driving an N-set split
    n_dropped_rows: int = 0  # rows skipped because x/y did not parse as numbers


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def sniff_point_table(path: str | Path) -> PointTablePreview:
    """Inspect a file and report its detected format without fully loading it."""

    p = Path(path)
    if p.suffix.lower() == ".json":
        return _sniff_json(p)
    return _sniff_csv(p)


def load_point_table(
    path: str | Path,
    *,
    units: str | None = None,
    scan_range_m: tuple[float, float] | None = None,
    image_shape: tuple[int, int] | None = None,
    name: str | None = None,
) -> list[FeatureSet]:
    """Load a point file into one or more :class:`FeatureSet` objects.

    ``units`` / ``scan_range_m`` / ``image_shape`` override the values inferred by
    :func:`sniff_point_table` and are required for CSV files whose units cannot be
    inferred.  ProbeFlow JSON files carry their own calibration and ignore these
    unless explicitly overridden.  A CSV with a multi-image column
    (frame/slice/image/...) is split into one set per image, all sharing one
    calibration and one re-centring shift so the sets stay poolable.
    """

    p = Path(path)
    label = name or p.stem
    if p.suffix.lower() == ".json":
        preview = _sniff_json(p)
        if preview.kind == "feature_set_store_json":
            return list(
                FeatureSetStore.from_dict(json.loads(p.read_text(encoding="utf-8"))).all()
            )
        return [_load_probeflow_json(p, preview, scan_range_m, image_shape, label)]

    # CSV (generic or ProbeFlow)
    parsed = _parse_csv(p)
    resolved_units = units or parsed.units
    if resolved_units not in ACCEPTED_UNITS:
        raise ValueError(
            "could not infer position units; specify units as one of "
            f"{ACCEPTED_UNITS}"
        )
    if len(parsed.xy) == 0:
        if parsed.n_dropped_rows:
            raise ValueError(
                f"no usable point rows: {parsed.n_dropped_rows} row(s) failed to "
                "parse as numbers (check the delimiter and decimal separator)"
            )
        raise ValueError("no point rows found in file")
    if scan_range_m is None:
        scan_range_m = default_scan_range_m(_bbox(parsed.xy), resolved_units)
    if image_shape is None:
        image_shape = default_image_shape(scan_range_m)

    ny, nx = int(image_shape[0]), int(image_shape[1])
    px_size = np.array([scan_range_m[0] / nx, scan_range_m[1] / ny], dtype=float)
    if resolved_units == "px":
        points_m = parsed.xy * px_size
    else:
        points_m = parsed.xy * _UNIT_SCALE_M[resolved_units]
    # A single global shift keeps multi-frame sets aligned with each other.
    points_m, shift_m = _fit_points_to_field(points_m, scan_range_m)
    points_px = points_m / px_size

    metadata: dict[str, Any] = {"import_source": str(p), "import_units": resolved_units}
    if shift_m is not None:
        metadata["import_recentered_offset_m"] = list(shift_m)

    if parsed.frame_col is None:
        return [
            FeatureSet.from_points(
                name=label,
                points_px=points_px,
                points_m=points_m,
                scan_range_m=scan_range_m,
                image_shape=image_shape,
                source_type="imported_csv",
                metadata=metadata,
            )
        ]

    frame_name = parsed.header[parsed.frame_col]
    frames = np.asarray(parsed.frames, dtype=object)
    sets: list[FeatureSet] = []
    for key in dict.fromkeys(parsed.frames):
        selected = frames == key
        sets.append(
            FeatureSet.from_points(
                name=f"{label} · {frame_name} {key}",
                points_px=points_px[selected],
                points_m=points_m[selected],
                scan_range_m=scan_range_m,
                image_shape=image_shape,
                source_type="imported_csv",
                metadata={**metadata, "import_frame": str(key)},
            )
        )
    return sets


# --------------------------------------------------------------------------- #
# Calibration defaults (used by the import dialog and by load defaults)
# --------------------------------------------------------------------------- #
def default_image_shape(scan_range_m: tuple[float, float], *, max_side: int = 1024) -> tuple[int, int]:
    """Synthetic (ny, nx) pixel dims for an image-less import, keeping aspect."""

    w, h = float(scan_range_m[0]), float(scan_range_m[1])
    if w <= 0.0 or h <= 0.0:
        return (max_side, max_side)
    if w >= h:
        nx = max_side
        ny = max(1, round(max_side * h / w))
    else:
        ny = max_side
        nx = max(1, round(max_side * w / h))
    return (int(ny), int(nx))


def default_scan_range_m(
    bbox_raw: tuple[float, float, float, float],
    units: str,
    *,
    margin: float = 0.05,
    px_fallback_nm: float = 1.0,
) -> tuple[float, float]:
    """Default physical field size sized to the point extent (plus a margin).

    Sizing from the extent rather than the raw maxima keeps the field honest for
    tables with an offset or absolute coordinate origin (e.g. stage coordinates):
    anchoring such data at (0, 0) would dilute the point density with empty area
    and bias randomness verdicts toward "clustered". Points that do not fit the
    field are re-centred on load (:func:`_fit_points_to_field`).

    For pixel units with no pixel size, assumes ``px_fallback_nm`` per pixel; the
    import dialog lets the user override.
    """

    xmin, ymin, xmax, ymax = bbox_raw
    scale = _UNIT_SCALE_M.get(units, px_fallback_nm * 1e-9) if units != "px" else px_fallback_nm * 1e-9
    w = max(float(xmax) - float(xmin), 0.0) * scale * (1.0 + margin)
    h = max(float(ymax) - float(ymin), 0.0) * scale * (1.0 + margin)
    # Guard against degenerate (single point / zero extent) fields.
    w = w if w > 0.0 else scale
    h = h if h > 0.0 else scale
    return (w, h)


def _fit_points_to_field(
    points_m: np.ndarray,
    scan_range_m: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Centre points in the field when they fall outside ``[0, w] x [0, h]``.

    Imported tables may use an offset or absolute coordinate origin; analysing
    them against a field anchored at (0, 0) would leave points outside the
    analysis region. A rigid translation changes no inter-point distance, so
    the point-pattern statistics are unaffected. Points that already fit the
    field are left untouched so multi-file imports sharing a coordinate frame
    (e.g. a tested set plus a feature layer) stay aligned.

    Returns the (possibly shifted) points and the applied metre offset, or
    ``None`` when no shift was needed.
    """

    if len(points_m) == 0:
        return points_m, None
    field = np.asarray(scan_range_m, dtype=float)
    lo = points_m.min(axis=0)
    hi = points_m.max(axis=0)
    if bool((lo >= 0.0).all()) and bool((hi <= field).all()):
        return points_m, None
    shift = 0.5 * (field - (hi - lo)) - lo
    return points_m + shift, (float(shift[0]), float(shift[1]))


# --------------------------------------------------------------------------- #
# CSV internals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ParsedCsv:
    """One-pass parse shared by :func:`sniff_point_table` and the loader."""

    header: tuple[str, ...]  # normalised column names ("" entries when unnamed)
    raw_header: tuple[str, ...]
    data_rows: tuple[tuple[str, ...], ...]
    delimiter: str  # "," ";" "\t" or "ws" (any whitespace)
    has_header: bool
    has_id_column: bool
    x_col: int | None
    y_col: int | None
    units: str
    frame_col: int | None
    decimal_comma: bool
    n_dropped_rows: int
    xy: np.ndarray
    frames: tuple[str, ...]  # per-point frame key, "" when no frame column
    notes: tuple[str, ...]


def _parse_csv(p: Path) -> _ParsedCsv:
    # utf-8-sig strips the BOM that Excel prepends to CSV exports.
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    comment_lines = [
        line for line in lines if line.lstrip().startswith(_COMMENT_PREFIXES)
    ]
    data_lines = [
        line for line in lines if not line.lstrip().startswith(_COMMENT_PREFIXES)
    ]
    notes: list[str] = []

    delimiter = _detect_delimiter(data_lines)
    rows = [
        tuple(cell for cell in row)
        for row in _tokenize(data_lines, delimiter)
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        return _ParsedCsv(
            header=(),
            raw_header=(),
            data_rows=(),
            delimiter=delimiter,
            has_header=False,
            has_id_column=False,
            x_col=None,
            y_col=None,
            units="unknown",
            frame_col=None,
            decimal_comma=False,
            n_dropped_rows=0,
            xy=np.empty((0, 2)),
            frames=(),
            notes=("empty file",),
        )

    has_header = not _row_is_numeric(list(rows[0]))
    raw_header = tuple(c.strip() for c in rows[0]) if has_header else ()
    data_rows = rows[1:] if has_header else rows
    header = tuple(_normalise_header_name(c) for c in raw_header)
    if not header and comment_lines:
        # Gwyddion-style files name their columns in a trailing comment line
        # ("# x y"); recover it as a header when it matches the data width.
        candidate = _comment_header(comment_lines[-1], delimiter)
        if candidate and len(candidate) <= max(len(r) for r in data_rows):
            header = candidate
            raw_header = candidate
            notes.append("Column names read from a comment line.")

    x_col, y_col, units, x_unit_col, classify_notes = _classify_columns(list(header))
    notes.extend(classify_notes)
    has_id = _detect_id_column(list(raw_header), [list(r) for r in data_rows])
    if x_col is None or y_col is None:
        # Fall back to positional: first two numeric columns (after an id column).
        start = 1 if has_id else 0
        x_col, y_col = start, start + 1
        units = units if units in ACCEPTED_UNITS else "unknown"

    # Units from an explicit unit column (measurements export).
    if units == "phys" and x_unit_col is not None:
        units = _normalise_unit(_first_value([list(r) for r in data_rows], x_unit_col))
    if units == "phys":
        units = "unknown"

    decimal_comma = _detect_decimal_comma(data_rows, x_col, y_col, delimiter)
    if decimal_comma:
        notes.append("Decimal commas detected; values read as 1,25 = 1.25.")

    frame_col = _detect_frame_column(list(header), data_rows, exclude={x_col, y_col})
    xy, frames, n_dropped = _rows_to_xy_frames(
        data_rows, x_col, y_col, frame_col, decimal_comma=decimal_comma
    )
    if n_dropped:
        notes.append(
            f"{n_dropped} row(s) skipped: x/y values could not be read as numbers."
        )
    if frame_col is not None:
        n_frames = len(dict.fromkeys(frames))
        notes.append(
            f"Column '{header[frame_col]}' has {n_frames} values; importing as "
            f"{n_frames} feature sets (poolable)."
        )
    return _ParsedCsv(
        header=header,
        raw_header=raw_header,
        data_rows=data_rows,
        delimiter=delimiter,
        has_header=has_header,
        has_id_column=has_id,
        x_col=x_col,
        y_col=y_col,
        units=units,
        frame_col=frame_col,
        decimal_comma=decimal_comma,
        n_dropped_rows=n_dropped,
        xy=xy,
        frames=frames,
        notes=tuple(notes),
    )


def _sniff_csv(p: Path) -> PointTablePreview:
    parsed = _parse_csv(p)
    if len(parsed.xy) == 0 and not parsed.data_rows:
        return PointTablePreview(
            path=str(p), kind="generic_csv", n_points=0, notes=parsed.notes
        )
    n = len(parsed.xy)
    bbox = _bbox(parsed.xy) if n else None
    kind = "probeflow_csv" if _is_probeflow_csv(list(parsed.header)) else "generic_csv"
    notes = list(parsed.notes)
    if parsed.units == "unknown":
        notes.append("Units could not be inferred from headers; choose units on import.")
    if bbox is not None and _has_offset_origin(bbox):
        notes.append(
            "Coordinates carry an offset origin; points will be re-centred "
            "in the field on import."
        )
    pixel_size_m = _derive_pixel_size_m(parsed)
    if pixel_size_m is not None:
        notes.append(
            f"Pixel size {pixel_size_m[0] * 1e9:.4g} × {pixel_size_m[1] * 1e9:.4g} nm "
            "derived from the file's pixel and physical columns."
        )
    n_sets = len(dict.fromkeys(parsed.frames)) if parsed.frame_col is not None else 1
    return PointTablePreview(
        path=str(p),
        kind=kind,
        n_points=n,
        n_sets=max(1, n_sets),
        delimiter=parsed.delimiter,
        has_header=parsed.has_header,
        has_id_column=parsed.has_id_column,
        columns=parsed.raw_header,
        units=parsed.units,
        bbox_raw=bbox,
        needs_calibration=True,
        notes=tuple(notes),
        x_col=parsed.x_col,
        y_col=parsed.y_col,
        sample_rows=parsed.data_rows[:5],
        pixel_size_m=pixel_size_m,
        frame_column=(
            parsed.header[parsed.frame_col] if parsed.frame_col is not None else None
        ),
        n_dropped_rows=parsed.n_dropped_rows,
    )


def _detect_delimiter(data_lines: list[str]) -> str:
    """Delimiter of the data lines: "," ";" "\\t", or "ws" for any whitespace."""

    sample = "\n".join(data_lines[:20])[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        pass
    # Heuristic fallback: the first delimiter that splits the sampled lines
    # into a consistent field count of at least two wins; whitespace last.
    probe = data_lines[:10]
    for delimiter in (",", ";", "\t"):
        counts = {len([c for c in line.split(delimiter) if c.strip()]) for line in probe}
        if counts and min(counts) >= 2 and len(counts) == 1:
            return delimiter
    ws_counts = {len(line.split()) for line in probe}
    if ws_counts and min(ws_counts) >= 2 and len(ws_counts) == 1:
        return "ws"
    return ","


def _tokenize(lines: list[str], delimiter: str) -> list[list[str]]:
    if delimiter == "ws":
        return [line.split() for line in lines]
    return list(csv.reader(lines, delimiter=delimiter))


def _normalise_header_name(name: str) -> str:
    """Lower-case and unify unit decorations: "X (nm)" / "x [nm]" / "x.nm" -> x_nm."""

    text = str(name).replace("﻿", "").strip().lower()
    match = re.match(r"^(.*?)[\s\.,_]*[\(\[]\s*([^\)\]]+?)\s*[\)\]]$", text)
    if match and match.group(1).strip():
        base = re.sub(r"[\s\.,]+", "_", match.group(1).strip())
        unit = match.group(2).strip().replace(" ", "")
        return f"{base}_{unit}"
    return re.sub(r"[\s\.,]+", "_", text)


def _comment_header(comment_line: str, delimiter: str) -> tuple[str, ...]:
    stripped = comment_line.strip()
    for prefix in _COMMENT_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):].strip()
            break
    if not stripped:
        return ()
    tokens = _tokenize([stripped], delimiter)[0] if delimiter != "ws" else stripped.split()
    if delimiter != "ws" and len(tokens) <= 1:
        tokens = stripped.split()
    header = tuple(_normalise_header_name(t) for t in tokens if t.strip())
    x_col, y_col, _units, _unit_col, _notes = _classify_columns(list(header))
    return header if x_col is not None and y_col is not None else ()


def _classify_columns(
    header: list[str],
) -> tuple[int | None, int | None, str, int | None, tuple[str, ...]]:
    """Return (x_col, y_col, units, x_unit_col, notes) from normalised names."""

    if not header:
        return None, None, "unknown", None, ()
    lut = {name: idx for idx, name in enumerate(header)}
    x_unit_col = lut.get("x_unit")
    # Priority: explicit physical units > phys(+unit column) > px > ambiguous > bare.
    for names, unit in (
        (_NM_NAMES, "nm"),
        (_M_NAMES, "m"),
        (_A_NAMES, "angstrom"),
        (_PM_NAMES, "pm"),
        (_UM_NAMES, "um"),
        (_PHYS_NAMES, "phys"),
        (_PX_NAMES, "px"),
    ):
        for xname, yname in names.items():
            if xname in lut and yname in lut:
                return lut[xname], lut[yname], unit, x_unit_col, ()
    for xname, yname in _AMBIGUOUS_NAMES.items():
        if xname in lut and yname in lut:
            return (
                lut[xname],
                lut[yname],
                "unknown",
                x_unit_col,
                (
                    "Columns XM/YM found (ImageJ style); their unit depends on "
                    "the image calibration — choose units on import.",
                ),
            )
    for xname, yname in _BARE_NAMES.items():
        if xname in lut and yname in lut:
            return lut[xname], lut[yname], "unknown", x_unit_col, ()
    return None, None, "unknown", x_unit_col, ()


def _detect_id_column(header: list[str], data_rows: list[list[str]]) -> bool:
    if header and _normalise_header_name(header[0]) in _ID_HEADERS:
        return True
    if header:  # named first column that isn't an id header
        return False
    # Headerless: treat a leading 0/1-based integer sequence as an id column.
    if len(data_rows) < 2:
        return False
    try:
        first_col = [int(float(r[0])) for r in data_rows if r and r[0].strip() != ""]
    except (ValueError, IndexError):
        return False
    if len(first_col) < 2:
        return False
    seq0 = first_col == list(range(len(first_col)))
    seq1 = first_col == list(range(1, len(first_col) + 1))
    return seq0 or seq1


def _detect_frame_column(
    header: list[str],
    data_rows: tuple[tuple[str, ...], ...],
    *,
    exclude: set[int | None],
) -> int | None:
    """Index of a multi-image column (frame/slice/...) worth splitting on.

    Requires at least two distinct values and at least one repeat — a column of
    all-distinct values is an index, and a constant column changes nothing.
    """

    for name in _FRAME_HEADERS:
        col = None
        for idx, column_name in enumerate(header):
            if column_name == name and idx not in exclude:
                col = idx
                break
        if col is None:
            continue
        values = [r[col].strip() for r in data_rows if len(r) > col and r[col].strip()]
        distinct = dict.fromkeys(values)
        if 2 <= len(distinct) < len(values):
            return col
    return None


def _detect_decimal_comma(
    data_rows: tuple[tuple[str, ...], ...],
    x_col: int,
    y_col: int,
    delimiter: str,
) -> bool:
    """True for European-style numbers (1,25) — only plausible when the comma
    cannot be the field delimiter."""

    if delimiter == ",":
        return False
    comma_pattern = re.compile(r"^-?\d+,\d+$")
    saw_comma = False
    for row in data_rows[:20]:
        for col in (x_col, y_col):
            if len(row) <= col:
                continue
            cell = row[col].strip()
            if not cell:
                continue
            if "." in cell:
                return False
            if comma_pattern.match(cell):
                saw_comma = True
            elif "," in cell:
                return False
    return saw_comma


def _to_float(cell: str, *, decimal_comma: bool) -> float:
    text = cell.strip()
    if decimal_comma and text.count(",") == 1:
        text = text.replace(",", ".")
    return float(text)


def _rows_to_xy_frames(
    data_rows: tuple[tuple[str, ...], ...],
    x_col: int,
    y_col: int,
    frame_col: int | None,
    *,
    decimal_comma: bool,
) -> tuple[np.ndarray, tuple[str, ...], int]:
    out: list[list[float]] = []
    frames: list[str] = []
    dropped = 0
    for row in data_rows:
        if len(row) <= max(x_col, y_col):
            dropped += 1
            continue
        try:
            out.append([
                _to_float(row[x_col], decimal_comma=decimal_comma),
                _to_float(row[y_col], decimal_comma=decimal_comma),
            ])
        except ValueError:
            dropped += 1
            continue
        frames.append(
            row[frame_col].strip() if frame_col is not None and len(row) > frame_col else ""
        )
    return np.asarray(out, dtype=float).reshape(-1, 2), tuple(frames), dropped


def _derive_pixel_size_m(parsed: _ParsedCsv) -> tuple[float, float] | None:
    """Per-axis pixel size from paired pixel+physical columns (ProbeFlow CSVs).

    Our own exports carry both x_px/y_px and x_nm/y_nm; their ratio recovers the
    pixel size that was lost when the CSV was written, so re-imports can prefill
    a faithful calibration instead of a guessed one.
    """

    lut = {name: idx for idx, name in enumerate(parsed.header)}
    px_x, px_y = lut.get("x_px"), lut.get("y_px")
    if px_x is None or px_y is None:
        return None
    for xname, yname, scale in (("x_nm", "y_nm", 1e-9), ("x_m", "y_m", 1.0)):
        phys_x, phys_y = lut.get(xname), lut.get(yname)
        if phys_x is None or phys_y is None:
            continue
        ratios_x: list[float] = []
        ratios_y: list[float] = []
        for row in parsed.data_rows[:50]:
            if len(row) <= max(px_x, px_y, phys_x, phys_y):
                continue
            try:
                px = _to_float(row[px_x], decimal_comma=parsed.decimal_comma)
                py = _to_float(row[px_y], decimal_comma=parsed.decimal_comma)
                fx = _to_float(row[phys_x], decimal_comma=parsed.decimal_comma)
                fy = _to_float(row[phys_y], decimal_comma=parsed.decimal_comma)
            except ValueError:
                continue
            if abs(px) > 1e-12 and abs(py) > 1e-12:
                ratios_x.append(fx / px)
                ratios_y.append(fy / py)
        if len(ratios_x) >= 2:
            size_x = float(np.median(ratios_x)) * scale
            size_y = float(np.median(ratios_y)) * scale
            if size_x > 0.0 and size_y > 0.0:
                return (size_x, size_y)
    return None


def _is_probeflow_csv(header: list[str]) -> bool:
    names = set(header)
    return {"x_px", "y_px", "x_nm", "y_nm"} <= names or {"x_phys", "y_phys", "x_unit"} <= names


# --------------------------------------------------------------------------- #
# JSON internals
# --------------------------------------------------------------------------- #
def _sniff_json(p: Path) -> PointTablePreview:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "feature_sets" in data:
        sets = data.get("feature_sets") or []
        total = sum(len(s.get("points_m", []) or []) for s in sets)
        return PointTablePreview(
            path=str(p),
            kind="feature_set_store_json",
            n_points=int(total),
            n_sets=len(sets),
            needs_calibration=False,
            notes=("Saved ProbeFlow feature-set file; calibration is embedded.",),
        )
    if isinstance(data, dict) and "items" in data:
        meta = data.get("meta") or {}
        items = data.get("items") or []
        scan_range = meta.get("scan_range_m")
        pixels = meta.get("pixels")
        scan_range_m = (
            (float(scan_range[0]), float(scan_range[1])) if scan_range else None
        )
        image_shape = (int(pixels[1]), int(pixels[0])) if pixels else None  # (ny, nx)
        return PointTablePreview(
            path=str(p),
            kind="probeflow_json",
            n_points=len(items),
            columns=(str(meta.get("kind", "items")),),
            units="m",
            scan_range_m=scan_range_m,
            image_shape=image_shape,
            needs_calibration=scan_range_m is None,
            notes=("ProbeFlow analysis JSON; calibration read from its meta block.",),
        )
    raise ValueError(f"unrecognised JSON structure in {p.name}")


def _load_probeflow_json(
    p: Path,
    preview: PointTablePreview,
    scan_range_m: tuple[float, float] | None,
    image_shape: tuple[int, int] | None,
    label: str,
) -> FeatureSet:
    data = json.loads(p.read_text(encoding="utf-8"))
    items = list(data.get("items") or [])
    scan_range_m = scan_range_m or preview.scan_range_m
    image_shape = image_shape or preview.image_shape
    if scan_range_m is None:
        # Derive from the metre coordinates if the meta block lacked a field size.
        xy = _items_xy_m(items)
        bbox = _bbox(xy)
        scan_range_m = default_scan_range_m(bbox, "m")
    if image_shape is None:
        image_shape = default_image_shape(scan_range_m)

    points_px, points_m = _items_to_px_m(items, scan_range_m, image_shape)
    points_m, shift_m = _fit_points_to_field(points_m, scan_range_m)
    metadata: dict[str, Any] = {"import_source": str(p)}
    if shift_m is not None:
        ny, nx = image_shape
        points_px = points_m / np.array(
            [scan_range_m[0] / nx, scan_range_m[1] / ny], dtype=float
        )
        metadata["import_recentered_offset_m"] = list(shift_m)
    return FeatureSet.from_points(
        name=label,
        points_px=points_px,
        points_m=points_m,
        scan_range_m=scan_range_m,
        image_shape=image_shape,
        source_type="imported_json",
        metadata=metadata,
    )


def feature_items_to_feature_set(
    items: Any,
    *,
    scan_range_m: tuple[float, float],
    image_shape: tuple[int, int],
    name: str,
    source_type: str = "feature_counting",
    image_label: str = "",
    metadata: dict[str, Any] | None = None,
) -> FeatureSet:
    """Build a :class:`FeatureSet` from Feature Counting particles/detections.

    Accepts dataclass items (with ``to_dict``) or plain dicts. Positions are
    converted via the AdStat adapter's ``feature_counting_to_particle_table``
    (see :func:`_items_to_px_m`), so the live Feature Counting → Particle
    Statistics path exercises that converter rather than leaving it stranded.
    """

    dicts = [it.to_dict() if hasattr(it, "to_dict") else dict(it) for it in items]
    points_px, points_m = _items_to_px_m(dicts, scan_range_m, image_shape)
    return FeatureSet.from_points(
        name=name,
        points_px=points_px,
        points_m=points_m,
        scan_range_m=scan_range_m,
        image_shape=image_shape,
        source_type=source_type,
        image_label=image_label,
        metadata=dict(metadata or {}),
    )


def _items_to_px_m(
    items: list[dict],
    scan_range_m: tuple[float, float],
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Feature Counting items to (points_px, points_m).

    Routes through the AdStat adapter's ``feature_counting_to_particle_table`` so
    that previously stranded code is exercised; falls back to a direct field read
    when AdStat is not installed.
    """

    ny, nx = image_shape
    scan = SimpleNamespace(scan_range_m=scan_range_m, dims=(nx, ny))
    try:
        from probeflow.analysis.adstat_adapter import (
            feature_counting_to_particle_table,
            scan_calibration_to_adstat,
        )

        calibration = scan_calibration_to_adstat(scan)
        table = feature_counting_to_particle_table(items, calibration=calibration)
        points_m = np.asarray(table.xy_nm, dtype=float) * 1e-9
        points_px = np.asarray(
            [[float(par.x_px), float(par.y_px)] for par in table.particles],
            dtype=float,
        )
        return points_px, points_m
    except Exception:
        # Direct fallback (no AdStat): read metre coordinates and synthesise px.
        points_m = _items_xy_m(items)
        px_x_m = scan_range_m[0] / nx
        px_y_m = scan_range_m[1] / ny
        points_px = points_m / np.array([px_x_m, px_y_m], dtype=float)
        return points_px, points_m


def _items_xy_m(items: list[dict]) -> np.ndarray:
    """Metre coordinates of accepted Feature Counting items (no-AdStat fallback).

    Applies the same accepted/manual status filter as the adapter's
    ``feature_counting_to_particle_table`` so the imported point count does not
    depend on whether the optional AdStat package is installed.
    """

    from probeflow.analysis.adstat_adapter import KEEP_STATUSES

    out = []
    for it in items:
        status = it.get("status", "accepted")
        if status is not None and str(status) not in KEEP_STATUSES:
            continue
        if it.get("centroid_x_m") is not None:
            out.append([float(it["centroid_x_m"]), float(it["centroid_y_m"])])
        elif it.get("x_m") is not None:
            out.append([float(it["x_m"]), float(it["y_m"])])
        elif it.get("x_nm") is not None:
            out.append([float(it["x_nm"]) * 1e-9, float(it["y_nm"]) * 1e-9])
    return np.asarray(out, dtype=float).reshape(-1, 2)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _row_is_numeric(row: list[str]) -> bool:
    numeric = 0
    for cell in row:
        cell = cell.strip()
        if cell == "":
            continue
        try:
            float(cell)
            numeric += 1
        except ValueError:
            return False
    return numeric > 0


def _has_offset_origin(bbox: tuple[float, float, float, float]) -> bool:
    """True when points would not fit the default (extent-sized) field at (0, 0).

    Mirrors the 5% margin used by :func:`default_scan_range_m`, so this predicts
    exactly when :func:`_fit_points_to_field` will re-centre the points.
    """

    xmin, ymin, xmax, ymax = bbox
    if xmin < 0.0 or ymin < 0.0:
        return True
    extent_x = float(xmax) - float(xmin)
    extent_y = float(ymax) - float(ymin)
    return (extent_x > 0.0 and xmin > 0.05 * extent_x) or (
        extent_y > 0.0 and ymin > 0.05 * extent_y
    )


def _bbox(xy: np.ndarray) -> tuple[float, float, float, float]:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return (
        float(xy[:, 0].min()),
        float(xy[:, 1].min()),
        float(xy[:, 0].max()),
        float(xy[:, 1].max()),
    )


def _first_value(data_rows: list[list[str]], col: int) -> str:
    for r in data_rows:
        if len(r) > col and r[col].strip():
            return r[col].strip()
    return ""


def _normalise_unit(raw: str) -> str:
    u = raw.strip().lower()
    if u in {"nm", "nanometer", "nanometre"}:
        return "nm"
    if u in {"um", "µm", "micron", "micrometer", "micrometre"}:
        return "um"
    if u in {"m", "meter", "metre"}:
        return "m"
    if u in {"å", "a", "ang", "angstrom", "angstroem", "ångström"}:
        return "angstrom"
    if u in {"pm", "picometer", "picometre"}:
        return "pm"
    if u in {"px", "pixel", "pixels"}:
        return "px"
    return "phys"
