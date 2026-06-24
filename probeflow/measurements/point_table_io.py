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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from probeflow.measurements.feature_sets import FeatureSet, FeatureSetStore


# metres per unit; ``px`` is special (needs a pixel size, handled separately)
_UNIT_SCALE_M: dict[str, float] = {"m": 1.0, "nm": 1e-9, "um": 1e-6, "µm": 1e-6}
ACCEPTED_UNITS = ("px", "nm", "um", "m")

# Header-name hints, normalised (lower-cased, stripped).
_PX_NAMES = {"x_px": "y_px", "xpx": "ypx", "x_pixel": "y_pixel", "col": "row", "px_x": "px_y"}
_NM_NAMES = {"x_nm": "y_nm", "xnm": "ynm"}
_M_NAMES = {"x_m": "y_m", "xm": "ym", "centroid_x_m": "centroid_y_m", "x_meter": "y_meter"}
_UM_NAMES = {"x_um": "y_um", "xum": "yum", "x_µm": "y_µm"}
_PHYS_NAMES = {"x_phys": "y_phys", "x_physical": "y_physical"}
_BARE_NAMES = {"x": "y", "x_pos": "y_pos", "posx": "posy", "pos_x": "pos_y", "xpos": "ypos"}

_ID_HEADERS = {"index", "id", "point_id", "particle", "particle_id", "n", "#", "no", "num"}


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
    units: str = "unknown"  # px | nm | um | m | unknown
    scan_range_m: tuple[float, float] | None = None  # embedded calibration if present
    image_shape: tuple[int, int] | None = None
    bbox_raw: tuple[float, float, float, float] | None = None  # xmin,ymin,xmax,ymax in `units`
    needs_calibration: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)


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
    unless explicitly overridden.
    """

    p = Path(path)
    preview = sniff_point_table(p)
    label = name or p.stem

    if preview.kind == "feature_set_store_json":
        return list(FeatureSetStore.from_dict(json.loads(p.read_text(encoding="utf-8"))).all())

    if preview.kind == "probeflow_json":
        return [_load_probeflow_json(p, preview, scan_range_m, image_shape, label)]

    # CSV (generic or ProbeFlow)
    resolved_units = units or preview.units
    if resolved_units not in ACCEPTED_UNITS:
        raise ValueError(
            "could not infer position units; specify units as one of "
            f"{ACCEPTED_UNITS}"
        )
    if preview.bbox_raw is None or preview.n_points == 0:
        raise ValueError("no point rows found in file")
    if scan_range_m is None:
        scan_range_m = default_scan_range_m(preview.bbox_raw, resolved_units)
    if image_shape is None:
        image_shape = default_image_shape(scan_range_m)
    xy_raw = _read_csv_points(p, preview)
    fs = _build_feature_set(
        xy_raw,
        units=resolved_units,
        scan_range_m=scan_range_m,
        image_shape=image_shape,
        name=label,
        source_type="imported_csv",
        metadata={"import_source": str(p), "import_units": resolved_units},
    )
    return [fs]


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
    """Default physical field size that contains all points (origin at 0,0).

    For pixel units with no pixel size, assumes ``px_fallback_nm`` per pixel; the
    import dialog lets the user override.
    """

    _, _, xmax, ymax = bbox_raw
    scale = _UNIT_SCALE_M.get(units, px_fallback_nm * 1e-9) if units != "px" else px_fallback_nm * 1e-9
    w = max(float(xmax), 0.0) * scale * (1.0 + margin)
    h = max(float(ymax), 0.0) * scale * (1.0 + margin)
    # Guard against degenerate (single point / zero extent) fields.
    w = w if w > 0.0 else scale
    h = h if h > 0.0 else scale
    return (w, h)


# --------------------------------------------------------------------------- #
# CSV internals
# --------------------------------------------------------------------------- #
def _sniff_csv(p: Path) -> PointTablePreview:
    text = p.read_text(encoding="utf-8", errors="replace")
    sample = text[:4096]
    delimiter = ","
    has_header = False
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample and "," not in sample else ","
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delimiter) if r]
    if not rows:
        return PointTablePreview(path=str(p), kind="generic_csv", n_points=0, notes=("empty file",))

    first = rows[0]
    has_header = not _row_is_numeric(first)
    header = [c.strip() for c in first] if has_header else []
    data_rows = rows[1:] if has_header else rows

    x_col, y_col, units, x_unit_col = _classify_columns(header)
    has_id = _detect_id_column(header, data_rows)
    if x_col is None or y_col is None:
        # Fall back to positional: first two numeric columns (after an id column).
        start = 1 if has_id else 0
        x_col, y_col = start, start + 1
        units = units if units in ACCEPTED_UNITS else "unknown"

    # Units from an explicit unit column (measurements export).
    if units == "phys" and x_unit_col is not None:
        units = _normalise_unit(_first_value(data_rows, x_unit_col))
    if units == "phys":
        units = "unknown"

    xy = _rows_to_xy(data_rows, x_col, y_col)
    n = len(xy)
    bbox = _bbox(xy) if n else None
    kind = "probeflow_csv" if (has_header and _is_probeflow_csv(header)) else "generic_csv"
    notes = []
    if units == "unknown":
        notes.append("Units could not be inferred from headers; choose units on import.")
    return PointTablePreview(
        path=str(p),
        kind=kind,
        n_points=n,
        delimiter=delimiter,
        has_header=has_header,
        has_id_column=has_id,
        columns=tuple(header),
        units=units,
        bbox_raw=bbox,
        needs_calibration=True,
        notes=tuple(notes),
    )


def _read_csv_points(p: Path, preview: PointTablePreview) -> np.ndarray:
    text = p.read_text(encoding="utf-8", errors="replace")
    delimiter = preview.delimiter or ","
    rows = [r for r in csv.reader(text.splitlines(), delimiter=delimiter) if r]
    data_rows = rows[1:] if preview.has_header else rows
    header = [c.strip() for c in rows[0]] if preview.has_header else []
    x_col, y_col, _units, _unit_col = _classify_columns(header)
    if x_col is None or y_col is None:
        start = 1 if preview.has_id_column else 0
        x_col, y_col = start, start + 1
    return _rows_to_xy(data_rows, x_col, y_col)


def _classify_columns(
    header: list[str],
) -> tuple[int | None, int | None, str, int | None]:
    """Return (x_col, y_col, units, x_unit_col) from header names."""

    if not header:
        return None, None, "unknown", None
    lut = {name.strip().lower(): idx for idx, name in enumerate(header)}
    x_unit_col = lut.get("x_unit")
    # Priority: nm > m > um > phys(+unit) > px > bare.
    for names, unit in (
        (_NM_NAMES, "nm"),
        (_M_NAMES, "m"),
        (_UM_NAMES, "um"),
        (_PHYS_NAMES, "phys"),
        (_PX_NAMES, "px"),
        (_BARE_NAMES, "bare"),
    ):
        for xname, yname in names.items():
            if xname in lut and yname in lut:
                resolved = "unknown" if unit == "bare" else unit
                return lut[xname], lut[yname], resolved, x_unit_col
    return None, None, "unknown", x_unit_col


def _detect_id_column(header: list[str], data_rows: list[list[str]]) -> bool:
    if header and header[0].strip().lower() in _ID_HEADERS:
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


def _is_probeflow_csv(header: list[str]) -> bool:
    lower = {c.strip().lower() for c in header}
    return {"x_px", "y_px", "x_nm", "y_nm"} <= lower or {"x_phys", "y_phys", "x_unit"} <= lower


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
    return FeatureSet.from_points(
        name=label,
        points_px=points_px,
        points_m=points_m,
        scan_range_m=scan_range_m,
        image_shape=image_shape,
        source_type="imported_json",
        metadata={"import_source": str(p)},
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
    out = []
    for it in items:
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
def _build_feature_set(
    xy_raw: np.ndarray,
    *,
    units: str,
    scan_range_m: tuple[float, float],
    image_shape: tuple[int, int],
    name: str,
    source_type: str,
    metadata: dict[str, Any],
) -> FeatureSet:
    xy = np.asarray(xy_raw, dtype=float).reshape(-1, 2)
    ny, nx = image_shape
    px_x_m = scan_range_m[0] / nx
    px_y_m = scan_range_m[1] / ny
    if units == "px":
        points_px = xy
        points_m = xy * np.array([px_x_m, px_y_m], dtype=float)
    else:
        scale = _UNIT_SCALE_M[units]
        points_m = xy * scale
        points_px = points_m / np.array([px_x_m, px_y_m], dtype=float)
    return FeatureSet.from_points(
        name=name,
        points_px=points_px,
        points_m=points_m,
        scan_range_m=scan_range_m,
        image_shape=image_shape,
        source_type=source_type,
        metadata=metadata,
    )


def _rows_to_xy(data_rows: list[list[str]], x_col: int, y_col: int) -> np.ndarray:
    out = []
    for r in data_rows:
        if len(r) <= max(x_col, y_col):
            continue
        try:
            out.append([float(r[x_col]), float(r[y_col])])
        except ValueError:
            continue
    return np.asarray(out, dtype=float).reshape(-1, 2)


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
    if u in {"px", "pixel", "pixels"}:
        return "px"
    return "phys"
