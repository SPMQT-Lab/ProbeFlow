"""GUI-free export of Particle Statistics (AdStat) results to simple formats.

The Particle Statistics result is an AdStat ``ResultViewSpec``: a tuple of
``PanelSpec`` panels (each carrying the plotted ``x`` / ``observed`` /
``band_low`` / ``central`` / ``band_high`` arrays) plus verdict rows. These
helpers turn that into plain CSV (one file per curve/table statistic, so the
plots can be reproduced in any program) and a single JSON snapshot of the whole
result for provenance.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Curve columns to emit when present, in order: (panel attribute, CSV header).
_CURVE_COLUMNS = (
    ("observed", "observed"),
    ("band_low", "model_low"),
    ("central", "model_central"),
    ("band_high", "model_high"),
)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_1d(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr


def panel_curve_csv_text(panel: Any) -> str | None:
    """CSV text for a 1-D curve panel (g(r), nearest-neighbour, Ripley L, …).

    Returns ``None`` for panels that are not simple x/y curves (heatmaps,
    real-space scatter, etc.).
    """

    x = _as_1d(_field(panel, "x"))
    if x is None:
        return None
    columns: list[tuple[str, np.ndarray]] = []
    x_label = str(_field(panel, "x_label") or "x")
    columns.append((x_label, x))
    for attr, header in _CURVE_COLUMNS:
        col = _as_1d(_field(panel, attr))
        if col is not None and len(col) == len(x):
            columns.append((header, col))
    if len(columns) < 2:  # x with no measured/model column is not worth a file
        return None

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([name for name, _ in columns])
    for row_idx in range(len(x)):
        writer.writerow([f"{col[row_idx]:.10g}" for _, col in columns])
    return out.getvalue()


def panel_table_csv_text(panel: Any) -> str | None:
    """CSV text for a table-style panel (``table_columns`` / ``table_rows``)."""

    cols = _field(panel, "table_columns")
    rows = _field(panel, "table_rows")
    if not cols or not rows:
        return None
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([str(c) for c in cols])
    for row in rows:
        writer.writerow([str(v) for v in row])
    return out.getvalue()


def verdict_rows_csv_text(spec: Any) -> str | None:
    """CSV text for the verdict table, or ``None`` if there are no verdicts."""

    rows = tuple(_field(spec, "verdict_rows", ()) or ())
    if not rows:
        return None
    out = io.StringIO()
    writer = csv.writer(out)
    header = ("model", "statistic", "verdict", "alpha", "statistic_value", "rank", "n_simulations")
    width = max(len(r) for r in rows)
    writer.writerow(header[:width] if width <= len(header) else [f"col{i}" for i in range(width)])
    for row in rows:
        writer.writerow([str(v) for v in row])
    return out.getvalue()


def export_result_csvs(spec: Any, out_dir: str | Path, *, base: str = "particle_statistics") -> list[Path]:
    """Write one CSV per curve/table statistic plus a verdicts CSV.

    Returns the list of files written.  Heatmap and real-space panels are skipped
    here (they are preserved in the JSON export); curve panels reproduce the
    plotted lines, including the model envelope.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(base)
    written: list[Path] = []

    for panel in tuple(_field(spec, "panels", ()) or ()):
        statistic = _slug(str(_field(panel, "statistic", "") or _field(panel, "title", "panel")))
        text = panel_curve_csv_text(panel) or panel_table_csv_text(panel)
        if not text:
            continue
        path = out_dir / f"{slug}_{statistic}.csv"
        path.write_text(text, encoding="utf-8")
        written.append(path)

    verdicts = verdict_rows_csv_text(spec)
    if verdicts:
        path = out_dir / f"{slug}_verdicts.csv"
        path.write_text(verdicts, encoding="utf-8")
        written.append(path)

    return written


def result_spec_to_plain(spec: Any) -> Any:
    """Recursively convert a view spec into JSON-serialisable Python objects."""

    return _plain(spec)


def export_result_json(spec: Any, out_path: str | Path) -> Path:
    """Write the whole result view spec to a single JSON file."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result_spec_to_plain(spec), indent=2, sort_keys=True), encoding="utf-8"
    )
    return out_path


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _plain(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    return value


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip()).strip("_")
    return slug or "particle_statistics"
