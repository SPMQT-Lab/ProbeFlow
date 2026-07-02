"""Calibration/format confirmation dialog for importing a point table.

Shown after :func:`probeflow.measurements.point_table_io.sniff_point_table`
detects the file's shape, prefilled with the sniffed guesses, so the user can
confirm units and physical field size before the points become a feature set.
ProbeFlow JSON files that carry their own calibration skip this dialog entirely.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from probeflow.measurements.point_table_io import (
    ACCEPTED_UNITS,
    PointTablePreview,
    default_image_shape,
    default_scan_range_m,
)

_UNIT_LABELS = {
    "px": "pixels",
    "nm": "nanometres (nm)",
    "angstrom": "ångström (Å)",
    "pm": "picometres (pm)",
    "um": "micrometres (µm)",
    "m": "metres (m)",
}

_ACCEPTED_FORMATS_NOTE = (
    "Accepted: CSV/TSV position tables (comma/semicolon/tab/space delimited, "
    "#-comments and decimal commas handled; units inferred from x_px / x_nm / "
    "x_A / x_pm / x_um / x_m / x_phys or 'x (nm)'-style headers, or chosen "
    "here), ProbeFlow Feature Finder / measurements CSV, and ProbeFlow JSON "
    "(Feature Counting exports and saved feature-set files). A frame/slice/"
    "image column imports as one poolable set per image."
)


class ImportPointsDialog(QDialog):
    """Confirm units + physical field size for an imported point table."""

    def __init__(self, preview: PointTablePreview, *, theme: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("importPointsDialog")
        self.setWindowTitle("Import points")
        self._preview = preview

        units = preview.units if preview.units in ACCEPTED_UNITS else "nm"
        scan_range_m, image_shape = _initial_calibration(preview, units)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        summary = QLabel(
            f"<b>Detected:</b> {_kind_label(preview.kind)} — "
            f"{preview.n_points} point(s)"
            + (f" in {preview.n_sets} sets" if preview.n_sets > 1 else "")
            + (", leading id column" if preview.has_id_column else "")
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        preview_table = _sample_table(preview, self)
        if preview_table is not None:
            layout.addWidget(preview_table)
            legend = QLabel("Highlighted columns will be read as x and y.")
            legend.setStyleSheet("color: palette(mid); font-size: 11px;")
            layout.addWidget(legend)

        if preview.notes:
            notes_lbl = QLabel("<br>".join(f"• {note}" for note in preview.notes))
            notes_lbl.setObjectName("importPointsNotes")
            notes_lbl.setWordWrap(True)
            notes_lbl.setStyleSheet(
                "color: #d4a72c; border: 1px solid rgba(212, 167, 44, 0.45); "
                "padding: 4px 6px;"
            )
            layout.addWidget(notes_lbl)

        form = QFormLayout()
        form.setSpacing(6)

        self._units_cb = QComboBox(self)
        self._units_cb.setObjectName("importPointsUnits")
        for u in ACCEPTED_UNITS:
            self._units_cb.addItem(_UNIT_LABELS[u], u)
        idx = self._units_cb.findData(units)
        if idx >= 0:
            self._units_cb.setCurrentIndex(idx)
        form.addRow("Position units:", self._units_cb)

        self._field_w = _nm_spin(scan_range_m[0] * 1e9)
        self._field_w.setObjectName("importPointsFieldW")
        form.addRow("Field width (nm):", self._field_w)
        self._field_h = _nm_spin(scan_range_m[1] * 1e9)
        self._field_h.setObjectName("importPointsFieldH")
        form.addRow("Field height (nm):", self._field_h)

        self._img_w = _px_spin(image_shape[1])
        self._img_w.setObjectName("importPointsImgW")
        form.addRow("Image width (px):", self._img_w)
        self._img_h = _px_spin(image_shape[0])
        self._img_h.setObjectName("importPointsImgH")
        form.addRow("Image height (px):", self._img_h)
        layout.addLayout(form)

        hint = QLabel(
            "The field size sets the analysis region (and therefore the expected "
            "random density). The image size is synthetic for an image-less import "
            "and only affects the pixel-resolution note."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        layout.addWidget(hint)

        note = QLabel(_ACCEPTED_FORMATS_NOTE)
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_calibration(self) -> tuple[str, tuple[float, float], tuple[int, int]]:
        """Return (units, scan_range_m, image_shape) chosen by the user."""
        units = str(self._units_cb.currentData())
        scan_range_m = (self._field_w.value() * 1e-9, self._field_h.value() * 1e-9)
        image_shape = (int(self._img_h.value()), int(self._img_w.value()))
        return units, scan_range_m, image_shape


def _sample_table(preview: PointTablePreview, parent) -> QTableWidget | None:
    """Read-only table of the first data rows, x/y columns highlighted."""

    rows = tuple(preview.sample_rows)
    if not rows:
        return None
    n_cols = max(len(row) for row in rows)
    table = QTableWidget(len(rows), n_cols, parent)
    table.setObjectName("importPointsSample")
    headers = [
        (preview.columns[c] if c < len(preview.columns) and preview.columns[c] else f"col {c}")
        for c in range(n_cols)
    ]
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionMode(QTableWidget.NoSelection)
    table.setFocusPolicy(Qt.NoFocus)
    highlight = QColor(47, 129, 247, 60)
    for r, row in enumerate(rows):
        for c in range(n_cols):
            item = QTableWidgetItem(row[c] if c < len(row) else "")
            if c in (preview.x_col, preview.y_col):
                item.setBackground(highlight)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()
    table.setMaximumHeight(
        table.horizontalHeader().height() + table.rowHeight(0) * len(rows) + 8
    )
    return table


def _initial_calibration(
    preview: PointTablePreview, units: str
) -> tuple[tuple[float, float], tuple[int, int]]:
    if preview.scan_range_m is not None:
        sr = preview.scan_range_m
        img = preview.image_shape or default_image_shape(sr)
        return sr, img
    if preview.bbox_raw is None:
        return (100e-9, 100e-9), (512, 512)
    if units == "px":
        # Pixel coordinates: size the image to the extent; pixel size from the
        # file's own px/physical column pair when available, else 1 nm/px.
        xmin, ymin, xmax, ymax = preview.bbox_raw
        nx = int(math.ceil(xmax - xmin)) + 1
        ny = int(math.ceil(ymax - ymin)) + 1
        px_x, px_y = preview.pixel_size_m or (1e-9, 1e-9)
        return (nx * px_x, ny * px_y), (ny, nx)
    sr = default_scan_range_m(preview.bbox_raw, units)
    if preview.pixel_size_m is not None:
        # Physical coordinates with a known pixel size (ProbeFlow CSV re-import):
        # propose image dims that reproduce that pixel size exactly, so the
        # resolution floor matches the original scan.
        px_x, px_y = preview.pixel_size_m
        if px_x > 0.0 and px_y > 0.0:
            nx = max(1, round(sr[0] / px_x))
            ny = max(1, round(sr[1] / px_y))
            return (nx * px_x, ny * px_y), (int(ny), int(nx))
    return sr, default_image_shape(sr)


def _kind_label(kind: str) -> str:
    return {
        "generic_csv": "generic CSV",
        "probeflow_csv": "ProbeFlow CSV",
        "probeflow_json": "ProbeFlow JSON",
        "feature_set_store_json": "saved feature-set JSON",
    }.get(kind, kind)


def _nm_spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(1e-3, 1e9)
    spin.setDecimals(3)
    spin.setValue(max(float(value), 1e-3))
    spin.setAlignment(Qt.AlignRight)
    return spin


def _px_spin(value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(1, 1_000_000)
    spin.setValue(max(int(value), 1))
    spin.setAlignment(Qt.AlignRight)
    return spin
