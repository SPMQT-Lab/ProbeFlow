"""Calibration/format confirmation dialog for importing a point table.

Shown after :func:`probeflow.measurements.point_table_io.sniff_point_table`
detects the file's shape, prefilled with the sniffed guesses, so the user can
confirm units and physical field size before the points become a feature set.
ProbeFlow JSON files that carry their own calibration skip this dialog entirely.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from probeflow.measurements.point_table_io import (
    ACCEPTED_UNITS,
    PointTablePreview,
    default_image_shape,
    default_scan_range_m,
)

_UNIT_LABELS = {"px": "pixels", "nm": "nanometres (nm)", "um": "micrometres (µm)", "m": "metres (m)"}

_ACCEPTED_FORMATS_NOTE = (
    "Accepted: CSV position tables (with or without a leading particle-number "
    "column; units inferred from x_px / x_nm / x_m / x_phys headers or chosen "
    "here), ProbeFlow Feature Finder / measurements CSV, and ProbeFlow JSON "
    "(Feature Counting exports and saved feature-set files)."
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
            + (", leading id column" if preview.has_id_column else "")
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

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
        # Pixel coordinates: size the image to the extent, default 1 nm/px.
        _, _, xmax, ymax = preview.bbox_raw
        nx = int(math.ceil(xmax)) + 1
        ny = int(math.ceil(ymax)) + 1
        return (nx * 1e-9, ny * 1e-9), (ny, nx)
    sr = default_scan_range_m(preview.bbox_raw, units)
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
