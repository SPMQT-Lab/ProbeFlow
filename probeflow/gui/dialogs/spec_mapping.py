from __future__ import annotations

from typing import Optional

from probeflow.gui.typography import ui_font
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from probeflow.core.scan_loader import load_scan


class SpecMappingDialog(QDialog):
    """Folder-level spec→image mapping editor.

    Lists every loaded .VERT spec file with a dropdown of every loaded
    .sxm image in the same folder (plus a leading "(none)" entry). The
    user picks the parent image for each spectrum; the result is returned
    as a ``dict[spec_stem, image_stem]`` containing only the assigned
    rows. Unassigned spectra are simply omitted from the result.

    A "Suggest all" button populates dropdowns by reading each scan's
    physical extent (offset, range, angle) and picking the smallest
    image whose scan-frame contains the spec's recorded coordinates.
    The suggestion is just a starting point — the user can change any
    row before accepting.
    """

    NONE_LABEL = "(none)"

    def __init__(self, sxm_entries: list, vert_entries: list,
                 current_map: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map spectra to images")
        self.resize(720, 520)
        self._sxm_entries = list(sxm_entries)
        self._vert_entries = list(vert_entries)
        self._current = dict(current_map or {})
        self._combos: dict[str, QComboBox] = {}
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        hdr = QLabel(
            "Pick the parent image for each spectrum. Unassigned spectra "
            "show no marker on any image.")
        hdr.setFont(ui_font(10))
        hdr.setWordWrap(True)
        v.addWidget(hdr)

        if not self._vert_entries:
            v.addWidget(QLabel("No spectroscopy files in the current folder."))
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            grid = QGridLayout(inner)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(4)
            grid.addWidget(QLabel("Spectrum"), 0, 0)
            grid.addWidget(QLabel("Parent image"), 0, 1)

            image_options = [self.NONE_LABEL] + [e.stem for e in self._sxm_entries]
            for i, vert in enumerate(self._vert_entries, start=1):
                grid.addWidget(QLabel(vert.stem), i, 0)
                cb = QComboBox()
                cb.addItems(image_options)
                cur = self._current.get(vert.stem)
                if cur and cur in image_options:
                    cb.setCurrentText(cur)
                else:
                    cb.setCurrentText(self.NONE_LABEL)
                grid.addWidget(cb, i, 1)
                self._combos[vert.stem] = cb
            grid.setColumnStretch(1, 1)
            scroll.setWidget(inner)
            v.addWidget(scroll, 1)

        # Action row
        btn_row = QHBoxLayout()
        suggest_btn = QPushButton("Suggest all (by coordinates)")
        suggest_btn.setToolTip(
            "For each spectrum, look at its recorded (x,y) position and pick "
            "the smallest loaded scan whose frame contains it. Existing "
            "selections are overwritten.")
        suggest_btn.clicked.connect(self._on_suggest)
        btn_row.addWidget(suggest_btn)
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(self._on_clear_all)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        ok_btn = QPushButton("Apply")
        ok_btn.setObjectName("accentBtn")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        v.addLayout(btn_row)

    def _on_clear_all(self):
        for cb in self._combos.values():
            cb.setCurrentText(self.NONE_LABEL)

    def _on_suggest(self):
        """Pick the smallest containing image per spec, by coordinate."""
        # We avoid importing this at module top because it pulls in scan I/O
        # — keeping the dialog responsive on large folders matters more
        # than saving the import here.
        from probeflow.io.spectroscopy import read_spec_file, _position_from_createc_header
        from probeflow.analysis.spec_plot import spec_position_to_pixel, _parse_sxm_offset

        # Pre-load image headers once (slow if many files).
        scan_info = []
        for img in self._sxm_entries:
            try:
                _scan = load_scan(img.path)
                shape = _scan.planes[0].shape if _scan.planes else None
                if shape is None or _scan.scan_range_m is None:
                    continue
                hdr = _scan.header or {}
                offset_m = (0.0, 0.0)
                angle_deg = 0.0
                if img.source_format == "sxm" and hdr:
                    offset_m = _parse_sxm_offset(hdr)
                    raw = hdr.get("SCAN_ANGLE", "0").strip()
                    try:
                        angle_deg = float(raw) if raw else 0.0
                    except ValueError:
                        angle_deg = 0.0
                elif img.source_format == "dat" and hdr:
                    # Createc scan-frame centre shares the OffsetX/OffsetY DAC
                    # coordinate system with the .VERT spec positions.
                    offset_m = _position_from_createc_header(hdr)
                    raw = str(hdr.get("Rotation", "0")).strip()
                    try:
                        angle_deg = float(raw) if raw else 0.0
                    except ValueError:
                        angle_deg = 0.0
                # "Size" used to break ties when several images contain a spec:
                # smaller scan range = better localisation = preferred.
                rng_m = _scan.scan_range_m
                area_m2 = float(rng_m[0]) * float(rng_m[1])
                scan_info.append((img.stem, shape, rng_m, offset_m, angle_deg, area_m2))
            except Exception:
                continue

        for vert in self._vert_entries:
            try:
                spec = read_spec_file(vert.path)
                x_m, y_m = spec.position
            except Exception:
                continue
            best: Optional[tuple[float, str]] = None  # (area, stem)
            for stem, shape, rng_m, offset_m, angle_deg, area in scan_info:
                hit = spec_position_to_pixel(
                    x_m, y_m,
                    scan_shape=shape,
                    scan_range_m=rng_m,
                    scan_offset_m=offset_m,
                    scan_angle_deg=angle_deg,
                )
                if hit is None:
                    continue
                if best is None or area < best[0]:
                    best = (area, stem)
            if best is not None and vert.stem in self._combos:
                self._combos[vert.stem].setCurrentText(best[1])

    def get_mapping(self) -> dict[str, str]:
        """Return the user's selection as ``{spec_stem: image_stem}``."""
        out: dict[str, str] = {}
        for spec_stem, cb in self._combos.items():
            sel = cb.currentText()
            if sel and sel != self.NONE_LABEL:
                out[spec_stem] = sel
        return out


class ViewerSpecMappingDialog(QDialog):
    """In-viewer mapping editor for ONE image.

    Lists all .VERT spec files in the same folder; the user ticks which
    spectra belong to the currently displayed image. Multiple images can
    share a parent in principle (e.g. different planes of the same scan)
    but our mapping is one-to-one, so ticking a spec here moves it from
    any prior parent to the current image.
    """

    def __init__(self, image_stem: str, vert_entries: list,
                 current_map: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Map spectra to {image_stem}")
        self.resize(420, 460)
        self._image_stem  = image_stem
        self._vert_entries = list(vert_entries)
        self._current     = dict(current_map or {})
        self._checks: dict[str, QCheckBox] = {}
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        hdr = QLabel(
            f"Tick the spectra to associate with <b>{self._image_stem}</b>. "
            "Ticking one that is already mapped to a different image will move it.")
        hdr.setFont(ui_font(9))
        hdr.setWordWrap(True)
        v.addWidget(hdr)

        if not self._vert_entries:
            v.addWidget(QLabel("No .VERT files in the current folder."))
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            inner_lay = QVBoxLayout(inner)
            inner_lay.setContentsMargins(0, 0, 0, 0)
            inner_lay.setSpacing(2)
            for vert in self._vert_entries:
                cb = QCheckBox(vert.stem)
                cb.setChecked(self._current.get(vert.stem) == self._image_stem)
                # Annotate other-image assignments so the user knows what
                # ticking this row will displace.
                other = self._current.get(vert.stem)
                if other and other != self._image_stem:
                    cb.setText(f"{vert.stem}   (currently → {other})")
                inner_lay.addWidget(cb)
                self._checks[vert.stem] = cb
            inner_lay.addStretch()
            scroll.setWidget(inner)
            v.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Apply")
        ok_btn.setObjectName("accentBtn")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        v.addLayout(btn_row)

    def updated_map(self) -> dict[str, str]:
        """Return a NEW mapping dict reflecting the user's choices."""
        out = dict(self._current)
        for spec_stem, cb in self._checks.items():
            if cb.isChecked():
                out[spec_stem] = self._image_stem
            else:
                # Only clear if this row WAS pointing at the current image.
                if out.get(spec_stem) == self._image_stem:
                    out.pop(spec_stem, None)
        return out

