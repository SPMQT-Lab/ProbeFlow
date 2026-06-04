"""Control panel for reciprocal-space lattice overlays."""

from __future__ import annotations

import math
from dataclasses import replace

from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from probeflow.analysis.lattice_grid import (
    LatticeGrid,
    LatticeKind,
    ReciprocalCalibration,
)
from probeflow.gui.no_wheel import install_no_wheel_spinboxes

from .fft_overlay import FFTLatticeOverlay

# Compact width for numeric value fields so they don't stretch across the wide
# sidebar column (keeps the two label/field pairs reading as a tidy two-up grid).
_FIELD_W = 96

# ── FFT panel ─────────────────────────────────────────────────────────────────

class FFTLatticePanel(QWidget):
    """Control and measurement panel for an FFT lattice overlay."""

    def __init__(
        self,
        overlay: FFTLatticeOverlay,
        calibration: ReciprocalCalibration,
        image_w: int,
        image_h: int,
        parent=None,
        on_change=None,
    ):
        super().__init__(parent)
        self._overlay = overlay
        self._cal = calibration
        self._image_w = image_w
        self._image_h = image_h
        self._external_on_change = on_change
        self._updating_controls = False
        self._build()
        overlay.set_on_change(self._on_grid_changed)
        self.sync_from_model()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Square", "Rectangular", "Hexagonal"])
        self._type_combo.setFont(ui_font(9))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo, 1)
        lay.addLayout(type_row)

        self._lock_cb = QCheckBox("Lock lattice constraint")
        self._lock_cb.setFont(ui_font(9))
        self._lock_cb.toggled.connect(self._on_locked_changed)
        lay.addWidget(self._lock_cb)

        params_grp = QGroupBox("Parameters")
        params_grp.setFont(ui_font(9))
        params_lay = QGridLayout(params_grp)
        params_lay.setHorizontalSpacing(8)
        params_lay.setVerticalSpacing(4)
        params_lay.setContentsMargins(8, 8, 8, 6)

        def _make_label(label):
            lb = QLabel(label)
            lb.setFont(ui_font(9))
            lb.setMinimumWidth(64)
            return lb

        def _double_cell(row, col, label, lo, hi, step, dec, sfx=""):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            if sfx:
                sp.setSuffix(f" {sfx}")
            sp.setFont(ui_font(9))
            sp.setFixedHeight(22)
            sp.setMaximumWidth(_FIELD_W)
            params_lay.addWidget(_make_label(label), row, col * 2)
            params_lay.addWidget(sp, row, col * 2 + 1)
            return sp

        def _int_cell(row, col, label, lo, hi, value):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(value)
            sp.setFont(ui_font(9))
            sp.setFixedHeight(22)
            sp.setMaximumWidth(_FIELD_W)
            params_lay.addWidget(_make_label(label), row, col * 2)
            params_lay.addWidget(sp, row, col * 2 + 1)
            return sp

        # Three label/field pairs per row, using the panel width. |g1|/|g2|/angle
        # are live measurements (they update as the grid handles are dragged);
        # origin and rotation were dropped — use "Reset origin to FFT centre" and
        # drag the handles instead.
        self._g1_spin  = _double_cell(0, 0, "|g1|:", 0.001, 1000.0, 0.01, 3, "nm⁻¹")
        self._g2_spin  = _double_cell(0, 1, "|g2|:", 0.001, 1000.0, 0.01, 3, "nm⁻¹")
        self._angle_ab_spin = _double_cell(0, 2, "Angle g1-g2:", 1.0, 179.0, 0.1, 2, "°")
        self._angle_ab_spin.setEnabled(False)

        self._cells_spin = _int_cell(1, 0, "Cells ±:", 1, 200, 12)
        self._line_width_spin = _double_cell(1, 1, "Line width:", 0.25, 10.0, 0.25, 2, "px")
        self._line_width_spin.setValue(1.5)
        params_lay.setColumnStretch(6, 1)

        lay.addWidget(params_grp)

        self._g1_spin.valueChanged.connect(self._on_g1_changed)
        self._g2_spin.valueChanged.connect(self._on_g2_changed)
        self._angle_ab_spin.valueChanged.connect(self._on_angle_ab_changed)
        self._cells_spin.valueChanged.connect(self._on_cells_changed)
        self._line_width_spin.valueChanged.connect(self._on_line_width_changed)
        install_no_wheel_spinboxes(params_grp)

        disp_grp = QGroupBox("Display")
        disp_grp.setFont(ui_font(9))
        disp_lay = QHBoxLayout(disp_grp)
        disp_lay.setSpacing(10)
        disp_lay.setContentsMargins(8, 8, 8, 6)
        self._show_grid_cb    = QCheckBox("Show grid")
        self._show_handles_cb = QCheckBox("Show handles")
        self._show_labels_cb  = QCheckBox("Show labels")
        for cb in (self._show_grid_cb, self._show_handles_cb, self._show_labels_cb):
            cb.setFont(ui_font(9))
            cb.setChecked(True)
            cb.toggled.connect(self._on_visibility_changed)
            disp_lay.addWidget(cb)
        disp_lay.addStretch(1)
        lay.addWidget(disp_grp)

        reset_btn = QPushButton("Reset origin to FFT centre")
        reset_btn.setFont(ui_font(9))
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self._on_reset_origin)
        lay.addWidget(reset_btn)

        fft_note = QLabel(
            "FFT-derived direct lattice measurements can be used for affine "
            "image correction from the FFT viewer."
        )
        fft_note.setFont(ui_font(8))
        fft_note.setWordWrap(True)
        fft_note.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(fft_note)
        lay.addStretch(1)

        # Default: start unlocked so the user can freely adjust both vectors.
        self._overlay.set_locked(False)

    def _on_grid_changed(self, grid: LatticeGrid) -> None:
        self.sync_from_model()
        if self._external_on_change is not None:
            self._external_on_change(grid)

    def sync_from_model(self) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None:
            return
        self._updating_controls = True
        try:
            self._angle_ab_spin.setValue(grid.angle_deg())
            self._cells_spin.setValue(self._overlay._cells)
            self._line_width_spin.setValue(self._overlay._line_width_px)

            g1 = self._cal.vec_length_q(grid.a_px)
            g2 = self._cal.vec_length_q(grid.b_px)
            self._g1_spin.setValue(g1)
            self._g2_spin.setValue(g2)

            kind = grid.kind
            idx = {"square": 0, "rectangular": 1, "hexagonal": 2}.get(kind, 0)
            if self._type_combo.currentIndex() != idx:
                self._type_combo.setCurrentIndex(idx)
            locked = self._lock_cb.isChecked()
            self._g2_spin.setEnabled(kind == "rectangular" or not locked)
            self._angle_ab_spin.setEnabled(not locked)
        finally:
            self._updating_controls = False

    def _on_g1_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None:
            return
        old_g1 = self._cal.vec_length_q(grid.a_px)
        if old_g1 < 1e-12:
            return
        factor = value / old_g1
        new_grid = grid.set_a_length_px(grid.a_length_px() * factor)
        self._overlay.set_grid(new_grid)

    def _on_g2_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None or grid.kind != "rectangular":
            return
        old_g2 = self._cal.vec_length_q(grid.b_px)
        if old_g2 < 1e-12:
            return
        factor = value / old_g2
        new_grid = grid.set_b_length_px(grid.b_length_px() * factor)
        self._overlay.set_grid(new_grid)

    def _on_cells_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self._overlay.set_cells(value)

    def _on_locked_changed(self, checked: bool) -> None:
        self._overlay.set_locked(checked)
        self._angle_ab_spin.setEnabled(not checked)
        g = self._overlay.grid()
        if g is not None:
            self._g2_spin.setEnabled(g.kind == "rectangular" or not checked)
            if checked:
                new_g = g.with_a_vector(g.a_px)
                self._overlay.set_grid(new_g)
                self.sync_from_model()

    def _on_angle_ab_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None:
            return
        lb_px = grid.b_length_px()
        if lb_px < 1e-9:
            return
        b_angle_rad = math.radians(grid.a_angle_deg() + value)
        new_b = (lb_px * math.cos(b_angle_rad), lb_px * math.sin(b_angle_rad))
        self._overlay.set_grid(replace(grid, b_px=new_b))

    def _on_line_width_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        self._overlay.set_line_width(value)

    def _on_type_changed(self, idx: int) -> None:
        if self._updating_controls:
            return
        kinds: list[LatticeKind] = ["square", "rectangular", "hexagonal"]
        new_kind = kinds[idx]
        g = self._overlay.grid()
        if g is None:
            return
        la = g.a_length_px()
        lb = g.b_length_px()
        angle_a = g.a_angle_deg()
        ca = math.cos(math.radians(angle_a))
        sa = math.sin(math.radians(angle_a))
        if new_kind == "square":
            new_g = replace(g, kind="square",
                a_px=(la * ca, la * sa),
                b_px=(la * math.cos(math.radians(angle_a + 90)),
                      la * math.sin(math.radians(angle_a + 90))))
        elif new_kind == "rectangular":
            new_g = replace(g, kind="rectangular",
                b_px=(lb * math.cos(math.radians(angle_a + 90)),
                      lb * math.sin(math.radians(angle_a + 90))))
        else:
            avg_l = (la + lb) * 0.5
            new_g = replace(g, kind="hexagonal",
                a_px=(avg_l * ca, avg_l * sa),
                b_px=(avg_l * math.cos(math.radians(angle_a + 60)),
                      avg_l * math.sin(math.radians(angle_a + 60))))
        self._overlay.set_grid(new_g)
        self.sync_from_model()

    def _on_visibility_changed(self) -> None:
        grid = self._overlay.grid()
        if grid is None:
            return
        self._overlay.set_grid(replace(
            grid,
            visible=self._show_grid_cb.isChecked(),
            show_handles=self._show_handles_cb.isChecked(),
            show_labels=self._show_labels_cb.isChecked(),
        ))

    def _on_reset_origin(self) -> None:
        grid = self._overlay.grid()
        if grid is None:
            return
        self._overlay.set_grid(
            grid.reset_origin(self._image_w / 2.0, self._image_h / 2.0)
        )
        self.sync_from_model()
