"""Real-space lattice grid control panel."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np

from probeflow.gui.typography import mono_font, ui_font
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QInputDialog,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from probeflow.analysis.lattice_correction_workflow import (
    lattice_correction_matrix_px,
    lattice_correction_operation_params,
)
from probeflow.analysis.lattice_distortion import (
    IdealLattice,
    LatticeCorrection,
    MeasuredLattice,
    compute_correction,
)
from probeflow.analysis.lattice_grid import (
    LatticeKind,
    RealSpaceCalibration,
    format_real_space_measurements,
)
from probeflow.gui.lattice_correction_ui import (
    KnownStructure,
    correction_main_lines,
    delete_structure,
    ideal_lattice_from_structure,
    load_known_structures,
    save_known_structures,
    upsert_structure,
)
from probeflow.gui.no_wheel import install_no_wheel_spinboxes

from .controller import LatticeGridController
from .graphics_item import LatticeGridItem
from .stored_grids import StoredGrid, StoredGridList, next_stored_color

# ── unit helpers ──────────────────────────────────────────────────────────────

def _choose_display_unit(cal: RealSpaceCalibration) -> tuple[float, str]:
    """
    Return (unit_scale_to_m, unit_label) for the most readable atomic-scale unit.

    unit_scale_to_m: multiply spinbox value by this to get metres.
    """
    typical_m = cal.px_size_x * cal.image_width * 0.1
    if typical_m < 5e-9:    # less than ~5 nm → use Å
        return (1e-10, "Å")
    return (1e-9, "nm")


# ── LatticeGridPanel ──────────────────────────────────────────────────────────

class LatticeGridPanel(QWidget):
    """
    Interactive control + measurement panel for a real-space lattice grid.

    Communicates with LatticeGridItem and LatticeGridController.
    """

    def __init__(
        self,
        item: LatticeGridItem,
        controller: LatticeGridController,
        calibration: RealSpaceCalibration,
        image_w: int,
        image_h: int,
        parent=None,
        get_image_fn=None,
        apply_correction_fn=None,
        preview_image_fn=None,
        clear_preview_fn=None,
    ):
        super().__init__(parent)
        self._item = item
        self._ctrl = controller
        self._cal = calibration
        self._image_w = image_w
        self._image_h = image_h
        self._get_image_fn = get_image_fn
        self._apply_correction_fn = apply_correction_fn
        self._preview_image_fn = preview_image_fn
        self._clear_preview_fn = clear_preview_fn
        self._correction: Optional[LatticeCorrection] = None
        self._preview_active: bool = False
        self._updating_structure = False
        self._known_structures = load_known_structures()
        self._active_known_structure = self._known_structures[0]

        self._unit_scale, self._unit_label = _choose_display_unit(calibration)
        self._updating_controls = False
        self._stored: list[StoredGrid] = []
        self._stored_items: list[LatticeGridItem] = []
        self._stored_color_count = 0

        self._build()
        self.sync_from_model()

    def cleanup(self) -> None:
        """Release transient viewer state owned by this panel."""
        self._clear_preview_if_active()
        for item in self._stored_items:
            try:
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            except RuntimeError:
                pass
        self._stored_items = []
        self._stored = []
        try:
            self._ctrl.set_active(False)
        except RuntimeError:
            pass
        try:
            self._ctrl.uninstall()
        except RuntimeError:
            pass

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer_lay = QVBoxLayout(self)
        outer_lay.setContentsMargins(0, 0, 0, 0)
        outer_lay.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setFont(ui_font(9))
        outer_lay.addWidget(self._tabs)

        # ── helpers ───────────────────────────────────────────────────────────

        def _make_spin_row(target_lay: QVBoxLayout):
            def _spin_row(
                label: str, lo: float, hi: float,
                step: float, decimals: int, suffix: str = "",
            ) -> QDoubleSpinBox:
                row = QHBoxLayout()
                lbl = QLabel(label)
                lbl.setFont(ui_font(9))
                lbl.setMinimumWidth(68)
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setSingleStep(step)
                spin.setDecimals(decimals)
                if suffix:
                    spin.setSuffix(f" {suffix}")
                spin.setFont(ui_font(9))
                spin.setFixedHeight(22)
                row.addWidget(lbl)
                row.addWidget(spin, 1)
                target_lay.addLayout(row)
                return spin
            return _spin_row

        # ════════════════════════════════════════════════════════════════════
        # Grid tab
        # ════════════════════════════════════════════════════════════════════
        grid_tab = QWidget()
        lay = QVBoxLayout(grid_tab)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── edit active ───────────────────────────────────────────────────────
        self._edit_cb = QCheckBox("Edit grid (drag handles)")
        self._edit_cb.setFont(ui_font(9, weight=QFont.Bold))
        self._edit_cb.setChecked(True)
        self._edit_cb.toggled.connect(self._on_active_toggled)
        lay.addWidget(self._edit_cb)
        self._ctrl.set_active(True)

        # ── lattice type + constraints ────────────────────────────────────────
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
        self._lock_cb.setChecked(True)
        self._lock_cb.toggled.connect(self._on_locked_changed)
        lay.addWidget(self._lock_cb)

        self._ab_equal_cb = QCheckBox("a = b  (equal vector lengths)")
        self._ab_equal_cb.setFont(ui_font(9))
        self._ab_equal_cb.setChecked(False)
        self._ab_equal_cb.toggled.connect(self._on_ab_equal_changed)
        lay.addWidget(self._ab_equal_cb)

        # ── parameters group ──────────────────────────────────────────────────
        params_grp = QGroupBox("Parameters")
        params_grp.setFont(ui_font(9))
        params_lay = QVBoxLayout(params_grp)
        params_lay.setSpacing(3)
        params_lay.setContentsMargins(6, 6, 6, 4)
        _spin_row = _make_spin_row(params_lay)

        self._ox_spin = _spin_row("Origin x:", 0.0, float(self._image_w), 0.5, 1, "px")
        self._oy_spin = _spin_row("Origin y:", 0.0, float(self._image_h), 0.5, 1, "px")

        a_max = self._cal.px_size_x * self._image_w / self._unit_scale
        self._a_spin = _spin_row(
            f"|a| ({self._unit_label}):", 0.001, a_max * 2, 0.01, 3, self._unit_label,
        )
        self._b_spin = _spin_row(
            f"|b| ({self._unit_label}):", 0.001, a_max * 2, 0.01, 3, self._unit_label,
        )

        self._rot_spin = _spin_row("Rotation:", -180.0, 180.0, 0.1, 1, "°")

        self._angle_ab_spin = _spin_row("Angle a-b:", 1.0, 179.0, 0.1, 2, "°")
        self._angle_ab_spin.setEnabled(False)

        cells_row = QHBoxLayout()
        cells_lbl = QLabel("Cells ±:")
        cells_lbl.setFont(ui_font(9))
        cells_lbl.setMinimumWidth(68)
        self._cells_spin = QSpinBox()
        self._cells_spin.setRange(1, 200)
        self._cells_spin.setValue(self._item.cells)
        self._cells_spin.setFont(ui_font(9))
        self._cells_spin.setFixedHeight(22)
        cells_row.addWidget(cells_lbl)
        cells_row.addWidget(self._cells_spin, 1)
        params_lay.addLayout(cells_row)

        self._line_width_spin = _spin_row("Line width:", 0.25, 10.0, 0.25, 2, "px")
        self._line_width_spin.setValue(1.5)

        lay.addWidget(params_grp)

        self._ox_spin.valueChanged.connect(self._on_origin_changed)
        self._oy_spin.valueChanged.connect(self._on_origin_changed)
        self._a_spin.valueChanged.connect(self._on_a_length_changed)
        self._b_spin.valueChanged.connect(self._on_b_length_changed)
        self._rot_spin.valueChanged.connect(self._on_rotation_changed)
        self._angle_ab_spin.valueChanged.connect(self._on_angle_ab_changed)
        self._cells_spin.valueChanged.connect(self._on_cells_changed)
        self._line_width_spin.valueChanged.connect(self._on_line_width_changed)

        # ── display group ─────────────────────────────────────────────────────
        disp_grp = QGroupBox("Display")
        disp_grp.setFont(ui_font(9))
        disp_lay = QVBoxLayout(disp_grp)
        disp_lay.setSpacing(2)
        disp_lay.setContentsMargins(6, 6, 6, 4)

        self._show_grid_cb    = QCheckBox("Show grid")
        self._show_handles_cb = QCheckBox("Show handles")
        self._show_labels_cb  = QCheckBox("Show labels")
        for cb in (self._show_grid_cb, self._show_handles_cb, self._show_labels_cb):
            cb.setFont(ui_font(9))
            cb.setChecked(True)
            cb.toggled.connect(self._on_visibility_changed)
            disp_lay.addWidget(cb)

        lay.addWidget(disp_grp)

        reset_btn = QPushButton("Reset origin to centre")
        reset_btn.setFont(ui_font(9))
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self._on_reset_origin)
        reset_row = QHBoxLayout()
        reset_row.setSpacing(4)
        reset_row.addWidget(reset_btn)
        # Stored grid layers: park the current grid as a coloured static
        # overlay and fit another lattice; Edit swaps a layer back in.
        self._stored_list = StoredGridList(store_button_row=reset_row)
        self._stored_list.store_requested.connect(self._on_store_grid)
        self._stored_list.edit_requested.connect(self._on_edit_stored)
        self._stored_list.remove_requested.connect(self._on_remove_stored)
        lay.addLayout(reset_row)
        lay.addWidget(self._stored_list)

        meas_grp = QGroupBox("Measured")
        meas_grp.setFont(ui_font(9))
        meas_lay = QVBoxLayout(meas_grp)
        meas_lay.setContentsMargins(6, 6, 6, 4)
        self._meas_lbl = QLabel("")
        self._meas_lbl.setFont(mono_font(8))
        self._meas_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._meas_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._meas_lbl.setWordWrap(True)
        meas_lay.addWidget(self._meas_lbl)
        lay.addWidget(meas_grp)

        lay.addStretch(1)
        exp_row = QHBoxLayout()
        exp_with_btn = QPushButton("Export with grid…")
        exp_with_btn.setFont(ui_font(9))
        exp_with_btn.setFixedHeight(24)
        exp_with_btn.clicked.connect(self._on_export_with_grid)
        exp_grid_btn = QPushButton("Export grid only…")
        exp_grid_btn.setFont(ui_font(9))
        exp_grid_btn.setFixedHeight(24)
        exp_grid_btn.clicked.connect(self._on_export_grid_only)
        exp_row.addWidget(exp_with_btn)
        exp_row.addWidget(exp_grid_btn)
        lay.addLayout(exp_row)

        self._tabs.addTab(grid_tab, "Grid")

        # ════════════════════════════════════════════════════════════════════
        # Distortion tab
        # ════════════════════════════════════════════════════════════════════
        dist_tab = QWidget()
        dist_lay = QVBoxLayout(dist_tab)
        dist_lay.setContentsMargins(6, 6, 6, 6)
        dist_lay.setSpacing(4)

        # ── ideal lattice group ───────────────────────────────────────────────
        ideal_grp = QGroupBox("Known structure")
        ideal_grp.setFont(ui_font(9))
        ideal_lay = QVBoxLayout(ideal_grp)
        ideal_lay.setSpacing(3)
        ideal_lay.setContentsMargins(6, 6, 6, 4)
        _ideal_spin = _make_spin_row(ideal_lay)

        structure_row = QHBoxLayout()
        structure_lbl = QLabel("Structure:")
        structure_lbl.setFont(ui_font(9))
        structure_lbl.setMinimumWidth(68)
        self._structure_combo = QComboBox()
        self._structure_combo.setFont(ui_font(9))
        self._structure_combo.setFixedHeight(22)
        self._structure_combo.setToolTip("Known surface lattice used as the correction target.")
        self._structure_combo.currentIndexChanged.connect(self._on_structure_selected)
        self._structure_save_btn = QPushButton("Save")
        self._structure_update_btn = QPushButton("Update")
        self._structure_delete_btn = QPushButton("Delete")
        for btn in (
            self._structure_save_btn,
            self._structure_update_btn,
            self._structure_delete_btn,
        ):
            btn.setFont(ui_font(8))
            btn.setFixedHeight(22)
        self._structure_save_btn.clicked.connect(self._on_save_structure)
        self._structure_update_btn.clicked.connect(self._on_update_structure)
        self._structure_delete_btn.clicked.connect(self._on_delete_structure)
        structure_row.addWidget(structure_lbl)
        structure_row.addWidget(self._structure_combo, 1)
        structure_row.addWidget(self._structure_save_btn)
        structure_row.addWidget(self._structure_update_btn)
        structure_row.addWidget(self._structure_delete_btn)
        ideal_lay.addLayout(structure_row)

        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Preset:")
        preset_lbl.setFont(ui_font(9))
        preset_lbl.setMinimumWidth(68)
        self._ideal_preset_combo = QComboBox()
        self._ideal_preset_combo.setFont(ui_font(9))
        self._ideal_preset_combo.setFixedHeight(22)
        self._ideal_preset_combo.addItems([
            "Match grid", "Square", "Rectangular", "Hexagonal", "Custom",
        ])
        preset_row.addWidget(preset_lbl)
        preset_row.addWidget(self._ideal_preset_combo, 1)
        ideal_lay.addLayout(preset_row)

        self._ideal_a_spin = _ideal_spin(
            f"|a| ({self._unit_label}):", 0.001, a_max * 2, 0.01, 3, self._unit_label,
        )
        self._ideal_b_spin = _ideal_spin(
            f"|b| ({self._unit_label}):", 0.001, a_max * 2, 0.01, 3, self._unit_label,
        )
        self._ideal_ab_cb = QCheckBox("ideal a = b")
        self._ideal_ab_cb.setFont(ui_font(9))
        self._ideal_ab_cb.setChecked(False)
        self._ideal_ab_cb.setVisible(False)
        ideal_lay.addWidget(self._ideal_ab_cb)

        self._ideal_angle_spin = _ideal_spin("Angle:", 1.0, 179.0, 0.1, 2, "°")
        self._ideal_angle_spin.setValue(90.0)
        self._refresh_structure_combo(self._active_known_structure.name)
        self._apply_known_structure(self._active_known_structure, refresh=False)

        dist_lay.addWidget(ideal_grp)

        self._ideal_preset_combo.currentIndexChanged.connect(self._on_ideal_preset_changed)
        self._ideal_a_spin.valueChanged.connect(self._on_ideal_a_changed)
        self._ideal_b_spin.valueChanged.connect(self._on_ideal_b_changed)
        self._ideal_ab_cb.toggled.connect(self._on_ideal_ab_changed)
        self._ideal_angle_spin.valueChanged.connect(self._on_ideal_angle_changed)

        # ── measured lattice display ──────────────────────────────────────────
        meas_dist_grp = QGroupBox("Measured lattice")
        meas_dist_grp.setFont(ui_font(9))
        meas_dist_lay = QVBoxLayout(meas_dist_grp)
        meas_dist_lay.setContentsMargins(6, 6, 6, 4)
        self._measured_lbl = QLabel("(tune grid above)")
        self._measured_lbl.setFont(mono_font(8))
        self._measured_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._measured_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meas_dist_lay.addWidget(self._measured_lbl)
        dist_lay.addWidget(meas_dist_grp)

        # ── correction display ────────────────────────────────────────────────
        corr_grp = QGroupBox("Correction: measured → ideal")
        corr_grp.setFont(ui_font(9))
        corr_lay = QVBoxLayout(corr_grp)
        corr_lay.setContentsMargins(6, 6, 6, 4)
        self._correction_lbl = QLabel("(enter ideal lattice above)")
        self._correction_lbl.setFont(mono_font(8))
        self._correction_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._correction_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._correction_lbl.setWordWrap(True)
        corr_lay.addWidget(self._correction_lbl)
        dist_lay.addWidget(corr_grp)

        dist_lay.addStretch(1)

        # ── correction options ────────────────────────────────────────────────
        opts_grp = QGroupBox("Advanced correction options")
        opts_grp.setFont(ui_font(9))
        opts_grp.setCheckable(True)
        opts_grp.setChecked(False)
        opts_lay = QVBoxLayout(opts_grp)
        opts_lay.setSpacing(3)
        opts_lay.setContentsMargins(6, 6, 6, 4)

        self._expand_cb = QCheckBox("Expand canvas to fit corrected image")
        self._expand_cb.setFont(ui_font(9))
        self._expand_cb.setChecked(True)
        opts_lay.addWidget(self._expand_cb)

        self._preserve_orientation_cb = QCheckBox("Preserve image orientation")
        self._preserve_orientation_cb.setFont(ui_font(9))
        self._preserve_orientation_cb.setChecked(True)
        self._preserve_orientation_cb.setToolTip(
            "Remove the global rotation from the measured-to-ideal correction.\n"
            "This corrects lattice shear and scale while keeping the scan\n"
            "approximately aligned with the original image axes."
        )
        opts_lay.addWidget(self._preserve_orientation_cb)
        self._preserve_orientation_cb.toggled.connect(self._on_preserve_orientation_changed)

        interp_row = QHBoxLayout()
        interp_lbl = QLabel("Interpolation:")
        interp_lbl.setFont(ui_font(9))
        interp_lbl.setMinimumWidth(85)
        self._interp_combo = QComboBox()
        self._interp_combo.setFont(ui_font(9))
        self._interp_combo.setFixedHeight(22)
        self._interp_combo.addItems(["Bilinear", "Nearest", "Bicubic"])
        interp_row.addWidget(interp_lbl)
        interp_row.addWidget(self._interp_combo, 1)
        opts_lay.addLayout(interp_row)

        fill_row = QHBoxLayout()
        fill_lbl = QLabel("Fill:")
        fill_lbl.setFont(ui_font(9))
        fill_lbl.setMinimumWidth(85)
        self._fill_combo = QComboBox()
        self._fill_combo.setFont(ui_font(9))
        self._fill_combo.setFixedHeight(22)
        self._fill_combo.addItems(["NaN", "Background", "Zero"])
        fill_row.addWidget(fill_lbl)
        fill_row.addWidget(self._fill_combo, 1)
        opts_lay.addLayout(fill_row)

        dist_lay.addWidget(opts_grp)

        # ── preview status ────────────────────────────────────────────────────
        self._preview_status_lbl = QLabel("")
        self._preview_status_lbl.setFont(ui_font(8))
        self._preview_status_lbl.setWordWrap(True)
        self._preview_status_lbl.setVisible(False)
        dist_lay.addWidget(self._preview_status_lbl)

        # ── action buttons ────────────────────────────────────────────────────
        self._preview_btn = QPushButton("Preview correction")
        self._preview_btn.setFont(ui_font(9))
        self._preview_btn.setFixedHeight(24)
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview)
        self._clear_preview_btn = QPushButton("Clear preview")
        self._clear_preview_btn.setFont(ui_font(9))
        self._clear_preview_btn.setFixedHeight(24)
        self._clear_preview_btn.setEnabled(False)
        self._clear_preview_btn.clicked.connect(self._on_clear_preview)
        self._apply_btn = QPushButton("Apply correction")
        self._apply_btn.setFont(ui_font(9))
        self._apply_btn.setFixedHeight(24)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        dist_lay.addWidget(self._preview_btn)
        dist_lay.addWidget(self._clear_preview_btn)
        dist_lay.addWidget(self._apply_btn)

        self._tabs.addTab(dist_tab, "Distortion")
        install_no_wheel_spinboxes(self)

    # ── model↔UI sync ─────────────────────────────────────────────────────────

    def _refresh_structure_combo(self, selected_name: str | None = None) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None:
            return
        self._updating_structure = True
        try:
            combo.clear()
            for structure in self._known_structures:
                combo.addItem(structure.name, structure)
            if selected_name:
                for idx, structure in enumerate(self._known_structures):
                    if structure.name == selected_name:
                        combo.setCurrentIndex(idx)
                        break
        finally:
            self._updating_structure = False

    def _on_structure_selected(self, idx: int) -> None:
        if self._updating_structure:
            return
        combo = getattr(self, "_structure_combo", None)
        if combo is None or idx < 0:
            return
        structure = combo.itemData(idx)
        if isinstance(structure, KnownStructure):
            self._apply_known_structure(structure)

    def _apply_known_structure(
        self,
        structure: KnownStructure,
        *,
        refresh: bool = True,
    ) -> None:
        self._active_known_structure = structure
        measured_angle = None
        try:
            measured_angle = self._measured_lattice_values()[2]
        except Exception:
            pass
        ideal = ideal_lattice_from_structure(structure, measured_angle_deg=measured_angle)
        preset = {
            "square": "Square",
            "rectangular": "Rectangular",
            "hexagonal": "Hexagonal",
        }.get(structure.symmetry, "Custom")
        self._updating_controls = True
        try:
            self._ideal_preset_combo.setCurrentText(preset)
            self._ideal_a_spin.setValue(max(self._ideal_a_spin.minimum(), ideal.a_nm / (self._unit_scale * 1e9)))
            self._ideal_b_spin.setValue(max(self._ideal_b_spin.minimum(), ideal.b_nm / (self._unit_scale * 1e9)))
            self._ideal_angle_spin.setValue(min(179.0, max(1.0, ideal.angle_deg)))
        finally:
            self._updating_controls = False
        self._refresh_ideal_control_state()
        if refresh:
            self._clear_preview_if_active()
            self._refresh_correction_label()

    def _structure_from_ideal_controls(self, name: str) -> KnownStructure:
        a_nm = self._ideal_a_spin.value() * self._unit_scale * 1e9
        b_nm = self._ideal_b_spin.value() * self._unit_scale * 1e9
        angle = self._ideal_angle_spin.value()
        preset = self._ideal_preset_combo.currentText()
        symmetry = {
            "Square": "square",
            "Rectangular": "rectangular",
            "Hexagonal": "hexagonal",
        }.get(preset, "custom")
        if symmetry == "square":
            b_nm, angle = a_nm, 90.0
        elif symmetry == "hexagonal":
            b_nm, angle = a_nm, 60.0
        elif symmetry == "rectangular":
            angle = 90.0
        return KnownStructure(name, symmetry, a_nm, b_nm, angle, self._unit_label)

    def _persist_known_structures(self, selected_name: str) -> None:
        save_known_structures(self._known_structures)
        self._refresh_structure_combo(selected_name)

    def _on_save_structure(self) -> None:
        default = getattr(self, "_active_known_structure", self._known_structures[0]).name
        name, ok = QInputDialog.getText(self, "Save known structure", "Structure name:", text=default)
        if not ok or not name.strip():
            return
        structure = self._structure_from_ideal_controls(name.strip())
        self._known_structures = upsert_structure(self._known_structures, structure)
        self._persist_known_structures(structure.name)
        self._apply_known_structure(structure)

    def _on_update_structure(self) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return
        name = combo.currentText().strip()
        if not name:
            return
        structure = self._structure_from_ideal_controls(name)
        self._known_structures = upsert_structure(self._known_structures, structure)
        self._persist_known_structures(structure.name)
        self._apply_known_structure(structure)

    def _on_delete_structure(self) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return
        self._known_structures = delete_structure(self._known_structures, combo.currentText())
        selected = self._known_structures[0]
        self._persist_known_structures(selected.name)
        self._apply_known_structure(selected)

    def sync_from_model(self) -> None:
        """Update all spinboxes and measurement display from the current grid."""
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            grid = self._item.grid()
            ox, oy = grid.origin_px

            a_len_m = self._cal.vector_length_m(grid.a_px)
            b_len_m = self._cal.vector_length_m(grid.b_px)
            a_len_u = a_len_m / self._unit_scale
            b_len_u = b_len_m / self._unit_scale

            self._ox_spin.setValue(ox)
            self._oy_spin.setValue(oy)
            self._a_spin.setValue(a_len_u)
            self._b_spin.setValue(b_len_u)
            self._rot_spin.setValue(grid.a_angle_deg())
            self._angle_ab_spin.setValue(grid.angle_deg())
            self._cells_spin.setValue(self._item.cells)
            self._line_width_spin.setValue(self._item._line_width_px)

            kind = grid.kind
            idx = {"square": 0, "rectangular": 1, "hexagonal": 2}.get(kind, 0)
            if self._type_combo.currentIndex() != idx:
                self._type_combo.setCurrentIndex(idx)

            locked = self._lock_cb.isChecked()
            # b-length editable if rectangular or tunable
            self._b_spin.setEnabled(kind == "rectangular" or not locked)
            # angle_ab editable only in tunable mode
            self._angle_ab_spin.setEnabled(not locked)

            self._sync_ideal_preset_from_grid()

        finally:
            self._updating_controls = False

        self._refresh_measurement_label()
        self._refresh_correction_label()

    def _refresh_measurement_label(self) -> None:
        grid = self._item.grid()
        try:
            d = format_real_space_measurements(grid, self._cal)
            lines = [
                f"|a| = {d['a_length']}",
                f"|b| = {d['b_length']}",
                f"angle = {d['angle']}",
                f"area = {d['area']}",
                f"origin = {d['origin_phys']}",
            ]
        except Exception as exc:
            lines = [f"(error: {exc})"]
        self._meas_lbl.setText("\n".join(lines))

    # ── spinbox slots ─────────────────────────────────────────────────────────

    def _on_origin_changed(self, _val: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        new_grid = grid.reset_origin(self._ox_spin.value(), self._oy_spin.value())
        self._item.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_a_length_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        new_a_m = value * self._unit_scale
        old_a_m = self._cal.vector_length_m(grid.a_px)
        old_a_px = grid.a_length_px()
        if old_a_m < 1e-25 or old_a_px < 1e-9:
            return
        new_a_px = new_a_m * old_a_px / old_a_m
        new_grid = grid.set_a_length_px(new_a_px)
        # If a=b is active, also force b to same pixel length (preserve b direction)
        if self._ab_equal_cb.isChecked():
            lb = new_grid.b_length_px()
            if lb > 1e-9:
                bx, by = new_grid.b_px
                new_grid = replace(new_grid, b_px=(bx * new_a_px / lb, by * new_a_px / lb))
        self._item.set_grid(new_grid)
        self._updating_controls = True
        try:
            b_m = self._cal.vector_length_m(new_grid.b_px)
            self._b_spin.setValue(b_m / self._unit_scale)
        finally:
            self._updating_controls = False
        self._refresh_measurement_label()
        self._refresh_correction_label()

    def _on_b_length_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        if grid.kind != "rectangular" and not self._ab_equal_cb.isChecked():
            return
        new_b_m = value * self._unit_scale
        old_b_m = self._cal.vector_length_m(grid.b_px)
        old_b_px = grid.b_length_px()
        if old_b_m < 1e-25 or old_b_px < 1e-9:
            return
        new_b_px = new_b_m * old_b_px / old_b_m
        # Raw rescale: preserve b direction
        bx, by = grid.b_px
        new_grid = replace(grid, b_px=(bx * new_b_px / old_b_px, by * new_b_px / old_b_px))
        # If a=b, also force a to same length
        if self._ab_equal_cb.isChecked():
            la = new_grid.a_length_px()
            if la > 1e-9:
                ax, ay = new_grid.a_px
                new_grid = replace(new_grid, a_px=(ax * new_b_px / la, ay * new_b_px / la))
        self._item.set_grid(new_grid)
        self._refresh_measurement_label()
        self._refresh_correction_label()

    def _on_rotation_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        new_grid = grid.set_rotation_deg(value)
        self._item.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_cells_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self._item.set_cells(value)

    def _on_locked_changed(self, checked: bool) -> None:
        self._ctrl.set_locked(checked)
        self._angle_ab_spin.setEnabled(not checked)
        g = self._item.grid()
        self._b_spin.setEnabled(g.kind == "rectangular" or not checked)
        if checked:
            new_g = g.with_a_vector(g.a_px)
            self._item.set_grid(new_g)
            self.sync_from_model()

    def _on_angle_ab_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        lb_px = grid.b_length_px()
        if lb_px < 1e-9:
            return
        b_angle_rad = math.radians(grid.a_angle_deg() + value)
        new_b = (lb_px * math.cos(b_angle_rad), lb_px * math.sin(b_angle_rad))
        self._item.set_grid(replace(grid, b_px=new_b))
        self._refresh_measurement_label()

    def _on_line_width_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        self._item.set_line_width(value)

    def _on_ab_equal_changed(self, checked: bool) -> None:
        self._ctrl.set_ab_equal(checked)
        if checked:
            # Immediately force |b| = |a|
            g = self._item.grid()
            la = g.a_length_px()
            lb = g.b_length_px()
            if la > 1e-9 and lb > 1e-9 and abs(la - lb) > 1e-9:
                bx, by = g.b_px
                self._item.set_grid(replace(g, b_px=(bx * la / lb, by * la / lb)))
        self.sync_from_model()

    def _measured_lattice_values(self) -> tuple[float, float, float]:
        grid = self._item.grid()
        a_len_u = self._cal.vector_length_m(grid.a_px) / self._unit_scale
        b_len_u = self._cal.vector_length_m(grid.b_px) / self._unit_scale
        angle = self._cal.vector_angle_deg(grid.a_px, grid.b_px)
        return a_len_u, b_len_u, angle

    def _set_ideal_values(self, a_value: float, b_value: float, angle: float) -> None:
        self._ideal_a_spin.setValue(max(self._ideal_a_spin.minimum(), a_value))
        self._ideal_b_spin.setValue(max(self._ideal_b_spin.minimum(), b_value))
        self._ideal_angle_spin.setValue(min(179.0, max(1.0, angle)))

    def _sync_ideal_preset_from_grid(self, *, initialize: bool = False) -> None:
        preset = self._ideal_preset_combo.currentText()
        a_val, b_val, angle = self._measured_lattice_values()
        if preset == "Match grid":
            self._set_ideal_values(a_val, b_val, angle)
        elif initialize and preset == "Square":
            side = 0.5 * (a_val + b_val)
            self._set_ideal_values(side, side, 90.0)
        elif initialize and preset == "Rectangular":
            self._set_ideal_values(a_val, b_val, 90.0)
        elif initialize and preset == "Hexagonal":
            side = 0.5 * (a_val + b_val)
            self._set_ideal_values(side, side, 60.0)
        self._refresh_ideal_control_state()

    def _refresh_ideal_control_state(self) -> None:
        preset = self._ideal_preset_combo.currentText()
        custom = preset == "Custom"
        match_grid = preset == "Match grid"
        equal_lengths = preset in {"Square", "Hexagonal"}
        fixed_angle = preset in {"Square", "Rectangular", "Hexagonal", "Match grid"}
        self._ideal_a_spin.setEnabled(not match_grid)
        self._ideal_b_spin.setEnabled(custom or preset == "Rectangular")
        self._ideal_angle_spin.setEnabled(custom)
        self._ideal_ab_cb.setVisible(custom)
        self._ideal_ab_cb.setEnabled(custom)
        if equal_lengths and not self._updating_controls:
            self._ideal_b_spin.setValue(self._ideal_a_spin.value())
        if fixed_angle and preset != "Match grid" and not self._updating_controls:
            angle = 60.0 if preset == "Hexagonal" else 90.0
            self._ideal_angle_spin.setValue(angle)

    def _on_ideal_preset_changed(self, _idx: int) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self._sync_ideal_preset_from_grid(initialize=True)
        finally:
            self._updating_controls = False
        self._clear_preview_if_active()
        self._refresh_correction_label()

    def _on_ideal_a_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        if (
            self._ideal_ab_cb.isChecked()
            or self._ideal_preset_combo.currentText() in {"Square", "Hexagonal"}
        ):
            self._updating_controls = True
            try:
                self._ideal_b_spin.setValue(value)
            finally:
                self._updating_controls = False
        self._clear_preview_if_active()
        self._refresh_correction_label()

    def _on_ideal_b_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        if self._ideal_ab_cb.isChecked():
            self._updating_controls = True
            try:
                self._ideal_a_spin.setValue(value)
            finally:
                self._updating_controls = False
        self._clear_preview_if_active()
        self._refresh_correction_label()

    def _on_ideal_ab_changed(self, checked: bool) -> None:
        if checked:
            self._updating_controls = True
            try:
                self._ideal_b_spin.setValue(self._ideal_a_spin.value())
            finally:
                self._updating_controls = False
        self._clear_preview_if_active()
        self._refresh_correction_label()

    def _on_ideal_angle_changed(self, _value: float) -> None:
        if self._updating_controls:
            return
        self._clear_preview_if_active()
        self._refresh_correction_label()

    def _refresh_correction_label(self) -> None:
        """Compute and display the measured-vs-ideal affine correction."""
        grid = self._item.grid()
        try:
            if (
                not self._updating_controls
                and self._ideal_preset_combo.currentText() == "Match grid"
            ):
                self._updating_controls = True
                try:
                    self._sync_ideal_preset_from_grid()
                finally:
                    self._updating_controls = False
            ideal_a_m = self._ideal_a_spin.value() * self._unit_scale
            ideal_b_m = self._ideal_b_spin.value() * self._unit_scale
            ideal_angle = self._ideal_angle_spin.value()

            # Physical measured vectors in nm (always update measured label)
            px_nm_x = self._cal.px_size_x * 1e9
            px_nm_y = self._cal.px_size_y * 1e9
            ax_px, ay_px = grid.a_px
            bx_px, by_px = grid.b_px
            m_a_nm = (ax_px * px_nm_x, ay_px * px_nm_y)
            m_b_nm = (bx_px * px_nm_x, by_px * px_nm_y)
            m_la = math.hypot(*m_a_nm)
            m_lb = math.hypot(*m_b_nm)
            m_angle = self._cal.vector_angle_deg(grid.a_px, grid.b_px)
            unit = self._unit_label
            s = self._unit_scale * 1e9

            self._measured_lbl.setText(
                f"|a| = {m_la / s:.4g} {unit}\n"
                f"|b| = {m_lb / s:.4g} {unit}\n"
                f"angle = {m_angle:.2f}°"
            )

            if ideal_a_m < 1e-25 or ideal_b_m < 1e-25:
                self._correction = None
                self._preview_btn.setEnabled(False)
                self._apply_btn.setEnabled(False)
                self._correction_lbl.setText("(enter ideal lattice above)")
                return

            measured = MeasuredLattice(a_nm=m_a_nm, b_nm=m_b_nm)
            ideal = IdealLattice(
                a_nm=ideal_a_m * 1e9,
                b_nm=ideal_b_m * 1e9,
                angle_deg=ideal_angle,
            )
            result = compute_correction(measured, ideal)

            if isinstance(result, str):
                self._correction = None
                self._preview_btn.setEnabled(False)
                self._apply_btn.setEnabled(False)
                self._correction_lbl.setText(f"Cannot compute correction:\n{result}")
                return

            self._correction = result
            self._preview_btn.setEnabled(
                self._get_image_fn is not None or self._preview_image_fn is not None
            )
            self._apply_btn.setEnabled(self._apply_correction_fn is not None)

            self._correction_lbl.setText("\n".join(
                correction_main_lines(
                    result,
                    preserve_orientation=self._preserve_orientation_cb.isChecked(),
                )
            ))
        except Exception as exc:
            self._correction = None
            self._preview_btn.setEnabled(False)
            self._apply_btn.setEnabled(False)
            self._correction_lbl.setText(f"(error: {exc})")

    # ── correction action helpers ─────────────────────────────────────────────

    def _correction_matrix_px(self) -> Optional[np.ndarray]:
        """Return the pixel-space correction matrix to apply to the image.

        Uses stretch_matrix (orientation-preserving) when the checkbox is on,
        otherwise uses the full measured-to-ideal matrix.
        """
        if self._correction is None:
            return None
        return lattice_correction_matrix_px(
            self._correction,
            pixel_size_x_m=self._cal.px_size_x,
            pixel_size_y_m=self._cal.px_size_y,
            preserve_orientation=self._preserve_orientation_cb.isChecked(),
        )

    def _correction_options(self) -> dict:
        interp_map = {"Bilinear": "bilinear", "Nearest": "nearest", "Bicubic": "bicubic"}
        fill_map = {"NaN": "nan", "Background": "background", "Zero": "zero"}
        return {
            "interpolation": interp_map.get(self._interp_combo.currentText(), "bilinear"),
            "fill_mode": fill_map.get(self._fill_combo.currentText(), "nan"),
            "expand_canvas": self._expand_cb.isChecked(),
            "preserve_orientation": self._preserve_orientation_cb.isChecked(),
        }

    def _on_preserve_orientation_changed(self) -> None:
        self._clear_preview_if_active()
        self._refresh_correction_label()

    def _set_preview_state(self, active: bool) -> None:
        self._preview_active = active
        self._clear_preview_btn.setEnabled(active)
        if active:
            self._preview_status_lbl.setText(
                "Previewing lattice correction. "
                "Click Apply to commit or Clear preview to return."
            )
        else:
            self._preview_status_lbl.setText("")
        self._preview_status_lbl.setVisible(active)

    def _clear_preview_if_active(self) -> None:
        if self._preview_active and self._clear_preview_fn is not None:
            self._clear_preview_fn()
            self._set_preview_state(False)

    def _on_preview(self) -> None:
        if self._correction is None:
            return
        get_arr = self._get_image_fn
        if get_arr is None:
            return
        T_px = self._correction_matrix_px()
        if T_px is None:
            return
        arr = get_arr()
        if arr is None:
            return
        from probeflow.processing.image import affine_lattice_correction
        try:
            opts = self._correction_options()
            corrected = affine_lattice_correction(
                arr,
                T_px,
                expand_canvas=opts["expand_canvas"],
                interpolation=opts["interpolation"],
                fill_mode=opts["fill_mode"],
            )
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Preview failed", str(exc))
            return

        if self._preview_image_fn is not None:
            self._preview_image_fn(corrected)
            self._set_preview_state(True)
        else:
            # Fallback: open comparison dialog when no in-viewer preview hook available
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from PySide6.QtWidgets import QDialog, QVBoxLayout as _VBox
            dlg = QDialog(self)
            dlg.setWindowTitle("Lattice correction preview")
            dlg.resize(700, 380)
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            vlo = float(np.nanpercentile(arr, 2))
            vhi = float(np.nanpercentile(arr, 98))
            axes[0].set_title("Before")
            axes[0].imshow(arr, origin="upper", cmap="gray", vmin=vlo, vmax=vhi)
            axes[0].axis("off")
            axes[1].set_title("After (preview)")
            axes[1].imshow(corrected, origin="upper", cmap="gray", vmin=vlo, vmax=vhi)
            axes[1].axis("off")
            fig.tight_layout()
            canvas = FigureCanvasQTAgg(fig)
            vbox = _VBox(dlg)
            vbox.addWidget(canvas)
            dlg.exec()
            plt.close(fig)

    def _on_clear_preview(self) -> None:
        if self._clear_preview_fn is not None:
            self._clear_preview_fn()
        self._set_preview_state(False)

    def _on_apply(self) -> None:
        if self._apply_correction_fn is None or self._correction is None:
            return
        if self._correction_matrix_px() is None:
            return

        # Clear preview before applying so the viewer is in sync
        if self._preview_active and self._clear_preview_fn is not None:
            self._clear_preview_fn()
        self._set_preview_state(False)

        opts = self._correction_options()
        corr = self._correction
        op_params = lattice_correction_operation_params(
            corr,
            pixel_size_x_m=self._cal.px_size_x,
            pixel_size_y_m=self._cal.px_size_y,
            expand_canvas=opts["expand_canvas"],
            interpolation=opts["interpolation"],
            fill_mode=opts["fill_mode"],
            preserve_orientation=opts["preserve_orientation"],
        )
        if op_params is None:
            return
        structure = getattr(self, "_active_known_structure", None)
        if isinstance(structure, KnownStructure):
            op_params["known_structure"] = structure.as_dict()
        self._apply_correction_fn("affine_lattice_correction", op_params)

        # Hide the grid overlay — it was measured on the pre-correction image
        self._item.setVisible(False)
        self._correction_lbl.setText(
            "Correction applied.\n"
            "Grid hidden: it was measured on the pre-correction image."
        )

    def _on_type_changed(self, idx: int) -> None:
        if self._updating_controls:
            return
        kinds: list[LatticeKind] = ["square", "rectangular", "hexagonal"]
        new_kind = kinds[idx]
        g = self._item.grid()
        la = g.a_length_px()
        lb = g.b_length_px()
        angle_a = g.a_angle_deg()
        ca, sa = math.cos(math.radians(angle_a)), math.sin(math.radians(angle_a))

        if new_kind == "square":
            new_g = replace(g, kind="square",
                a_px=(la * ca, la * sa),
                b_px=(la * math.cos(math.radians(angle_a + 90)),
                      la * math.sin(math.radians(angle_a + 90))))
        elif new_kind == "rectangular":
            new_g = replace(g, kind="rectangular",
                b_px=(lb * math.cos(math.radians(angle_a + 90)),
                      lb * math.sin(math.radians(angle_a + 90))))
        else:  # hexagonal
            avg_l = (la + lb) * 0.5
            new_g = replace(g, kind="hexagonal",
                a_px=(avg_l * ca, avg_l * sa),
                b_px=(avg_l * math.cos(math.radians(angle_a + 60)),
                      avg_l * math.sin(math.radians(angle_a + 60))))

        self._item.set_grid(new_g)
        self.sync_from_model()

    def _on_active_toggled(self, checked: bool) -> None:
        self._ctrl.set_active(checked)

    def _on_visibility_changed(self) -> None:
        g = self._item.grid()
        self._item.set_grid(replace(
            g,
            visible=self._show_grid_cb.isChecked(),
            show_handles=self._show_handles_cb.isChecked(),
            show_labels=self._show_labels_cb.isChecked(),
        ))

    def _on_reset_origin(self) -> None:
        g = self._item.grid()
        self._item.set_grid(g.reset_origin(self._image_w / 2.0, self._image_h / 2.0))
        self.sync_from_model()

    # ── stored grid layers ────────────────────────────────────────────────────

    def _grid_summary(self, grid) -> str:
        try:
            d = format_real_space_measurements(grid, self._cal)
            return f"{d['a_length']} · {d['b_length']} · {d['angle']}"
        except Exception:
            return "grid"

    def _grid_detail(self, grid) -> str:
        try:
            d = format_real_space_measurements(grid, self._cal)
            return (
                f"|a| = {d['a_length']}\n|b| = {d['b_length']}\n"
                f"angle = {d['angle']}\narea = {d['area']}"
            )
        except Exception:
            return ""

    def _stored_entry_from(self, grid, color: str) -> StoredGrid:
        return StoredGrid(
            grid=replace(grid, show_handles=False, show_labels=False),
            cells=self._item.cells,
            line_width_px=float(self._line_width_spin.value()),
            color=color,
            summary=self._grid_summary(grid),
            detail=self._grid_detail(grid),
        )

    def _on_store_grid(self) -> None:
        scene = self._item.scene()
        if scene is None:
            return
        entry = self._stored_entry_from(
            self._item.grid(), next_stored_color(self._stored_color_count)
        )
        self._stored_color_count += 1
        item = LatticeGridItem(
            entry.grid, self._image_w, self._image_h,
            cells=entry.cells, color=entry.color,
        )
        item.set_line_width(entry.line_width_px)
        item.setZValue(45)  # under the active grid (50) so its handles stay on top
        scene.addItem(item)
        self._stored.append(entry)
        self._stored_items.append(item)
        self._stored_list.set_entries(self._stored)

    def _on_edit_stored(self, index: int) -> None:
        """Swap a stored layer into the editor; the active grid takes its slot."""
        if not (0 <= index < len(self._stored)):
            return
        stored = self._stored[index]
        active = self._item.grid()
        parked = self._stored_entry_from(active, stored.color)
        self._stored[index] = parked
        self._stored_items[index].set_grid(parked.grid)
        self._stored_items[index].set_cells(parked.cells)
        self._stored_items[index].set_line_width(parked.line_width_px)
        self._item.set_grid(replace(
            stored.grid,
            visible=active.visible,
            show_handles=active.show_handles,
            show_labels=active.show_labels,
        ))
        self._stored_list.set_entries(self._stored)
        self.sync_from_model()

    def _on_remove_stored(self, index: int) -> None:
        if not (0 <= index < len(self._stored)):
            return
        item = self._stored_items.pop(index)
        del self._stored[index]
        scene = item.scene()
        if scene is not None:
            scene.removeItem(item)
        self._stored_list.set_entries(self._stored)

    def _on_export_with_grid(self) -> None:
        self._export(grid_only=False)

    def _on_export_grid_only(self) -> None:
        self._export(grid_only=True)

    def _export(self, grid_only: bool) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "",
            "PNG image (*.png);;PDF document (*.pdf)",
        )
        if not path:
            return
        from probeflow.gui.lattice_export import export_grid
        export_grid(self._item, path, include_grid=True, grid_only=grid_only)
