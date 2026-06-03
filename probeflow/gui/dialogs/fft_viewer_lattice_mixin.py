"""FFT-derived lattice-correction (Grid / Correction) tab for the FFT viewer.

Mixin split out of ``fft_viewer.py``. It relies on attributes owned by
``FFTViewerDialog`` (the Bragg/known-structure controls, the ``_fft_ideal_*`` and
``_fft_*`` correction widgets, the reciprocal-grid overlay/panel, ``self._arr`` /
``self._qx`` / ``self._qy`` / ``self._scan_range_m``, the ``_get_image_fn`` /
``_apply_correction_fn`` callbacks, and the shared helpers ``_show_fft_preview`` /
``_hide_fft_preview`` / ``_recompute_fft`` / ``_redraw`` / ``_update_info_panel`` /
``_resolve_source_array`` / ``_on_bragg_changed``).
"""

from __future__ import annotations

import math
import weakref

import numpy as np
from probeflow.analysis.lattice_correction_workflow import (
    lattice_correction_matrix_px,
    lattice_correction_operation_params,
)
from probeflow.analysis.lattice_distortion import (
    IdealLattice,
    MeasuredLattice,
    compute_correction,
)
from probeflow.analysis.lattice_grid import direct_lattice_vectors_from_reciprocal_grid
from probeflow.gui.lattice_correction_ui import (
    KnownStructure,
    correction_main_lines,
    delete_structure,
    ideal_lattice_from_structure,
    save_known_structures,
    structure_display_value_nm,
    upsert_structure,
)
from PySide6.QtWidgets import QFileDialog, QInputDialog, QWidget


class FFTViewerLatticeMixin:
    """Known-structure presets, reciprocal-grid measurement, and affine correction."""

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
        if not isinstance(structure, KnownStructure):
            return
        self._apply_known_structure_to_fft(structure)

    def _apply_known_structure_to_fft(
        self,
        structure: KnownStructure,
        *,
        refresh: bool = True,
    ) -> None:
        self._active_known_structure = structure
        self._updating_structure = True
        try:
            if structure.symmetry in {"square", "hexagonal"}:
                self._bragg_sym_combo.setCurrentIndex(0 if structure.symmetry == "square" else 1)
                if not self._bragg_enable_cb.isChecked():
                    self._bragg_enable_cb.setChecked(True)
            else:
                self._bragg_enable_cb.setChecked(False)
            unit = "Å" if structure.unit == "Å" else "nm"
            self._bragg_unit_combo.setCurrentText(unit)
            self._bragg_a_spin.setValue(structure_display_value_nm(structure))
        finally:
            self._updating_structure = False

        if hasattr(self, "_fft_ideal_combo"):
            target = {
                "square": "Square",
                "rectangular": "Rectangular",
                "hexagonal": "Hexagonal",
            }.get(structure.symmetry, "Custom")
            self._updating_fft_ideal = True
            try:
                self._fft_ideal_combo.setCurrentText(target)
                self._fft_ideal_a_spin.setValue(max(self._fft_ideal_a_spin.minimum(), structure.a_nm))
                self._fft_ideal_b_spin.setValue(max(self._fft_ideal_b_spin.minimum(), structure.b_nm))
                self._fft_ideal_angle_spin.setValue(
                    min(179.0, max(1.0, structure.angle_deg))
                )
            finally:
                self._updating_fft_ideal = False
        if refresh:
            self._on_bragg_changed()
            self._refresh_fft_correction_ui()

    def _structure_from_fft_controls(self, name: str) -> KnownStructure:
        active = getattr(self, "_active_known_structure", None)
        symmetry = active.symmetry if isinstance(active, KnownStructure) else "hexagonal"
        if symmetry in {"square", "hexagonal"}:
            symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hexagonal"
        a_nm = self._reference_lattice_a_nm()
        if symmetry == "square":
            return KnownStructure(name, "square", a_nm, a_nm, 90.0, self._bragg_unit_combo.currentText())
        if symmetry == "hexagonal":
            return KnownStructure(name, "hexagonal", a_nm, a_nm, 60.0, self._bragg_unit_combo.currentText())
        b_nm = getattr(self, "_fft_ideal_b_spin", None)
        angle = getattr(self, "_fft_ideal_angle_spin", None)
        return KnownStructure(
            name,
            symmetry,
            a_nm,
            float(b_nm.value()) if b_nm is not None else a_nm,
            float(angle.value()) if angle is not None else 90.0,
            self._bragg_unit_combo.currentText(),
        )

    def _persist_known_structures(self, selected_name: str) -> None:
        save_known_structures(self._known_structures)
        self._refresh_structure_combo(selected_name)

    def _on_save_structure(self) -> None:
        default = getattr(self, "_active_known_structure", self._known_structures[0]).name
        name, ok = QInputDialog.getText(self, "Save known structure", "Structure name:", text=default)
        if not ok or not name.strip():
            return
        structure = self._structure_from_fft_controls(name.strip())
        self._known_structures = upsert_structure(self._known_structures, structure)
        self._persist_known_structures(structure.name)
        self._apply_known_structure_to_fft(structure)

    def _on_update_structure(self) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return
        name = combo.currentText().strip()
        if not name:
            return
        structure = self._structure_from_fft_controls(name)
        self._known_structures = upsert_structure(self._known_structures, structure)
        self._persist_known_structures(structure.name)
        self._apply_known_structure_to_fft(structure)

    def _on_delete_structure(self) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return
        name = combo.currentText().strip()
        self._known_structures = delete_structure(self._known_structures, name)
        selected = self._known_structures[0]
        self._persist_known_structures(selected.name)
        self._apply_known_structure_to_fft(selected)

    def _on_bragg_symmetry_changed(self, _idx: int) -> None:
        if self._updating_structure:
            return
        self._on_bragg_changed()
        self._sync_fft_ideal_from_symmetry()
        self._refresh_fft_correction_ui()

    def _on_fft_grid_changed(self, _grid=None) -> None:
        self._clear_fft_preview_if_active()
        self._refresh_grid_measure_lbl()
        self._refresh_fft_correction_ui()

    def _refresh_grid_measure_lbl(self) -> None:
        lbl = getattr(self, "_grid_measure_lbl", None)
        if lbl is None:
            return
        panel = getattr(self, "_fft_lattice_panel", None)
        if panel is None:
            lbl.setText("No grid — click Draw Grid to start")
            return
        try:
            from probeflow.analysis.lattice_grid import format_reciprocal_measurements
            grid = panel._overlay.grid()
            if grid is None:
                return
            d = format_reciprocal_measurements(grid, panel._cal)
            lbl.setText(f"{d['g1']}    {d['g2']}    ∠ {d['angle']}")
        except Exception:
            pass

    def _on_grid_extent_changed(self, value: int) -> None:
        overlay = getattr(self, "_fft_lattice_overlay", None)
        if overlay is not None:
            overlay.set_cells(value)
        panel = getattr(self, "_fft_lattice_panel", None)
        spin = getattr(panel, "_cells_spin", None)
        if spin is not None and spin.value() != value:
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(value)
            finally:
                spin.blockSignals(blocked)

    def _fft_pixel_sizes_m(self) -> tuple[float, float] | None:
        Ny, Nx = self._arr.shape[:2]
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
        except Exception:
            return None
        if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
            return None
        return (w_m / Nx, h_m / Ny)

    def _fft_measured_direct_vectors_nm(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        overlay = getattr(self, "_fft_lattice_overlay", None)
        if overlay is None:
            return None
        grid = overlay.grid()
        panel = getattr(self, "_fft_lattice_panel", None)
        cal = getattr(panel, "_cal", None)
        if grid is None or cal is None:
            return None
        return direct_lattice_vectors_from_reciprocal_grid(grid, cal)

    @staticmethod
    def _vec_len_nm(vec: tuple[float, float]) -> float:
        return math.hypot(float(vec[0]), float(vec[1]))

    @staticmethod
    def _vec_angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la <= 0 or lb <= 0:
            return float("nan")
        cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        return math.degrees(math.acos(cosang))

    def _sync_fft_ideal_from_symmetry(self) -> None:
        combo = getattr(self, "_fft_ideal_combo", None)
        if combo is None or combo.currentText() == "Custom":
            return
        symmetry = "Square" if self._bragg_sym_combo.currentIndex() == 0 else "Hexagonal"
        if combo.currentText() != symmetry:
            self._updating_fft_ideal = True
            try:
                combo.setCurrentText(symmetry)
            finally:
                self._updating_fft_ideal = False

    def _reference_lattice_a_nm(self) -> float:
        value = float(self._bragg_a_spin.value())
        return value * 0.1 if self._bragg_unit_combo.currentText() == "Å" else value

    def _sync_fft_ideal_values(self, a_nm: float, b_nm: float, angle: float) -> None:
        combo = self._fft_ideal_combo.currentText()
        structure = getattr(self, "_active_known_structure", None)
        structure_combo = {
            "square": "Square",
            "rectangular": "Rectangular",
            "hexagonal": "Hexagonal",
            "custom": "Custom",
        }.get(structure.symmetry if isinstance(structure, KnownStructure) else "", "")
        if isinstance(structure, KnownStructure) and combo == structure_combo:
            ideal = ideal_lattice_from_structure(structure, measured_angle_deg=angle)
            values = (ideal.a_nm, ideal.b_nm, ideal.angle_deg)
        elif combo == "Custom":
            return
        elif combo == "Square":
            ref_a_nm = self._reference_lattice_a_nm()
            side = ref_a_nm
            values = (side, side, 90.0)
        elif combo == "Rectangular":
            ref_a_nm = self._reference_lattice_a_nm()
            values = (ref_a_nm, b_nm, 90.0)
        elif combo == "Hexagonal":
            ref_a_nm = self._reference_lattice_a_nm()
            side = ref_a_nm
            values = (side, side, 120.0 if angle >= 90.0 else 60.0)
        else:
            values = (a_nm, b_nm, angle)
        self._updating_fft_ideal = True
        try:
            self._fft_ideal_a_spin.setValue(max(self._fft_ideal_a_spin.minimum(), values[0]))
            self._fft_ideal_b_spin.setValue(max(self._fft_ideal_b_spin.minimum(), values[1]))
            self._fft_ideal_angle_spin.setValue(min(179.0, max(1.0, values[2])))
            self._fft_ideal_a_spin.setEnabled(combo not in {"Match measured"})
            self._fft_ideal_b_spin.setEnabled(combo in {"Rectangular", "Custom"})
            self._fft_ideal_angle_spin.setEnabled(combo == "Custom")
        finally:
            self._updating_fft_ideal = False

    def _on_fft_ideal_changed(self, *_args) -> None:
        if self._updating_fft_ideal:
            return
        self._clear_fft_preview_if_active()
        self._refresh_fft_correction_ui()

    def _refresh_fft_correction_ui(self) -> None:
        measured_lbl = getattr(self, "_fft_measured_lbl", None)
        corr_lbl = getattr(self, "_fft_correction_lbl", None)
        status_lbl = getattr(self, "_fft_correction_status_lbl", None)
        if measured_lbl is None or corr_lbl is None:
            return
        self._fft_correction = None
        for btn_name in ("_fft_preview_btn", "_fft_apply_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(False)

        vectors = self._fft_measured_direct_vectors_nm()
        if vectors is None:
            measured_lbl.setText(
                "Create a reciprocal grid, then drag g1/g2 handles until the grid "
                "tracks the visible Bragg peaks."
            )
            corr_lbl.setText(
                "Step 2: Click 'Create/Edit reciprocal grid' below and drag the "
                "g₁/g₂ handles onto two Bragg peaks in the FFT."
            )
            if status_lbl is not None:
                status_lbl.setText("No reciprocal grid yet")
            return

        a_vec, b_vec = vectors
        a_len = self._vec_len_nm(a_vec)
        b_len = self._vec_len_nm(b_vec)
        angle = self._vec_angle_deg(a_vec, b_vec)
        self._sync_fft_ideal_values(a_len, b_len, angle)

        ideal = IdealLattice(
            a_nm=self._fft_ideal_a_spin.value(),
            b_nm=self._fft_ideal_b_spin.value(),
            angle_deg=self._fft_ideal_angle_spin.value(),
        )
        da_pct = 100.0 * (a_len / ideal.a_nm - 1.0) if ideal.a_nm > 0 else float("nan")
        db_pct = 100.0 * (b_len / ideal.b_nm - 1.0) if ideal.b_nm > 0 else float("nan")
        measured_lbl.setText(
            f"Measured |a|={a_len:.4g} nm |b|={b_len:.4g} nm angle={angle:.2f}°\n"
            f"Target |a|={ideal.a_nm:.4g} nm |b|={ideal.b_nm:.4g} nm "
            f"angle={ideal.angle_deg:.2f}°   "
            f"da {da_pct:+.2f}% db {db_pct:+.2f}% dAngle {angle - ideal.angle_deg:+.2f}°"
        )
        result = compute_correction(MeasuredLattice(a_nm=a_vec, b_nm=b_vec), ideal)
        if isinstance(result, str):
            corr_lbl.setText(f"Cannot compute correction:\n{result}")
            if status_lbl is not None:
                status_lbl.setText("Correction unavailable")
            return

        self._fft_correction = result
        correction_lines = correction_main_lines(
            result,
            preserve_orientation=self._fft_preserve_orientation_cb.isChecked(),
        )
        corr_lbl.setText(
            "\n".join(correction_lines)
            + "\n→ Click 'Preview corrected image' to verify."
        )
        if status_lbl is not None:
            status_lbl.setText("Correction ready")
        if self._get_image_fn is not None:
            self._fft_preview_btn.setEnabled(True)
        if self._apply_correction_fn is not None:
            self._fft_apply_btn.setEnabled(True)

    def _fft_correction_options(self) -> dict:
        interp_map = {"Bilinear": "bilinear", "Nearest": "nearest", "Bicubic": "bicubic"}
        fill_map = {"NaN": "nan", "Background": "background", "Zero": "zero"}
        return {
            "interpolation": interp_map.get(self._fft_interp_combo.currentText(), "bilinear"),
            "fill_mode": fill_map.get(self._fft_fill_combo.currentText(), "nan"),
            "expand_canvas": self._fft_expand_cb.isChecked(),
            "preserve_orientation": self._fft_preserve_orientation_cb.isChecked(),
        }

    def _fft_correction_matrix_px(self) -> np.ndarray | None:
        if self._fft_correction is None:
            return None
        px = self._fft_pixel_sizes_m()
        if px is None:
            return None
        opts = self._fft_correction_options()
        return lattice_correction_matrix_px(
            self._fft_correction,
            pixel_size_x_m=px[0],
            pixel_size_y_m=px[1],
            preserve_orientation=opts["preserve_orientation"],
        )

    def _clear_fft_preview_if_active(self) -> None:
        if self._fft_preview_active:
            self._hide_fft_preview()

    def _on_fft_preview_correction(self) -> None:
        if self._fft_correction is None or self._get_image_fn is None:
            return
        T_px = self._fft_correction_matrix_px()
        if T_px is None:
            return
        arr = self._get_image_fn()
        if arr is None:
            return
        from probeflow.processing.image import affine_lattice_correction
        opts = self._fft_correction_options()
        try:
            corrected = affine_lattice_correction(
                arr,
                T_px,
                expand_canvas=opts["expand_canvas"],
                interpolation=opts["interpolation"],
                fill_mode=opts["fill_mode"],
            )
        except Exception as exc:
            self._fft_correction_lbl.setText(f"Preview failed: {exc}")
            return
        self._show_fft_preview(corrected)
        self._fft_preview_active = True
        self._fft_clear_preview_btn.setEnabled(True)
        self._fft_correction_status_lbl.setText("Preview shown")

    def _on_fft_clear_preview(self) -> None:
        self._hide_fft_preview()
        self._refresh_fft_correction_ui()

    def _on_fft_apply_correction(self) -> None:
        if self._apply_correction_fn is None or self._fft_correction is None:
            return
        px = self._fft_pixel_sizes_m()
        if px is None or self._fft_correction_matrix_px() is None:
            return
        if self._fft_preview_active:
            self._hide_fft_preview()
        opts = self._fft_correction_options()
        op_params = lattice_correction_operation_params(
            self._fft_correction,
            pixel_size_x_m=px[0],
            pixel_size_y_m=px[1],
            expand_canvas=opts["expand_canvas"],
            interpolation=opts["interpolation"],
            fill_mode=opts["fill_mode"],
            preserve_orientation=opts["preserve_orientation"],
        )
        if op_params is None:
            return
        op_params["source"] = "fft_reciprocal_grid"
        # Record which region the FFT (and hence the measured lattice) came from.
        # The correction itself is valid for the full image — lattice vectors are
        # physical (nm) — but recording the source removes ambiguity when the
        # provenance is revisited.
        op_params["fft_source"] = self._fft_source
        if self._fft_source == "active_roi" and self._roi_id is not None:
            op_params["fft_roi_id"] = self._roi_id
        structure = getattr(self, "_active_known_structure", None)
        if isinstance(structure, KnownStructure):
            op_params["known_structure"] = structure.as_dict()
        self._apply_correction_fn("affine_lattice_correction", op_params)
        # Recompute FFT from the now-corrected image so the display reflects the
        # undistorted real-space data.  _apply_correction_fn calls
        # _refresh_processing_display() synchronously, so _get_image_fn() already
        # returns the corrected array by the time we reach here.
        if self._get_image_fn is not None:
            updated = self._get_image_fn()
            if updated is not None and np.asarray(updated).ndim == 2:
                # Preserve original pixel sizes across the canvas-size change.
                # Canvas expansion increases row/col count without changing the
                # physical size of each pixel.  If we kept _scan_range_m the same
                # while Nx/Ny grew, _recompute_fft would divide the same physical
                # range over more pixels, computing a smaller dx/dy and a
                # spuriously higher Nyquist frequency in the expanded direction.
                orig_ny, orig_nx = self._full_arr.shape
                px_x_nm = self._full_scan_range_m[0] * 1e9 / orig_nx if orig_nx > 0 else 1.0
                px_y_nm = self._full_scan_range_m[1] * 1e9 / orig_ny if orig_ny > 0 else 1.0
                self._full_arr = np.asarray(updated, dtype=np.float64)
                new_ny, new_nx = self._full_arr.shape
                self._full_scan_range_m = (px_x_nm * new_nx * 1e-9,
                                           px_y_nm * new_ny * 1e-9)
                # The corrected (possibly canvas-expanded) image invalidates the
                # ROI bbox fitted on the pre-correction image; revert to whole image.
                self._roi_bounds_px = None
                self._fft_source = "whole_image"
                if hasattr(self, "_fft_source_combo"):
                    self._fft_source_combo.blockSignals(True)
                    self._fft_source_combo.setCurrentIndex(0)
                    self._fft_source_combo.model().item(1).setEnabled(False)
                    self._fft_source_combo.blockSignals(False)
                self._arr, self._scan_range_m = self._resolve_source_array()
                self._recompute_fft()
                self._redraw()
                self._update_info_panel()
        # Clear the stale grid overlay — it was fitted on the pre-correction FFT
        # and no longer aligns with the corrected diffraction pattern.
        self._on_clear_fft_lattice()
        self._fft_correction_lbl.setText(
            "Correction applied. FFT recomputed from corrected image.\n"
            "Bragg peaks should now lie on the inner shell ring."
        )
        self._fft_correction_status_lbl.setText("Correction applied")
        self._fft_preview_btn.setEnabled(False)
        self._fft_clear_preview_btn.setEnabled(False)
        self._fft_apply_btn.setEnabled(False)

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FFT view", "fft_view.png",
            "PNG image (*.png);;All files (*)"
        )
        if path:
            self._fig_fft.savefig(path, dpi=150, bbox_inches="tight")

    def _clear_grid_tab(self) -> None:
        """Reset the embedded Grid tab panel area to its placeholder state."""
        lay = getattr(self, "_grid_tab_lay", None)
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        lay.addWidget(self._grid_placeholder_lbl)
        draw_btn = getattr(self, "_grid_draw_btn", None)
        if draw_btn is not None:
            lay.addWidget(draw_btn)
        lbl = getattr(self, "_grid_measure_lbl", None)
        if lbl is not None:
            lbl.setText("No grid — click Draw Grid to start")

    def _set_grid_tab_panel(self, panel: QWidget) -> None:
        """Install the reciprocal-grid controls into the Grid tab panel area."""
        lay = getattr(self, "_grid_tab_lay", None)
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        panel.setParent(self)
        lay.addWidget(panel)

    def _on_clear_fft_lattice(self):
        """Remove the FFT lattice overlay and reset its embedded controls."""
        self._clear_fft_preview_if_active()
        overlay = getattr(self, "_fft_lattice_overlay", None)
        if overlay is not None:
            overlay.clear()
            self._fft_lattice_overlay = None
        panel = getattr(self, "_fft_lattice_panel", None)
        if panel is not None:
            panel.setParent(None)
            panel.deleteLater()
            self._fft_lattice_panel = None
        self._fft_lattice_dock = None
        self._clear_grid_tab()
        if getattr(self, "_clear_grid_btn", None) is not None:
            self._clear_grid_btn.setEnabled(False)
        self._refresh_fft_correction_ui()

    def _on_edit_reciprocal_grid(self) -> None:
        self._on_open_fft_lattice(select_advanced=False)

    def _on_open_fft_lattice(self, select_advanced: bool = True):
        from probeflow.gui.lattice_grid import open_fft_tool

        existing = getattr(self, "_fft_lattice_overlay", None)
        existing_panel = getattr(self, "_fft_lattice_panel", None)
        if existing is not None and existing_panel is not None:
            if select_advanced:
                self._tab_widget.setCurrentIndex(self._grid_tab_index)
            return
        if self._qx is None or self._qy is None:
            return

        Ny, Nx = self._arr.shape[:2]
        overlay, panel = open_fft_tool(
            self._ax_fft, self._canvas_fft,
            self._qx, self._qy,
            (Ny, Nx), parent=self,
            on_change=self._on_fft_grid_changed,
        )
        self._fft_lattice_overlay = overlay
        self._fft_lattice_panel = panel
        self._fft_lattice_dock = None
        if hasattr(overlay, "set_drag_state_callback"):
            # Use WeakMethod so the overlay does not prevent this dialog from being
            # garbage-collected after it is closed. The overlay's callback wrapper
            # must handle the case where the weak reference has expired (returns None).
            _cb_ref = weakref.WeakMethod(self._on_fft_lattice_drag_state_changed)

            def _drag_state_cb(dragging: bool, _ref: weakref.ref = _cb_ref) -> None:
                cb = _ref()
                if cb is not None:
                    cb(dragging)

        overlay.set_drag_state_callback(_drag_state_cb)

        self._set_grid_tab_panel(panel)
        self._on_grid_extent_changed(self._grid_extent_spin.value())
        if select_advanced:
            self._tab_widget.setCurrentIndex(self._grid_tab_index)
        if getattr(self, "_clear_grid_btn", None) is not None:
            self._clear_grid_btn.setEnabled(True)
        self._refresh_grid_measure_lbl()
        self._refresh_fft_correction_ui()
