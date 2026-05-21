"""ROI and spectroscopy-marker orchestration for ImageViewerDialog."""

from __future__ import annotations

from probeflow.core import AREA_ROI_KINDS
from probeflow.gui.models import SxmFile
from probeflow.gui.roi_context import active_line_roi_context
from probeflow.gui.viewer import (
    activate_roi,
    active_roi,
    delete_active_roi,
    delete_all_rois,
    delete_roi,
    has_roi_aware_local_filter,
    invert_active_roi,
    invert_roi,
    load_roi_set,
    plot_roi_line_profile,
    rename_roi,
    roi_canvas_created,
    roi_canvas_moved,
    roi_line_endpoint_changed,
    roi_line_set_width,
    save_roi_set,
    select_nth_roi,
    selected_or_active_roi_id,
    show_roi_fft,
    show_roi_histogram,
)


class ImageViewerRoiMixin:
    # ── Spec position overlay ─────────────────────────────────────────────────
    def _load_spec_markers(self, entry):
        self._spec_overlay.load(
            entry,
            self._scan_range_m,
            self._scan_shape,
            self._scan_format,
            self._scan_header,
            show=self._spec_show_cb.isChecked(),
        )

    def _on_spec_show_toggled(self, checked: bool):
        self._spec_overlay.apply_visibility(checked)

    # ── Image-level ROI set ───────────────────────────────────────────────────

    def _load_image_roi_set(self, entry: "SxmFile") -> None:
        """Load ROIs from <stem>.rois.json sidecar if it exists, else create empty set."""
        self._image_roi_set, _err = load_roi_set(entry.path)
        self._zoom_lbl.set_roi_set(self._image_roi_set)
        if hasattr(self, "_roi_dock"):
            self._roi_dock.refresh(self._image_roi_set)
        self._sync_viewer_menu_actions()

    def _save_image_roi_set(self) -> None:
        """Persist the current ROISet to its sidecar file."""
        if self._image_roi_set is None:
            return
        entry = self._entries[self._idx]
        err = save_roi_set(self._image_roi_set, entry.path)
        if err and hasattr(self, "_status_lbl"):
            self._status_lbl.setText(err)

    def _on_image_roi_set_changed(self) -> None:
        self._zoom_lbl.set_roi_set(self._image_roi_set)
        self._save_image_roi_set()
        if hasattr(self, "_roi_dock"):
            self._roi_dock.refresh(self._image_roi_set)
        self._sync_line_profile_visibility()
        self._sync_viewer_menu_actions()

    def _on_pixel_hovered(self, col: int, row: int, val) -> None:
        if not hasattr(self, "_coord_lbl"):
            return
        if val is None:
            self._coord_lbl.setText(f"({col}, {row})")
        else:
            scale, unit, _ = self._channel_unit()
            val_disp = float(val) * scale
            unit_str = f" {unit}" if unit else ""
            self._coord_lbl.setText(f"({col}, {row}): {val_disp:.4g}{unit_str}")

    # ── Canvas drawing-tool callbacks ─────────────────────────────────────────

    def _on_canvas_roi_created(self, roi) -> None:
        roi_canvas_created(
            self._image_roi_set, roi,
            self._on_image_roi_set_changed, self._set_drawing_tool,
        )

    def _on_canvas_roi_move(self, roi_id: str, dx: int, dy: int) -> None:
        roi_canvas_moved(
            self._image_roi_set, roi_id, dx, dy, self._on_image_roi_set_changed,
        )

    def _on_canvas_tool_changed(self, kind: str) -> None:
        """Canvas emitted tool_changed (e.g. after Escape or drawing completion)."""
        if hasattr(self, "_quick_toolbar"):
            self._quick_toolbar.set_active_mode(kind)
        self._sync_line_profile_visibility(kind)
        from probeflow.gui.tool_manager import _TOOL_HINTS
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(_TOOL_HINTS.get(kind, ""))
        self._sync_viewer_menu_actions()

    def _on_roi_canvas_context_menu(self, roi_id: str, global_pos) -> None:
        """Right-click on a ROI in the canvas — show a small ROI action menu."""
        from PySide6.QtWidgets import QMenu
        roi_set = self._image_roi_set
        roi = roi_set.get(roi_id) if roi_set else None
        if roi is None:
            return
        is_area = roi.kind in AREA_ROI_KINDS
        is_line = roi.kind == "line"
        menu = QMenu(self)
        act_active = menu.addAction("Set Active")
        act_active.triggered.connect(lambda: self._set_active_image_roi(roi_id))
        act_rename = menu.addAction("Rename…")
        act_rename.triggered.connect(lambda: self._rename_image_roi(roi_id))
        act_delete = menu.addAction("Delete")
        act_delete.triggered.connect(lambda: self._delete_image_roi(roi_id))
        act_invert = menu.addAction("Invert")
        act_invert.setEnabled(is_area)
        act_invert.triggered.connect(lambda: self._invert_image_roi(roi_id))
        menu.addSeparator()
        act_stm_bg = menu.addAction("STM Background fit from ROI...")
        act_stm_bg.setEnabled(is_area)
        act_stm_bg.triggered.connect(lambda: self._open_stm_background_for_roi(roi_id))
        act_fft = menu.addAction("FFT this region")
        act_fft.setEnabled(is_area)
        act_fft.triggered.connect(lambda: self._on_roi_fft(roi_id))
        act_hist = menu.addAction("Histogram of this region")
        act_hist.setEnabled(is_area)
        act_hist.triggered.connect(lambda: self._on_roi_histogram(roi_id))
        act_stats = menu.addAction("Add ROI statistics to measurements")
        act_stats.setEnabled(is_area)
        act_stats.triggered.connect(
            lambda: self._image_measurements.add_roi_stats_measurement(roi_id)
        )
        act_maxima = menu.addAction("Detect maxima in this region")
        act_maxima.setEnabled(is_area)
        act_maxima.triggered.connect(
            lambda: self._image_measurements.detect_feature_maxima_for_roi(roi_id)
        )
        act_profile = menu.addAction("Line profile")
        act_profile.setEnabled(is_line)
        act_profile.triggered.connect(lambda: self._on_roi_line_profile(roi_id))
        act_profile_measure = menu.addAction("Add line profile measurement")
        act_profile_measure.setEnabled(is_line)
        act_profile_measure.triggered.connect(
            lambda: self._image_measurements.add_line_profile_measurement_for_roi(roi_id)
        )
        menu.exec(global_pos)

    # ── ROI helper actions ────────────────────────────────────────────────────

    def _set_active_image_roi(self, roi_id: str) -> None:
        activate_roi(self._image_roi_set, roi_id, self._on_image_roi_set_changed)

    def _rename_image_roi(self, roi_id: str) -> None:
        rename_roi(self._image_roi_set, roi_id, self._on_image_roi_set_changed, parent=self)

    def _delete_image_roi(self, roi_id: str) -> None:
        delete_roi(self._image_roi_set, roi_id, self._on_image_roi_set_changed)

    def _delete_active_image_roi(self) -> None:
        delete_active_roi(self._image_roi_set, self._on_image_roi_set_changed)

    def _clear_all_image_marks(self) -> None:
        delete_all_rois(self._image_roi_set, self._on_image_roi_set_changed)
        if self._angle_overlay is not None:
            scene = self._zoom_lbl.scene()
            self._angle_overlay.remove_from_scene(scene)
            self._angle_overlay = None

    def _to_dock_result(self, legacy_r, measurement_id: str):
        """Convert a legacy MeasurementResult to the newer dock format."""
        from probeflow.measurements.adapters import legacy_measurement_to_result

        return legacy_measurement_to_result(legacy_r, measurement_id)

    def _add_dialog_measurement_result(self, result) -> None:
        from dataclasses import replace

        from probeflow.measurements.models import MeasurementResult

        mid = self._measurement_table.next_measurement_id()
        if isinstance(result, MeasurementResult):
            dock_result = replace(result, measurement_id=mid)
        else:
            dock_result = self._to_dock_result(result, mid)
        self._measurement_table.add_result(dock_result)
        self._measurement_dock.show()
        self._measurement_dock.raise_()

    def _invert_image_roi(self, roi_id: str) -> None:
        invert_roi(
            self._image_roi_set, roi_id,
            self._current_array_shape(), self._on_image_roi_set_changed,
        )

    def _invert_active_image_roi(self) -> None:
        had_area = (
            self._active_image_roi() is not None
            and self._active_image_roi().kind in AREA_ROI_KINDS
        )
        invert_active_roi(
            self._image_roi_set, self._current_array_shape(), self._on_image_roi_set_changed,
        )
        if had_area and hasattr(self, "_scope_cb"):
            self._scope_cb.setCurrentText("ROI filters only")
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(
                    "ROI inverted. Filters will apply inside the inverted area."
                )

    def _select_nth_image_roi(self, n: int) -> None:
        select_nth_roi(self._image_roi_set, n, self._on_image_roi_set_changed)

    # ── ROI operation callbacks ───────────────────────────────────────────────

    def _on_roi_fft(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or self._display_arr is None:
            return
        show_roi_fft(roi, self._display_arr, parent=self)

    def _on_roi_histogram(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or self._display_arr is None:
            return
        show_roi_histogram(roi, self._display_arr, self._channel_unit, parent=self)

    def _on_roi_line_profile(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None or roi.kind != "line" or self._display_arr is None:
            return
        self._line_profile_panel.set_width(int(roi.geometry.get("width", 1)))
        plot_roi_line_profile(
            roi, self._display_arr,
            self._pixel_size_xy_m(),
            self._channel_unit,
            self._line_profile_panel,
            self._t,
        )

    def _on_line_roi_preview(
        self, roi_id: str, x1: float, y1: float, x2: float, y2: float,
    ) -> None:
        """Live endpoint drag: update profile without touching the data model."""
        if self._display_arr is None:
            return
        from probeflow.core.roi import ROI as _ROI
        tmp_roi = _ROI(
            id=roi_id, name="", kind="line",
            geometry={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )
        plot_roi_line_profile(
            tmp_roi, self._display_arr,
            self._pixel_size_xy_m(),
            self._channel_unit,
            self._line_profile_panel,
            self._t,
        )

    def _on_line_roi_geometry_changed(
        self, roi_id: str, x1: float, y1: float, x2: float, y2: float,
    ) -> None:
        """Endpoint drag released: commit new geometry and persist."""
        roi_line_endpoint_changed(
            self._image_roi_set, roi_id, x1, y1, x2, y2,
            self._on_image_roi_set_changed,
        )

    def _on_line_profile_width_changed(self, width: int) -> None:
        """Width spinbox changed: update active line ROI geometry and re-plot."""
        roi_id = self._active_line_roi_id()
        if roi_id is None:
            return
        roi_line_set_width(
            self._image_roi_set, roi_id, width,
            self._on_image_roi_set_changed,
        )

    def _on_canvas_roi_activate(self, roi_id: str) -> None:
        if self._image_roi_set is None:
            return
        self._image_roi_set.set_active(roi_id)
        self._on_image_roi_set_changed()

    def _on_canvas_roi_delete(self, roi_id: str) -> None:
        if self._image_roi_set is None:
            return
        self._image_roi_set.remove(roi_id)
        self._on_image_roi_set_changed()

    def _on_canvas_roi_copy(self, roi_id: str) -> None:
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is not None:
            self._copy_roi_buffer = roi

    def _on_canvas_roi_paste(self) -> None:
        roi = self._copy_roi_buffer
        if roi is None or self._image_roi_set is None:
            return
        from probeflow.core.roi import ROI as _ROI, translate as _translate
        # Create a new ROI (new id) offset by 10 pixels so it doesn't overlap
        offset_roi = _translate(roi, 10.0, 10.0)
        pasted = _ROI.new(
            offset_roi.kind, offset_roi.geometry,
            name=f"{roi.name}_copy",
        )
        self._image_roi_set.add(pasted)
        self._image_roi_set.set_active(pasted.id)
        self._on_image_roi_set_changed()

    def _on_map_spectra_here(self):
        """Open the per-image spec→this-image mapping dialog."""
        entry = self._entries[self._idx]
        accepted, n = self._spec_overlay.open_map_dialog(entry, self)
        if not accepted and n == 0 and not self._spec_image_map:
            self._status_lbl.setText(
                "No spectroscopy files found alongside this image.")
            return
        if accepted:
            self._status_lbl.setText(
                f"{n} spec(s) mapped to this image. Reloading markers…")
            self._load_spec_markers(entry)

    def _on_marker_clicked(self, entry):
        self._spec_overlay.open_spec_viewer(entry, self._t, self)

    def _current_array_shape(self) -> tuple[int, int] | None:
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        return None if arr is None else arr.shape

    def _active_image_roi(self):
        return active_roi(self._image_roi_set)

    def _processing_has_roi_aware_local_filter(self, state: dict) -> bool:
        return has_roi_aware_local_filter(state)

    def _selected_or_active_image_roi_id(self) -> "str | None":
        return selected_or_active_roi_id(
            getattr(self, "_image_roi_set", None), getattr(self, "_roi_dock", None),
        )

    def _rename_active_image_roi(self) -> None:
        roi_id = self._selected_or_active_image_roi_id()
        if roi_id:
            self._rename_image_roi(roi_id)

    def _set_selected_or_active_image_roi(self) -> None:
        roi_id = self._selected_or_active_image_roi_id()
        if roi_id:
            self._set_active_image_roi(roi_id)
