"""Analysis and tool-launch handlers for ImageViewerDialog."""

from __future__ import annotations

import logging

import numpy as np

from PySide6.QtWidgets import QDialog

_log = logging.getLogger(__name__)

from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
from probeflow.gui.dialogs.periodic_filter import PeriodicFilterDialog
from probeflow.gui.roi_context import (
    active_area_roi_bounds,
    active_area_roi_context,
    area_roi_mask,
    collect_point_source_records,
    point_source_arrays_m,
    point_source_arrays_px,
    point_source_metadata,
    selected_or_active_area_roi_context,
)
from probeflow.gui.viewer.tool_launch import (
    feature_lattice_launch_context,
    lattice_grid_launch_context,
    pair_correlation_launch_context,
)


class ImageViewerToolsMixin:
    def _current_viewer_array(self):
        arr = getattr(self, "_display_arr", None)
        if arr is None:
            arr = getattr(self, "_raw_arr", None)
        return arr

    def _clear_lattice_grid_overlay(self, *, close_panel: bool = True) -> bool:
        """Remove the real-space lattice grid overlay and optionally close its
        sidebar tool panel."""
        removed = False
        panel = getattr(self, "_lattice_grid_panel", None)
        if panel is not None:
            try:
                panel.cleanup()
            except RuntimeError:
                pass
            self._lattice_grid_panel = None

        item = getattr(self, "_lattice_grid_item", None)
        if item is not None:
            removed = True
            try:
                scene = item.scene()
            except RuntimeError:
                scene = None
            if scene is not None:
                try:
                    scene.removeItem(item)
                except RuntimeError:
                    pass
            self._lattice_grid_item = None

        if close_panel and hasattr(self, "_close_sidebar_tool"):
            self._close_sidebar_tool()

        if hasattr(self, "_sync_viewer_menu_actions"):
            self._sync_viewer_menu_actions()
        return removed

    def _on_clear_lattice_grid(self) -> None:
        if self._clear_lattice_grid_overlay(close_panel=True):
            self._status_lbl.setText("Cleared lattice grid overlay.")
        else:
            self._status_lbl.setText("No lattice grid overlay to clear.")

    def _on_open_lattice_grid(self):
        arr = getattr(self, "_display_arr", None)
        if arr is None:
            arr = getattr(self, "_raw_arr", None)
        context = lattice_grid_launch_context(arr, scan_range_m=self._scan_range_m)
        if not context.ready:
            self._status_lbl.setText(str(context.status_message))
            return
        from probeflow.gui.lattice_grid import open_real_space_tool

        self._clear_lattice_grid_overlay(close_panel=True)

        def _get_image():
            return self._display_arr if self._display_arr is not None else self._raw_arr

        def _preview_lattice_correction(corrected_arr) -> None:
            self._display_arr = corrected_arr
            self._update_export_summary()
            self._refresh_viewer_pixmap(reset_zoom=False)

        def _clear_lattice_correction_preview() -> None:
            self._refresh_display_array(reset_zoom_if_shape_changed=False)
            self._refresh_viewer_pixmap(reset_zoom=False)

        def _apply_lattice_correction(op_name: str, op_params: dict) -> None:
            ops = list(self._processing.get("geometric_ops") or [])
            ops.append({"op": op_name, "params": op_params})
            self._processing["geometric_ops"] = ops
            self._refresh_processing_display()

        item, panel = open_real_space_tool(
            self._zoom_lbl, context.scan_range_m, context.image_shape, parent=self,
            get_image_fn=_get_image,
            apply_correction_fn=_apply_lattice_correction,
            preview_image_fn=_preview_lattice_correction,
            clear_preview_fn=_clear_lattice_correction_preview,
        )
        self._lattice_grid_item = item
        self._lattice_grid_panel = panel
        # Host the controls in the sidebar (single right column) rather than a
        # separate dock that competes with the image for width.
        self._show_sidebar_tool(
            "Lattice grid", panel,
            on_close=lambda: self._clear_lattice_grid_overlay(close_panel=False),
        )
        self._sync_viewer_menu_actions()

    def _on_open_feature_finder(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        px_x_m, px_y_m = self._pixel_size_xy_m()
        roi_mask = None
        roi_ctx = selected_or_active_area_roi_context(
            self._image_roi_set,
            getattr(self, "_roi_panel", None),
        )
        if roi_ctx.roi is not None:
            roi_mask = area_roi_mask(roi_ctx.roi, arr.shape[:2])
        value_scale, value_unit, _ = self._channel_unit()
        from probeflow.gui.dialogs.feature_finder import FeatureFinderDialog
        dlg = FeatureFinderDialog(
            arr,
            pixel_size_x_m=px_x_m,
            pixel_size_y_m=px_y_m,
            roi_mask=roi_mask,
            theme=self._t,
            value_scale=value_scale,
            value_unit=value_unit,
            on_send_to_particle_statistics=None,
            parent=self,
        )
        self._feature_finder_dlg = dlg
        self._present_modal_tool(dlg)

    def _on_open_image_operations(self) -> None:
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return

        from probeflow.gui.dialogs.image_arithmetic import ImageArithmeticDialog

        scale, unit_label, _axis_label = self._channel_unit()
        active_area_roi_id = active_area_roi_context(self._image_roi_set).roi_id
        dlg = ImageArithmeticDialog(
            self._entries,
            current_entry_index=self._idx,
            current_plane_idx=self._ch_cb.currentIndex(),
            current_shape=arr.shape,
            current_scan_range_m=self._scan_range_m,
            display_scale=scale,
            display_unit=unit_label,
            has_active_area_roi=active_area_roi_id is not None,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        spec = dlg.operation_spec()
        if not spec:
            return

        scope = dlg.scope()
        op_spec = {
            "op": "arithmetic",
            "params": dict(spec.get("params", {})),
        }
        if scope == ImageArithmeticDialog.ACTIVE_AREA_ROI:
            active_roi = self._active_image_roi()
            if active_roi is not None and active_area_roi_id is None:
                self._status_lbl.setText(
                    f"Active {active_roi.kind} ROI is not valid for image arithmetic; "
                    "select an area ROI or use Whole image."
                )
                return
            if active_area_roi_id is None:
                self._status_lbl.setText(
                    "Select an active area ROI before using ROI-scoped image arithmetic."
                )
                return
            op_spec["roi_id"] = active_area_roi_id
            area_roi = (
                self._image_roi_set.get(active_area_roi_id)
                if self._image_roi_set is not None else None
            )
            if area_roi is not None:
                # Freeze geometry + pipeline position at commit time, at
                # parity with ROI-scoped filters: the step must not retarget
                # when the live ROI later moves, and must replay in the
                # display frame it was committed in (the adapter interleaves
                # it at this position; review: scope-replay ordering).
                op_spec["frozen_geometry"] = {
                    "kind": area_roi.kind,
                    "geometry": dict(area_roi.geometry),
                    "coord_system": area_roi.coord_system,
                }
                op_spec["after_geometric_ops"] = len(
                    self._processing.get("geometric_ops") or []
                )

        self._push_proc_undo_snapshot()
        ops = list(self._processing.get("arithmetic_ops") or [])
        ops.append(op_spec)
        self._processing["arithmetic_ops"] = ops
        self._clear_bad_line_preview()
        self._refresh_processing_display()
        op = str(op_spec["params"].get("operation", "add")).replace("_", " ")
        operand = str(op_spec["params"].get("operand_type", "constant"))
        target = "active ROI" if "roi_id" in op_spec else "whole image"
        self._status_lbl.setText(f"Applied image arithmetic: {op} {operand} ({target}).")

    def _on_measure_distance(self) -> None:
        """Measure length/angle of the active line ROI → new panel."""
        roi_id = self._active_line_roi_id()
        if roi_id is None:
            self._status_lbl.setText("Select a line ROI first.")
            return
        roi = self._image_roi_set.get(roi_id) if self._image_roi_set else None
        if roi is None:
            return
        from probeflow.analysis.simple_measurements import measure_line_distance
        px_x_m, px_y_m = self._pixel_size_xy_m()
        mid = self._measurement_table.next_measurement_id()
        _, ch_unit, _ = self._channel_unit()
        result = measure_line_distance(
            roi, px_x_m, px_y_m,
            measurement_id=mid,
            source=self._source_label(),
            channel=ch_unit,
        )
        self._measurement_table.add_result(result)
        self._show_measurements()
        self._status_lbl.setText(str(result.context.get("summary") or ""))

    def _on_measure_angle(self) -> None:
        """Switch to the 3-point angle tool; handles emitted from angle_points_ready."""
        self._zoom_lbl.set_tool("angle")
        self._status_lbl.setText("Click P1, P2 (vertex), P3 to measure angle")

    def _angle_measurement_result(self, mid: str, deg: float):
        """Build an angle MeasurementResult for *deg* under measurement id *mid*."""
        from probeflow.measurements.models import MeasurementResult as R
        return R(
            measurement_id=mid,
            kind="angle",
            source_label=self._source_label(),
            source_path=self._source_label(),
            channel=None,
            x_unit="°",
            y_unit=None,
            z_unit=None,
            values={"angle_deg": deg},
            context={},
            notes="",
        )

    def _on_angle_points_ready(self, p1, p2, p3) -> None:
        """Create angle overlay and record result from the 3-point angle tool."""
        from probeflow.gui.angle_overlay import AngleOverlayItem
        scene = self._zoom_lbl.scene()
        if self._angle_overlay is not None:
            self._angle_overlay.remove_from_scene(scene)
        self._angle_overlay = AngleOverlayItem(p1, p2, p3, scene)
        deg = self._angle_overlay.angle_deg
        mid = self._measurement_table.next_measurement_id()
        self._angle_measurement_id = mid
        self._measurement_table.add_result(self._angle_measurement_result(mid, deg))
        self._show_measurements()
        self._sync_viewer_menu_actions()
        self._status_lbl.setText(
            f"Angle: {deg:.2f}°  — drag handles to adjust, then 'Update angle "
            "measurement'."
        )

    def _on_update_angle_measurement(self) -> None:
        """Rewrite the current angle measurement with the adjusted overlay value."""
        if self._angle_overlay is None:
            self._status_lbl.setText(
                "No angle on the image. Use Measure → Angle to place one first."
            )
            return
        deg = self._angle_overlay.angle_deg
        mid = self._angle_measurement_id
        # If the tracked measurement was cleared (or never created), add a fresh one.
        if mid is None:
            mid = self._measurement_table.next_measurement_id()
            self._angle_measurement_id = mid
            self._measurement_table.add_result(self._angle_measurement_result(mid, deg))
            self._status_lbl.setText(f"Angle measurement {mid}: {deg:.2f}°.")
            return
        result = self._angle_measurement_result(mid, deg)
        if not self._measurement_table.update_result(result):
            # Row was removed from the table — re-add it.
            self._measurement_table.add_result(result)
        self._show_measurements()
        self._status_lbl.setText(f"Updated angle measurement {mid}: {deg:.2f}°.")

    def _on_measure_roi_stats(self) -> None:
        """Compute statistics for the active area ROI → new panel."""
        roi_set = self._image_roi_set
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        if roi_set is None:
            self._status_lbl.setText("No ROIs loaded.")
            return
        roi_ctx = selected_or_active_area_roi_context(
            roi_set,
            getattr(self, "_roi_panel", None),
        )
        roi = roi_ctx.roi
        if roi is None:
            self._status_lbl.setText("Select an area ROI first.")
            return
        mask = area_roi_mask(roi, arr.shape[:2])
        if mask is None:
            self._status_lbl.setText("Could not create a non-empty ROI mask.")
            return
        scale, ch_unit, _ = self._channel_unit()
        phys_arr = arr.astype(float) * float(scale)
        from probeflow.analysis.roi_statistics import compute_roi_statistics
        px_x_m, px_y_m = self._pixel_size_xy_m()
        mid = self._measurement_table.next_measurement_id()
        result = compute_roi_statistics(
            phys_arr, mask,
            pixel_size_x_m=px_x_m,
            pixel_size_y_m=px_y_m,
            z_unit=ch_unit,
            measurement_id=mid,
            source=self._source_label(),
            channel=ch_unit,
            roi_id=roi.id,
            roi_name=roi.name,
        )
        self._measurement_table.add_result(result)
        self._show_measurements()
        self._status_lbl.setText(str(result.context.get("summary") or ""))

    def _on_show_image_info(self) -> None:
        """Open the Image Info dialog (modeless) for the current image."""
        from probeflow.gui.dialogs.image_info import ImageInfoDialog
        from probeflow.provenance import display_lines

        # Try to load lightweight metadata for the current entry
        metadata = None
        try:
            entry = self._entries[self._idx]
            from probeflow.core.metadata import read_scan_metadata
            metadata = read_scan_metadata(entry.path)
        except Exception:
            _log.warning("Could not read scan metadata; Image Info opens "
                         "without it", exc_info=True)

        # Build processing history text
        history = getattr(self, "_processing_history", None)
        if history is not None:
            try:
                hist_text = "\n".join(display_lines(history))
            except Exception:
                hist_text = "(Processing history unavailable)"
        else:
            hist_text = "(No processing history)"

        arr = self._current_viewer_array()
        current_shape = arr.shape if arr is not None else None

        dlg = ImageInfoDialog(
            metadata=metadata,
            processing_history_text=hist_text,
            current_shape=current_shape,
            parent=self,
        )
        self._present_modal_tool(dlg)

    def _source_label(self) -> str:
        """Short label for the currently loaded file, for measurement provenance."""
        try:
            return self._entries[self._idx].stem
        except (AttributeError, IndexError, TypeError):
            return ""

    def _point_source_records(self):
        px_x, px_y = self._pixel_size_xy_m()
        ff_dlg = getattr(self, "_feature_finder_dlg", None)
        measure_ctrl = getattr(self, "_image_measurements", None)
        dock = getattr(self, "_roi_panel", None)
        sel_ids = list(dock.selected_roi_ids()) if dock and hasattr(dock, "selected_roi_ids") else []
        return collect_point_source_records(
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
            feature_finder_result=getattr(ff_dlg, "result", None),
            measurement_points=getattr(measure_ctrl, "feature_points", []) or [],
            measurement_metadata=getattr(measure_ctrl, "feature_metadata", {}) or {},
            roi_set=self._image_roi_set,
            selected_roi_ids=sel_ids,
        )

    def _collect_point_sources_m(self) -> dict[str, "np.ndarray"]:
        """Collect available point sources as (N,2) arrays in metres."""
        return point_source_arrays_m(self._point_source_records())

    def _collect_point_sources_px(self) -> dict[str, "np.ndarray"]:
        """Collect available point sources as (N,2) arrays in pixel coordinates."""
        return point_source_arrays_px(self._point_source_records())

    def _collect_point_source_metadata(self) -> dict[str, dict[str, object]]:
        """Collect metadata for available point sources."""
        return point_source_metadata(self._point_source_records())

    def _on_open_pair_correlation(self) -> None:
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        px_x, px_y = self._pixel_size_xy_m()
        context = pair_correlation_launch_context(
            self._point_source_records(),
            active_area_roi=self._active_image_roi(),
            image_shape=arr.shape[:2] if arr is not None else None,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
        )
        if not context.ready:
            self._status_lbl.setText(str(context.status_message))
            return
        _, ch_unit, _ = self._channel_unit()
        entries = getattr(self, "_entries", [])
        entry = entries[self._idx] if entries else None
        from probeflow.gui.dialogs.pair_correlation import PairCorrelationDialog

        def _add(result):
            self._add_dialog_measurement_result(result)

        dlg = PairCorrelationDialog(
            context.sources_m,
            roi_area_m2=context.roi_area_m2,
            pixel_size_x_m=context.pixel_size_x_m,
            pixel_size_y_m=context.pixel_size_y_m,
            source_label=self._source_label(),
            source_path=str(entry.path) if entry is not None else None,
            channel=ch_unit,
            source_metadata=context.source_metadata,
            on_add_result=_add,
            theme=self._t,
            parent=self,
        )
        self._present_modal_tool(dlg)
        self._status_lbl.setText("Pair correlation opened.")

    def _feature_set_store(self):
        """Lazily-created viewer-session store of named feature sets."""
        store = getattr(self, "_feature_set_store_obj", None)
        if store is None:
            from probeflow.measurements.feature_sets import FeatureSetStore

            store = FeatureSetStore()
            self._feature_set_store_obj = store
        return store

    def _on_open_feature_lattice(self) -> None:
        item = getattr(self, "_lattice_grid_item", None)
        grid = item.grid() if item is not None else None
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        px_x, px_y = self._pixel_size_xy_m()
        context = feature_lattice_launch_context(
            self._point_source_records(),
            lattice_grid=grid,
            image_shape=arr.shape[:2] if arr is not None else None,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
        )
        if not context.ready:
            self._status_lbl.setText(str(context.status_message))
            return
        _, ch_unit, _ = self._channel_unit()
        entries = getattr(self, "_entries", [])
        entry = entries[self._idx] if entries else None
        from probeflow.gui.dialogs.feature_lattice_dialog import FeatureLatticeDialog

        def _add(result):
            self._add_dialog_measurement_result(result)

        dlg = FeatureLatticeDialog(
            context.sources_px,
            lattice_origin_px=context.lattice_origin_px,
            a_px=context.a_px,
            b_px=context.b_px,
            pixel_size_x_m=context.pixel_size_x_m,
            pixel_size_y_m=context.pixel_size_y_m,
            image_shape=context.image_shape,
            source_label=self._source_label(),
            source_path=str(entry.path) if entry is not None else None,
            channel=ch_unit,
            source_metadata=context.source_metadata,
            on_add_result=_add,
            theme=self._t,
            parent=self,
        )
        self._present_modal_tool(dlg)

    def _on_open_fft_viewer(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        def _get_image():
            return self._display_arr if self._display_arr is not None else self._raw_arr

        def _preview_fft_correction(corrected_arr) -> None:
            self._display_arr = corrected_arr
            self._update_export_summary()
            self._refresh_viewer_pixmap(reset_zoom=False)

        def _clear_fft_correction_preview() -> None:
            self._refresh_display_array(reset_zoom_if_shape_changed=False)
            self._refresh_viewer_pixmap(reset_zoom=False)

        def _apply_fft_correction(op_name: str, op_params: dict) -> None:
            ops = list(self._processing.get("geometric_ops") or [])
            ops.append({"op": op_name, "params": op_params})
            self._processing["geometric_ops"] = ops
            self._refresh_processing_display()
            self._status_lbl.setText("Applied FFT-derived lattice correction.")

        # Resolve the active area ROI (if any) so the FFT can optionally be
        # computed from its bounding box. The dialog still defaults to the whole
        # image; the ROI is opt-in via its "Source" selector.
        roi_ctx = active_area_roi_context(self._image_roi_set)
        roi_bounds = active_area_roi_bounds(self._image_roi_set, arr.shape[:2])
        roi_id = roi_ctx.roi_id if roi_bounds is not None else None
        roi_name = (
            getattr(roi_ctx.roi, "name", None) if roi_bounds is not None else None
        )

        def _open_fft_result_image(result_arr, result_scan_range_m, provenance) -> None:
            """Open generated ROI inverse-FFT output in its own lightweight viewer."""
            try:
                from probeflow.gui.dialogs.array_image import ArrayImageDialog

                label = f" - {roi_name}" if roi_name else ""
                result_dlg = ArrayImageDialog(
                    result_arr,
                    scan_range_m=tuple(result_scan_range_m),
                    title=f"Inverse FFT result{label}",
                    colormap=getattr(self, "_viewer_colormap", self._colormap),
                    theme=self._t,
                    provenance=provenance,
                    parent=self,
                )
            except Exception as exc:
                self._status_lbl.setText(f"Could not open inverse FFT result: {exc}")
                return
            self._track_modeless_child(result_dlg)
            result_dlg.show()
            self._status_lbl.setText("Opened inverse FFT result image.")

        fft_scan_range = self._display_scan_range_m or self._scan_range_m or (1e-9, 1e-9)
        # Estimate the fast-scan speed from the header so the Mains tab can
        # predict where 50/60 Hz pickup lands (None → the user enters it).
        from probeflow.processing.mains_pickup import estimate_fast_scan_speed_m_per_s
        scan_speed = estimate_fast_scan_speed_m_per_s(
            getattr(self, "_scan_header", None) or {},
            scan_range_m=fft_scan_range, image_shape=arr.shape[:2],
        )

        dlg = FFTViewerDialog(
            arr,
            fft_scan_range,
            colormap=self._colormap,
            theme=self._t,
            channel_unit=self._channel_unit(),
            get_image_fn=_get_image,
            apply_correction_fn=_apply_fft_correction,
            preview_image_fn=_preview_fft_correction,
            clear_preview_fn=_clear_fft_correction_preview,
            roi_bounds_px=roi_bounds,
            roi_id=roi_id,
            roi_name=roi_name,
            scan_speed_m_per_s=scan_speed,
            new_image_fn=_open_fft_result_image,
            parent=self,
        )
        self._track_modeless_child(dlg)
        dlg.show()

    def _on_periodic_filter(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image data available for FFT filtering.")
            return
        dlg = PeriodicFilterDialog(
            arr,
            peaks=self._processing.get("periodic_notches", []),
            radius_px=float(self._processing.get("periodic_notch_radius", 3.0)),
            scan_range_m=self._scan_range_m,
            colormap=self._colormap,
            theme=self._t,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        peaks = dlg.selected_peaks()
        if peaks:
            self._processing["periodic_notches"] = peaks
            self._processing["periodic_notch_radius"] = dlg.radius_px()
            self._status_lbl.setText(f"Periodic FFT filter: {len(peaks)} peak(s) selected.")
        else:
            self._processing.pop("periodic_notches", None)
            self._processing.pop("periodic_notch_radius", None)
            self._status_lbl.setText("Periodic FFT filter cleared.")
        self._refresh_processing_display()
