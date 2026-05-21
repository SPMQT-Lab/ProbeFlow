"""Image-viewer measurement controller.

This module keeps image measurement orchestration out of the legacy viewer
class.  Numerical work stays in :mod:`probeflow.measurements`; this controller
only gathers viewer state, records results, updates point overlays, and handles
small GUI export actions.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QFileDialog

from probeflow.measurements.export import (
    feature_points_to_csv_text,
    feature_points_to_json_text,
)
from probeflow.measurements.features import (
    detect_local_maxima,
    feature_maxima_result,
)
from probeflow.measurements.fft_points import (
    fft_from_point_mask,
    point_fft_summary_result,
    point_fft_to_csv_text,
    point_mask_to_csv_text,
    points_to_mask,
)
from probeflow.measurements.image import (
    line_periodicity_measurement,
    line_profile_delta_measurement,
    line_profile_measurement,
    roi_statistics,
    step_height_from_rois,
)

from probeflow.core import AREA_ROI_KINDS
from probeflow.gui.roi_context import (
    ROIContext,
    active_line_roi_context,
    selected_area_roi_contexts,
    selected_or_active_area_roi_context,
    selected_or_active_roi_context,
    selected_roi_ids_for_context,
)


def _csv_text(rows: list[list[str]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerows(rows)
    return out.getvalue()


def _metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _format_period_bound_nm(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 1e9:.12g} nm"
    except (TypeError, ValueError):
        return str(value)


class ImageMeasurementController:
    """Coordinate image measurements between viewer state and result widgets."""

    def __init__(
        self,
        viewer: Any,
        table: Any,
        feature_panel: Any | None = None,
        point_mask_panel: Any | None = None,
        line_periodicity_panel: Any | None = None,
    ):
        self._viewer = viewer
        self._table = table
        self._feature_panel = feature_panel
        self._point_mask_panel = point_mask_panel
        self._line_periodicity_panel = line_periodicity_panel
        self._feature_points: list[Any] = []
        self._feature_metadata: dict[str, object] = {}
        self._last_periodicity_result: Any = None
        self._last_periodicity_diag: Any = None
        self._last_periodicity_settings: dict = {}
        if feature_panel is not None:
            feature_panel.detectRequested.connect(self.detect_feature_maxima_for_active_roi)
            feature_panel.copyPointsRequested.connect(self.copy_feature_points)
            feature_panel.exportCsvRequested.connect(self.export_feature_points_csv)
            feature_panel.exportJsonRequested.connect(self.export_feature_points_json)
            feature_panel.clearRequested.connect(self.clear_feature_points)
        if point_mask_panel is not None:
            point_mask_panel.exportMaskCsvRequested.connect(self.export_point_mask_csv)
            point_mask_panel.computeFftRequested.connect(self.compute_point_mask_fft)
            point_mask_panel.exportFftCsvRequested.connect(self.export_point_fft_csv)
        if line_periodicity_panel is not None:
            line_periodicity_panel.findPeriodicityRequested.connect(
                self.find_periodicity_for_active_line_roi
            )
            line_periodicity_panel.copyResultRequested.connect(self.copy_periodicity_result)
            line_periodicity_panel.exportProfileCsvRequested.connect(
                self.export_periodicity_profile_csv
            )

    @property
    def feature_points(self) -> list[Any]:
        """Return the currently detected feature points."""
        return list(self._feature_points)

    @property
    def feature_metadata(self) -> dict[str, object]:
        """Return metadata for the currently detected feature points."""
        return dict(self._feature_metadata)

    def action_enabled_state(self) -> dict[str, bool]:
        """Return enabled states for viewer measurement menu actions."""
        roi_ctx = self._selected_or_active_roi_context()
        roi = roi_ctx.roi
        is_area = roi is not None and roi.kind in AREA_ROI_KINDS
        is_line = roi is not None and roi.kind == "line"
        selected_ids = self.selected_roi_ids()
        selected_area_pair = len(selected_ids) == 2 and len(self._selected_area_rois()) == 2
        has_line = is_line or self._active_line_roi_id() is not None
        return {
            "distance": has_line,
            "roi_stats": is_area,
            "step_height": selected_area_pair,
            "line_profile": has_line,
            "line_periodicity": has_line,
            "feature_maxima": True,  # works on full image when no area ROI is active
            "copy_points": bool(self._feature_points),
            "export_points_csv": bool(self._feature_points),
            "export_points_json": bool(self._feature_points),
            "export_point_mask": bool(self._feature_points),
            "compute_point_fft": bool(self._feature_points),
            "export_point_fft": bool(self._feature_points),
            "clear_points": bool(self._feature_points),
        }

    def add_detected_point_menu_actions(
        self,
        menu: Any,
        action_owner: Any,
        action_map: dict[str, QAction],
    ) -> None:
        """Install menu actions for point-list, mask, and FFT operations."""
        for label, key, callback in [
            ("Copy detected points", "copy_points", self.copy_feature_points),
            ("Export detected points CSV", "export_points_csv", self.export_feature_points_csv),
            ("Export detected points JSON", "export_points_json", self.export_feature_points_json),
            ("Export point mask CSV", "export_point_mask", self.export_point_mask_csv),
            ("Compute point-mask FFT", "compute_point_fft", self.compute_point_mask_fft),
            ("Export point-mask FFT CSV", "export_point_fft", self.export_point_fft_csv),
            ("Clear detected points", "clear_points", self.clear_feature_points),
        ]:
            action = QAction(label, action_owner)
            action.triggered.connect(lambda _checked=False, cb=callback: cb())
            action_map[key] = action
            menu.addAction(action)

    def selected_roi_ids(self) -> list[str]:
        """Return selected ROI IDs from the dock, falling back to active ROI."""
        return selected_roi_ids_for_context(self._roi_set(), self._roi_dock())

    def add_active_roi_stats_measurement(self) -> None:
        roi_ctx = self._selected_or_active_area_roi_context()
        if roi_ctx.roi_id is None:
            self._set_status("Select an area ROI first.")
            return
        self.add_roi_stats_measurement(roi_ctx.roi_id)

    def add_roi_stats_measurement(self, roi_id: str) -> None:
        roi = self._roi(roi_id)
        arr = self._display_arr()
        if roi is None or arr is None:
            return
        if roi.kind not in AREA_ROI_KINDS:
            self._set_status("ROI statistics require an area ROI.")
            return
        try:
            entry, scale, unit, channel, source_label = self._source_info()
            px_x_nm, px_y_nm = self._pixel_size_nm()
            result = roi_statistics(
                arr * scale,
                measurement_id=self._table.next_measurement_id(),
                source_label=source_label,
                source_path=str(entry.path),
                channel=channel,
                roi=roi,
                pixel_size_x=px_x_nm,
                pixel_size_y=px_y_nm,
                x_unit="nm",
                y_unit="nm",
                height_unit=unit or None,
                notes=f"ROI statistics for {roi.name}",
            )
            self._record(result)
        except Exception as exc:
            self._set_status(f"Could not add ROI statistics: {exc}")

    def add_selected_step_height_measurement(self) -> None:
        self.add_step_height_measurement_for_rois(self.selected_roi_ids())

    def add_step_height_measurement_for_rois(self, roi_ids: list[str]) -> None:
        arr = self._display_arr()
        if arr is None or len(roi_ids) != 2:
            self._set_status("Select exactly two area ROIs for step height.")
            return
        roi_a = self._roi(roi_ids[0])
        roi_b = self._roi(roi_ids[1])
        if (
            roi_a is None or roi_b is None
            or roi_a.kind not in AREA_ROI_KINDS
            or roi_b.kind not in AREA_ROI_KINDS
        ):
            self._set_status("Step height requires two area ROIs.")
            return
        try:
            entry, scale, unit, channel, source_label = self._source_info()
            result = step_height_from_rois(
                arr * scale,
                roi_a,
                roi_b,
                measurement_id=self._table.next_measurement_id(),
                source_label=source_label,
                source_path=str(entry.path),
                channel=channel,
                x_unit="nm",
                y_unit="nm",
                height_unit=unit or None,
                notes=f"Step height: {roi_a.name} to {roi_b.name}",
            )
            self._record(result)
        except Exception as exc:
            self._set_status(f"Could not add step-height measurement: {exc}")

    def add_current_line_profile_measurement(self) -> None:
        roi_id = self._active_line_roi_id()
        if roi_id is None:
            self._set_status(
                "Set a saved line ROI active before adding a line-profile measurement."
            )
            return
        self.add_line_profile_measurement_for_roi(roi_id)

    def find_periodicity_for_active_line_roi(self) -> None:
        from probeflow.analysis.line_periodicity import estimate_line_periodicity, format_result_text

        roi_id = self._active_line_roi_id()
        if roi_id is None:
            self._set_status("Draw or select a line ROI first.")
            if self._line_periodicity_panel is not None:
                self._line_periodicity_panel.show_message("Draw or select a line ROI first.")
            return

        roi = self._roi(roi_id)
        arr = self._display_arr()
        if roi is None or roi.kind != "line" or arr is None:
            return

        settings = (
            self._line_periodicity_panel.settings()
            if self._line_periodicity_panel is not None
            else {}
        )
        method = str(settings.get("method", "autocorrelation"))
        background = str(settings.get("background", "linear"))
        smoothing = str(settings.get("smoothing", "light_gaussian"))
        width_px = float(settings.get("width_px", 1.0))
        min_period_m = settings.get("min_period_m")
        max_period_m = settings.get("max_period_m")

        try:
            self._viewer._on_roi_line_profile(roi_id)
            distance_m, values = self._line_profile_m(roi, width_px=width_px)
            result, diag = estimate_line_periodicity(
                distance_m,
                values,
                method=method,
                background=background,
                smoothing=smoothing,
                min_period_m=min_period_m,
                max_period_m=max_period_m,
            )
            self._last_periodicity_result = result
            self._last_periodicity_diag = diag
            self._last_periodicity_settings = {
                "method": method,
                "background": background,
                "smoothing": smoothing,
                "width_px": width_px,
                "min_period_m": min_period_m,
                "max_period_m": max_period_m,
                "roi_id": roi.id,
                "roi_name": roi.name,
            }
            if self._line_periodicity_panel is not None:
                self._line_periodicity_panel.set_result(result)

            entry, _scale, _unit, channel, source_label = self._source_info()
            self._last_periodicity_settings.update({
                "source_label": source_label,
                "source_path": str(entry.path),
                "channel": channel,
                "quality": result.quality,
                "message": result.message,
            })
            meas = line_periodicity_measurement(
                result,
                measurement_id=self._table.next_measurement_id(),
                source_label=source_label,
                source_path=str(entry.path),
                channel=channel,
                roi_id=roi.id,
                roi_name=roi.name,
                background=background,
                smoothing=smoothing,
                width_px=width_px,
                notes=f"Line periodicity for {roi.name}",
            )
            self._record(meas)
            self._show_periodicity_plot_dialog(result, diag)
        except Exception as exc:
            msg = f"Could not estimate periodicity: {exc}"
            self._set_status(msg)
            if self._line_periodicity_panel is not None:
                self._line_periodicity_panel.show_message(msg)

    def copy_periodicity_result(self) -> None:
        from probeflow.analysis.line_periodicity import format_result_text

        if self._last_periodicity_result is None:
            return
        s = self._last_periodicity_settings
        text = format_result_text(
            self._last_periodicity_result,
            background=s.get("background", ""),
            smoothing=s.get("smoothing", ""),
        )
        context = self._periodicity_context_lines()
        if context:
            text = f"{text}\n" + "\n".join(context)
        QApplication.clipboard().setText(text)
        self._set_status("Copied periodicity result.")

    def export_periodicity_profile_csv(self) -> None:
        if self._last_periodicity_diag is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._viewer,
            "Export periodicity profile CSV",
            str(Path.home() / "probeflow_periodicity_profile.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        diag = self._last_periodicity_diag
        rows = self._periodicity_csv_rows("probeflow_line_periodicity_profile")
        rows.append(["s_m", "s_nm", "z_raw", "z_processed"])
        for s, z_r, z_p in zip(diag.s_m, diag.z_raw, diag.z_processed):
            rows.append([
                f"{s:.6e}",
                f"{s * 1e9:.6e}",
                f"{z_r:.6e}",
                f"{z_p:.6e}",
            ])
        Path(path).write_text(_csv_text(rows), encoding="utf-8")
        # Optionally export autocorrelation if available
        if diag.autocorr_lag_m is not None and diag.autocorr is not None:
            ac_path = Path(path).with_stem(Path(path).stem + "_autocorr")
            ac_rows = self._periodicity_csv_rows("probeflow_line_periodicity_autocorr")
            ac_rows.append(["lag_m", "lag_nm", "autocorrelation"])
            for lag, ac in zip(diag.autocorr_lag_m, diag.autocorr):
                ac_rows.append([f"{lag:.6e}", f"{lag * 1e9:.6e}", f"{ac:.6e}"])
            ac_path.write_text(_csv_text(ac_rows), encoding="utf-8")
        self._set_status(f"Periodicity profile → {path}")

    def _periodicity_context_lines(self) -> list[str]:
        s = self._last_periodicity_settings
        lines: list[str] = []
        roi_name = s.get("roi_name")
        roi_id = s.get("roi_id")
        if roi_name:
            lines.append(f"ROI: {roi_name}")
        elif roi_id:
            lines.append(f"ROI id: {roi_id}")
        if s.get("source_label"):
            lines.append(f"Source: {s['source_label']}")
        if s.get("channel"):
            lines.append(f"Channel: {s['channel']}")
        if s.get("width_px") is not None:
            lines.append(f"Width: {_metadata_value(s['width_px'])} px")
        min_period = _format_period_bound_nm(s.get("min_period_m"))
        max_period = _format_period_bound_nm(s.get("max_period_m"))
        if min_period or max_period:
            lines.append(f"Period bounds: {min_period or 'none'} to {max_period or 'none'}")
        if s.get("quality"):
            lines.append(f"Quality: {s['quality']}")
        if s.get("message"):
            lines.append(f"Message: {s['message']}")
        return lines

    def _periodicity_csv_rows(self, export_type: str) -> list[list[str]]:
        s = self._last_periodicity_settings
        rows: list[list[str]] = [["# export_type", export_type]]
        for key in (
            "source_label",
            "source_path",
            "channel",
            "roi_id",
            "roi_name",
            "method",
            "background",
            "smoothing",
            "width_px",
            "min_period_m",
            "max_period_m",
            "quality",
            "message",
        ):
            rows.append([f"# {key}", _metadata_value(s.get(key))])
        return rows

    def _show_periodicity_plot_dialog(self, result, diag) -> None:
        from probeflow.gui.dialogs.line_periodicity_plot import PeriodicityPlotDialog

        existing = getattr(self, "_periodicity_plot_dialog", None)
        if existing is not None and not existing.isHidden():
            existing.update_plot(result, diag)
            existing.raise_()
            return
        dialog = PeriodicityPlotDialog(
            result, diag,
            theme=getattr(self._viewer, "_t", None),
            parent=self._viewer,
        )
        self._periodicity_plot_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _line_profile_m(self, roi, *, width_px: float = 1.0) -> tuple:
        """Return raw (distance_m, values) without unit conversion."""
        from probeflow.processing.image import line_profile

        arr = self._display_arr()
        px_x, px_y = self._viewer._pixel_size_xy_m()
        distance_m, values = line_profile(
            arr,
            roi=roi,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
            width_px=max(1.0, width_px),
        )
        return (
            np.asarray(distance_m, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )

    def add_current_line_profile_delta_measurement(self) -> None:
        panel = getattr(self._viewer, "_line_profile_panel", None)
        if panel is None:
            self._set_status("No line profile panel available.")
            return
        delta = panel.meas_delta()
        if delta is None:
            self._set_status("Select two points on the line profile first.")
            return
        roi_id = self._active_line_roi_id()
        roi = self._roi(roi_id)
        try:
            entry, _scale, _unit, channel, source_label = self._source_info()
        except Exception as exc:
            self._set_status(f"Could not read source info: {exc}")
            return
        result = line_profile_delta_measurement(
            delta_x=delta["delta_x"],
            delta_y=delta["delta_y"],
            p1_distance=delta["p1_distance"],
            p1_height=delta["p1_height"],
            p2_distance=delta["p2_distance"],
            p2_height=delta["p2_height"],
            measurement_id=self._table.next_measurement_id(),
            source_label=source_label,
            source_path=str(entry.path),
            channel=channel,
            x_unit=delta["x_unit"] or None,
            y_unit=delta["y_unit"] or None,
            roi_id=roi.id if roi else None,
            roi_name=roi.name if roi else None,
            notes=f"Line profile Δ{' for ' + roi.name if roi else ''}",
        )
        self._record(result)

    def add_line_profile_measurement_for_roi(self, roi_id: str) -> None:
        roi = self._roi(roi_id)
        arr = self._display_arr()
        if roi is None or roi.kind != "line" or arr is None:
            return
        try:
            self._viewer._on_roi_line_profile(roi_id)
            entry, _scale, _unit, channel, source_label = self._source_info()
            distance, profile, x_unit, y_unit, width_px = self._line_profile_data(roi)
            result = line_profile_measurement(
                distance,
                profile,
                measurement_id=self._table.next_measurement_id(),
                source_label=source_label,
                source_path=str(entry.path),
                channel=channel,
                x_unit=x_unit,
                y_unit=y_unit or None,
                p0=(
                    float(roi.geometry.get("x1", 0.0)),
                    float(roi.geometry.get("y1", 0.0)),
                ),
                p1=(
                    float(roi.geometry.get("x2", 0.0)),
                    float(roi.geometry.get("y2", 0.0)),
                ),
                roi_id=roi.id,
                roi_name=roi.name,
                swath_width=width_px,
                averaging_method="perpendicular mean",
                notes=f"Line profile for {roi.name}",
            )
            self._record(result)
        except Exception as exc:
            self._set_status(f"Could not add line profile measurement: {exc}")

    def detect_feature_maxima_for_active_roi(self) -> None:
        """Detect maxima inside the active area ROI, or over the full image if none."""
        roi_ctx = self._selected_or_active_area_roi_context()
        roi_id = roi_ctx.roi_id
        settings = self._feature_panel.settings() if self._feature_panel else {}
        self._run_feature_maxima(roi_id=roi_id, settings=settings)

    def detect_feature_maxima_for_roi(
        self,
        roi_id: str,
        *,
        settings: dict[str, object] | None = None,
    ) -> None:
        roi = self._roi(roi_id)
        arr = self._display_arr()
        if roi is None or arr is None:
            return
        if roi.kind not in AREA_ROI_KINDS:
            self._set_status("Feature maxima detection requires an area ROI.")
            return
        if settings is None and self._feature_panel is not None:
            settings = self._feature_panel.settings()
        self._run_feature_maxima(roi_id=roi_id, settings=settings)

    def _run_feature_maxima(
        self,
        *,
        roi_id: str | None,
        settings: dict[str, object] | None = None,
    ) -> None:
        """Run feature maxima detection with optional ROI, falling back to full image."""
        arr = self._display_arr()
        if arr is None:
            self._set_status("No image is loaded.")
            return
        roi = self._roi(roi_id)
        settings = dict(settings or {})
        try:
            entry, scale, unit, channel, source_label = self._source_info()
            px_x_nm, px_y_nm = self._pixel_size_nm()
            threshold_mode = str(settings.get("threshold_mode", "percentile"))
            threshold_value = float(settings.get("threshold_value", 95.0))
            min_distance_px = int(settings.get("min_distance_px", 2))
            smoothing_sigma = settings.get("smoothing_sigma")
            max_peaks = settings.get("max_peaks")
            exclude_border = int(settings.get("exclude_border", 0))
            scope = "roi" if roi is not None else "full_image"
            scope_label = roi.name if roi is not None else "full image"
            points = detect_local_maxima(
                arr * scale,
                threshold_mode=threshold_mode,
                threshold_value=threshold_value,
                min_distance_px=min_distance_px,
                smoothing_sigma=(
                    float(smoothing_sigma) if smoothing_sigma is not None else None
                ),
                max_peaks=(int(max_peaks) if max_peaks is not None else None),
                exclude_border=exclude_border,
                roi=roi,
                pixel_size_x=px_x_nm,
                pixel_size_y=px_y_nm,
                channel=channel,
                source_label=source_label,
                roi_id=roi_id,
            )
            self._feature_points = list(points)
            self._feature_metadata = {
                "source_label": source_label,
                "source_path": str(entry.path),
                "channel": channel,
                "x_unit": "nm",
                "y_unit": "nm",
                "z_unit": unit or None,
                "roi_id": roi_id,
                "roi_name": roi.name if roi is not None else None,
                "selection_scope": scope,
                "threshold_mode": threshold_mode,
                "threshold_value": threshold_value,
                "min_distance_px": min_distance_px,
                "smoothing_sigma": smoothing_sigma,
                "exclude_border": exclude_border,
            }
            self._push_feature_overlay()
            self._sync_actions()
            if self._feature_panel is not None:
                self._feature_panel.set_points_count(len(points), roi_name=scope_label)
            if (
                self._point_mask_panel is not None
                and hasattr(self._point_mask_panel, "set_points_available")
            ):
                self._point_mask_panel.set_points_available(bool(points))
            result = feature_maxima_result(
                points,
                measurement_id=self._table.next_measurement_id(),
                source_label=source_label,
                source_path=str(entry.path),
                channel=channel,
                x_unit="nm",
                y_unit="nm",
                threshold_mode=threshold_mode,
                threshold_value=threshold_value,
                min_distance_px=min_distance_px,
                smoothing_sigma=(
                    float(smoothing_sigma) if smoothing_sigma is not None else None
                ),
                roi_id=roi_id,
                roi_name=roi.name if roi is not None else None,
                selection_scope=scope,
                exclude_border=exclude_border,
                notes=f"Local maxima in {scope_label}; z unit: {unit or 'data units'}",
            )
            self._record(result)
            self._set_status(
                f"Detected {len(points)} maxima ({scope_label}) → {result.measurement_id}."
            )
        except Exception as exc:
            self.clear_feature_points(silent=True)
            self._set_status(f"Could not detect maxima: {exc}")
            if self._feature_panel is not None:
                self._feature_panel.show_message(str(exc))

    def copy_feature_points(self) -> None:
        if not self._feature_points:
            return
        QApplication.clipboard().setText(
            feature_points_to_csv_text(
                self._feature_points,
                metadata=self._feature_metadata,
            )
        )
        self._set_status(f"Copied {len(self._feature_points)} feature points.")

    def export_feature_points_csv(self) -> None:
        if not self._feature_points:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._viewer,
            "Export feature points CSV",
            str(Path.home() / "probeflow_feature_points.csv"),
            "CSV files (*.csv)",
        )
        if path:
            Path(path).write_text(
                feature_points_to_csv_text(
                    self._feature_points,
                    metadata=self._feature_metadata,
                ),
                encoding="utf-8",
            )
            self._set_status(f"Feature points -> {path}")

    def export_feature_points_json(self) -> None:
        if not self._feature_points:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._viewer,
            "Export feature points JSON",
            str(Path.home() / "probeflow_feature_points.json"),
            "JSON files (*.json)",
        )
        if path:
            Path(path).write_text(
                feature_points_to_json_text(
                    self._feature_points,
                    metadata=self._feature_metadata,
                ),
                encoding="utf-8",
            )
            self._set_status(f"Feature points -> {path}")

    def export_point_mask_csv(self) -> None:
        """Export a derived binary mask from the current feature points."""
        try:
            mask, _radius_px, _shape_mode = self._point_mask()
        except ValueError as exc:
            self._set_status(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self._viewer,
            "Export point mask CSV",
            str(Path.home() / "probeflow_feature_point_mask.csv"),
            "CSV files (*.csv)",
        )
        if path:
            metadata = self._point_mask_export_metadata(
                mask,
                radius_px=_radius_px,
                shape_mode=_shape_mode,
            )
            Path(path).write_text(
                point_mask_to_csv_text(mask, metadata=metadata),
                encoding="utf-8",
            )
            self._set_status(f"Point mask -> {path}")

    def compute_point_mask_fft(self, *, show_dialog: bool = True) -> None:
        """Compute an FFT from the derived feature-point mask."""
        try:
            mask, radius_px, shape_mode = self._point_mask()
            fft_result = self._point_fft(mask, radius_px=radius_px)
            entry, _scale, _unit, channel, source_label = self._source_info()
            result = point_fft_summary_result(
                fft_result,
                measurement_id=self._table.next_measurement_id(),
                source_label=source_label,
                source_path=str(entry.path),
                channel=channel,
                mask_pixels=int(np.count_nonzero(mask)),
                shape_mode=shape_mode,
                notes="FFT of derived binary mask from detected feature maxima.",
            )
            self._record(result)
            self._set_status(
                f"Computed point-mask FFT from {len(self._feature_points)} points."
            )
            if show_dialog:
                self._show_point_fft_dialog(mask, fft_result)
        except Exception as exc:
            self._set_status(f"Could not compute point-mask FFT: {exc}")

    def export_point_fft_csv(self) -> None:
        """Export the current derived point-mask FFT as long-form CSV."""
        try:
            mask, radius_px, _shape_mode = self._point_mask()
            fft_result = self._point_fft(mask, radius_px=radius_px)
        except ValueError as exc:
            self._set_status(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self._viewer,
            "Export point-mask FFT CSV",
            str(Path.home() / "probeflow_point_mask_fft.csv"),
            "CSV files (*.csv)",
        )
        if path:
            metadata = self._point_mask_export_metadata(
                mask,
                radius_px=radius_px,
                shape_mode=_shape_mode,
            )
            metadata.update({
                "export_type": "probeflow_point_mask_fft",
                "fft_units": fft_result.units,
            })
            Path(path).write_text(
                point_fft_to_csv_text(fft_result, metadata=metadata),
                encoding="utf-8",
            )
            self._set_status(f"Point-mask FFT -> {path}")

    def clear_feature_points(self, *, silent: bool = False) -> None:
        self._feature_points = []
        self._feature_metadata = {}
        self._push_feature_overlay()
        if self._feature_panel is not None:
            self._feature_panel.set_points_count(0)
        if self._point_mask_panel is not None and hasattr(self._point_mask_panel, "set_points_available"):
            self._point_mask_panel.set_points_available(False)
        self._sync_actions()
        if not silent:
            self._set_status("Cleared feature maxima overlay.")

    def _point_mask_export_metadata(
        self,
        mask: np.ndarray,
        *,
        radius_px: int,
        shape_mode: str,
    ) -> dict[str, object]:
        px_x_nm, px_y_nm = self._pixel_size_nm()
        metadata = {
            "export_type": "probeflow_feature_point_mask",
            "source_label": self._feature_metadata.get("source_label"),
            "source_path": self._feature_metadata.get("source_path"),
            "channel": self._feature_metadata.get("channel"),
            "roi_id": self._feature_metadata.get("roi_id"),
            "roi_name": self._feature_metadata.get("roi_name"),
            "selection_scope": self._feature_metadata.get("selection_scope"),
            "threshold_mode": self._feature_metadata.get("threshold_mode"),
            "threshold_value": self._feature_metadata.get("threshold_value"),
            "min_distance_px": self._feature_metadata.get("min_distance_px"),
            "smoothing_sigma": self._feature_metadata.get("smoothing_sigma"),
            "exclude_border": self._feature_metadata.get("exclude_border"),
            "point_count": len(self._feature_points),
            "mask_pixels": int(np.count_nonzero(mask)),
            "mask_shape_y": int(mask.shape[0]),
            "mask_shape_x": int(mask.shape[1]),
            "radius_px": int(radius_px),
            "shape_mode": shape_mode,
            "pixel_size_x_nm": px_x_nm,
            "pixel_size_y_nm": px_y_nm,
            "x_unit": "nm",
            "y_unit": "nm",
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def _record(self, result) -> None:
        self._table.add_result(result)
        if hasattr(self._viewer, "_show_measurements"):
            self._viewer._show_measurements()
        self._set_status(f"Added {result.kind} measurement {result.measurement_id}.")

    def _push_feature_overlay(self) -> None:
        canvas = getattr(self._viewer, "_zoom_lbl", None)
        if canvas is not None and hasattr(canvas, "set_feature_points"):
            canvas.set_feature_points(self._feature_points)

    def _point_mask(self) -> tuple[np.ndarray, int, str]:
        if not self._feature_points:
            raise ValueError("Detect feature maxima before generating a point mask.")
        arr = self._display_arr()
        if arr is None:
            raise ValueError("No image is loaded for point-mask generation.")
        settings = (
            self._point_mask_panel.mask_settings()
            if self._point_mask_panel is not None and hasattr(self._point_mask_panel, "mask_settings")
            else {}
        )
        radius_px = int(settings.get("radius_px", 0))
        shape_mode = str(settings.get("shape_mode", "disk"))
        mask = points_to_mask(
            self._feature_points,
            np.asarray(arr).shape,
            radius_px=radius_px,
            shape_mode=shape_mode,
        )
        return mask, radius_px, shape_mode

    def _point_fft(self, mask: np.ndarray, *, radius_px: int):
        px_x_nm, px_y_nm = self._pixel_size_nm()
        return fft_from_point_mask(
            mask,
            pixel_size_x=px_x_nm,
            pixel_size_y=px_y_nm,
            spatial_unit="nm",
            n_points=len(self._feature_points),
            radius_px=radius_px,
        )

    def _show_point_fft_dialog(self, mask: np.ndarray, fft_result) -> None:
        from probeflow.gui.dialogs.point_fft import PointMaskFFTDialog

        dialog = PointMaskFFTDialog(
            mask,
            fft_result,
            theme=getattr(self._viewer, "_t", None),
            parent=self._viewer,
        )
        self._point_fft_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _line_profile_data(self, roi) -> tuple[np.ndarray, np.ndarray, str, str, float]:
        from probeflow.analysis.spec_plot import choose_display_unit
        from probeflow.processing.image import line_profile

        arr = self._display_arr()
        px_x, px_y = self._viewer._pixel_size_xy_m()
        width_px = float(roi.geometry.get("width", 1)) if roi.geometry else 1.0
        distance_m, values = line_profile(
            arr,
            roi=roi,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
            width_px=max(1.0, width_px),
        )
        scale, unit, _channel = self._viewer._channel_unit()
        x_scale, x_unit = choose_display_unit("m", distance_m)
        return (
            np.asarray(distance_m, dtype=np.float64) * x_scale,
            np.asarray(values, dtype=np.float64) * scale,
            x_unit,
            unit,
            width_px,
        )

    def _source_info(self):
        entry = self._viewer._entries[self._viewer._idx]
        scale, unit, axis_label = self._viewer._channel_unit()
        channel = axis_label or self._viewer._ch_cb.currentText()
        source_label = f"{entry.stem}:{channel}" if channel else entry.stem
        return entry, scale, unit, channel, source_label

    def _pixel_size_nm(self) -> tuple[float, float]:
        px_x_m, px_y_m = self._viewer._pixel_size_xy_m()
        return float(px_x_m) * 1e9, float(px_y_m) * 1e9

    def _display_arr(self):
        return getattr(self._viewer, "_display_arr", None)

    def _roi(self, roi_id: str | None):
        roi_set = self._roi_set()
        return roi_set.get(roi_id) if roi_set is not None and roi_id else None

    def _selected_or_active_roi_id(self) -> str | None:
        return self._selected_or_active_roi_context().roi_id

    def _active_line_roi_id(self) -> str | None:
        if hasattr(self._viewer, "_active_line_roi_id"):
            return self._viewer._active_line_roi_id()
        return active_line_roi_context(self._roi_set()).roi_id

    def _selected_or_active_roi_context(self):
        if hasattr(self._viewer, "_selected_or_active_image_roi_id"):
            roi_id = self._viewer._selected_or_active_image_roi_id()
            return ROIContext(
                roi_id=roi_id,
                roi=self._roi(roi_id),
                source="viewer" if roi_id else "none",
            )
        return selected_or_active_roi_context(self._roi_set(), self._roi_dock())

    def _selected_or_active_area_roi_context(self):
        return selected_or_active_area_roi_context(self._roi_set(), self._roi_dock())

    def _selected_area_rois(self):
        return selected_area_roi_contexts(self._roi_set(), self._roi_dock())

    def _roi_set(self):
        return getattr(self._viewer, "_image_roi_set", None)

    def _roi_dock(self):
        return getattr(self._viewer, "_roi_dock", None)

    def _set_status(self, message: str) -> None:
        status = getattr(self._viewer, "_status_lbl", None)
        if status is not None and hasattr(status, "setText"):
            status.setText(str(message))
        if self._feature_panel is not None:
            self._feature_panel.show_message(str(message))
        if self._point_mask_panel is not None and hasattr(self._point_mask_panel, "show_message"):
            self._point_mask_panel.show_message(str(message))

    def _sync_actions(self) -> None:
        if hasattr(self._viewer, "_sync_viewer_menu_actions"):
            self._viewer._sync_viewer_menu_actions()
