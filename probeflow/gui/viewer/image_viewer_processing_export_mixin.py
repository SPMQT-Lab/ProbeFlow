"""Processing, export, geometric-operation, and close handlers for ImageViewerDialog."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

from PySide6.QtWidgets import QFileDialog

from probeflow.gui.config import load_config, save_config
from probeflow.gui.roi_context import active_area_roi_context
from probeflow.gui.viewer import (
    export_line_profile,
    save_viewer_png,
    transform_mask_set_for_display_op,
    transform_roi_set_for_display_op,
)
from probeflow.gui.viewer.processed_export import (
    build_processed_scan_for_export,
    save_processed_image,
    save_provenance_json,
)
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.processing.state import (
    apply_processing_state,
    assert_roi_references_resolved,
)
from probeflow.provenance import build_export_record, display_lines


class ImageViewerProcessingExportMixin:
    def _on_apply_processing(self):
        panel_state = self._processing_panel.state()
        panel_state.update(self._advanced_processing_state())
        has_local_filter = self._processing_has_roi_aware_local_filter(panel_state)

        # Auto-scope: a quick selection wins; otherwise a named ROI explicitly
        # chosen as the filter scope; otherwise whole image.  The old
        # whole-image / ROI dropdown is retired.
        selection = self._active_quick_selection()
        roi_scope_id = None
        if selection is None:
            roi_scope_id = self._active_roi_filter_scope_id()
            if (
                roi_scope_id is not None
                and has_local_filter
                and active_area_roi_context(self._image_roi_set).roi_id != roi_scope_id
            ):
                self._status_lbl.setText(
                    "The ROI chosen as filter scope is not an area ROI; pick an "
                    "area ROI or clear the scope (Esc) to process the whole image."
                )
                return
        # Snapshot for undo before any mutation. Validation has passed; this
        # apply is going to change the state.
        self._push_proc_undo_snapshot()
        preserve = {
            key: self._processing[key]
            for key in (
                "set_zero_xy",
                "set_zero_plane_points",
                "set_zero_patch",
                "periodic_notches",
                "periodic_notch_radius",
                "geometric_ops",
                "arithmetic_ops",
                "stm_background",
                "plane_bg",
                # Durable scope-filter lists must survive the panel-state rebuild
                # so previously committed ROI/region/mask filters are not dropped.
                "roi_filter_ops",
                "mask_filter_ops",
            )
            if key in self._processing
        }
        self._processing = panel_state
        self._processing.update(preserve)
        if selection is not None:
            # Anonymous, frozen region step (quick selection → not a managed ROI).
            self._commit_region_scoped_filters(selection)
        elif roi_scope_id is not None:
            # Named ROI scope: durable, frozen-geometry steps.
            self._commit_roi_scoped_filters(roi_scope_id)
        # The single global ROI scope is no longer written (kept readable for
        # back-compat with previously saved states).
        self._processing.pop("processing_scope", None)
        self._processing.pop("processing_roi_id", None)
        self._clear_bad_line_preview()
        self._refresh_processing_display()

    def _active_roi_filter_scope_id(self) -> "str | None":
        """Return the ROI id explicitly set as filter scope, if it still exists."""
        rid = getattr(self, "_roi_filter_scope_id", None)
        if rid is None or self._image_roi_set is None:
            return None
        if self._image_roi_set.get(rid) is None:
            self._roi_filter_scope_id = None
            return None
        return rid

    # Secondary GUI keys that only matter alongside a committed filter's
    # trigger key; cleared on commit so the live panel does not re-emit them.
    _ROI_FILTER_COMPANION_KEYS = (
        "edge_sigma",
        "edge_sigma2",
        "fft_cutoff",
        "fft_window",
        "fft_soft_mode",
        "fft_soft_cutoff",
        "fft_soft_border_frac",
    )

    def _commit_roi_scoped_filters(self, area_roi_id: str) -> None:
        """Bake the panel's ROI-eligible filters into durable, frozen steps.

        Each committed entry snapshots the active area ROI's geometry so it
        rasterises independently of the live ROI.  The committed filters are
        then stripped from the live panel (and the panel widgets resynced) so a
        later Apply neither re-commits them nor emits a duplicate global step.
        """
        from probeflow.processing.gui_adapter import roi_eligible_filter_specs

        specs = roi_eligible_filter_specs(self._processing)
        if not specs:
            return
        area_roi = (
            self._image_roi_set.get(area_roi_id)
            if self._image_roi_set is not None else None
        )
        if area_roi is None:
            return
        frozen = {
            "kind": area_roi.kind,
            "geometry": dict(area_roi.geometry),
            "coord_system": area_roi.coord_system,
        }
        committed = list(self._processing.get("roi_filter_ops") or [])
        for op_name, params in specs:
            committed.append({
                "op": op_name,
                "params": params,
                "roi_id": area_roi_id,
                "frozen_geometry": frozen,
                # The geometry above is in the *current display* frame — after
                # every geometric op applied so far. Recording the count lets
                # replay re-insert this step at the same pipeline position
                # (review: scope-replay ordering).
                "after_geometric_ops": len(self._processing.get("geometric_ops") or []),
            })
        self._processing["roi_filter_ops"] = committed
        self._strip_committed_filter_keys()
        self._roi_status_lbl.setText(
            f"Committed {len(specs)} ROI-scoped filter(s) to "
            f"'{area_roi.name}' (frozen geometry)."
        )

    def _commit_region_scoped_filters(self, selection: dict) -> None:
        """Bake the panel's eligible filters into an anonymous frozen region step.

        The quick selection's geometry is frozen into each step (``scope_kind=
        "region"``, no ``roi_id``) so the region processes reproducibly without
        ever becoming a managed ROI.  The selection itself persists (ImageJ-style)
        so further filters can be applied to the same region.
        """
        from probeflow.processing.gui_adapter import roi_eligible_filter_specs

        specs = roi_eligible_filter_specs(self._processing)
        if not specs:
            return
        frozen = {
            "kind": str(selection["kind"]),
            "geometry": dict(selection["geometry"]),
            "coord_system": "pixel",
        }
        committed = list(self._processing.get("roi_filter_ops") or [])
        for op_name, params in specs:
            committed.append({
                "op": op_name,
                "params": params,
                "scope_kind": "region",
                "frozen_geometry": frozen,
                # Selection coordinates are in the *current display* frame;
                # see _commit_roi_scoped_filters.
                "after_geometric_ops": len(self._processing.get("geometric_ops") or []),
            })
        self._processing["roi_filter_ops"] = committed
        self._strip_committed_filter_keys()
        self._roi_status_lbl.setText(
            f"Committed {len(specs)} region filter(s) (frozen selection)."
        )

    def _strip_committed_filter_keys(self) -> None:
        """Remove just-committed panel filters and resync the panel widgets."""
        from probeflow.processing.gui_adapter import ROI_ELIGIBLE_FILTER_TRIGGER_KEYS

        for key in (*ROI_ELIGIBLE_FILTER_TRIGGER_KEYS, *self._ROI_FILTER_COMPANION_KEYS):
            self._processing.pop(key, None)
        self._processing_panel.set_state(self._processing)
        self._set_advanced_processing_state(self._processing)

    def _on_reset_processing(self):
        """Clear all processing for the current image and reload raw data."""
        has_zero = bool(self._zero_ctrl.points)
        if not self._processing and not has_zero:
            self._status_lbl.setText("Already showing the original — nothing to reset.")
            return
        # Snapshot for undo before clearing.
        self._push_proc_undo_snapshot()
        self._processing = {}
        self._processing_panel.set_state({})
        self._set_advanced_processing_state({})
        self._clear_bad_line_preview()
        # Untoggle any active set-zero pick modes so we don't re-pick on reload.
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        self._zero_ctrl.clear()
        self._set_selection_tool("none")
        self._clear_quick_selection()
        self._roi_filter_scope_id = None
        self._roi_status_lbl.setText("Processing scope: whole image")
        self._refresh_zero_markers()
        self._status_lbl.setText("Reset: showing original on-disk data.")
        self._refresh_processing_display()

    # ── Processing undo / redo ────────────────────────────────────────────────

    def _push_proc_undo_snapshot(self) -> None:
        self._proc_undo_ctrl.push(self._processing)

    def _restore_processing_state(self, state: dict) -> None:
        """Apply a snapshot to ``self._processing`` and resync the GUI."""
        self._processing = copy.deepcopy(state)
        self._processing_panel.set_state(self._processing)
        self._set_advanced_processing_state(self._processing)
        self._refresh_processing_display()

    def _on_undo_processing(self) -> None:
        state = self._proc_undo_ctrl.undo(self._processing)
        if state is None:
            return
        self._restore_processing_state(state)
        self._status_lbl.setText("Undo: restored previous processing.")

    def _on_redo_processing(self) -> None:
        state = self._proc_undo_ctrl.redo(self._processing)
        if state is None:
            return
        self._restore_processing_state(state)
        self._status_lbl.setText("Redo: reapplied processing.")

    def _update_undo_redo_buttons(self) -> None:
        if self._proc_undo_ctrl is not None:
            self._proc_undo_ctrl.update_buttons()

    def _current_export_array(self):
        return self._display_arr if self._display_arr is not None else self._raw_arr

    def _default_viewer_export_path(self, suffix: str) -> Path:
        entry = self._entries[self._idx]
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return Path.home() / f"{entry.stem}_viewer{suffix}"

    def _export_provenance_enabled(self) -> bool:
        checkbox = getattr(self, "_export_provenance_chk", None)
        return True if checkbox is None else bool(checkbox.isChecked())

    def _export_scalebar_enabled(self) -> bool:
        checkbox = getattr(self, "_export_scalebar_chk", None)
        return True if checkbox is None else bool(checkbox.isChecked())

    def _source_format_for_export(self) -> str:
        entry = self._entries[self._idx]
        return str(
            getattr(self, "_scan_format", "")
            or getattr(entry, "source_format", None)
            or ""
        ).lower()

    def _update_export_format_controls(self) -> None:
        sxm_btn = getattr(self, "_save_sxm_btn", None)
        if sxm_btn is None:
            return
        fmt = self._source_format_for_export()
        sxm_enabled = fmt in {"dat", "createc_dat"}
        sxm_btn.setEnabled(sxm_enabled)
        if sxm_enabled:
            sxm_btn.setToolTip("Export the current Createc .dat view as a Nanonis .sxm file.")
        else:
            sxm_btn.setToolTip(
                "SXM conversion from the viewer is intended for Createc .dat scans; "
                "SM4 and existing SXM sources should use PNG/PDF/GWY or the multi-format dialog."
            )

    def _export_precision_text(self) -> str:
        bits = None
        for op in (self._processing or {}).get("geometric_ops") or []:
            if not isinstance(op, dict) or op.get("op") != "quantize_bit_depth":
                continue
            try:
                bits = int((op.get("params") or {}).get("bits"))
            except (TypeError, ValueError):
                bits = None
        if bits is not None:
            return f"{bits}-bit quantized"
        arr = self._current_export_array()
        if arr is None:
            return "--"
        return f"{arr.dtype} data"

    @staticmethod
    def _format_export_bias(bias_mv) -> str:
        if bias_mv is None:
            return "--"
        try:
            bias_mv = float(bias_mv)
        except (TypeError, ValueError):
            return "--"
        if abs(bias_mv) >= 1000:
            return f"{bias_mv / 1000:.4g} V"
        return f"{bias_mv:.4g} mV"

    @staticmethod
    def _format_export_current(current_pa) -> str:
        if current_pa is None:
            return "--"
        try:
            current_pa = float(current_pa)
        except (TypeError, ValueError):
            return "--"
        if abs(current_pa) >= 1000:
            return f"{current_pa / 1000:.4g} nA"
        return f"{current_pa:.4g} pA"

    def _update_export_summary(self) -> None:
        if not hasattr(self, "_export_png_size_lbl"):
            return
        arr = self._current_export_array()
        if arr is None or getattr(arr, "ndim", 0) < 2:
            size_text = "--"
        else:
            h, w = arr.shape[:2]
            size_text = f"{int(w)}x{int(h)} px"

        entry = self._entries[self._idx]
        self._export_png_size_lbl.setText(size_text)
        filename = self._default_viewer_export_path(".png").name
        file_lbl = self._export_png_file_lbl
        if hasattr(file_lbl, "set_full_text"):
            file_lbl.set_full_text(filename)
        else:
            file_lbl.setText(filename)
            file_lbl.setToolTip(filename)
        self._export_bias_lbl.setText(self._format_export_bias(getattr(entry, "bias_mv", None)))
        self._export_current_lbl.setText(
            self._format_export_current(getattr(entry, "current_pa", None))
        )
        self._export_precision_lbl.setText(self._export_precision_text())
        self._update_export_format_controls()

    def _ask_save_path(self, title: str, default: str, file_filter: str) -> str:
        """Save-file dialog that keeps the viewer active afterwards.

        On macOS the native save sheet can hand focus to the parent browser
        window when it closes (even on Cancel); re-activating the viewer keeps
        the user where they were.
        """
        out_path, _ = QFileDialog.getSaveFileName(self, title, str(default), file_filter)
        self.raise_()
        self.activateWindow()
        return out_path

    def _on_save_png(self):
        entry = self._entries[self._idx]
        if not self._assert_exportable_processing():
            return
        out_path = self._ask_save_path(
            "Save PNG", str(self._default_viewer_export_path(".png")),
            "PNG images (*.png)")
        if not out_path:
            return
        arr = self._current_export_array()
        if arr is None:
            self._status_lbl.setText("No data to save.")
            return

        msg = save_viewer_png(
            arr, out_path, entry.path,
            self._viewer_colormap, self._clip_low, self._clip_high,
            self._drs, self._processing, self._image_roi_set,
            self._ch_cb.currentIndex(), self._ch_cb.currentText() or None,
            processing_history=(
                self._processing_history.to_dict()
                if self._processing_history is not None else None
            ),
            add_scalebar=self._export_scalebar_enabled(),
            include_provenance=self._export_provenance_enabled(),
            image_mask_set=getattr(self, "_image_mask_set", None),
        )
        if msg.startswith("Saved") and self._processing_history is not None:
            self._mark_history_export(out_path, export_parameters={"export_kind": "viewer_png"})
        self._status_lbl.setText(msg)

    def _processing_state_has_image_arithmetic_operand(self, state) -> bool:
        for step in state.steps:
            if (
                step.op == "arithmetic"
                and step.params.get("operand_type") == "image"
            ):
                return True
            if step.op == "roi":
                nested = step.params.get("step")
                if not isinstance(nested, dict):
                    continue
                params = nested.get("params", {})
                if (
                    nested.get("op") == "arithmetic"
                    and isinstance(params, dict)
                    and params.get("operand_type") == "image"
                ):
                    return True
        return False

    def _assert_exportable_processing(self) -> bool:
        if getattr(self, "_processing_roi_error", ""):
            self._status_lbl.setText(
                f"Cannot export while processing has stale ROI references. {self._processing_roi_error}"
            )
            return False
        if getattr(self, "_processing_error", ""):
            self._status_lbl.setText(f"Export blocked: {self._processing_error}")
            return False
        mask_set = getattr(self, "_image_mask_set", None)
        try:
            ps = processing_state_from_gui(self._processing or {})
            # Validate both ROI and mask scope references; a mask step that
            # points at a deleted mask must block export rather than silently
            # exporting an unscoped or stale result (review: mask export safety).
            assert_roi_references_resolved(ps, self._image_roi_set, mask_set)
        except ValueError as _roi_err:
            self._status_lbl.setText(f"Export blocked: {_roi_err}")
            return False
        if not self._assert_mask_scopes_current(ps):
            return False
        if (
            self._raw_arr is not None
            and self._processing_state_has_image_arithmetic_operand(ps)
        ):
            try:
                # Forward calibration (review image-proc #1) — preflight
                # must mirror what _refresh_display_array does so it
                # validates the same pipeline that will execute.
                psx, psy = self._processing_pixel_sizes_m()
                apply_processing_state(
                    self._raw_arr, ps, self._image_roi_set,
                    mask_set=mask_set,
                    pixel_size_x_m=psx, pixel_size_y_m=psy,
                )
            except Exception as exc:
                self._status_lbl.setText(f"Export blocked: Processing failed: {exc}")
                return False
        return True

    def _assert_mask_scopes_current(self, ps) -> bool:
        """Block export when a ``mask`` step's raster is stale (shape mismatch).

        Masks are persistent rasters: after a same-shape display op (h/v flip,
        180° rotate) an un-transformed mask would still match the array shape but
        sit over the wrong pixels.  The transform path keeps overlay masks in
        sync; here we additionally refuse export when a mask referenced by a
        processing step no longer matches the array shape it will rasterise
        against, rather than exporting a silently mislocated result.
        """
        mask_set = getattr(self, "_image_mask_set", None)
        arr = self._current_export_array()
        if mask_set is None or arr is None:
            return True
        target_shape = arr.shape[:2]
        for step in ps.steps:
            if step.op != "mask":
                continue
            # Frozen mask steps replay their own raster snapshot and never
            # consult the live mask, so they cannot go stale.
            if step.params.get("frozen_mask") is not None:
                continue
            mask_id = step.params.get("mask_id")
            if mask_id is None:
                continue
            mask = mask_set.get(str(mask_id))
            if mask is None and hasattr(mask_set, "get_by_name"):
                mask = mask_set.get_by_name(str(mask_id))
            if mask is None:
                continue  # already caught by assert_roi_references_resolved
            if mask.shape != target_shape:
                self._status_lbl.setText(
                    f"Export blocked: mask '{mask.name}' shape {mask.shape} no "
                    f"longer matches the image {target_shape}; re-create or "
                    "re-apply the mask before exporting."
                )
                return False
        return True

    def _current_display_settings(self) -> dict:
        from probeflow.provenance.export import png_display_state

        add_scalebar = self._export_scalebar_enabled()
        return png_display_state(
            self._drs,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            colormap=self._viewer_colormap,
            add_scalebar=add_scalebar,
            scalebar_unit="nm",
            scalebar_pos="bottom-right",
        )

    def _processed_scan_for_export(self):
        entry = self._entries[self._idx]
        arr = self._current_export_array()
        # Post-processing scan_range_m must reflect any shape-changing step
        # (rotate_arbitrary / shear / affine_lattice_correction expand the
        # canvas; without this the exported PNG scale bar and FFT k-axes
        # silently use the raw scan_range_m on a now-larger array — review
        # image-proc #4).
        return build_processed_scan_for_export(
            entry.path, self._ch_cb.currentIndex(), arr, self._processing or {},
            scan_range_m=self._processed_scan_range_m(),
        )

    def _processed_scan_range_m(self) -> tuple[float, float] | None:
        """Walk the current processing state to compute post-processing scan_range_m.

        Returns ``None`` if the raw scan calibration is unknown or there is no
        raw array to walk.  When the pipeline contains no shape-changing step
        the returned value equals the raw ``self._scan_range_m`` (modulo float
        coercion).
        """
        from probeflow.processing.state import (
            apply_processing_state_with_calibration,
        )

        raw_range = getattr(self, "_scan_range_m", None)
        if raw_range is None or self._raw_arr is None:
            return None
        try:
            state = processing_state_from_gui(self._processing or {})
        except Exception:
            return (float(raw_range[0]), float(raw_range[1]))
        if not state.steps:
            return (float(raw_range[0]), float(raw_range[1]))
        try:
            _, new_range = apply_processing_state_with_calibration(
                self._raw_arr, state, self._image_roi_set,
                mask_set=getattr(self, "_image_mask_set", None),
                scan_range_m=(float(raw_range[0]), float(raw_range[1])),
            )
        except Exception:
            return (float(raw_range[0]), float(raw_range[1]))
        return new_range

    def _on_save_processed_image(self):
        if not self._assert_exportable_processing():
            return
        entry = self._entries[self._idx]
        out_path = self._ask_save_path(
            "Save processed image",
            str(Path.home() / f"{entry.stem}_processed.sxm"),
            (
                "Supported images (*.sxm *.png *.csv *.pdf *.gwy);;"
                "Nanonis SXM (*.sxm);;PNG images (*.png);;"
                "CSV grids (*.csv);;PDF figures (*.pdf);;Gwyddion (*.gwy)"
            ),
        )
        if not out_path:
            return
        out = Path(out_path)
        if not out.suffix:
            out = out.with_suffix(".sxm")
        try:
            scan, plane_idx = self._processed_scan_for_export()
        except ValueError as exc:
            self._status_lbl.setText(str(exc))
            return
        include_provenance = self._export_provenance_enabled()
        msg = save_processed_image(
            scan, plane_idx, out,
            colormap=self._viewer_colormap,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            display_settings=(
                self._current_display_settings() if include_provenance else None
            ),
            roi_set=self._image_roi_set,
            mask_set=getattr(self, "_image_mask_set", None),
            processing_history=(
                self._processing_history.to_dict()
                if self._processing_history is not None else None
            ),
            include_provenance=include_provenance,
            add_scalebar=self._export_scalebar_enabled(),
        )
        self._status_lbl.setText(msg)

    def _on_save_current_view_as(self, suffix: str, title: str, file_filter: str) -> None:
        if not self._assert_exportable_processing():
            return
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        sxm_btn = getattr(self, "_save_sxm_btn", None)
        if suffix == ".sxm" and sxm_btn is not None and not sxm_btn.isEnabled():
            self._status_lbl.setText(
                "SXM export from the viewer is available for Createc .dat scans."
            )
            return
        out_path = self._ask_save_path(
            title,
            str(self._default_viewer_export_path(suffix)),
            file_filter,
        )
        if not out_path:
            return
        out = Path(out_path)
        if not out.suffix:
            out = out.with_suffix(suffix)
        try:
            scan, plane_idx = self._processed_scan_for_export()
        except ValueError as exc:
            self._status_lbl.setText(str(exc))
            return
        include_provenance = self._export_provenance_enabled()
        msg = save_processed_image(
            scan, plane_idx, out,
            colormap=self._viewer_colormap,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            display_settings=(
                self._current_display_settings() if include_provenance else None
            ),
            roi_set=self._image_roi_set,
            mask_set=getattr(self, "_image_mask_set", None),
            processing_history=(
                self._processing_history.to_dict()
                if self._processing_history is not None else None
            ),
            include_provenance=include_provenance,
            add_scalebar=self._export_scalebar_enabled(),
        )
        self._status_lbl.setText(msg)

    def _on_save_pdf(self):
        self._on_save_current_view_as(".pdf", "Save PDF", "PDF figures (*.pdf)")

    def _on_save_sxm(self):
        self._on_save_current_view_as(".sxm", "Save SXM", "Nanonis SXM (*.sxm)")

    def _on_save_gwy(self):
        self._on_save_current_view_as(".gwy", "Save GWY", "Gwyddion (*.gwy)")

    def _on_save_provenance(self):
        if not self._assert_exportable_processing():
            return
        if self._processing_history is None:
            self._status_lbl.setText("No provenance available to save.")
            return
        entry = self._entries[self._idx]
        out_path = self._ask_save_path(
            "Save provenance",
            str(Path.home() / f"{entry.stem}.probeflow.json"),
            "ProbeFlow provenance (*.probeflow.json *.json)",
        )
        if not out_path:
            return
        out = Path(out_path)
        if not out.suffix:
            out = out.with_suffix(".probeflow.json")
        try:
            msg, record = save_provenance_json(
                self._processing_history,
                out,
                display_settings=self._current_display_settings(),
            )
            self._last_export_record = record
            self._history_text.setText(
                "\n".join(display_lines(record.processing_history))
            )
            self._status_lbl.setText(msg)
        except Exception as exc:
            self._status_lbl.setText(f"Save provenance error: {exc}")

    def _mark_history_export(self, out_path: str, export_parameters: dict | None = None) -> None:
        try:
            record = build_export_record(
                self._processing_history,
                export_path=out_path,
                export_format="png",
                display_settings=self._current_display_settings(),
                export_parameters=export_parameters,
            )
            self._last_export_record = record
            self._history_text.setText(
                "\n".join(display_lines(record.processing_history))
            )
        except Exception:
            _log.warning("Could not build export provenance record; history "
                         "panel not updated", exc_info=True)

    def _on_send_to_features(self):
        self._deferred.action = "features"
        self._deferred.plane_idx = self._ch_cb.currentIndex()
        # Emit immediately so the parent can load data without closing the viewer.
        # The viewer stays open; the parent clears _deferred so _on_closed won't fire it again.
        self.immediate_action_requested.emit("features")

    def _on_send_to_tv(self):
        self._deferred.action = "tv"
        self._deferred.plane_idx = self._ch_cb.currentIndex()
        self.immediate_action_requested.emit("tv")

    def _on_image_context_menu(self, pos):
        from probeflow.gui.viewer.context_menus import build_blank_image_context_menu

        menu = build_blank_image_context_menu(self)
        menu.exec(pos)

    def _on_threshold(self) -> None:
        """Open the Threshold dialog (modeless) for the current image."""
        from probeflow.gui.dialogs.threshold_dialog import ThresholdDialog
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return

        # Snapshot of the pipeline-output array at dialog-open time.
        # _preview must restore _display_arr to this value immediately after
        # submitting the render job so that other concurrent modeless operations
        # (Scale, Shear, etc.) read the real pipeline state, not a transient
        # preview result.
        _pipeline_arr = arr

        def _preview(result_arr):
            # Temporarily install the preview array so _refresh_viewer_pixmap
            # reads its range for vmin/vmax and passes it to the ViewerLoader.
            self._display_arr = result_arr
            self._refresh_viewer_pixmap(reset_zoom=False)
            # Restore immediately — ViewerLoader captured its own reference to
            # result_arr at construction; restoring here is safe and prevents
            # other handlers from seeing the transient preview data.
            self._display_arr = _pipeline_arr

        def _preview_pixmap(pixmap):
            # Coloured highlight preview — set pixmap directly, bypass async loader.
            self._zoom_lbl.setText("")
            self._zoom_lbl.set_source(pixmap, reset_zoom=False)

        def _clear_preview():
            # Restore the full processing-pipeline display.
            self._refresh_processing_display()

        dlg = ThresholdDialog(
            arr,
            preview_fn=_preview,
            preview_pixmap_fn=_preview_pixmap,
            clear_preview_fn=_clear_preview,
            theme=getattr(self, "_t", None),
            parent=self,
        )
        dlg.applied.connect(self._on_threshold_applied)
        # Track via the shared modeless-child registry (review gui-arch #22)
        # so closeEvent closes the dialog before the viewer tears down,
        # preventing queued HistogramPanel signals from firing into
        # partially-destroyed viewer widgets.
        self._threshold_dialog = dlg
        self._present_modal_tool(dlg)

    def _on_threshold_applied(self, params: dict) -> None:
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "image_threshold", "params": params})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        mode = params.get("mode", "clip")
        self._status_lbl.setText(f"Threshold applied ({mode} mode).")

    def _on_scale_image(self) -> None:
        """Open the Scale dialog to resample the image to new pixel dimensions."""
        from probeflow.gui.dialogs.scale_dialog import ScaleDialog
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        dlg = ScaleDialog(
            arr.shape,
            scan_range_m=self._scan_range_m,
            parent=self,
        )
        dlg.applied.connect(self._on_scale_image_applied)
        dlg.exec()

    def _on_scale_image_applied(self, params: dict) -> None:
        # ROI vector geometry scales exactly with the resample; raster masks
        # cannot (invalidated with a status message) and the quick selection
        # scales like the ROIs — same path as flips/rotations, which used to
        # bypass scale entirely and leave every overlay silently mislocated.
        self._transform_image_roi_set_for_display_op("scale_image", params)
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "scale_image", "params": params})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        w, h = params["new_width"], params["new_height"]
        self._status_lbl.setText(f"Scaled to {w} × {h} px (ROIs rescaled to match).")

    def _on_shear(self) -> None:
        """Open the Shear dialog to apply a 2-component shear correction."""
        from probeflow.gui.dialogs.shear_dialog import ShearDialog
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        dlg = ShearDialog(parent=self)
        dlg.applied.connect(self._on_shear_applied)
        dlg.exec()

    def _on_shear_applied(self, params: dict) -> None:
        # Shear invalidates ROIs, masks, and the quick selection (rectangles/
        # ellipses cannot represent the sheared shape) — same policy as
        # rotate_arbitrary. Previously nothing was transformed or even warned
        # about, leaving stale overlays over the sheared image.
        roi_set = self._image_roi_set
        mask_set = getattr(self, "_image_mask_set", None)
        n_overlays = (
            (len(roi_set.rois) if roi_set is not None else 0)
            + (len(mask_set.masks) if mask_set is not None else 0)
        )
        self._transform_image_roi_set_for_display_op("shear", params)
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "shear", "params": params})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        sx, sy = params.get("shear_x", 0.0), params.get("shear_y", 0.0)
        msg = f"Shear applied (x={sx:.4f}, y={sy:.4f})."
        if n_overlays:
            msg += f" {n_overlays} ROI(s)/mask(s) removed (cannot be sheared)."
        self._status_lbl.setText(msg)

    def _on_convert_bit_depth(self, bits: int) -> None:
        """Quantize the current image to *bits*-bit precision as a processing step."""
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "quantize_bit_depth", "params": {"bits": bits}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        n = 2 ** bits
        self._status_lbl.setText(f"Converted to {bits}-bit ({n:,} levels).")

    def _on_geometric_op(self, op_name: str) -> None:
        self._transform_image_roi_set_for_display_op(op_name)
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": op_name, "params": {}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()

    def _on_rotate_arbitrary(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        angle, ok = QInputDialog.getDouble(
            self, "Rotate Arbitrary",
            "Angle (degrees, positive = counter-clockwise):",
            0.0, -360.0, 360.0, 1,
        )
        if not ok:
            return
        self._transform_image_roi_set_for_display_op(
            "rotate_arbitrary",
            {"angle_degrees": angle},
        )
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "rotate_arbitrary", "params": {"angle_degrees": angle}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()

    def _transform_image_roi_set_for_display_op(
        self,
        op_name: str,
        params: dict | None = None,
    ) -> None:
        status_fn = (
            self._status_lbl.setText if hasattr(self, "_status_lbl") else None
        )
        array_shape = self._current_array_shape()
        transform_roi_set_for_display_op(
            self._image_roi_set,
            op_name,
            params,
            array_shape,
            status_fn=status_fn,
            roi_changed_fn=self._on_image_roi_set_changed,
        )
        # Masks are rasters and must follow the same geometric op, or a
        # same-shape flip/180°-rotate would leave them silently stale.
        mask_set = getattr(self, "_image_mask_set", None)
        mask_changed_fn = getattr(self, "_on_image_mask_set_changed", None)
        if mask_set is not None and mask_changed_fn is not None:
            transform_mask_set_for_display_op(
                mask_set,
                op_name,
                params,
                array_shape,
                status_fn=status_fn,
                mask_changed_fn=mask_changed_fn,
            )
        # The quick selection is geometry too — transform it (or drop it if the
        # op invalidates it) so the marquee never sits over the wrong pixels.
        if hasattr(self, "_transform_quick_selection_for_display_op"):
            self._transform_quick_selection_for_display_op(op_name, params)

    def _on_export_line_profile_csv(self):
        prof = self._line_profile_panel.profile_data()
        if prof is None:
            self._status_lbl.setText("No line profile to export (draw a line first).")
            return
        x_vals, y_vals, x_label, y_label = prof
        entry = self._entries[self._idx]
        ok, msg = export_line_profile(
            x_vals, y_vals, x_label, y_label,
            entry.stem,
            self._scan_header or {},
            parent=self,
        )
        if msg:
            self._status_lbl.setText(msg)

    def _track_modeless_child(self, dlg) -> None:
        """Register a modeless child dialog so :meth:`closeEvent` can close it.

        Without this, child dialogs created via ``dlg.show()`` outlive the
        viewer (Qt holds them as parented children until garbage collection)
        and their queued signal handlers can fire on a partially-destroyed
        viewer — the f6aac4a class of RuntimeError (review gui-arch #22).

        The dialog is removed from the list automatically when Qt destroys
        it (via the ``destroyed`` signal) so the list does not accumulate
        dangling references over the viewer's lifetime.
        """
        if dlg is None:
            return
        try:
            self._modeless_children.append(dlg)
        except AttributeError:
            self._modeless_children = [dlg]
        try:
            dlg.destroyed.connect(
                lambda _obj=None, _dlg=dlg: self._untrack_modeless_child(_dlg)
            )
        except Exception:
            _log.warning("Could not connect destroyed() for modeless child "
                         "tracking", exc_info=True)

    def _untrack_modeless_child(self, dlg) -> None:
        children = getattr(self, "_modeless_children", None)
        if not children:
            return
        try:
            children.remove(dlg)
        except ValueError:
            pass

    def _close_modeless_children(self) -> None:
        """Close every tracked modeless child dialog.

        Tolerates already-deleted C++ objects (``RuntimeError`` from PySide6
        when the underlying QWidget was already deallocated) and dialogs
        whose ``close()`` raises — we want best-effort teardown, not a
        cascade of exceptions during viewer shutdown.
        """
        children = list(getattr(self, "_modeless_children", ()) or ())
        for dlg in children:
            if dlg is None:
                continue
            try:
                visible = dlg.isVisible()
            except RuntimeError:
                # Underlying QWidget already destroyed.
                continue
            except Exception:
                continue
            if not visible:
                continue
            try:
                dlg.close()
            except Exception:
                pass

    def closeEvent(self, event):
        try:
            cfg = load_config()
            self._save_viewer_desktop_layout_into(cfg)
            save_config(cfg)
        except Exception:
            pass
        # Invalidate the in-flight worker token so any pending loaded() signal
        # is dropped rather than delivered to widgets that are being torn down.
        self._token = object()
        # Close every modeless child dialog tracked via _track_modeless_child;
        # they hold references to self / the currently displayed Scan and
        # their queued signal handlers must not fire on partially-destroyed
        # viewer widgets (review gui-arch #22).
        self._close_modeless_children()
        super().closeEvent(event)
