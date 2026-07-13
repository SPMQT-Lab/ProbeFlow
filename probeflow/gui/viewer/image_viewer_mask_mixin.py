"""Active-mask layer orchestration for ImageViewerDialog.

Mirrors :class:`probeflow.gui.viewer.image_viewer_roi_mixin.ImageViewerRoiMixin`:
the viewer owns a :class:`~probeflow.core.mask.MaskSet` per image, persisted to a
``<stem>.masks.json`` sidecar, with one active mask shown as a canvas overlay
and consumed by downstream statistics / background exclusion.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from probeflow.gui.roi_context import area_roi_mask
from probeflow.gui.viewer import load_mask_set, save_mask_set
from probeflow.gui.viewer.geometric_ops import processing_changes_coordinate_frame

_log = logging.getLogger(__name__)


class ImageViewerMaskMixin:
    # ── Image-level mask set ──────────────────────────────────────────────────

    def _load_image_mask_set(self, entry) -> None:
        """Load masks from the ``<stem>.masks.json`` sidecar, else an empty set."""
        self._image_mask_set, err = load_mask_set(entry.path)
        self._raw_image_mask_payload = copy.deepcopy(self._image_mask_set.to_dict())
        self._processed_mask_edits_pending = False
        if err and hasattr(self, "_status_lbl"):
            # Mirror of the ROI loader: a corrupt sidecar must be surfaced,
            # not read as "no masks".
            self._status_lbl.setText(err)
        self._refresh_image_mask_set_ui()

    def _save_image_mask_set(self) -> None:
        if self._image_mask_set is None:
            return
        if processing_changes_coordinate_frame(getattr(self, "_processing", {})):
            self._processed_mask_edits_pending = True
            warn = getattr(self, "_warn_processed_overlay_edit", None)
            if callable(warn):
                warn("mask")
            return
        entry = self._entries[self._idx]
        err = save_mask_set(self._image_mask_set, entry.path)
        if err and hasattr(self, "_status_lbl"):
            self._status_lbl.setText(err)
        elif err is None:
            self._raw_image_mask_payload = copy.deepcopy(self._image_mask_set.to_dict())
            self._processed_mask_edits_pending = False

    def _refresh_image_mask_set_ui(self) -> None:
        """Refresh the overlay/manager and re-run mask-aware display."""
        self._refresh_mask_overlay()
        if hasattr(self, "_mask_panel"):
            self._mask_panel.refresh(self._image_mask_set)
        # The active mask restricts statistics, so refresh the readout.
        if hasattr(self, "_refresh_measurements"):
            try:
                self._refresh_measurements()
            except Exception:
                _log.warning("Measurement refresh after mask change failed; "
                             "readout may show stale statistics", exc_info=True)

    def _on_image_mask_set_changed(self) -> None:
        """Refresh the live mask model and persist it when in the raw frame."""
        self._refresh_image_mask_set_ui()
        self._save_image_mask_set()

    # ── Active mask access (consumed by stats / background) ────────────────────

    def _active_mask_array(self) -> "np.ndarray | None":
        """Return the active mask's boolean array, or None.

        Returns None when the mask shape no longer matches the displayed array
        (e.g. after a shape-changing processing step) so callers fall back to
        the whole image rather than crashing.
        """
        ms = getattr(self, "_image_mask_set", None)
        if ms is None:
            return None
        mask = ms.active()
        if mask is None:
            return None
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is not None and mask.shape != arr.shape[:2]:
            return None
        return mask.data

    def _refresh_mask_overlay(self) -> None:
        """Show the active mask as a canvas overlay, or clear it."""
        if not hasattr(self, "_zoom_lbl"):
            return
        data = self._active_mask_array()
        if data is None or not data.any():
            self._zoom_lbl.clear_mask_overlay()
        else:
            self._zoom_lbl.set_mask_overlay(data)
        self._warn_if_mask_channel_mismatch()

    def _warn_if_mask_channel_mismatch(self) -> None:
        """Non-blocking warning when the active mask was made on another channel.

        The mask still applies (shape is the only hard requirement), but a
        same-shape mask from a different channel/processing state is
        semantically stale — surface that rather than applying it silently.
        """
        ms = getattr(self, "_image_mask_set", None)
        mask = ms.active() if ms is not None else None
        if mask is None or not hasattr(self, "_status_lbl"):
            return
        recorded = mask.parameters.get("source_channel")
        current = self._edge_source_channel()
        if recorded and current and recorded != current:
            self._status_lbl.setText(
                f"Note: active mask “{mask.name}” was made on channel "
                f"“{recorded}”, now viewing “{current}”."
            )

    # ── Advanced Edge Detection dialog ─────────────────────────────────────────

    def _on_open_advanced_edge(self) -> None:
        from probeflow.gui.dialogs.edge_detection import EdgeDetectionDialog

        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("Advanced Edge Detection: no image loaded.")
            return
        roi_mask = area_roi_mask(self._active_image_roi(), arr.shape[:2])
        px_x_m, px_y_m = self._processing_pixel_sizes_m()
        px_x_nm = px_x_m * 1e9 if px_x_m else None
        px_y_nm = px_y_m * 1e9 if px_y_m else None

        dlg = EdgeDetectionDialog(
            arr,
            theme=self._t,
            pixel_size_nm=px_x_nm,
            pixel_size_x_nm=px_x_nm,
            pixel_size_y_nm=px_y_nm,
            active_roi_mask=roi_mask,
            source_channel=self._edge_source_channel(),
            parent=self,
        )
        dlg.overlay_requested.connect(self._on_edge_overlay_requested)
        dlg.overlay_cleared.connect(self._refresh_mask_overlay)
        dlg.mask_created.connect(self._on_edge_mask_created)
        dlg.rois_created.connect(self._on_edge_rois_created)
        dlg.image_created.connect(self._on_edge_image_created)
        self._edge_detection_dialog = dlg
        self._present_modal_tool(dlg)

    # ── Dialog output handlers ──────────────────────────────────────────────────

    def _on_edge_overlay_requested(self, result) -> None:
        mask = getattr(result, "edge_mask", None)
        if mask is None or not np.asarray(mask).any():
            self._status_lbl.setText(
                "Overlay: no binary edges to show (enable thresholding for Sobel/Scharr)."
            )
            return
        # Transient preview overlay (cyan) distinct from the active mask (red).
        self._zoom_lbl.set_mask_overlay(mask, color=(0, 229, 255), alpha=120)

    def _on_edge_mask_created(self, image_mask) -> None:
        from probeflow.core.mask import MaskSet
        entry = self._entries[self._idx]
        if self._image_mask_set is None:
            self._image_mask_set = MaskSet(image_id=entry.path.name)
        # Record source context so a mask made from a processed channel is not
        # mistaken for raw-data-derived later (it only matches by shape).
        image_mask.parameters.setdefault("source_path", entry.path.name)
        image_mask.parameters.setdefault("source_channel", self._edge_source_channel())
        image_mask.parameters.setdefault("data_basis", "processed_image")
        self._image_mask_set.add(image_mask)
        self._image_mask_set.set_active(image_mask.id)
        self._on_image_mask_set_changed()
        self._status_lbl.setText(f"Created active mask “{image_mask.name}”.")

    def _edge_source_channel(self) -> "str | None":
        try:
            _scale, _unit, axis_label = self._channel_unit()
        except Exception:
            return None
        return axis_label or (self._ch_cb.currentText() if hasattr(self, "_ch_cb") else None)

    def _on_edge_rois_created(self, rois) -> None:
        if self._image_roi_set is None or not rois:
            return
        for roi in rois:
            self._image_roi_set.add(roi)
        self._image_roi_set.set_active(rois[-1].id)
        self._on_image_roi_set_changed()
        self._status_lbl.setText(f"Added {len(rois)} ROI(s) from edge mask.")

    # ── Masks-manager callbacks ────────────────────────────────────────────────

    def _convert_mask_to_rois(self, mask_id: str) -> None:
        from probeflow.core.roi import roi_from_mask

        ms = getattr(self, "_image_mask_set", None)
        mask = ms.get(mask_id) if ms is not None else None
        if mask is None:
            return
        rois = roi_from_mask(mask.data, min_size_px=0, name_prefix=mask.name)
        if not rois:
            self._status_lbl.setText("No closed regions in mask to convert to ROI(s).")
            return
        self._on_edge_rois_created(rois)

    def _add_active_mask_stats(self) -> None:
        if hasattr(self, "_image_measurements"):
            self._image_measurements.add_active_mask_stats_measurement()

    def _export_mask_to_file(self, mask_id: str) -> None:
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

        from probeflow.io.mask_sidecar import save_mask_set_sidecar
        from probeflow.core.mask import MaskSet

        ms = getattr(self, "_image_mask_set", None)
        mask = ms.get(mask_id) if ms is not None else None
        if mask is None:
            return
        default = f"{Path(self._entries[self._idx].path).stem}.{mask.name}.masks.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export mask", default, "Mask JSON (*.masks.json *.json)"
        )
        if not path:
            return
        single = MaskSet(image_id=ms.image_id)
        single.add(mask)
        single.set_active(mask.id)
        try:
            save_mask_set_sidecar(single, path, sidecar=path)
            self._status_lbl.setText(f"Exported mask to {path}.")
        except Exception as exc:
            self._status_lbl.setText(f"Could not export mask: {exc}")

    def _on_edge_image_created(self, arr, provenance) -> None:
        try:
            from probeflow.gui.dialogs.array_image import ArrayImageDialog
            scan_range = self._display_scan_range_m or self._scan_range_m or (1e-9, 1e-9)
            dlg = ArrayImageDialog(
                np.asarray(arr, dtype=float),
                scan_range_m=tuple(scan_range),
                title="Edge detection result",
                colormap=getattr(self, "_viewer_colormap", self._colormap),
                theme=self._t,
                provenance=provenance,
                parent=self,
            )
        except Exception as exc:
            self._status_lbl.setText(f"Could not open edge result image: {exc}")
            return
        self._track_modeless_child(dlg)
        dlg.show()
        self._status_lbl.setText("Opened edge detection result image.")
