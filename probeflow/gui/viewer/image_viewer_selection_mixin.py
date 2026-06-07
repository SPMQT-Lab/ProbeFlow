"""Quick-selection (ImageJ-style ephemeral region) handlers for ImageViewerDialog.

A *quick selection* is a single, unnamed, provenance-light region drawn with the
area tools (rectangle/ellipse/polygon/freehand).  It is **not** a managed ROI:
it is not added to the ROISet, not persisted to a ``.rois.json`` sidecar, and
carries no on-image name label.  Processing applied inside it is still recorded
(as an anonymous, frozen ``scope_kind="region"`` step) so the image stays
reproducible — see ``_commit_region_scoped_filters`` in the processing mixin.

The canvas (:class:`probeflow.gui.image_canvas.ImageCanvas`) owns the selection
state and marquee; this mixin reacts to its signals and provides the
promote / clear / transform lifecycle.
"""

from __future__ import annotations


class ImageViewerSelectionMixin:
    # ── Canvas signal handlers ────────────────────────────────────────────────

    def _on_canvas_selection_drawn(self, kind: str, geometry: dict) -> None:
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(
                f"Quick {kind} selection — Apply processes inside it. "
                "Right-click → Promote to ROI to keep it."
            )
        self._sync_viewer_menu_actions()

    def _on_canvas_selection_cleared(self) -> None:
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText("Selection cleared.")
        self._sync_viewer_menu_actions()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def _active_quick_selection(self) -> "dict | None":
        """Return the current quick selection ``{"kind", "geometry"}`` or None.

        The canvas is the single source of truth for the selection.
        """
        zoom = getattr(self, "_zoom_lbl", None)
        if zoom is None or not hasattr(zoom, "selection"):
            return None
        return zoom.selection()

    def _has_quick_selection(self) -> bool:
        return self._active_quick_selection() is not None

    def _clear_quick_selection(self) -> None:
        zoom = getattr(self, "_zoom_lbl", None)
        if zoom is not None and hasattr(zoom, "clear_selection"):
            zoom.clear_selection(emit=False)

    # ── Promote to a managed ROI ──────────────────────────────────────────────

    def _promote_selection_to_roi(self) -> None:
        """Turn the current quick selection into a managed, named ROI."""
        sel = self._active_quick_selection()
        if sel is None:
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText("No selection to promote — draw a region first.")
            return
        from probeflow.core.roi import ROI
        from probeflow.gui.viewer import roi_canvas_created

        roi = ROI.new(sel["kind"], dict(sel["geometry"]))
        roi_canvas_created(
            self._image_roi_set, roi,
            self._on_image_roi_set_changed, self._set_drawing_tool,
        )
        self._clear_quick_selection()
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(f"Promoted selection to ROI '{roi.name}'.")
        self._sync_viewer_menu_actions()

    # ── Display-op lifecycle ──────────────────────────────────────────────────

    def _transform_quick_selection_for_display_op(
        self, op_name: str, params: dict | None = None,
    ) -> None:
        """Transform the quick selection alongside a geometric display op.

        Reuses :meth:`probeflow.core.roi.ROI.transform`; drops the selection
        when the op invalidates it (e.g. ``rotate_arbitrary``).
        """
        sel = self._active_quick_selection()
        if sel is None:
            return
        shape = self._current_array_shape() if hasattr(self, "_current_array_shape") else None
        if shape is None:
            return
        from probeflow.core.roi import ROI

        tmp = ROI(id="selection", name="selection",
                  kind=sel["kind"], geometry=dict(sel["geometry"]))
        try:
            transformed = tmp.transform(op_name, params or {}, shape)
        except Exception:
            transformed = None
        if transformed is None:
            self._clear_quick_selection()
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(f"{op_name} cleared the quick selection.")
            return
        self._zoom_lbl.set_selection(transformed.kind, dict(transformed.geometry))
