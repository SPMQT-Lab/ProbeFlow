"""Set-zero-plane controller extracted from ImageViewerDialog."""

from __future__ import annotations

from typing import Callable

import numpy as np


class SetZeroPlaneController:
    """Manages the interactive 3-point zero-plane levelling mode.

    The controller owns the transient pick state (``_points_px``,
    ``_markers_hidden``) and drives the canvas marker overlay.  All mutations
    to the processing dict and the full re-render are delegated back to the
    viewer via callables so this class carries no direct dependency on
    ``ImageViewerDialog``.
    """

    def __init__(self, zoom_lbl):
        self._zoom_lbl = zoom_lbl
        self._points_px: list[tuple[int, int]] = []
        self._markers_hidden: bool = False

    # ── Public state ──────────────────────────────────────────────────────────

    @property
    def points(self) -> list[tuple[int, int]]:
        return list(self._points_px)

    def restore_points(self, points: list[tuple[int, int]]) -> None:
        """Restore pick points from a saved processing state (e.g. on undo)."""
        self._points_px = list(points)
        self._markers_hidden = False

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def toggle(
        self,
        enabled: bool,
        set_selection_tool_fn: Callable[[str], None],
    ) -> str:
        """Handle the Set Zero Plane button being toggled.

        Returns a status-bar message for the caller to display.
        """
        if enabled:
            set_selection_tool_fn("none")
            self._points_px = []
            self._markers_hidden = False
            return "Click 3 reference points to define the zero plane."

        if len(self._points_px) < 3:
            self._points_px = []
            self._zoom_lbl.set_zero_markers([])
        return ""

    # ── Canvas pick ───────────────────────────────────────────────────────────

    def on_canvas_pick(
        self,
        frac_x: float,
        frac_y: float,
        display_arr: np.ndarray | None,
        processing: dict,
        mode_btn_checked: bool,
    ) -> tuple[bool, str]:
        """Handle an image click while zero-plane mode is active.

        ``display_arr`` is the array the user is actually clicking on (the
        displayed, possibly processed one): the click fraction is converted
        to pixel coordinates in *that* frame, and the completed pick set is
        stamped with the current geometric-op count so replay applies the
        zero plane in the same frame (2026-06-12 workflow review: mapping
        fractions onto the raw shape anchored the plane at the mirrored /
        wrong feature once a flip or rotation was in the pipeline, while the
        markers — drawn from the same fractions — still showed the clicked
        spots).

        Returns ``(trigger_rerender, status_message)``.  When
        ``trigger_rerender`` is ``True`` the caller should call
        ``_refresh_processing_display()`` and un-toggle the mode button.
        """
        if display_arr is None:
            return False, ""

        Ny, Nx = display_arr.shape
        x_px = max(0, min(int(round(frac_x * (Nx - 1))), Nx - 1))
        y_px = max(0, min(int(round(frac_y * (Ny - 1))), Ny - 1))

        if not mode_btn_checked:
            return False, ""

        self._markers_hidden = False
        self._points_px.append((x_px, y_px))
        n = len(self._points_px)
        self.refresh_markers(display_arr, processing)

        if n < 3:
            return False, (
                f"Zero plane point {n}/3 set at ({x_px}, {y_px}); "
                f"click {3 - n} more."
            )

        processing["set_zero_plane_points"] = self._points_px[:3]
        processing["set_zero_patch"] = 1
        # Frame stamp: replay re-inserts the set-zero step after this many
        # geometric ops, the frame the pixel coordinates were picked in.
        processing["set_zero_after_geometric_ops"] = len(
            processing.get("geometric_ops") or ())
        processing.pop("set_zero_xy", None)
        return True, "Zero plane set from 3 reference points."

    # ── Marker refresh ────────────────────────────────────────────────────────

    def refresh_markers(self, display_arr: np.ndarray | None, processing: dict) -> None:
        """Push the current pick state into the canvas as zero markers.

        Fractions are computed against the displayed array's shape — the
        frame the points were picked in.
        """
        if display_arr is None or self._markers_hidden:
            self._zoom_lbl.set_zero_markers([])
            return

        Ny, Nx = display_arr.shape
        denom_x = max(1, Nx - 1)
        denom_y = max(1, Ny - 1)

        def _to_marker(pt, label: str) -> dict:
            x_px, y_px = pt
            return {
                "frac_x": float(x_px) / denom_x,
                "frac_y": float(y_px) / denom_y,
                "label": label,
            }

        markers: list[dict] = []
        if self._points_px:
            for i, pt in enumerate(self._points_px[:3]):
                markers.append(_to_marker(pt, str(i + 1)))
        elif processing.get("set_zero_plane_points"):
            for i, pt in enumerate(processing["set_zero_plane_points"][:3]):
                markers.append(_to_marker(pt, str(i + 1)))
        elif processing.get("set_zero_xy") is not None:
            markers.append(_to_marker(processing["set_zero_xy"], "0"))

        self._zoom_lbl.set_zero_markers(markers)

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self) -> str:
        """Hide all zero markers without modifying the processing dict.

        Returns a status-bar message for the caller to display.
        """
        self._points_px = []
        self._markers_hidden = True
        self._zoom_lbl.set_zero_markers([])
        return (
            "Zero-plane reference points hidden. "
            "Processing is unchanged; use Reset to original to undo leveling."
        )
