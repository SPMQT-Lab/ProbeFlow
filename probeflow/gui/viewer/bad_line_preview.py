"""Bad-line detection preview controller extracted from ImageViewerDialog."""

from __future__ import annotations

from typing import Callable


class BadLinePreviewController:
    """Owns the transient bad-line detection preview state.

    Drives the canvas segment overlay and the processing-panel summary label.
    All caller-visible feedback (status-bar text) is returned as strings so
    the controller carries no dependency on the dialog's status label.
    """

    def __init__(
        self,
        zoom_lbl,
        processing_panel,
        get_arr_fn: Callable,
    ) -> None:
        self._zoom_lbl = zoom_lbl
        self._processing_panel = processing_panel
        self._get_arr = get_arr_fn
        self._active: bool = False
        self._segments: list = []

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Public API ────────────────────────────────────────────────────────────

    def on_settings_changed(self) -> str | None:
        """Handle a parameter change in the bad-line panel.

        Re-runs detection when a preview is already showing.  Returns a status
        message when a re-run happened, otherwise ``None``.
        """
        if self._active:
            return self.run()
        method = self._processing_panel.bad_line_method()
        if method is None:
            self._processing_panel.set_bad_line_preview_summary("Preview: select a method")
        else:
            self._processing_panel.set_bad_line_preview_summary("Preview: run detection")
        if hasattr(self._zoom_lbl, "clear_bad_segment_overlay"):
            self._zoom_lbl.clear_bad_segment_overlay()
        return None

    def run(self) -> str:
        """Run bad-line detection and update the canvas overlay.

        Returns a summary string for the caller to display in the status bar.
        Returns ``""`` on early exit (no image, no method, or error).
        """
        arr = self._get_arr()
        if arr is None:
            self._processing_panel.set_bad_line_preview_summary("Preview: no image")
            return ""
        method = self._processing_panel.bad_line_method()
        if method is None:
            self.clear("Preview: select a method")
            return ""
        try:
            from probeflow.processing import (
                detect_bad_scanline_segments,
                repair_bad_scanline_segments,
            )
            segments = detect_bad_scanline_segments(
                arr,
                threshold=self._processing_panel.bad_line_threshold(),
                method=method,
                polarity=self._processing_panel.bad_line_polarity(),
                min_segment_length_px=(
                    self._processing_panel.bad_line_min_segment_length_px()
                ),
                max_adjacent_bad_lines=(
                    self._processing_panel.bad_line_max_adjacent_bad_lines()
                ),
            )
            _, preview_info = repair_bad_scanline_segments(
                arr,
                segments,
                max_adjacent_bad_lines=(
                    self._processing_panel.bad_line_max_adjacent_bad_lines()
                ),
                threshold=self._processing_panel.bad_line_threshold(),
                polarity=self._processing_panel.bad_line_polarity(),
                min_segment_length_px=(
                    self._processing_panel.bad_line_min_segment_length_px()
                ),
            )
        except Exception as exc:
            self.clear(f"Preview error: {exc}")
            return ""

        self._segments = list(segments)
        self._active = True
        if hasattr(self._zoom_lbl, "set_bad_segment_overlay"):
            self._zoom_lbl.set_bad_segment_overlay(segments)

        n = len(segments)
        n_lines = len({seg.line_index for seg in segments})
        skipped = len(preview_info.skipped_segments)
        skipped_lines = len({seg.line_index for seg in preview_info.skipped_segments})

        if n == 0:
            summary = "Detected 0 segments"
        elif n == 1:
            summary = "Detected 1 segment on 1 scan line"
        else:
            summary = f"Detected {n} segments across {n_lines} scan lines"
        if skipped:
            summary += (
                f"; skipped {skipped} segment{'s' if skipped != 1 else ''} "
                f"across {skipped_lines} line{'s' if skipped_lines != 1 else ''} "
                "because the adjacent-line limit was exceeded"
            )
        self._processing_panel.set_bad_line_preview_summary(summary)
        return summary

    def clear(self, summary: str = "Preview: not run") -> None:
        """Reset preview state and clear the canvas overlay."""
        self._active = False
        self._segments = []
        self._processing_panel.set_bad_line_preview_summary(summary)
        if hasattr(self._zoom_lbl, "clear_bad_segment_overlay"):
            self._zoom_lbl.clear_bad_segment_overlay()
