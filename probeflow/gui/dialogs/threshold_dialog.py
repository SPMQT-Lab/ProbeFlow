"""Threshold dialog — clip or binarize a scan plane by height value.

Provides an ImageJ-style histogram with two draggable lines (lower / upper
bounds) so the user can set the threshold visually rather than by typing
raw SI values.  An optional highlight colour overlays out-of-range pixels
in the preview so the user can see exactly what would be clipped.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


def _display_range_from_finite(finite: np.ndarray) -> tuple[float, float]:
    if finite.size == 0:
        return 0.0, 1.0
    raw_min = float(finite.min())
    raw_max = float(finite.max())
    display_min = float(np.percentile(finite, 1.0))
    display_max = float(np.percentile(finite, 99.0))
    if display_min >= display_max:
        return raw_min, raw_max
    return display_min, display_max


# ── Scientific-notation spinbox ───────────────────────────────────────────────

class _SciSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that stores full float64 precision and displays as ``:.6g``.

    Standard QDoubleSpinBox with ``setDecimals(6)`` truncates values like
    ``-1.38e-9`` to ``0.000000``.  This subclass uses 15 internal decimal
    places and overrides the text/validate round-trip so that scientific
    notation values entered by the user (e.g. ``-1.38e-9``) are accepted
    and displayed cleanly.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDecimals(15)
        self.setRange(-1e15, 1e15)

    # ── Display ───────────────────────────────────────────────────────────────

    def textFromValue(self, value: float) -> str:  # type: ignore[override]
        return f"{value:.6g}"

    def valueFromText(self, text: str) -> float:  # type: ignore[override]
        try:
            return float(text.strip())
        except ValueError:
            # Return the current stored value so Qt treats the bad input as
            # a no-op rather than silently snapping to 0.0.
            return self.value()

    def validate(self, text: str, pos: int):  # type: ignore[override]
        stripped = text.strip()
        # Empty / sign only / decimal started / exponent started → intermediate
        if stripped in ("", "-", "+", ".", "-.", "+."):
            return (QValidator.State.Intermediate, text, pos)
        if stripped and stripped[-1].lower() == "e":
            return (QValidator.State.Intermediate, text, pos)
        if len(stripped) > 1 and stripped[-2].lower() == "e" and stripped[-1] in "+-":
            return (QValidator.State.Intermediate, text, pos)
        try:
            float(stripped)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            return (QValidator.State.Intermediate, text, pos)


# ── Main dialog ───────────────────────────────────────────────────────────────

class ThresholdDialog(QDialog):
    """Modeless dialog to apply a data threshold to the current image.

    Shows the image histogram with two draggable clip lines (red = lower,
    green = upper).  Dragging a line or adjusting the Min/Max sliders fires
    a live preview automatically.

    Modes
    -----
    *Clip*: values outside ``[lower, upper]`` become NaN.
    *Binarize*: pixels inside the range become 1.0, outside 0.0.

    Highlight
    ---------
    When a highlight colour is chosen the preview shows the *original* image
    in greyscale with out-of-range pixels tinted in that colour (ImageJ
    style).  Pass a *preview_pixmap_fn* callback to enable this path; it
    receives a ``QPixmap`` instead of a float array.

    Signals
    -------
    applied(dict):
        Emitted when the user clicks **Apply**.  The dict has keys
        ``"mode"`` (str), ``"lower"`` (float), and ``"upper"`` (float).
    """

    applied: Signal = Signal(dict)

    # Default fallback theme (light-neutral; overridden when the viewer theme
    # is passed in via the *theme* constructor argument)
    _FALLBACK_THEME = {
        "bg":        "#f5f5f5",
        "fg":        "#222222",
        "accent_bg": "#4c8bcc",
        "sep":       "#cccccc",
    }

    # Highlight colour table (name → normalised RGB triple)
    _HIGHLIGHT_COLORS: dict[str, tuple[float, float, float]] = {
        "Red":    (1.00, 0.15, 0.15),
        "Blue":   (0.15, 0.45, 1.00),
        "Green":  (0.15, 1.00, 0.25),
        "Yellow": (1.00, 1.00, 0.15),
        "Cyan":   (0.15, 1.00, 0.95),
    }

    def __init__(
        self,
        arr: np.ndarray,
        *,
        preview_fn: "Callable | None" = None,
        preview_pixmap_fn: "Callable | None" = None,
        clear_preview_fn: "Callable | None" = None,
        theme: dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Threshold")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumWidth(420)

        self._arr = arr
        self._preview_fn = preview_fn
        self._preview_pixmap_fn = preview_pixmap_fn
        self._clear_preview_fn = clear_preview_fn
        self._theme = theme or self._FALLBACK_THEME

        # ── Data range ────────────────────────────────────────────────────────
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            self._finite = finite
            self._vmin = float(finite.min())
            self._vmax = float(finite.max())
            self._display_vmin, self._display_vmax = _display_range_from_finite(finite)
        else:
            self._finite = np.array([0.0, 1.0])
            self._vmin, self._vmax = 0.0, 1.0
            self._display_vmin, self._display_vmax = 0.0, 1.0

        # Initial bounds: 1st/99th percentile (ImageJ "Auto" equivalent)
        lo_init = float(np.percentile(self._finite, 1.0))
        hi_init = float(np.percentile(self._finite, 99.0))
        if lo_init >= hi_init:
            lo_init, hi_init = self._vmin, self._vmax

        # Internal authoritative lo / hi (spinbox values are display-only when
        # the data range is smaller than spinbox precision allows)
        self._lo: float = lo_init
        self._hi: float = hi_init

        # ── Histogram panel ───────────────────────────────────────────────────
        from probeflow.gui.viewer.histogram import HistogramPanel
        self._hist = HistogramPanel(parent=self)
        self._hist.set_threshold_mode(True)

        self._hist.render(
            flat_phys=self._finite,
            lo_phys=lo_init,
            hi_phys=hi_init,
            unit="",
            axis_label="Value",
            theme=self._theme,
            scale=1.0,
            data_min_phys=self._vmin,
            data_max_phys=self._vmax,
        )

        # Connect histogram interactions → spinboxes + preview
        self._hist.rangeReleased.connect(self._on_hist_range_released)
        self._hist.minReleased.connect(self._on_min_slider_released)
        self._hist.maxReleased.connect(self._on_max_slider_released)

        # ── Spinboxes (precise numeric entry, scientific notation) ────────────
        self._lower_spin = _SciSpinBox()
        self._lower_spin.setValue(lo_init)
        self._lower_spin.setFont(ui_font(8))
        self._upper_spin = _SciSpinBox()
        self._upper_spin.setValue(hi_init)
        self._upper_spin.setFont(ui_font(8))

        self._lower_spin.editingFinished.connect(self._on_spinbox_changed)
        self._upper_spin.editingFinished.connect(self._on_spinbox_changed)

        spin_row = QHBoxLayout()
        spin_row.addWidget(QLabel("Lower:"))
        spin_row.addWidget(self._lower_spin, 1)
        spin_row.addSpacing(8)
        spin_row.addWidget(QLabel("Upper:"))
        spin_row.addWidget(self._upper_spin, 1)

        # ── % in range label ──────────────────────────────────────────────────
        self._pct_lbl = QLabel()
        self._pct_lbl.setFont(ui_font(8))
        self._pct_lbl.setAlignment(Qt.AlignCenter)
        self._update_pct_label(lo_init, hi_init)

        # ── Mode selector ─────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode:")
        mode_lbl.setFont(ui_font(8))
        self._mode_cb = QComboBox()
        self._mode_cb.setFont(ui_font(8))
        self._mode_cb.addItem("Clip (set out-of-range to NaN)", "clip")
        self._mode_cb.addItem("Binarize (0 outside, 1 inside)", "binarize")
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self._mode_cb, 1)

        # ── Highlight colour selector ─────────────────────────────────────────
        hl_row = QHBoxLayout()
        hl_lbl = QLabel("Highlight:")
        hl_lbl.setFont(ui_font(8))
        self._highlight_cb = QComboBox()
        self._highlight_cb.setFont(ui_font(8))
        self._highlight_cb.addItem("None", None)
        for name in self._HIGHLIGHT_COLORS:
            self._highlight_cb.addItem(name, name)
        self._highlight_cb.setCurrentIndex(1)   # default to Red
        self._highlight_cb.currentIndexChanged.connect(self._on_highlight_changed)
        hl_row.addWidget(hl_lbl)
        hl_row.addWidget(self._highlight_cb, 1)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._do_apply)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)

        # ── Root layout ───────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setSpacing(4)
        root.addWidget(self._hist)
        root.addLayout(spin_row)
        root.addWidget(self._pct_lbl)
        root.addLayout(mode_row)
        root.addLayout(hl_row)
        root.addLayout(btn_row)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _update_pct_label(self, lo: float, hi: float) -> None:
        if self._finite.size == 0:
            self._pct_lbl.setText("")
            return
        in_range = int(np.sum((self._finite >= lo) & (self._finite <= hi)))
        pct = 100.0 * in_range / self._finite.size
        self._pct_lbl.setText(f"{pct:.1f}% of pixels in range")

    def _current_params(self) -> dict:
        """Return threshold parameters using the authoritative internal floats."""
        return {
            "mode":  self._mode_cb.currentData(),
            "lower": self._lo,
            "upper": self._hi,
        }

    # ── Preview ────────────────────────────────────────────────────────────────

    def _do_preview(self) -> None:
        """Fire the live preview if the current bounds form a valid range."""
        if self._lo >= self._hi:
            return   # invalid or collapsed range — skip silently

        highlight = self._highlight_cb.currentData()   # str colour key or None

        if highlight is not None and self._preview_pixmap_fn is not None:
            pixmap = self._build_highlight_pixmap(self._lo, self._hi, highlight)
            self._preview_pixmap_fn(pixmap)
        elif self._preview_fn is not None:
            from probeflow.processing.geometry import threshold_image
            params = self._current_params()
            result = threshold_image(
                self._arr,
                lower=params["lower"],
                upper=params["upper"],
                mode=params["mode"],
            )
            self._preview_fn(result)

    def _build_highlight_pixmap(
        self, lo: float, hi: float, color: str
    ) -> QPixmap:
        """Render the original image in greyscale with out-of-range pixels tinted.

        This mirrors the ImageJ threshold visualisation: pixels outside the
        chosen bounds are overlaid with a semi-transparent highlight colour.
        """
        arr = self._arr
        span = self._display_vmax - self._display_vmin
        if span > 0:
            norm = np.clip((arr - self._display_vmin) / span, 0.0, 1.0)
        else:
            norm = np.zeros_like(arr, dtype=np.float32)

        # Start with greyscale RGBA
        g = norm.astype(np.float32)
        rgba = np.stack([g, g, g, np.ones_like(g, dtype=np.float32)], axis=-1)

        r_c, g_c, b_c = self._HIGHLIGHT_COLORS.get(color, (1.0, 0.15, 0.15))

        # Out-of-range mask (NaN pixels count as out-of-range)
        finite_mask = np.isfinite(arr)
        outside = (~finite_mask) | (arr < lo) | (arr > hi)

        alpha = 0.65
        rgba[outside, 0] = alpha * r_c + (1.0 - alpha) * rgba[outside, 0]
        rgba[outside, 1] = alpha * g_c + (1.0 - alpha) * rgba[outside, 1]
        rgba[outside, 2] = alpha * b_c + (1.0 - alpha) * rgba[outside, 2]

        # Pure-NaN pixels → dark grey so they don't float as highlight colour
        rgba[~finite_mask] = [0.12, 0.12, 0.12, 1.0]

        img8 = np.ascontiguousarray(np.clip(rgba, 0.0, 1.0) * 255, dtype=np.uint8)
        h, w = img8.shape[:2]
        qimg = QImage(img8.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        return QPixmap.fromImage(qimg)

    def _do_apply(self) -> None:
        if self._lo >= self._hi:
            # Degenerate range — nothing meaningful to apply; let the user fix it.
            return
        self.applied.emit(self._current_params())
        self.close()

    # ── Signal handlers ────────────────────────────────────────────────────────

    def _on_hist_range_released(self, lo: float, hi: float) -> None:
        """Histogram drag-line released → sync internal state, spinboxes, preview."""
        self._lo = lo
        self._hi = hi
        self._lower_spin.blockSignals(True)
        self._upper_spin.blockSignals(True)
        self._lower_spin.setValue(lo)
        self._upper_spin.setValue(hi)
        self._lower_spin.blockSignals(False)
        self._upper_spin.blockSignals(False)
        self._update_pct_label(lo, hi)
        self._do_preview()

    def _on_min_slider_released(self, v: int) -> None:
        """Min slider released → clamp to stay below _hi, sync spinbox, preview."""
        lo = self._hist.sl_to_si(v)   # scale=1.0 → SI == physical
        # Clamp: min slider must stay strictly below the current upper bound.
        if lo >= self._hi:
            lo = self._hi - abs(self._hi - self._vmin) * 1e-6 or self._hi - 1e-15
        self._lo = lo
        self._lower_spin.blockSignals(True)
        self._lower_spin.setValue(lo)
        self._lower_spin.blockSignals(False)
        self._update_pct_label(lo, self._hi)
        self._do_preview()

    def _on_max_slider_released(self, v: int) -> None:
        """Max slider released → clamp to stay above _lo, sync spinbox, preview."""
        hi = self._hist.sl_to_si(v)
        # Clamp: max slider must stay strictly above the current lower bound.
        if hi <= self._lo:
            hi = self._lo + abs(self._vmax - self._lo) * 1e-6 or self._lo + 1e-15
        self._hi = hi
        self._upper_spin.blockSignals(True)
        self._upper_spin.setValue(hi)
        self._upper_spin.blockSignals(False)
        self._update_pct_label(self._lo, hi)
        self._do_preview()

    def _on_spinbox_changed(self) -> None:
        """Spinbox editing finished → sync internal state, histogram, preview.

        If the entered value would produce an invalid range (lo >= hi) the
        spinbox is reverted to the last accepted value so the display never
        disagrees with the internal state.
        """
        lo = self._lower_spin.value()
        hi = self._upper_spin.value()
        if lo >= hi:
            # Revert the spinbox that was just edited back to the last valid value
            # so the display stays consistent with self._lo / self._hi.
            sender = self.sender()
            if sender is self._lower_spin:
                self._lower_spin.blockSignals(True)
                self._lower_spin.setValue(self._lo)
                self._lower_spin.blockSignals(False)
            elif sender is self._upper_spin:
                self._upper_spin.blockSignals(True)
                self._upper_spin.setValue(self._hi)
                self._upper_spin.blockSignals(False)
            return
        self._lo = lo
        self._hi = hi
        self._hist.update_drag_lines(lo, hi)
        lo_sl = self._hist.si_to_sl(lo)
        hi_sl = self._hist.si_to_sl(hi)
        self._hist.set_slider_positions(
            lo_sl, hi_sl,
            self._hist.brightness_value,
            self._hist.contrast_value,
        )
        self._update_pct_label(lo, hi)
        self._do_preview()

    def _on_highlight_changed(self) -> None:
        """Highlight colour changed → re-fire preview with new colour."""
        self._do_preview()

    def closeEvent(self, event) -> None:
        if self._clear_preview_fn is not None:
            self._clear_preview_fn()
        super().closeEvent(event)
