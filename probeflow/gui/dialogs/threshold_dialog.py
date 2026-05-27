"""Threshold dialog — clip or binarize a scan plane by height value.

Provides an ImageJ-style histogram with two draggable lines (lower / upper
bounds) so the user can set the threshold visually rather than by typing
raw SI values.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ThresholdDialog(QDialog):
    """Modeless dialog to apply a data threshold to the current image.

    Shows the image histogram with two draggable clip lines (red = lower,
    green = upper).  Dragging a line or adjusting the Min/Max sliders fires
    a live preview automatically.

    Modes
    -----
    *Clip*: values outside ``[lower, upper]`` become NaN.
    *Binarize*: pixels inside the range become 1.0, outside 0.0.

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

    def __init__(
        self,
        arr: np.ndarray,
        *,
        preview_fn: "Callable | None" = None,
        clear_preview_fn: "Callable | None" = None,
        theme: dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Threshold")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumWidth(400)

        self._arr = arr
        self._preview_fn = preview_fn
        self._clear_preview_fn = clear_preview_fn
        self._theme = theme or self._FALLBACK_THEME

        # ── Data range ────────────────────────────────────────────────────────
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            self._finite = finite
            self._vmin = float(finite.min())
            self._vmax = float(finite.max())
        else:
            self._finite = np.array([0.0, 1.0])
            self._vmin, self._vmax = 0.0, 1.0

        # Initial bounds: 1st/99th percentile (ImageJ "Auto" equivalent)
        lo_init = float(np.percentile(self._finite, 1.0))
        hi_init = float(np.percentile(self._finite, 99.0))
        if lo_init >= hi_init:
            lo_init, hi_init = self._vmin, self._vmax

        # ── Histogram panel ───────────────────────────────────────────────────
        from probeflow.gui.viewer.histogram import HistogramPanel
        self._hist = HistogramPanel(parent=self)
        # Hide controls not relevant to thresholding
        self._hist._brightness_w.hide()
        self._hist._contrast_w.hide()
        self._hist._auto_btn.hide()
        self._hist._reset_btn.hide()

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

        # ── Spinboxes (precise numeric entry) ─────────────────────────────────
        def _spin() -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(-1e15, 1e15)
            s.setDecimals(6)
            s.setFont(QFont("Helvetica", 8))
            return s

        self._lower_spin = _spin()
        self._lower_spin.setValue(lo_init)
        self._upper_spin = _spin()
        self._upper_spin.setValue(hi_init)

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
        self._pct_lbl.setFont(QFont("Helvetica", 8))
        self._pct_lbl.setAlignment(Qt.AlignCenter)
        self._update_pct_label(lo_init, hi_init)

        # ── Mode selector ─────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Mode:")
        mode_lbl.setFont(QFont("Helvetica", 8))
        self._mode_cb = QComboBox()
        self._mode_cb.setFont(QFont("Helvetica", 8))
        self._mode_cb.addItem("Clip (set out-of-range to NaN)", "clip")
        self._mode_cb.addItem("Binarize (0 outside, 1 inside)", "binarize")
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self._mode_cb, 1)

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
        return {
            "mode":  self._mode_cb.currentData(),
            "lower": self._lower_spin.value(),
            "upper": self._upper_spin.value(),
        }

    def _do_preview(self) -> None:
        if self._preview_fn is None:
            return
        from probeflow.processing.geometry import threshold_image
        params = self._current_params()
        result = threshold_image(
            self._arr,
            lower=params.get("lower"),
            upper=params.get("upper"),
            mode=params.get("mode", "clip"),
        )
        self._preview_fn(result)

    def _do_apply(self) -> None:
        self.applied.emit(self._current_params())
        self.close()

    # ── Signal handlers ────────────────────────────────────────────────────────

    def _on_hist_range_released(self, lo: float, hi: float) -> None:
        """Histogram drag-line released → sync spinboxes + auto-preview."""
        self._lower_spin.blockSignals(True)
        self._upper_spin.blockSignals(True)
        self._lower_spin.setValue(lo)
        self._upper_spin.setValue(hi)
        self._lower_spin.blockSignals(False)
        self._upper_spin.blockSignals(False)
        self._update_pct_label(lo, hi)
        self._do_preview()

    def _on_min_slider_released(self, v: int) -> None:
        """Min slider released → sync lower spinbox + auto-preview."""
        lo = self._hist.sl_to_si(v)  # scale=1.0 → SI == physical
        self._lower_spin.blockSignals(True)
        self._lower_spin.setValue(lo)
        self._lower_spin.blockSignals(False)
        hi = self._upper_spin.value()
        self._update_pct_label(lo, hi)
        self._do_preview()

    def _on_max_slider_released(self, v: int) -> None:
        """Max slider released → sync upper spinbox + auto-preview."""
        hi = self._hist.sl_to_si(v)
        self._upper_spin.blockSignals(True)
        self._upper_spin.setValue(hi)
        self._upper_spin.blockSignals(False)
        lo = self._lower_spin.value()
        self._update_pct_label(lo, hi)
        self._do_preview()

    def _on_spinbox_changed(self) -> None:
        """Spinbox editing finished → sync histogram drag lines + auto-preview."""
        lo = self._lower_spin.value()
        hi = self._upper_spin.value()
        if lo >= hi:
            return  # ignore invalid range silently
        self._hist.update_drag_lines(lo, hi)
        # Sync Min/Max slider positions to match
        lo_sl = self._hist.si_to_sl(lo)
        hi_sl = self._hist.si_to_sl(hi)
        self._hist.set_slider_positions(
            lo_sl, hi_sl,
            self._hist.brightness_value,
            self._hist.contrast_value,
        )
        self._update_pct_label(lo, hi)
        self._do_preview()

    def closeEvent(self, event) -> None:
        if self._clear_preview_fn is not None:
            self._clear_preview_fn()
        super().closeEvent(event)
