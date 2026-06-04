"""Self-contained histogram + display-range-slider widget for the image viewer.

Owns the matplotlib canvas, drag lines, display sliders, and Auto/Reset buttons.
Emits signals for all user interactions; the parent dialog converts physical
values to SI and updates :class:`DisplayRangeController` accordingly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class HistogramPanel(QWidget):
    """Histogram canvas + display range sliders in a single reusable widget.

    Signals
    -------
    rangeReleased(lo_phys, hi_phys)
        User finished dragging a clip line.  Values are in physical display
        units (Å or pA); the caller converts to SI for :class:`DisplayRangeController`.
    autoClipRequested()
        Auto button clicked.
    resetRequested()
        Reset button clicked.
    contextMenuRequested(pos)
        Right-click on the canvas; ``pos`` is the local widget-space QPoint.
    minReleased(v), maxReleased(v), brightnessReleased(v), contrastReleased(v)
        Slider released — emits the integer slider value (0–1000).
    """

    rangeReleased        = Signal(float, float)
    autoClipRequested    = Signal()
    resetRequested       = Signal()
    contextMenuRequested = Signal(object)  # QPoint

    minReleased        = Signal(int)
    maxReleased        = Signal(int)
    brightnessReleased = Signal(int)
    contrastReleased   = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)

        # ── Matplotlib canvas ──────────────────────────────────────────────────
        self._fig = Figure(figsize=(2.8, 1.4), dpi=80)
        self._fig.patch.set_alpha(0)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFixedHeight(140)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self._canvas.customContextMenuRequested.connect(
            lambda pos: self.contextMenuRequested.emit(pos))
        lay.addWidget(self._canvas)

        # ── Drag state ─────────────────────────────────────────────────────────
        self._low_line  = None
        self._high_line = None
        self._dragging: Optional[str] = None   # 'low' | 'high' | None
        self.flat_phys: Optional[np.ndarray] = None   # finite physical-unit data
        self.unit: str = ""

        self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)

        # ── Slider data range (SI; populated when histogram is rendered) ────────
        self.data_min_si: Optional[float] = None
        self.data_max_si: Optional[float] = None
        self._scale: float = 1.0  # physical = SI * scale (for live label updates)

        # ── Display sliders ────────────────────────────────────────────────────
        def _slider_row(label: str):
            w = QWidget()
            rl = QHBoxLayout(w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setFont(ui_font(8))
            lbl.setFixedWidth(74)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 1000)
            val_lbl = QLabel("—")
            val_lbl.setFont(ui_font(8))
            val_lbl.setFixedWidth(48)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            rl.addWidget(lbl)
            rl.addWidget(sl, 1)
            rl.addWidget(val_lbl)
            return w, sl, val_lbl

        self._min_w,        self._min_sl,        self._min_lbl        = _slider_row("Min")
        self._max_w,        self._max_sl,        self._max_lbl        = _slider_row("Max")
        self._brightness_w, self._brightness_sl, self._brightness_lbl = _slider_row("Brightness")
        self._contrast_w,   self._contrast_sl,   self._contrast_lbl   = _slider_row("Contrast")

        self._min_sl.setValue(0)
        self._max_sl.setValue(1000)
        self._brightness_sl.setValue(500)
        self._contrast_sl.setValue(0)

        # Release signals → parent dialog updates DRS
        self._min_sl.sliderReleased.connect(
            lambda: self.minReleased.emit(self._min_sl.value()))
        self._max_sl.sliderReleased.connect(
            lambda: self.maxReleased.emit(self._max_sl.value()))
        self._brightness_sl.sliderReleased.connect(
            lambda: self.brightnessReleased.emit(self._brightness_sl.value()))
        self._contrast_sl.sliderReleased.connect(
            lambda: self.contrastReleased.emit(self._contrast_sl.value()))

        # Drag label updates handled internally so the dialog stays clean
        self._min_sl.valueChanged.connect(self._on_min_moved)
        self._max_sl.valueChanged.connect(self._on_max_moved)
        self._brightness_sl.valueChanged.connect(self._on_brightness_moved)
        self._contrast_sl.valueChanged.connect(self._on_contrast_moved)

        lay.addWidget(self._min_w)
        lay.addWidget(self._max_w)
        lay.addWidget(self._brightness_w)
        lay.addWidget(self._contrast_w)

        # ── Auto / Reset buttons ───────────────────────────────────────────────
        actions_row = QHBoxLayout()
        self._auto_btn = QPushButton("Auto")
        self._auto_btn.setFont(ui_font(8))
        self._auto_btn.setFixedHeight(22)
        self._auto_btn.setToolTip(
            "Autoscale display bounds to the current image's 1%–99% percentiles.")
        self._auto_btn.clicked.connect(self.autoClipRequested)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setFont(ui_font(8))
        self._reset_btn.setFixedHeight(22)
        self._reset_btn.setToolTip(
            "Reset display range to the default (1%–99% percentile).")
        self._reset_btn.clicked.connect(self.resetRequested)
        actions_row.addStretch()
        actions_row.addWidget(self._auto_btn)
        actions_row.addWidget(self._reset_btn)
        lay.addLayout(actions_row)

        # ── Clip value readout label ───────────────────────────────────────────
        self._clip_lbl = QLabel("")
        self._clip_lbl.setFont(ui_font(8))
        self._clip_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._clip_lbl)
        lay.addStretch()

    # ── Public update API ──────────────────────────────────────────────────────

    def render(
        self,
        flat_phys: np.ndarray,
        lo_phys: float,
        hi_phys: float,
        unit: str,
        axis_label: str,
        theme: dict,
        scale: float = 1.0,
        data_min_phys: Optional[float] = None,
        data_max_phys: Optional[float] = None,
    ) -> None:
        """Redraw the histogram with new data.

        Parameters
        ----------
        flat_phys:
            Flat array of finite pixel values in physical display units.
        lo_phys, hi_phys:
            Current display clip-line positions in physical units.
        unit:
            Physical unit string (e.g. "Å" or "pA").
        axis_label:
            Axis label prefix (e.g. "Height").
        theme:
            Dict with at least "bg", "fg", "accent_bg", "sep" keys.
        scale:
            Physical/SI conversion factor (physical = SI * scale).
            Stored for live slider-label updates.
        data_min_phys, data_max_phys:
            Histogram x-axis extent (0.1–99.9 percentile range in physical
            units).  If None, the panel falls back to the flat-array extremes.
        """
        self.unit   = unit
        self._scale = scale
        self.flat_phys = flat_phys

        self._ax.cla()
        self._low_line  = None
        self._high_line = None

        bg = theme.get("bg", "#1e1e2e")
        fg = theme.get("fg", "#cdd6f4")
        self._fig.patch.set_facecolor(bg)
        self._ax.set_facecolor(bg)

        try:
            from probeflow.processing.display import histogram_from_array as _hist
            counts, edges = _hist(flat_phys, bins=128, clip_percentiles=(0.1, 99.9))
        except (ValueError, Exception):
            counts, edges = np.histogram(flat_phys, bins=128)
            data_min_phys = data_max_phys = None

        x_min = float(edges[0])
        x_max = float(edges[-1])

        if data_min_phys is None:
            data_min_phys = x_min
        if data_max_phys is None:
            data_max_phys = x_max

        if scale and scale != 0:
            self.data_min_si = float(data_min_phys) / scale
            self.data_max_si = float(data_max_phys) / scale
        else:
            self.data_min_si = None
            self.data_max_si = None

        counts = np.maximum(counts, 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        self._ax.bar(centers, counts, width=widths,
                     color=theme.get("accent_bg", "#89b4fa"),
                     alpha=0.85, linewidth=0)
        self._ax.set_yscale("log")

        x0 = min(float(data_min_phys), float(lo_phys), float(hi_phys))
        x1 = max(float(data_max_phys), float(lo_phys), float(hi_phys))
        span = x1 - x0
        pad = 0.02 * span if span > 0 else max(abs(x0) * 0.02, 1.0)
        self._ax.set_xlim(x0 - pad, x1 + pad)

        self._low_line  = self._ax.axvline(lo_phys, color="#f38ba8",
                                            linewidth=1.6, picker=6)
        self._high_line = self._ax.axvline(hi_phys, color="#a6e3a1",
                                            linewidth=1.6, picker=6)

        self._ax.tick_params(axis="x", colors=fg, labelsize=7)
        self._ax.tick_params(axis="y", left=False, labelleft=False)
        self._ax.yaxis.set_visible(False)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(theme.get("sep", "#45475a"))
        self._ax.set_xlabel(f"{axis_label} [{unit}]", fontsize=7, color=fg)
        self._fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.22)
        self._canvas.draw_idle()

        self._clip_lbl.setText(f"{lo_phys:.3g} {unit}  →  {hi_phys:.3g} {unit}")

    def clear(self, theme: dict) -> None:
        """Draw a 'No finite data' placeholder and reset all drag/slider state."""
        self._ax.cla()
        self._low_line  = None
        self._high_line = None
        self.flat_phys  = None
        self.data_min_si = None
        self.data_max_si = None

        bg = theme.get("bg", "#1e1e2e")
        fg = theme.get("fg", "#cdd6f4")
        self._fig.patch.set_facecolor(bg)
        self._ax.set_facecolor(bg)
        self._ax.text(
            0.5, 0.5, "No finite data",
            transform=self._ax.transAxes,
            ha="center", va="center",
            fontsize=8, color=fg,
        )
        for spine in self._ax.spines.values():
            spine.set_edgecolor(theme.get("sep", "#45475a"))
        self._fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.12)
        self._canvas.draw_idle()
        self._clip_lbl.setText("")

    def set_slider_positions(
        self, min_v: int, max_v: int, brightness_v: int, contrast_v: int
    ) -> None:
        """Update all four sliders without triggering user-interaction signals."""
        for sl in (self._min_sl, self._max_sl, self._brightness_sl, self._contrast_sl):
            sl.blockSignals(True)
        self._min_sl.setValue(min_v)
        self._max_sl.setValue(max_v)
        self._brightness_sl.setValue(brightness_v)
        self._contrast_sl.setValue(contrast_v)
        for sl in (self._min_sl, self._max_sl, self._brightness_sl, self._contrast_sl):
            sl.blockSignals(False)

    def set_slider_labels(
        self, min_t: str, max_t: str, brightness_t: str, contrast_t: str
    ) -> None:
        """Update all four slider value labels."""
        self._min_lbl.setText(min_t)
        self._max_lbl.setText(max_t)
        self._brightness_lbl.setText(brightness_t)
        self._contrast_lbl.setText(contrast_t)

    def set_clip_text(self, text: str) -> None:
        """Update the clip value readout label."""
        self._clip_lbl.setText(text)

    def update_drag_lines(self, lo_phys: float, hi_phys: float) -> None:
        """Reposition clip lines after an external range update."""
        if self._low_line is not None:
            self._low_line.set_xdata([lo_phys, lo_phys])
        if self._high_line is not None:
            self._high_line.set_xdata([hi_phys, hi_phys])
        self._canvas.draw_idle()

    def set_threshold_mode(self, enabled: bool) -> None:
        """Hide brightness/contrast/auto/reset controls in threshold mode.

        ``ThresholdDialog`` reuses ``HistogramPanel`` to pick low/high clip
        values; the brightness/contrast sliders and the Auto/Reset buttons
        are not meaningful in that context.  Call ``set_threshold_mode(True)``
        right after construction to hide them.
        """
        visible = not enabled
        self._brightness_w.setVisible(visible)
        self._contrast_w.setVisible(visible)
        self._auto_btn.setVisible(visible)
        self._reset_btn.setVisible(visible)

    # ── Slider value accessors ─────────────────────────────────────────────────

    @property
    def min_value(self) -> int:
        return self._min_sl.value()

    @property
    def max_value(self) -> int:
        return self._max_sl.value()

    @property
    def brightness_value(self) -> int:
        return self._brightness_sl.value()

    @property
    def contrast_value(self) -> int:
        return self._contrast_sl.value()

    # ── SI ↔ slider integer conversion ────────────────────────────────────────

    def sl_to_si(self, v: int) -> float:
        """Map slider integer 0–1000 to an SI value using the stored data range."""
        lo, hi = self.data_min_si, self.data_max_si
        if lo is None or hi is None or hi <= lo:
            return 0.0
        return lo + v / 1000.0 * (hi - lo)

    def si_to_sl(self, si: float) -> int:
        """Map an SI value to slider integer 0–1000."""
        lo, hi = self.data_min_si, self.data_max_si
        if lo is None or hi is None or hi <= lo:
            return 0
        return max(0, min(1000, round((si - lo) / (hi - lo) * 1000)))

    # ── Internal drag handlers ─────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if (event.inaxes is not self._ax or event.xdata is None
                or event.button != 1
                or self._low_line is None or self._high_line is None):
            return
        lo = self._low_line.get_xdata()[0]
        hi = self._high_line.get_xdata()[0]
        x0, x1 = self._ax.get_xlim()
        tol = 0.04 * (x1 - x0) if x1 > x0 else 0.0
        d_lo = abs(event.xdata - lo)
        d_hi = abs(event.xdata - hi)
        self._dragging = 'low' if d_lo <= d_hi else 'high'
        if min(d_lo, d_hi) > tol and (lo <= event.xdata <= hi):
            self._dragging = None

    def _on_motion(self, event) -> None:
        if (self._dragging is None or event.inaxes is not self._ax
                or event.xdata is None
                or self._low_line is None or self._high_line is None):
            return
        x = float(event.xdata)
        lo = self._low_line.get_xdata()[0]
        hi = self._high_line.get_xdata()[0]
        if self._dragging == 'low':
            x = min(x, hi - 1e-12)
            self._low_line.set_xdata([x, x])
        else:
            x = max(x, lo + 1e-12)
            self._high_line.set_xdata([x, x])
        new_lo = self._low_line.get_xdata()[0]
        new_hi = self._high_line.get_xdata()[0]
        self._clip_lbl.setText(
            f"{new_lo:.3g} {self.unit}  →  {new_hi:.3g} {self.unit}")
        self._canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._dragging is None or self.flat_phys is None:
            self._dragging = None
            return
        lo_phys = float(self._low_line.get_xdata()[0])
        hi_phys = float(self._high_line.get_xdata()[0])
        self._dragging = None
        self.rangeReleased.emit(lo_phys, hi_phys)

    # ── Internal slider drag label handlers ────────────────────────────────────

    def _on_min_moved(self, v: int) -> None:
        if self.data_min_si is None:
            return
        self._min_lbl.setText(
            f"{self.sl_to_si(v) * self._scale:.3g} {self.unit}")

    def _on_max_moved(self, v: int) -> None:
        if self.data_min_si is None:
            return
        self._max_lbl.setText(
            f"{self.sl_to_si(v) * self._scale:.3g} {self.unit}")

    def _on_brightness_moved(self, v: int) -> None:
        if self.data_min_si is None:
            return
        self._brightness_lbl.setText(
            f"{self.sl_to_si(v) * self._scale:.3g} {self.unit}")

    def _on_contrast_moved(self, v: int) -> None:
        if self.data_min_si is None or self.data_max_si is None:
            return
        data_range_si = self.data_max_si - self.data_min_si
        width_frac = max(0.001, 1.0 - v / 1000.0)
        new_width_phys = data_range_si * width_frac * self._scale
        self._contrast_lbl.setText(
            f"{new_width_phys:.3g} {self.unit}")
