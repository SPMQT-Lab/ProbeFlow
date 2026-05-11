"""Viewer-only widgets used by ImageViewerDialog."""

from __future__ import annotations

from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

# ── Physical-axis ruler (top / left of the image) ───────────────────────────
class RulerWidget(QWidget):
    """Thin tick-mark ruler showing physical nm extent of the image.

    Placed in the scroll viewport alongside the image so it scrolls/zooms
    in step with it. ``orientation`` is "horizontal" (top, runs left→right)
    or "vertical" (left, runs top→bottom).
    """

    THICKNESS_PX = 26  # height for horizontal, width for vertical

    def __init__(self, orientation: str = "horizontal", parent=None):
        super().__init__(parent)
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")
        self._orient = orientation
        self._scan_nm: float = 0.0
        self._extent_px: int = 0  # image pixel extent in this direction
        if orientation == "horizontal":
            self.setFixedHeight(self.THICKNESS_PX)
        else:
            self.setFixedWidth(self.THICKNESS_PX)

    def set_extent(self, scan_nm: float, extent_px: int) -> None:
        """Bind to scan physical size and current pixmap extent (px)."""
        self._scan_nm = float(scan_nm) if scan_nm and scan_nm > 0 else 0.0
        self._extent_px = max(0, int(extent_px))
        if self._orient == "horizontal":
            self.setFixedWidth(max(1, self._extent_px))
        else:
            self.setFixedHeight(max(1, self._extent_px))
        self.update()

    @staticmethod
    def _nice_step(scan_nm: float) -> float:
        """Pick a tick step roughly scan/5, snapped to {1, 2, 5} × 10^k."""
        if scan_nm <= 0:
            return 1.0
        target = scan_nm / 5.0
        import math
        exp = math.floor(math.log10(target))
        base = target / (10 ** exp)
        if base < 1.5:
            mult = 1
        elif base < 3.5:
            mult = 2
        elif base < 7.5:
            mult = 5
        else:
            mult = 10
        return mult * (10 ** exp)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._scan_nm <= 0 or self._extent_px <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#cdd6f4"), 1))
        painter.setFont(QFont("Helvetica", 7))

        step = self._nice_step(self._scan_nm)
        if step <= 0:
            painter.end()
            return
        n_steps = int(self._scan_nm / step) + 1

        if self._orient == "horizontal":
            # Border line at bottom of the ruler.
            y_baseline = self.height() - 1
            painter.drawLine(0, y_baseline, self.width(), y_baseline)
            for i in range(n_steps + 1):
                nm = i * step
                if nm > self._scan_nm + 1e-9:
                    break
                x = int(round(nm / self._scan_nm * self._extent_px))
                tick_h = 6 if i % 1 == 0 else 3
                painter.drawLine(x, y_baseline - tick_h, x, y_baseline)
                lbl = f"{nm:g}"
                fm = painter.fontMetrics()
                w = fm.horizontalAdvance(lbl)
                tx = max(0, min(self.width() - w, x - w // 2))
                painter.drawText(tx, y_baseline - tick_h - 2, lbl)
            # Unit label at far right, drawn against the right edge.
            unit = "nm"
            fm = painter.fontMetrics()
            uw = fm.horizontalAdvance(unit)
            painter.drawText(self.width() - uw - 2, y_baseline - 2, unit)
        else:  # vertical
            x_baseline = self.width() - 1
            painter.drawLine(x_baseline, 0, x_baseline, self.height())
            for i in range(n_steps + 1):
                nm = i * step
                if nm > self._scan_nm + 1e-9:
                    break
                y = int(round(nm / self._scan_nm * self._extent_px))
                tick_w = 6 if i % 1 == 0 else 3
                painter.drawLine(x_baseline - tick_w, y, x_baseline, y)
                lbl = f"{nm:g}"
                fm = painter.fontMetrics()
                h = fm.height()
                # Right-align label to the tick start.
                w = fm.horizontalAdvance(lbl)
                tx = max(0, x_baseline - tick_w - 2 - w)
                ty = max(h, min(self.height(), y + h // 2 - 1))
                painter.drawText(tx, ty, lbl)
        painter.end()


# ── Scale-bar widget (lives below the image, separate from the pixmap) ──────
class ScaleBarWidget(QWidget):
    """Independent scale bar drawn underneath the image (not on the pixmap).

    Defaults to an integer-nm length computed as roughly 20-30% of the scan
    width, rounded down to the nearest integer nm. The user can override that
    with a custom value (which can be any positive integer nm). Hidden when
    ``visible`` is False; nothing is painted at all in that case.
    """

    BAR_HEIGHT_PX = 6
    LABEL_GAP_PX  = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan_nm: float = 0.0
        self._image_pixel_width: int = 0
        self._visible: bool = False
        self._custom_nm: Optional[float] = None  # None → auto integer length
        self.setFixedHeight(28)
        self.setMinimumWidth(40)

    def set_scan_size(self, scan_nm: float, image_pixel_width: int) -> None:
        """Tell the widget the scan's physical width and the on-screen width
        of the image pixmap (so the bar can be sized in proportion).
        """
        self._scan_nm = float(scan_nm) if scan_nm and scan_nm > 0 else 0.0
        self._image_pixel_width = max(0, int(image_pixel_width))
        self.update()

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        self.update()

    def set_custom_length_nm(self, length_nm: Optional[float]) -> None:
        """Override the auto length. Pass None to revert to auto."""
        if length_nm is None or length_nm <= 0:
            self._custom_nm = None
        else:
            self._custom_nm = float(length_nm)
        self.update()

    def auto_length_nm(self) -> float:
        """Default integer-nm bar length (~25% of scan), floored to integer."""
        if self._scan_nm <= 0:
            return 0.0
        target = self._scan_nm * 0.25
        if target >= 1.0:
            return float(int(target))  # floor to integer nm
        # Sub-nm: fall back to a single nm if scan_nm >= 1, else just half.
        if self._scan_nm >= 1.0:
            return 1.0
        return max(0.1, round(self._scan_nm * 0.25, 2))

    def current_length_nm(self) -> float:
        if self._custom_nm is not None:
            return self._custom_nm
        return self.auto_length_nm()

    def paintEvent(self, event):
        super().paintEvent(event)
        if (not self._visible) or self._scan_nm <= 0 or self._image_pixel_width <= 0:
            return
        length_nm = self.current_length_nm()
        if length_nm <= 0 or length_nm > self._scan_nm:
            return
        bar_px = int(round(length_nm / self._scan_nm * self._image_pixel_width))
        if bar_px <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Centre the bar horizontally over the image (image is centered in
        # the scroll area by Qt.AlignCenter; we approximate by centering on
        # this widget's full width — close enough since the widget is set
        # to match the scroll area width).
        x0 = (self.width() - bar_px) // 2
        y0 = 4
        painter.setPen(QPen(QColor("white"), 0))
        painter.setBrush(QBrush(QColor("black")))
        painter.drawRect(x0, y0, bar_px, self.BAR_HEIGHT_PX)

        # Label "X nm" centred below.
        if length_nm == int(length_nm):
            txt = f"{int(length_nm)} nm"
        else:
            txt = f"{length_nm:g} nm"
        painter.setPen(QPen(QColor("black"), 0))
        painter.setFont(QFont("Helvetica", 10, QFont.Bold))
        painter.drawText(QRect(x0, y0 + self.BAR_HEIGHT_PX + self.LABEL_GAP_PX,
                                bar_px, 16),
                         Qt.AlignCenter, txt)
        painter.end()


class LineProfilePanel(QWidget):
    """Compact live profile plot for viewer line selections."""

    export_csv_clicked = Signal()
    width_changed      = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(2)

        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        ctrl_row.setSpacing(4)
        _lbl = QLabel("Width:")
        _lbl.setFont(QFont("Helvetica", 8))
        ctrl_row.addWidget(_lbl)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 500)
        self._width_spin.setValue(1)
        self._width_spin.setSuffix(" px")
        self._width_spin.setFixedWidth(70)
        self._width_spin.setFont(QFont("Helvetica", 8))
        self._width_spin.setToolTip("Averaging width perpendicular to the line (pixels)")
        self._width_spin.valueChanged.connect(self.width_changed)
        ctrl_row.addWidget(self._width_spin)
        ctrl_row.addStretch()
        lay.addLayout(ctrl_row)

        self._fig = Figure(figsize=(5.0, 1.8), dpi=80)
        self._fig.patch.set_alpha(0)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFixedHeight(150)
        lay.addWidget(self._canvas)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.addStretch()
        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setFont(QFont("Helvetica", 8))
        self._export_btn.setFixedHeight(20)
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip("Export line profile data as CSV")
        self._export_btn.clicked.connect(self.export_csv_clicked)
        btn_row.addWidget(self._export_btn)
        lay.addLayout(btn_row)

        self._x_vals = None
        self._y_vals = None
        self._x_label = ""
        self._y_label = ""
        self._source_label = ""
        self.show_empty()

    def set_width(self, width: int) -> None:
        """Set spinbox to *width* without firing width_changed."""
        self._width_spin.blockSignals(True)
        self._width_spin.setValue(max(1, int(width)))
        self._width_spin.blockSignals(False)

    def profile_data(self):
        """Return (x_vals, y_vals, x_label, y_label) or None if no profile."""
        if self._x_vals is None:
            return None
        return self._x_vals, self._y_vals, self._x_label, self._y_label

    def show_empty(self, message: str = "Draw a line to show profile.",
                   theme: Optional[dict] = None) -> None:
        theme = theme or {}
        bg = theme.get("bg", "#1e1e2e")
        fg = theme.get("fg", "#cdd6f4")
        sep = theme.get("sep", "#45475a")
        self._fig.patch.set_facecolor(bg)
        self._ax.cla()
        self._ax.set_facecolor(bg)
        self._ax.text(0.5, 0.5, message, ha="center", va="center",
                      transform=self._ax.transAxes, color=fg, fontsize=9)
        self._ax.set_title("")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(sep)
        self._fig.tight_layout(pad=0.35)
        self._canvas.draw_idle()
        self._x_vals = None
        self._y_vals = None
        self._source_label = ""
        self._export_btn.setEnabled(False)

    def plot_profile(self, x_vals, values, *, x_label: str = "Distance [nm]",
                     y_label: str, theme: Optional[dict] = None) -> None:
        theme = theme or {}
        bg = theme.get("bg", "#1e1e2e")
        fg = theme.get("fg", "#cdd6f4")
        sep = theme.get("sep", "#45475a")
        accent = theme.get("accent_bg", "#89b4fa")
        self._fig.patch.set_facecolor(bg)
        self._ax.cla()
        self._ax.set_facecolor(bg)
        self._ax.plot(x_vals, values, color=accent, linewidth=1.1)
        self._ax.set_xlabel(x_label, fontsize=8, color=fg)
        self._ax.set_ylabel(y_label, fontsize=8, color=fg)
        self._ax.tick_params(colors=fg, labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(sep)
        self._fig.tight_layout(pad=0.35)
        self._canvas.draw_idle()
        self._x_vals = x_vals
        self._y_vals = values
        self._x_label = x_label
        self._y_label = y_label
        self._source_label = ""
        self._export_btn.setEnabled(True)

    def set_source_label(self, source_label: str | None,
                         theme: Optional[dict] = None) -> None:
        """Show the ROI or selection that produced the current profile."""
        if self._x_vals is None:
            return
        self._source_label = str(source_label or "")
        fg = (theme or {}).get("fg", "#cdd6f4")
        self._ax.set_title(self._source_label, fontsize=8, color=fg)
        self._fig.tight_layout(pad=0.35)
        self._canvas.draw_idle()
