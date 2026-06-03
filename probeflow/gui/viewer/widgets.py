"""Viewer-only widgets used by ImageViewerDialog."""

from __future__ import annotations

import warnings
from typing import Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen,
)
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

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
        painter.setFont(ui_font(7))

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
        painter.setFont(ui_font(10, weight=QFont.Bold))
        painter.drawText(QRect(x0, y0 + self.BAR_HEIGHT_PX + self.LABEL_GAP_PX,
                                bar_px, 16),
                         Qt.AlignCenter, txt)
        painter.end()


class LineProfilePanel(QWidget):
    """Compact live profile plot for viewer line selections.

    Hovering over the plot shows a crosshair.  Clicking places a coloured
    marker; a second click places a second marker and displays the lateral
    distance (Δx) and height difference (Δy) between the two points.  A
    third click starts a new measurement cycle.
    """

    export_csv_clicked = Signal()
    add_delta_measurement_clicked = Signal()
    add_profile_summary_clicked = Signal()

    # Catppuccin Mocha colours used for measurement markers
    _MEAS_COLORS = ("#f38ba8", "#a6e3a1")  # red, green
    _CONN_COLOR  = "#fab387"               # peach connecting line

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(2)

        self._fig = Figure(figsize=(5.0, 1.8), dpi=80)
        self._fig.patch.set_alpha(0)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFixedHeight(150)
        lay.addWidget(self._canvas)

        self._meas_label = QLabel("")
        self._meas_label.setFont(ui_font(8))
        self._meas_label.setAlignment(Qt.AlignCenter)
        self._meas_label.setStyleSheet("color: #cdd6f4;")
        self._meas_label.setVisible(False)
        lay.addWidget(self._meas_label)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self._add_delta_btn = QPushButton("Add Δ measurement")
        self._add_delta_btn.setFont(ui_font(8))
        self._add_delta_btn.setFixedHeight(20)
        self._add_delta_btn.setEnabled(False)
        self._add_delta_btn.setToolTip(
            "Save the Δx and Δy between the two selected profile points."
        )
        self._add_delta_btn.clicked.connect(self.add_delta_measurement_clicked)
        btn_row.addWidget(self._add_delta_btn)
        self._add_summary_btn = QPushButton("Add profile summary")
        self._add_summary_btn.setFont(ui_font(8))
        self._add_summary_btn.setFixedHeight(20)
        self._add_summary_btn.setEnabled(False)
        self._add_summary_btn.setToolTip(
            "Add statistics of the whole line profile to the measurements table."
        )
        self._add_summary_btn.clicked.connect(self.add_profile_summary_clicked)
        btn_row.addWidget(self._add_summary_btn)
        btn_row.addStretch()
        self._export_btn = QPushButton("Export CSV…")
        self._export_btn.setFont(ui_font(8))
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

        # Measurement state
        self._meas_pts: list = []       # [(x1,y1), (x2,y2)] in data coords
        self._meas_artists: list = []   # matplotlib artists (cleared on each cla())
        self._crosshair_v = None        # Line2D spanning full axis height
        self._crosshair_h = None        # Line2D spanning full axis width

        self.show_empty()

        # Connect matplotlib canvas events (persist for widget lifetime)
        self._canvas.mpl_connect("motion_notify_event", self._on_profile_motion)
        self._canvas.mpl_connect("button_press_event",  self._on_profile_click)
        self._canvas.mpl_connect("axes_leave_event",    self._on_profile_axes_leave)

    # ── data access ───────────────────────────────────────────────────────────

    def profile_data(self):
        """Return (x_vals, y_vals, x_label, y_label) or None if no profile."""
        if self._x_vals is None:
            return None
        return self._x_vals, self._y_vals, self._x_label, self._y_label

    # ── plot lifecycle ────────────────────────────────────────────────────────

    def show_empty(self, message: str = "Draw a line to show profile.",
                   theme: Optional[dict] = None) -> None:
        theme = theme or {}
        bg  = theme.get("bg",  "#1e1e2e")
        fg  = theme.get("fg",  "#cdd6f4")
        sep = theme.get("sep", "#45475a")
        self._fig.patch.set_facecolor(bg)
        self._ax.cla()
        # cla() destroys all artists — reset references before _setup_crosshair
        self._meas_pts.clear()
        self._meas_artists.clear()
        if hasattr(self, "_meas_label"):
            self._meas_label.setVisible(False)
        self._ax.set_facecolor(bg)
        self._ax.text(0.5, 0.5, message, ha="center", va="center",
                      transform=self._ax.transAxes, color=fg, fontsize=9)
        self._ax.set_title("")
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(sep)
        self._apply_layout()
        self._setup_crosshair()
        self._canvas.draw_idle()
        self._x_vals = None
        self._y_vals = None
        self._source_label = ""
        self._export_btn.setEnabled(False)
        self._add_summary_btn.setEnabled(False)
        self._add_delta_btn.setEnabled(False)

    def plot_profile(self, x_vals, values, *, x_label: str = "Distance [nm]",
                     y_label: str, theme: Optional[dict] = None) -> None:
        theme = theme or {}
        bg     = theme.get("bg",         "#1e1e2e")
        fg     = theme.get("fg",         "#cdd6f4")
        sep    = theme.get("sep",        "#45475a")
        accent = theme.get("accent_bg",  "#89b4fa")
        self._fig.patch.set_facecolor(bg)
        self._ax.cla()
        # cla() destroys all artists — clear stale measurement references
        self._meas_pts.clear()
        self._meas_artists.clear()
        self._meas_label.setVisible(False)
        self._ax.set_facecolor(bg)
        self._ax.plot(x_vals, values, color=accent, linewidth=1.1)
        self._ax.set_xlabel(x_label, fontsize=8, color=fg)
        self._ax.set_ylabel(y_label, fontsize=8, color=fg)
        self._ax.tick_params(colors=fg, labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(sep)
        self._apply_layout()
        self._setup_crosshair()
        self._canvas.draw_idle()
        self._x_vals = x_vals
        self._y_vals = values
        self._x_label = x_label
        self._y_label = y_label
        self._source_label = ""
        self._export_btn.setEnabled(True)
        self._add_summary_btn.setEnabled(True)
        self._add_delta_btn.setEnabled(False)

    def set_source_label(self, source_label: str | None,
                         theme: Optional[dict] = None) -> None:
        """Show the ROI or selection that produced the current profile."""
        if self._x_vals is None:
            return
        self._source_label = str(source_label or "")
        fg = (theme or {}).get("fg", "#cdd6f4")
        self._ax.set_title(self._source_label, fontsize=8, color=fg)
        self._apply_layout()
        self._canvas.draw_idle()

    def _apply_layout(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            self._fig.tight_layout(pad=0.35)

    # ── crosshair ─────────────────────────────────────────────────────────────

    def _setup_crosshair(self) -> None:
        """(Re-)create crosshair Line2D artists after each axes.cla() call."""
        from matplotlib.transforms import blended_transform_factory
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        tv = blended_transform_factory(self._ax.transData, self._ax.transAxes)
        th = blended_transform_factory(self._ax.transAxes,  self._ax.transData)
        (self._crosshair_v,) = self._ax.plot(
            [0, 0], [0, 1], transform=tv,
            color="#585b70", lw=0.9, ls="--", visible=False,
            scalex=False, scaley=False,
        )
        (self._crosshair_h,) = self._ax.plot(
            [0, 1], [0, 0], transform=th,
            color="#585b70", lw=0.9, ls="--", visible=False,
            scalex=False, scaley=False,
        )
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

    def _on_profile_motion(self, event) -> None:
        if self._crosshair_v is None or self._x_vals is None:
            return
        if event.inaxes != self._ax:
            if self._crosshair_v.get_visible():
                self._crosshair_v.set_visible(False)
                self._crosshair_h.set_visible(False)
                self._canvas.draw_idle()
            return
        self._canvas.setCursor(Qt.CrossCursor)
        self._crosshair_v.set_xdata([event.xdata, event.xdata])
        self._crosshair_h.set_ydata([event.ydata, event.ydata])
        self._crosshair_v.set_visible(True)
        self._crosshair_h.set_visible(True)
        self._canvas.draw_idle()

    def _on_profile_axes_leave(self, event) -> None:
        if self._crosshair_v is not None:
            self._crosshair_v.set_visible(False)
            self._crosshair_h.set_visible(False)
            self._canvas.draw_idle()
        self._canvas.setCursor(Qt.ArrowCursor)

    # ── two-point measurement ─────────────────────────────────────────────────

    def _on_profile_click(self, event) -> None:
        if self._x_vals is None or event.inaxes != self._ax or event.button != 1:
            return
        if len(self._meas_pts) >= 2:
            # Third click: begin a new measurement cycle
            self._meas_pts.clear()
            self._clear_meas_artists()
            self._meas_label.setVisible(False)
            self._add_delta_btn.setEnabled(False)
        self._meas_pts.append((event.xdata, event.ydata))
        self._redraw_meas()
        if len(self._meas_pts) == 2:
            self._update_meas_label()
            self._add_delta_btn.setEnabled(True)
        self._canvas.draw_idle()

    def meas_delta(self) -> dict | None:
        """Return current two-point delta measurement data, or None if not ready."""
        if len(self._meas_pts) != 2 or self._x_vals is None:
            return None
        (x1, y1), (x2, y2) = self._meas_pts
        return {
            "delta_x": abs(x2 - x1),
            "delta_y": y2 - y1,
            "p1_distance": x1,
            "p1_height": y1,
            "p2_distance": x2,
            "p2_height": y2,
            "x_unit": self._extract_unit(self._x_label),
            "y_unit": self._extract_unit(self._y_label),
        }

    def _clear_meas_artists(self) -> None:
        for a in self._meas_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._meas_artists.clear()

    def _redraw_meas(self) -> None:
        self._clear_meas_artists()
        for i, (x, y) in enumerate(self._meas_pts):
            col = self._MEAS_COLORS[i]
            vl = self._ax.axvline(x, color=col, lw=0.9, ls=":", alpha=0.9)
            (mk,) = self._ax.plot(x, y, "o", color=col, ms=5, zorder=5)
            self._meas_artists.extend([vl, mk])
        if len(self._meas_pts) == 2:
            (x1, y1), (x2, y2) = self._meas_pts
            (conn,) = self._ax.plot(
                [x1, x2], [y1, y2],
                color=self._CONN_COLOR, lw=0.9, ls="--", alpha=0.85,
            )
            self._meas_artists.append(conn)

    @staticmethod
    def _extract_unit(label: str) -> str:
        if "[" in label and "]" in label:
            return label[label.index("[") + 1 : label.rindex("]")]
        return ""

    def _update_meas_label(self) -> None:
        (x1, y1), (x2, y2) = self._meas_pts
        dx = abs(x2 - x1)
        dy = y2 - y1
        x_unit = self._extract_unit(self._x_label)
        y_unit = self._extract_unit(self._y_label)
        x_str = f"Δx = {dx:.4g} {x_unit}" if x_unit else f"Δx = {dx:.4g}"
        y_str = f"Δy = {dy:+.4g} {y_unit}" if y_unit else f"Δy = {dy:+.4g}"
        self._meas_label.setText(f"{x_str}     {y_str}")
        self._meas_label.setVisible(True)
