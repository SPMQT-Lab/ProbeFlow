"""
Lattice/Grid measurement overlay tool for ProbeFlow.

Architecture:
  LatticeGridItem         — pure display QGraphicsObject, no mouse handling
  LatticeGridController   — viewport event-filter for reliable handle dragging
  LatticeGridPanel        — control panel with spinboxes, live measurements
  FFTLatticeOverlay       — matplotlib overlay for FFT viewer
  FFTLatticePanel         — control panel for FFT overlay

Entry points:
  open_real_space_tool(canvas, scan_range_m, image_shape, parent)
  open_fft_tool(ax, canvas, qx_axis, qy_axis, image_shape, parent)
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
    QGraphicsObject,
)

from probeflow.analysis.lattice_grid import (
    LatticeGrid, LatticeKind, RealSpaceCalibration, ReciprocalCalibration,
    format_real_space_measurements, format_reciprocal_measurements,
)

# ── colours ───────────────────────────────────────────────────────────────────

_COL_GRID   = QColor("#89b4fa")  # blue  – lattice lines
_COL_A      = QColor("#a6e3a1")  # green – a vector / handle
_COL_B      = QColor("#fab387")  # peach – b vector / handle
_COL_ORIGIN = QColor("#f38ba8")  # red   – origin handle
_COL_ROT    = QColor("#cba6f7")  # purple – rotation handle
_COL_SCALE  = QColor("#f9e2af")  # yellow – scale handle
_COL_LABEL  = QColor("#cdd6f4")  # text labels

# ── handle IDs ────────────────────────────────────────────────────────────────

_HANDLE_NONE   = 0
_HANDLE_ORIGIN = 1
_HANDLE_A      = 2
_HANDLE_B      = 3
_HANDLE_ROT    = 4
_HANDLE_SCALE  = 5

# Screen-space hit radius used by both controller and FFT overlay
HIT_RADIUS_PX = 12

# Visual handle size in screen pixels (drawn via cosmetic pen / fixed size)
_HANDLE_SCREEN_R = 6.0


# ── LatticeGridItem — pure display ────────────────────────────────────────────

class LatticeGridItem(QGraphicsObject):
    """
    Draws the lattice grid on an ImageCanvas (QGraphicsScene).

    Scene coordinates match image pixel coordinates 1:1.
    All mouse interaction is handled by LatticeGridController.
    """

    grid_changed = Signal(object)   # emits LatticeGrid after any set_grid()

    def __init__(
        self,
        grid: LatticeGrid,
        image_w: int,
        image_h: int,
        cells: int = 12,
        parent=None,
    ):
        super().__init__(parent)
        self._grid = grid
        self._image_w = image_w
        self._image_h = image_h
        self._cells = cells

        self.setZValue(50)
        self.setFlag(QGraphicsObject.ItemIsMovable, False)
        self.setFlag(QGraphicsObject.ItemIsFocusable, False)
        self.setAcceptHoverEvents(False)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def cells(self) -> int:
        return self._cells

    def set_cells(self, n: int) -> None:
        self._cells = max(1, int(n))
        self.update()

    def grid(self) -> LatticeGrid:
        return self._grid

    def set_grid(self, grid: LatticeGrid) -> None:
        self._grid = grid
        self.update()
        self.grid_changed.emit(grid)

    # ── handle positions (scene / pixel coordinates) ───────────────────────────

    def handle_positions_scene(self) -> dict[int, tuple[float, float]]:
        """Return all handle positions in scene (= pixel) coordinates."""
        return _compute_handle_positions(self._grid)

    # ── QGraphicsItem interface ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        m = max(self._image_w, self._image_h) * (self._cells + 2.0)
        return QRectF(-m, -m, m * 3, m * 3)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._grid.visible:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        # Cosmetic pen: 1 screen pixel regardless of zoom
        pen = QPen(_COL_GRID)
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        self._paint_grid_lines(painter)

        self._paint_basis_vectors(painter)

        if self._grid.show_labels:
            self._paint_labels(painter)

        if self._grid.show_handles:
            self._paint_handles(painter)

    def _paint_grid_lines(self, painter: QPainter) -> None:
        grid = self._grid
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px
        c = self._cells

        # Lines parallel to b-vector (one line per n along a-direction)
        for n in range(-c, c + 1):
            sx = ox + n * ax - c * bx
            sy = oy + n * ay - c * by
            ex = ox + n * ax + c * bx
            ey = oy + n * ay + c * by
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

        # Lines parallel to a-vector (one line per m along b-direction)
        for m in range(-c, c + 1):
            sx = ox - c * ax + m * bx
            sy = oy - c * ay + m * by
            ex = ox + c * ax + m * bx
            ey = oy + c * ay + m * by
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

    def _paint_basis_vectors(self, painter: QPainter) -> None:
        grid = self._grid
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px

        pen_a = QPen(_COL_A)
        pen_a.setCosmetic(True)
        pen_a.setWidthF(2.0)
        painter.setPen(pen_a)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + ax, oy + ay))

        pen_b = QPen(_COL_B)
        pen_b.setCosmetic(True)
        pen_b.setWidthF(2.0)
        painter.setPen(pen_b)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + bx, oy + by))

    def _paint_handles(self, painter: QPainter) -> None:
        zoom = painter.worldTransform().m11()
        r = _HANDLE_SCREEN_R / max(zoom, 0.01)

        handle_cols = {
            _HANDLE_ORIGIN: _COL_ORIGIN,
            _HANDLE_A:      _COL_A,
            _HANDLE_B:      _COL_B,
            _HANDLE_ROT:    _COL_ROT,
            _HANDLE_SCALE:  _COL_SCALE,
        }
        for hid, (hx, hy) in _compute_handle_positions(self._grid).items():
            col = handle_cols[hid]
            painter.setPen(QPen(col.darker(130)))
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(hx, hy), r, r)

    def _paint_labels(self, painter: QPainter) -> None:
        zoom = painter.worldTransform().m11()
        font_pt = max(0.5, _HANDLE_SCREEN_R * 1.6 / max(zoom, 0.01))

        font = QFont("Helvetica")
        font.setPointSizeF(font_pt)
        painter.setFont(font)

        pen = QPen(_COL_LABEL)
        pen.setCosmetic(True)
        painter.setPen(pen)

        grid = self._grid
        handles = _compute_handle_positions(grid)
        off = _HANDLE_SCREEN_R * 1.2 / max(zoom, 0.01)

        a_lbl = "g1" if grid.space == "reciprocal" else "a"
        b_lbl = "g2" if grid.space == "reciprocal" else "b"

        ax_h, ay_h = handles[_HANDLE_A]
        bx_h, by_h = handles[_HANDLE_B]
        ox_h, oy_h = handles[_HANDLE_ORIGIN]

        painter.drawText(QPointF(ax_h + off, ay_h), a_lbl)
        painter.drawText(QPointF(bx_h + off, by_h), b_lbl)
        painter.drawText(QPointF(ox_h + off, oy_h), "O")


# ── handle position helper ────────────────────────────────────────────────────

def _compute_handle_positions(grid: LatticeGrid) -> dict[int, tuple[float, float]]:
    """Compute all handle positions in scene (pixel) coordinates."""
    ox, oy = grid.origin_px
    ax, ay = grid.a_px
    bx, by = grid.b_px
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    avg_l = (la + lb) * 0.5 or 1.0

    if la > 1e-6:
        # Rotation handle perpendicular to a (90° CCW from a)
        perp_x = -ay / la
        perp_y =  ax / la
    else:
        perp_x, perp_y = 0.0, 1.0

    rot_dist = avg_l * 0.65

    return {
        _HANDLE_ORIGIN: (ox, oy),
        _HANDLE_A:      (ox + ax, oy + ay),
        _HANDLE_B:      (ox + bx, oy + by),
        _HANDLE_ROT:    (ox + perp_x * rot_dist, oy + perp_y * rot_dist),
        _HANDLE_SCALE:  (ox + (ax + bx) * 0.55, oy + (ay + by) * 0.55),
    }


# ── LatticeGridController — viewport event filter ─────────────────────────────

class LatticeGridController(QObject):
    """
    Installs a viewport event filter on ImageCanvas for reliable handle dragging.

    Hit testing is done in screen pixels so it is zoom-invariant.
    Once a drag starts it is not stolen by the canvas's own pan/ROI handlers.
    """

    def __init__(self, item: LatticeGridItem, canvas, parent=None):
        super().__init__(parent)
        self._item = item
        self._canvas = canvas
        self._panel: Optional["LatticeGridPanel"] = None
        self._active: bool = False
        self._dragging: bool = False
        self._drag_handle: int = _HANDLE_NONE
        self._drag_start_scene: Optional[QPointF] = None
        self._drag_grid_start: Optional[LatticeGrid] = None

    def set_panel(self, panel: "LatticeGridPanel") -> None:
        self._panel = panel

    def install(self) -> None:
        self._canvas.viewport().installEventFilter(self)

    def uninstall(self) -> None:
        self._canvas.viewport().removeEventFilter(self)
        self._canvas.viewport().setCursor(Qt.ArrowCursor)

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._dragging = False
            self._drag_handle = _HANDLE_NONE
            self._canvas.viewport().setCursor(Qt.ArrowCursor)

    # ── screen-space hit testing ───────────────────────────────────────────────

    def hit_handle_screen(self, view_pos) -> int:
        """
        Find which handle (if any) is under view_pos.

        Uses screen-pixel distance so hit radius is zoom-invariant.
        """
        if not self._item.grid().show_handles:
            return _HANDLE_NONE
        for hid, (sx, sy) in self._item.handle_positions_scene().items():
            screen = self._canvas.mapFromScene(QPointF(sx, sy))
            dist = math.hypot(view_pos.x() - screen.x(), view_pos.y() - screen.y())
            if dist <= HIT_RADIUS_PX:
                return hid
        return _HANDLE_NONE

    # ── event filter ──────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if not self._active:
            return False

        t = event.type()

        if t == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
            vpos = event.pos()
            hid = self.hit_handle_screen(vpos)
            if hid != _HANDLE_NONE:
                self._drag_handle = hid
                self._drag_start_scene = self._canvas.mapToScene(vpos)
                self._drag_grid_start = self._item.grid()
                self._dragging = True
                event.accept()
                return True

        elif t == QEvent.Type.MouseMove:
            vpos = event.pos()
            if self._dragging:
                self._update_drag(self._canvas.mapToScene(vpos))
                event.accept()
                return True
            else:
                hid = self.hit_handle_screen(vpos)
                vp = self._canvas.viewport()
                if hid == _HANDLE_ORIGIN:
                    vp.setCursor(Qt.SizeAllCursor)
                elif hid != _HANDLE_NONE:
                    vp.setCursor(Qt.CrossCursor)
                else:
                    vp.setCursor(Qt.ArrowCursor)
                return False

        elif t == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._dragging:
                self._dragging = False
                self._drag_handle = _HANDLE_NONE
                self._drag_start_scene = None
                self._drag_grid_start = None
                event.accept()
                return True

        return False

    def _update_drag(self, scene_pos: QPointF) -> None:
        if self._drag_start_scene is None or self._drag_grid_start is None:
            return

        dx = scene_pos.x() - self._drag_start_scene.x()
        dy = scene_pos.y() - self._drag_start_scene.y()
        g0 = self._drag_grid_start
        hid = self._drag_handle

        if hid == _HANDLE_ORIGIN:
            new_grid = g0.translate(dx, dy)

        elif hid == _HANDLE_A:
            new_grid = g0.with_a_vector((g0.a_px[0] + dx, g0.a_px[1] + dy))

        elif hid == _HANDLE_B:
            new_grid = g0.with_b_vector((g0.b_px[0] + dx, g0.b_px[1] + dy))

        elif hid == _HANDLE_ROT:
            ox, oy = g0.origin_px
            angle_start = math.degrees(math.atan2(
                self._drag_start_scene.y() - oy,
                self._drag_start_scene.x() - ox,
            ))
            angle_now = math.degrees(math.atan2(scene_pos.y() - oy, scene_pos.x() - ox))
            new_grid = g0.rotate(angle_now - angle_start)

        elif hid == _HANDLE_SCALE:
            ox, oy = g0.origin_px
            d0 = math.hypot(
                self._drag_start_scene.x() - ox,
                self._drag_start_scene.y() - oy,
            )
            d1 = math.hypot(scene_pos.x() - ox, scene_pos.y() - oy)
            if d0 > 1e-6:
                new_grid = g0.scale(d1 / d0)
            else:
                return
        else:
            return

        self._item.set_grid(new_grid)
        if self._panel is not None:
            self._panel.sync_from_model()


# ── unit helpers ──────────────────────────────────────────────────────────────

def _choose_display_unit(cal: RealSpaceCalibration) -> tuple[float, str]:
    """
    Return (unit_scale_to_m, unit_label) for the most readable atomic-scale unit.

    unit_scale_to_m: multiply spinbox value by this to get metres.
    """
    typical_m = cal.px_size_x * cal.image_width * 0.1
    if typical_m < 5e-9:    # less than ~5 nm → use Å
        return (1e-10, "Å")
    return (1e-9, "nm")


# ── LatticeGridPanel ──────────────────────────────────────────────────────────

class LatticeGridPanel(QWidget):
    """
    Interactive control + measurement panel for a real-space lattice grid.

    Communicates with LatticeGridItem and LatticeGridController.
    """

    def __init__(
        self,
        item: LatticeGridItem,
        controller: LatticeGridController,
        calibration: RealSpaceCalibration,
        image_w: int,
        image_h: int,
        parent=None,
    ):
        super().__init__(parent)
        self._item = item
        self._ctrl = controller
        self._cal = calibration
        self._image_w = image_w
        self._image_h = image_h

        self._unit_scale, self._unit_label = _choose_display_unit(calibration)
        self._updating_controls = False

        self._build()
        self.sync_from_model()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── edit active ───────────────────────────────────────────────────────
        self._edit_cb = QCheckBox("Edit grid (drag handles)")
        self._edit_cb.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._edit_cb.setChecked(True)
        self._edit_cb.toggled.connect(self._on_active_toggled)
        lay.addWidget(self._edit_cb)
        # Activate immediately
        self._ctrl.set_active(True)

        # ── lattice type ──────────────────────────────────────────────────────
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Square", "Rectangular", "Hexagonal"])
        self._type_combo.setFont(QFont("Helvetica", 9))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo, 1)
        lay.addLayout(type_row)

        # ── parameters group ──────────────────────────────────────────────────
        params_grp = QGroupBox("Parameters")
        params_grp.setFont(QFont("Helvetica", 9))
        params_lay = QVBoxLayout(params_grp)
        params_lay.setSpacing(3)
        params_lay.setContentsMargins(6, 6, 6, 4)

        def _spin_row(label: str, lo: float, hi: float,
                      step: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 9))
            lbl.setMinimumWidth(68)
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            if suffix:
                spin.setSuffix(f" {suffix}")
            spin.setFont(QFont("Helvetica", 9))
            spin.setFixedHeight(22)
            row.addWidget(lbl)
            row.addWidget(spin, 1)
            params_lay.addLayout(row)
            return spin

        self._ox_spin = _spin_row("Origin x:", 0.0, float(self._image_w), 0.5, 1, "px")
        self._oy_spin = _spin_row("Origin y:", 0.0, float(self._image_h), 0.5, 1, "px")

        a_max = self._cal.px_size_x * self._image_w / self._unit_scale
        self._a_spin = _spin_row(
            f"|a| ({self._unit_label}):", 0.001, a_max * 2, 0.01, 3, self._unit_label,
        )
        self._b_spin = _spin_row(
            f"|b| ({self._unit_label}):", 0.001, a_max * 2, 0.01, 3, self._unit_label,
        )

        self._rot_spin = _spin_row("Rotation:", -180.0, 180.0, 0.1, 1, "°")

        # Cells spinbox (integer)
        cells_row = QHBoxLayout()
        cells_lbl = QLabel("Cells ±:")
        cells_lbl.setFont(QFont("Helvetica", 9))
        cells_lbl.setMinimumWidth(68)
        self._cells_spin = QSpinBox()
        self._cells_spin.setRange(1, 200)
        self._cells_spin.setValue(self._item.cells)
        self._cells_spin.setFont(QFont("Helvetica", 9))
        self._cells_spin.setFixedHeight(22)
        cells_row.addWidget(cells_lbl)
        cells_row.addWidget(self._cells_spin, 1)
        params_lay.addLayout(cells_row)

        lay.addWidget(params_grp)

        # Connect spinboxes after building (to avoid early triggers)
        self._ox_spin.valueChanged.connect(self._on_origin_changed)
        self._oy_spin.valueChanged.connect(self._on_origin_changed)
        self._a_spin.valueChanged.connect(self._on_a_length_changed)
        self._b_spin.valueChanged.connect(self._on_b_length_changed)
        self._rot_spin.valueChanged.connect(self._on_rotation_changed)
        self._cells_spin.valueChanged.connect(self._on_cells_changed)

        # ── display group ─────────────────────────────────────────────────────
        disp_grp = QGroupBox("Display")
        disp_grp.setFont(QFont("Helvetica", 9))
        disp_lay = QVBoxLayout(disp_grp)
        disp_lay.setSpacing(2)
        disp_lay.setContentsMargins(6, 6, 6, 4)

        self._show_grid_cb    = QCheckBox("Show grid")
        self._show_handles_cb = QCheckBox("Show handles")
        self._show_labels_cb  = QCheckBox("Show labels")
        for cb in (self._show_grid_cb, self._show_handles_cb, self._show_labels_cb):
            cb.setFont(QFont("Helvetica", 9))
            cb.setChecked(True)
            cb.toggled.connect(self._on_visibility_changed)
            disp_lay.addWidget(cb)

        lay.addWidget(disp_grp)

        # ── actions ───────────────────────────────────────────────────────────
        reset_btn = QPushButton("Reset origin to centre")
        reset_btn.setFont(QFont("Helvetica", 9))
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self._on_reset_origin)
        lay.addWidget(reset_btn)

        # ── measured values ───────────────────────────────────────────────────
        meas_grp = QGroupBox("Measured")
        meas_grp.setFont(QFont("Helvetica", 9))
        meas_lay = QVBoxLayout(meas_grp)
        meas_lay.setContentsMargins(6, 6, 6, 4)
        self._meas_lbl = QLabel("")
        self._meas_lbl.setFont(QFont("Courier", 8))
        self._meas_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._meas_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._meas_lbl.setWordWrap(True)
        meas_lay.addWidget(self._meas_lbl)
        lay.addWidget(meas_grp)

        # ── export ────────────────────────────────────────────────────────────
        lay.addStretch(1)
        exp_row = QHBoxLayout()
        exp_with_btn = QPushButton("Export with grid…")
        exp_with_btn.setFont(QFont("Helvetica", 9))
        exp_with_btn.setFixedHeight(24)
        exp_with_btn.clicked.connect(self._on_export_with_grid)
        exp_grid_btn = QPushButton("Export grid only…")
        exp_grid_btn.setFont(QFont("Helvetica", 9))
        exp_grid_btn.setFixedHeight(24)
        exp_grid_btn.clicked.connect(self._on_export_grid_only)
        exp_row.addWidget(exp_with_btn)
        exp_row.addWidget(exp_grid_btn)
        lay.addLayout(exp_row)

    # ── model↔UI sync ─────────────────────────────────────────────────────────

    def sync_from_model(self) -> None:
        """Update all spinboxes and measurement display from the current grid."""
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            grid = self._item.grid()
            ox, oy = grid.origin_px

            a_len_m = self._cal.vector_length_m(grid.a_px)
            b_len_m = self._cal.vector_length_m(grid.b_px)
            a_len_u = a_len_m / self._unit_scale
            b_len_u = b_len_m / self._unit_scale

            self._ox_spin.setValue(ox)
            self._oy_spin.setValue(oy)
            self._a_spin.setValue(a_len_u)
            self._b_spin.setValue(b_len_u)
            self._rot_spin.setValue(grid.a_angle_deg())
            self._cells_spin.setValue(self._item.cells)

            kind = grid.kind
            idx = {"square": 0, "rectangular": 1, "hexagonal": 2}.get(kind, 0)
            if self._type_combo.currentIndex() != idx:
                self._type_combo.setCurrentIndex(idx)

            # Disable b-length for symmetric lattices
            b_editable = (kind == "rectangular")
            self._b_spin.setEnabled(b_editable)

        finally:
            self._updating_controls = False

        self._refresh_measurement_label()

    def _refresh_measurement_label(self) -> None:
        grid = self._item.grid()
        try:
            d = format_real_space_measurements(grid, self._cal)
            lines = [
                f"|a| = {d['a_length']}",
                f"|b| = {d['b_length']}",
                f"angle = {d['angle']}",
                f"area = {d['area']}",
                f"origin = {d['origin_phys']}",
            ]
        except Exception as exc:
            lines = [f"(error: {exc})"]
        self._meas_lbl.setText("\n".join(lines))

    # ── spinbox slots ─────────────────────────────────────────────────────────

    def _on_origin_changed(self, _val: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        new_grid = grid.reset_origin(self._ox_spin.value(), self._oy_spin.value())
        self._item.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_a_length_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        new_a_m = value * self._unit_scale
        # Direction-aware px conversion
        old_a_m = self._cal.vector_length_m(grid.a_px)
        old_a_px = grid.a_length_px()
        if old_a_m < 1e-25 or old_a_px < 1e-9:
            return
        new_a_px = new_a_m * old_a_px / old_a_m
        new_grid = grid.set_a_length_px(new_a_px)
        self._item.set_grid(new_grid)
        self._updating_controls = True
        try:
            if grid.kind != "rectangular":
                b_m = self._cal.vector_length_m(new_grid.b_px)
                self._b_spin.setValue(b_m / self._unit_scale)
        finally:
            self._updating_controls = False
        self._refresh_measurement_label()

    def _on_b_length_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        if grid.kind != "rectangular":
            return
        new_b_m = value * self._unit_scale
        old_b_m = self._cal.vector_length_m(grid.b_px)
        old_b_px = grid.b_length_px()
        if old_b_m < 1e-25 or old_b_px < 1e-9:
            return
        new_b_px = new_b_m * old_b_px / old_b_m
        new_grid = grid.set_b_length_px(new_b_px)
        self._item.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_rotation_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._item.grid()
        new_grid = grid.set_rotation_deg(value)
        self._item.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_cells_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self._item.set_cells(value)

    def _on_type_changed(self, idx: int) -> None:
        if self._updating_controls:
            return
        kinds: list[LatticeKind] = ["square", "rectangular", "hexagonal"]
        new_kind = kinds[idx]
        g = self._item.grid()
        la = g.a_length_px()
        lb = g.b_length_px()
        angle_a = g.a_angle_deg()
        ca, sa = math.cos(math.radians(angle_a)), math.sin(math.radians(angle_a))

        if new_kind == "square":
            new_g = replace(g, kind="square",
                a_px=(la * ca, la * sa),
                b_px=(la * math.cos(math.radians(angle_a + 90)),
                      la * math.sin(math.radians(angle_a + 90))))
        elif new_kind == "rectangular":
            new_g = replace(g, kind="rectangular",
                b_px=(lb * math.cos(math.radians(angle_a + 90)),
                      lb * math.sin(math.radians(angle_a + 90))))
        else:  # hexagonal
            avg_l = (la + lb) * 0.5
            new_g = replace(g, kind="hexagonal",
                a_px=(avg_l * ca, avg_l * sa),
                b_px=(avg_l * math.cos(math.radians(angle_a + 60)),
                      avg_l * math.sin(math.radians(angle_a + 60))))

        self._item.set_grid(new_g)
        self.sync_from_model()

    def _on_active_toggled(self, checked: bool) -> None:
        self._ctrl.set_active(checked)

    def _on_visibility_changed(self) -> None:
        g = self._item.grid()
        self._item.set_grid(replace(
            g,
            visible=self._show_grid_cb.isChecked(),
            show_handles=self._show_handles_cb.isChecked(),
            show_labels=self._show_labels_cb.isChecked(),
        ))

    def _on_reset_origin(self) -> None:
        g = self._item.grid()
        self._item.set_grid(g.reset_origin(self._image_w / 2.0, self._image_h / 2.0))
        self.sync_from_model()

    def _on_export_with_grid(self) -> None:
        self._export(grid_only=False)

    def _on_export_grid_only(self) -> None:
        self._export(grid_only=True)

    def _export(self, grid_only: bool) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "",
            "PNG image (*.png);;PDF document (*.pdf)",
        )
        if not path:
            return
        from probeflow.gui.lattice_export import export_grid
        export_grid(self._item, path, include_grid=True, grid_only=grid_only)


# ── FFT matplotlib overlay ────────────────────────────────────────────────────

class FFTLatticeOverlay:
    """
    Lattice grid overlay on a matplotlib FFT axes.

    Hit testing uses display (screen) pixels via ax.transData.transform(),
    so the hit radius is zoom-invariant.
    """

    def __init__(
        self,
        ax,
        canvas,
        qx_axis: np.ndarray,
        qy_axis: np.ndarray,
        image_w: int,
        image_h: int,
    ):
        self._ax = ax
        self._canvas = canvas
        self._qx = qx_axis
        self._qy = qy_axis
        self._image_w = image_w
        self._image_h = image_h
        self._artists: list = []
        self._grid: Optional[LatticeGrid] = None
        self._cells: int = 12
        self._dragging: bool = False
        self._drag_handle: int = _HANDLE_NONE
        self._drag_start_q: Optional[tuple[float, float]] = None
        self._drag_grid_start: Optional[LatticeGrid] = None
        self._on_change_cb = None

        self._cid_press   = canvas.mpl_connect("button_press_event",   self._on_press)
        self._cid_release = canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion  = canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def set_grid(self, grid: LatticeGrid) -> None:
        self._grid = grid
        self.redraw()

    def grid(self) -> Optional[LatticeGrid]:
        return self._grid

    def set_cells(self, n: int) -> None:
        self._cells = max(1, int(n))
        self.redraw()

    def set_on_change(self, cb) -> None:
        self._on_change_cb = cb

    def disconnect(self) -> None:
        for cid in (self._cid_press, self._cid_release, self._cid_motion):
            try:
                self._canvas.mpl_disconnect(cid)
            except Exception:
                pass

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _px_to_q(self, ix: float, iy: float) -> tuple[float, float]:
        Nx, Ny = self._image_w, self._image_h
        dqx = (float(self._qx[-1]) - float(self._qx[0])) / max(1, Nx - 1)
        dqy = (float(self._qy[-1]) - float(self._qy[0])) / max(1, Ny - 1)
        return (float(self._qx[0]) + ix * dqx, float(self._qy[0]) + iy * dqy)

    def _q_to_px(self, qx: float, qy: float) -> tuple[float, float]:
        Nx, Ny = self._image_w, self._image_h
        dqx = (float(self._qx[-1]) - float(self._qx[0])) / max(1, Nx - 1)
        dqy = (float(self._qy[-1]) - float(self._qy[0])) / max(1, Ny - 1)
        ix = (qx - float(self._qx[0])) / dqx if dqx != 0 else 0.0
        iy = (qy - float(self._qy[0])) / dqy if dqy != 0 else 0.0
        return ix, iy

    def _handle_positions_q(self) -> dict[int, tuple[float, float]]:
        """Return handle positions in q-space."""
        if self._grid is None:
            return {}
        handles_px = _compute_handle_positions(self._grid)
        return {hid: self._px_to_q(px, py) for hid, (px, py) in handles_px.items()}

    def _hit_handle_display(self, mouse_x: float, mouse_y: float) -> int:
        """
        Hit test using display (screen) pixels.

        event.x / event.y are matplotlib display coordinates (pixels from bottom).
        """
        if self._grid is None:
            return _HANDLE_NONE
        for hid, (qx, qy) in self._handle_positions_q().items():
            try:
                disp = self._ax.transData.transform((qx, qy))
                dist = math.hypot(mouse_x - disp[0], mouse_y - disp[1])
                if dist <= HIT_RADIUS_PX:
                    return hid
            except Exception:
                pass
        return _HANDLE_NONE

    # ── matplotlib events ─────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax or self._grid is None:
            return
        hid = self._hit_handle_display(event.x, event.y)
        if hid == _HANDLE_NONE:
            return
        self._drag_handle = hid
        self._drag_start_q = (event.xdata, event.ydata)
        self._drag_grid_start = self._grid
        self._dragging = True

    def _on_release(self, event) -> None:
        self._dragging = False
        self._drag_handle = _HANDLE_NONE
        self._drag_start_q = None
        self._drag_grid_start = None

    def _on_motion(self, event) -> None:
        if (not self._dragging
                or event.inaxes is not self._ax
                or self._drag_start_q is None
                or self._drag_grid_start is None):
            return

        dqx = event.xdata - self._drag_start_q[0]
        dqy = event.ydata - self._drag_start_q[1]

        Nx, Ny = self._image_w, self._image_h
        q_range_x = float(self._qx[-1]) - float(self._qx[0])
        q_range_y = float(self._qy[-1]) - float(self._qy[0])
        dpx = dqx * Nx / q_range_x if q_range_x != 0 else 0.0
        dpy = dqy * Ny / q_range_y if q_range_y != 0 else 0.0

        g0 = self._drag_grid_start
        hid = self._drag_handle

        if hid == _HANDLE_ORIGIN:
            self._grid = g0.translate(dpx, dpy)
        elif hid == _HANDLE_A:
            self._grid = g0.with_a_vector((g0.a_px[0] + dpx, g0.a_px[1] + dpy))
        elif hid == _HANDLE_B:
            self._grid = g0.with_b_vector((g0.b_px[0] + dpx, g0.b_px[1] + dpy))
        elif hid == _HANDLE_ROT:
            ox_px, oy_px = g0.origin_px
            ox_q, oy_q = self._px_to_q(ox_px, oy_px)
            angle_start = math.degrees(math.atan2(
                self._drag_start_q[1] - oy_q,
                self._drag_start_q[0] - ox_q,
            ))
            angle_now = math.degrees(math.atan2(event.ydata - oy_q, event.xdata - ox_q))
            self._grid = g0.rotate(angle_now - angle_start)
        elif hid == _HANDLE_SCALE:
            ox_px, oy_px = g0.origin_px
            ox_q, oy_q = self._px_to_q(ox_px, oy_px)
            d0 = math.hypot(self._drag_start_q[0] - ox_q, self._drag_start_q[1] - oy_q)
            d1 = math.hypot(event.xdata - ox_q, event.ydata - oy_q)
            if d0 > 1e-12:
                self._grid = g0.scale(d1 / d0)

        self.redraw()
        if self._on_change_cb is not None:
            self._on_change_cb(self._grid)

    # ── drawing ───────────────────────────────────────────────────────────────

    def redraw(self) -> None:
        for art in self._artists:
            try:
                art.remove()
            except Exception:
                pass
        self._artists.clear()

        if self._grid is None or not self._grid.visible:
            self._canvas.draw_idle()
            return

        grid = self._grid
        ox_px, oy_px = grid.origin_px
        ax_px, ay_px = grid.a_px
        bx_px, by_px = grid.b_px
        c = self._cells

        def p2q(ix, iy):
            return self._px_to_q(ix, iy)

        # Grid lines — parallel to b
        for n in range(-c, c + 1):
            s_px = (ox_px + n * ax_px - c * bx_px, oy_px + n * ay_px - c * by_px)
            e_px = (ox_px + n * ax_px + c * bx_px, oy_px + n * ay_px + c * by_px)
            sq, eq = p2q(*s_px), p2q(*e_px)
            art, = self._ax.plot(
                [sq[0], eq[0]], [sq[1], eq[1]],
                color="#89b4fa", lw=0.8, alpha=0.7, zorder=5,
            )
            self._artists.append(art)

        # Grid lines — parallel to a
        for m in range(-c, c + 1):
            s_px = (ox_px - c * ax_px + m * bx_px, oy_px - c * ay_px + m * by_px)
            e_px = (ox_px + c * ax_px + m * bx_px, oy_px + c * ay_px + m * by_px)
            sq, eq = p2q(*s_px), p2q(*e_px)
            art, = self._ax.plot(
                [sq[0], eq[0]], [sq[1], eq[1]],
                color="#89b4fa", lw=0.8, alpha=0.7, zorder=5,
            )
            self._artists.append(art)

        # Basis vectors
        oq = p2q(ox_px, oy_px)
        aq = p2q(ox_px + ax_px, oy_px + ay_px)
        bq = p2q(ox_px + bx_px, oy_px + by_px)

        art, = self._ax.plot([oq[0], aq[0]], [oq[1], aq[1]],
                             color="#a6e3a1", lw=2.0, zorder=6)
        self._artists.append(art)
        art, = self._ax.plot([oq[0], bq[0]], [oq[1], bq[1]],
                             color="#fab387", lw=2.0, zorder=6)
        self._artists.append(art)

        # Handles + labels
        handle_cols = {
            _HANDLE_ORIGIN: "#f38ba8",
            _HANDLE_A:      "#a6e3a1",
            _HANDLE_B:      "#fab387",
            _HANDLE_ROT:    "#cba6f7",
            _HANDLE_SCALE:  "#f9e2af",
        }
        handle_labels = {
            _HANDLE_A:      "g1" if grid.space == "reciprocal" else "a",
            _HANDLE_B:      "g2" if grid.space == "reciprocal" else "b",
            _HANDLE_ORIGIN: "O",
        }
        handles_q = self._handle_positions_q()
        for hid, (hqx, hqy) in handles_q.items():
            if grid.show_handles:
                art, = self._ax.plot(
                    hqx, hqy, "o",
                    color=handle_cols[hid], markersize=9,
                    markeredgewidth=1.0, zorder=8,
                )
                self._artists.append(art)
            if grid.show_labels and hid in handle_labels:
                art = self._ax.text(
                    hqx, hqy, f" {handle_labels[hid]}",
                    color="#cdd6f4", fontsize=8, va="center", zorder=9,
                )
                self._artists.append(art)

        self._canvas.draw_idle()


# ── FFT panel ─────────────────────────────────────────────────────────────────

class FFTLatticePanel(QWidget):
    """Control and measurement panel for an FFT lattice overlay."""

    def __init__(
        self,
        overlay: FFTLatticeOverlay,
        calibration: ReciprocalCalibration,
        image_w: int,
        image_h: int,
        parent=None,
    ):
        super().__init__(parent)
        self._overlay = overlay
        self._cal = calibration
        self._image_w = image_w
        self._image_h = image_h
        self._updating_controls = False
        self._build()
        overlay.set_on_change(self._on_grid_changed)
        self.sync_from_model()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Square", "Rectangular", "Hexagonal"])
        self._type_combo.setFont(QFont("Helvetica", 9))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo, 1)
        lay.addLayout(type_row)

        params_grp = QGroupBox("Parameters")
        params_grp.setFont(QFont("Helvetica", 9))
        params_lay = QVBoxLayout(params_grp)
        params_lay.setSpacing(3)
        params_lay.setContentsMargins(6, 6, 6, 4)

        def _row(label, lo, hi, step, dec, sfx=""):
            r = QHBoxLayout()
            lb = QLabel(label)
            lb.setFont(QFont("Helvetica", 9))
            lb.setMinimumWidth(68)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            sp.setDecimals(dec)
            if sfx:
                sp.setSuffix(f" {sfx}")
            sp.setFont(QFont("Helvetica", 9))
            sp.setFixedHeight(22)
            r.addWidget(lb)
            r.addWidget(sp, 1)
            params_lay.addLayout(r)
            return sp

        self._ox_spin  = _row("Origin x:", 0, float(self._image_w), 1, 1, "px")
        self._oy_spin  = _row("Origin y:", 0, float(self._image_h), 1, 1, "px")
        self._g1_spin  = _row("|g1|:", 0.001, 1000.0, 0.01, 3, "nm⁻¹")
        self._g2_spin  = _row("|g2|:", 0.001, 1000.0, 0.01, 3, "nm⁻¹")
        self._rot_spin = _row("Rotation:", -180, 180, 0.1, 1, "°")

        cells_r = QHBoxLayout()
        cells_lb = QLabel("Cells ±:")
        cells_lb.setFont(QFont("Helvetica", 9))
        cells_lb.setMinimumWidth(68)
        self._cells_spin = QSpinBox()
        self._cells_spin.setRange(1, 200)
        self._cells_spin.setValue(12)
        self._cells_spin.setFont(QFont("Helvetica", 9))
        self._cells_spin.setFixedHeight(22)
        cells_r.addWidget(cells_lb)
        cells_r.addWidget(self._cells_spin, 1)
        params_lay.addLayout(cells_r)
        lay.addWidget(params_grp)

        self._ox_spin.valueChanged.connect(self._on_origin_changed)
        self._oy_spin.valueChanged.connect(self._on_origin_changed)
        self._g1_spin.valueChanged.connect(self._on_g1_changed)
        self._g2_spin.valueChanged.connect(self._on_g2_changed)
        self._rot_spin.valueChanged.connect(self._on_rotation_changed)
        self._cells_spin.valueChanged.connect(self._on_cells_changed)

        disp_grp = QGroupBox("Display")
        disp_grp.setFont(QFont("Helvetica", 9))
        disp_lay = QVBoxLayout(disp_grp)
        disp_lay.setSpacing(2)
        disp_lay.setContentsMargins(6, 6, 6, 4)
        self._show_grid_cb    = QCheckBox("Show grid")
        self._show_handles_cb = QCheckBox("Show handles")
        self._show_labels_cb  = QCheckBox("Show labels")
        for cb in (self._show_grid_cb, self._show_handles_cb, self._show_labels_cb):
            cb.setFont(QFont("Helvetica", 9))
            cb.setChecked(True)
            cb.toggled.connect(self._on_visibility_changed)
            disp_lay.addWidget(cb)
        lay.addWidget(disp_grp)

        reset_btn = QPushButton("Reset origin to FFT centre")
        reset_btn.setFont(QFont("Helvetica", 9))
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self._on_reset_origin)
        lay.addWidget(reset_btn)

        meas_grp = QGroupBox("Measured")
        meas_grp.setFont(QFont("Helvetica", 9))
        meas_lay = QVBoxLayout(meas_grp)
        meas_lay.setContentsMargins(6, 6, 6, 4)
        self._meas_lbl = QLabel("")
        self._meas_lbl.setFont(QFont("Courier", 8))
        self._meas_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._meas_lbl.setWordWrap(True)
        self._meas_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meas_lay.addWidget(self._meas_lbl)
        lay.addWidget(meas_grp)
        lay.addStretch(1)

    def _on_grid_changed(self, grid: LatticeGrid) -> None:
        self.sync_from_model()

    def sync_from_model(self) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None:
            return
        self._updating_controls = True
        try:
            ox, oy = grid.origin_px
            self._ox_spin.setValue(ox)
            self._oy_spin.setValue(oy)
            self._rot_spin.setValue(grid.a_angle_deg())
            self._cells_spin.setValue(self._overlay._cells)

            g1 = self._cal.vec_length_q(grid.a_px)
            g2 = self._cal.vec_length_q(grid.b_px)
            self._g1_spin.setValue(g1)
            self._g2_spin.setValue(g2)

            kind = grid.kind
            idx = {"square": 0, "rectangular": 1, "hexagonal": 2}.get(kind, 0)
            if self._type_combo.currentIndex() != idx:
                self._type_combo.setCurrentIndex(idx)
            self._g2_spin.setEnabled(kind == "rectangular")
        finally:
            self._updating_controls = False

        self._refresh_measurement_label()

    def _refresh_measurement_label(self) -> None:
        grid = self._overlay.grid()
        if grid is None:
            return
        try:
            d = format_reciprocal_measurements(grid, self._cal)
            lines = [d["g1"], d["g2"], f"angle = {d['angle']}", f"area = {d['area_q']}"]
        except Exception as exc:
            lines = [f"(error: {exc})"]
        self._meas_lbl.setText("\n".join(lines))

    def _on_origin_changed(self, _v: float) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None:
            return
        new_grid = grid.reset_origin(self._ox_spin.value(), self._oy_spin.value())
        self._overlay.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_g1_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None:
            return
        old_g1 = self._cal.vec_length_q(grid.a_px)
        if old_g1 < 1e-12:
            return
        factor = value / old_g1
        new_grid = grid.set_a_length_px(grid.a_length_px() * factor)
        self._overlay.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_g2_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None or grid.kind != "rectangular":
            return
        old_g2 = self._cal.vec_length_q(grid.b_px)
        if old_g2 < 1e-12:
            return
        factor = value / old_g2
        new_grid = grid.set_b_length_px(grid.b_length_px() * factor)
        self._overlay.set_grid(new_grid)
        self._refresh_measurement_label()

    def _on_rotation_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        grid = self._overlay.grid()
        if grid is None:
            return
        self._overlay.set_grid(grid.set_rotation_deg(value))
        self._refresh_measurement_label()

    def _on_cells_changed(self, value: int) -> None:
        if self._updating_controls:
            return
        self._overlay.set_cells(value)

    def _on_type_changed(self, idx: int) -> None:
        if self._updating_controls:
            return
        kinds: list[LatticeKind] = ["square", "rectangular", "hexagonal"]
        new_kind = kinds[idx]
        g = self._overlay.grid()
        if g is None:
            return
        la = g.a_length_px()
        lb = g.b_length_px()
        angle_a = g.a_angle_deg()
        ca = math.cos(math.radians(angle_a))
        sa = math.sin(math.radians(angle_a))
        if new_kind == "square":
            new_g = replace(g, kind="square",
                a_px=(la * ca, la * sa),
                b_px=(la * math.cos(math.radians(angle_a + 90)),
                      la * math.sin(math.radians(angle_a + 90))))
        elif new_kind == "rectangular":
            new_g = replace(g, kind="rectangular",
                b_px=(lb * math.cos(math.radians(angle_a + 90)),
                      lb * math.sin(math.radians(angle_a + 90))))
        else:
            avg_l = (la + lb) * 0.5
            new_g = replace(g, kind="hexagonal",
                a_px=(avg_l * ca, avg_l * sa),
                b_px=(avg_l * math.cos(math.radians(angle_a + 60)),
                      avg_l * math.sin(math.radians(angle_a + 60))))
        self._overlay.set_grid(new_g)
        self.sync_from_model()

    def _on_visibility_changed(self) -> None:
        grid = self._overlay.grid()
        if grid is None:
            return
        self._overlay.set_grid(replace(
            grid,
            visible=self._show_grid_cb.isChecked(),
            show_handles=self._show_handles_cb.isChecked(),
            show_labels=self._show_labels_cb.isChecked(),
        ))

    def _on_reset_origin(self) -> None:
        grid = self._overlay.grid()
        if grid is None:
            return
        self._overlay.set_grid(
            grid.reset_origin(self._image_w / 2.0, self._image_h / 2.0)
        )
        self.sync_from_model()


# ── public entry points ───────────────────────────────────────────────────────

def open_real_space_tool(
    canvas,
    scan_range_m: tuple,
    image_shape: tuple,
    parent=None,
) -> tuple[LatticeGridItem, LatticeGridPanel]:
    """
    Create a lattice grid overlay on an ImageCanvas.

    Installs the interaction controller immediately with edit mode active.
    Returns (item, panel); caller adds the panel to the UI (e.g. as a dock).
    """
    Ny, Nx = image_shape
    cx, cy = Nx / 2.0, Ny / 2.0
    size = min(Nx, Ny) * 0.15

    cal = RealSpaceCalibration.from_scan_range(scan_range_m, Nx, Ny)
    grid = LatticeGrid.make_square(cx, cy, size, space="real")

    item = LatticeGridItem(grid, Nx, Ny, cells=12)
    canvas.scene().addItem(item)

    controller = LatticeGridController(item, canvas)
    controller.install()

    panel = LatticeGridPanel(item, controller, cal, Nx, Ny, parent=parent)
    controller.set_panel(panel)
    item.grid_changed.connect(panel.sync_from_model)

    return item, panel


def open_fft_tool(
    ax,
    canvas,
    qx_axis: np.ndarray,
    qy_axis: np.ndarray,
    image_shape: tuple,
    parent=None,
) -> tuple[FFTLatticeOverlay, FFTLatticePanel]:
    """
    Create a reciprocal-space lattice grid overlay on an FFT matplotlib axes.

    Returns (overlay, panel).
    """
    Ny, Nx = image_shape
    cx_px, cy_px = Nx / 2.0, Ny / 2.0
    size_px = min(Nx, Ny) * 0.12

    cal = ReciprocalCalibration(
        qx_axis=qx_axis, qy_axis=qy_axis,
        image_width=Nx, image_height=Ny,
    )
    grid = LatticeGrid.make_square(cx_px, cy_px, size_px, space="reciprocal")

    overlay = FFTLatticeOverlay(ax, canvas, qx_axis, qy_axis, Nx, Ny)
    overlay.set_grid(grid)

    panel = FFTLatticePanel(overlay, cal, Nx, Ny, parent=parent)
    return overlay, panel
