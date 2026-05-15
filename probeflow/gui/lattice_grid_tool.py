"""
Lattice/Grid measurement overlay tool for ProbeFlow.

Entry points:
  open_real_space_tool(canvas, scan_range_m, image_shape, parent)
  open_fft_tool(fft_dialog)

Both return a LatticeGridPanel (QDockWidget) that can be shown/hidden.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor, QFont, QPainter, QPainterPath, QPen, QBrush,
    QCursor,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
    QGraphicsObject,
)

from probeflow.analysis.lattice_grid import (
    LatticeGrid, LatticeKind, RealSpaceCalibration, ReciprocalCalibration,
    format_real_space_measurements, format_reciprocal_measurements,
)

# ── colours ───────────────────────────────────────────────────────────────────

_COL_GRID    = QColor("#89b4fa")  # blue – lattice lines
_COL_A       = QColor("#a6e3a1")  # green – a vector
_COL_B       = QColor("#fab387")  # peach – b vector
_COL_ORIGIN  = QColor("#f38ba8")  # red   – origin handle
_COL_ROT     = QColor("#cba6f7")  # purple – rotation handle
_COL_SCALE   = QColor("#f9e2af")  # yellow – scale handle
_COL_LABEL   = QColor("#cdd6f4")  # text

# Handle screen radius in pixels
_HANDLE_R = 6.0
# Number of grid cells to draw on each side
_GRID_HALF = 12


# ── drag-handle enumeration ───────────────────────────────────────────────────

_HANDLE_NONE   = 0
_HANDLE_ORIGIN = 1
_HANDLE_A      = 2
_HANDLE_B      = 3
_HANDLE_ROT    = 4
_HANDLE_SCALE  = 5


# ── helper: lattice-line drawing range ───────────────────────────────────────

def _grid_range(origin_px, a_px, b_px, image_w: int, image_h: int):
    """
    Return (n_min, n_max, m_min, m_max) such that drawing lines
    n in [n_min, n_max] (parallel to b) and m in [m_min, m_max]
    (parallel to a) is guaranteed to cover the image.
    """
    ox, oy = origin_px
    ax, ay = a_px
    bx, by = b_px

    corners = [(0.0, 0.0), (image_w, 0.0), (0.0, image_h), (image_w, image_h)]
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)

    n_vals, m_vals = [], []
    for cx, cy in corners:
        dx, dy = cx - ox, cy - oy
        # Project on a-unit and b-unit
        if la > 1e-6:
            na = (dx * ax + dy * ay) / (la ** 2)
            n_vals.append(na)
        if lb > 1e-6:
            mb = (dx * bx + dy * by) / (lb ** 2)
            m_vals.append(mb)

    pad = 2
    if n_vals:
        n_min = max(-_GRID_HALF, math.floor(min(n_vals)) - pad)
        n_max = min( _GRID_HALF, math.ceil (max(n_vals)) + pad)
    else:
        n_min, n_max = -_GRID_HALF, _GRID_HALF
    if m_vals:
        m_min = max(-_GRID_HALF, math.floor(min(m_vals)) - pad)
        m_max = min( _GRID_HALF, math.ceil (max(m_vals)) + pad)
    else:
        m_min, m_max = -_GRID_HALF, _GRID_HALF

    return n_min, n_max, m_min, m_max


# ── QGraphicsObject overlay ────────────────────────────────────────────────────

class LatticeGridItem(QGraphicsObject):
    """
    Interactive lattice grid overlay for ImageCanvas (QGraphicsScene).

    Scene coordinates = image pixel coordinates (1 px = 1 scene unit).
    """

    grid_changed = Signal(object)   # emits LatticeGrid after any change

    def __init__(self, grid: LatticeGrid, image_w: int, image_h: int,
                 parent=None):
        super().__init__(parent)
        self._grid = grid
        self._image_w = image_w
        self._image_h = image_h

        self._active_handle = _HANDLE_NONE
        self._drag_start_scene: Optional[QPointF] = None
        self._drag_grid_start: Optional[LatticeGrid] = None

        self.setZValue(50)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsObject.ItemIsMovable, False)
        self.setFlag(QGraphicsObject.ItemIsFocusable, True)

    # ── grid access ───────────────────────────────────────────────────────────

    def grid(self) -> LatticeGrid:
        return self._grid

    def set_grid(self, grid: LatticeGrid) -> None:
        self._grid = grid
        self.update()

    def set_image_size(self, w: int, h: int) -> None:
        self._image_w = w
        self._image_h = h
        self.update()

    # ── QGraphicsItem interface ───────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        # Cover the whole scene (generous for line drawing)
        m = max(self._image_w, self._image_h) * 2.0
        return QRectF(-m, -m, m * 3, m * 3)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        if not self._grid.visible:
            return

        painter.setRenderHint(QPainter.Antialiasing)

        # Current zoom level (to keep handle screen size constant)
        zoom = painter.worldTransform().m11()
        r = _HANDLE_R / max(zoom, 0.01)   # handle radius in scene units

        grid = self._grid
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px

        self._paint_grid_lines(painter, grid, zoom)
        self._paint_basis_vectors(painter, grid, r)

        if grid.show_labels:
            self._paint_labels(painter, grid, r)

        if grid.show_handles:
            self._paint_handles(painter, grid, r)

    def _paint_grid_lines(self, painter: QPainter, grid: LatticeGrid, zoom: float) -> None:
        pen = QPen(_COL_GRID, 1.0 / max(zoom, 0.01))
        pen.setCosmetic(True)
        painter.setPen(pen)

        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px
        n_min, n_max, m_min, m_max = _grid_range(
            grid.origin_px, grid.a_px, grid.b_px, self._image_w, self._image_h
        )

        # Lines parallel to b-vector (indexed by n along a)
        for n in range(n_min, n_max + 1):
            sx = ox + n * ax + m_min * bx
            sy = oy + n * ay + m_min * by
            ex = ox + n * ax + m_max * bx
            ey = oy + n * ay + m_max * by
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

        # Lines parallel to a-vector (indexed by m along b)
        for m in range(m_min, m_max + 1):
            sx = ox + n_min * ax + m * bx
            sy = oy + n_min * ay + m * by
            ex = ox + n_max * ax + m * bx
            ey = oy + n_max * ay + m * by
            painter.drawLine(QPointF(sx, sy), QPointF(ex, ey))

    def _paint_basis_vectors(self, painter: QPainter, grid: LatticeGrid, r: float) -> None:
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px

        pen_a = QPen(_COL_A, r * 0.35)
        pen_a.setCosmetic(False)
        painter.setPen(pen_a)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + ax, oy + ay))

        pen_b = QPen(_COL_B, r * 0.35)
        pen_b.setCosmetic(False)
        painter.setPen(pen_b)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + bx, oy + by))

    def _paint_handles(self, painter: QPainter, grid: LatticeGrid, r: float) -> None:
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px

        handles = self._handle_positions(grid)

        colours = {
            _HANDLE_ORIGIN: _COL_ORIGIN,
            _HANDLE_A:      _COL_A,
            _HANDLE_B:      _COL_B,
            _HANDLE_ROT:    _COL_ROT,
            _HANDLE_SCALE:  _COL_SCALE,
        }

        for hid, (hx, hy) in handles.items():
            col = colours[hid]
            painter.setPen(QPen(col.darker(120), r * 0.2))
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(hx, hy), r, r)

    def _paint_labels(self, painter: QPainter, grid: LatticeGrid, r: float) -> None:
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px

        painter.setPen(QPen(_COL_LABEL))

        font = QFont("Helvetica", 1)
        font_size = max(0.5, r * 1.8)
        font.setPointSizeF(font_size)
        painter.setFont(font)

        prefix = "g1" if grid.space == "reciprocal" else "a"
        painter.drawText(
            QPointF(ox + ax + r * 0.5, oy + ay), prefix
        )
        prefix = "g2" if grid.space == "reciprocal" else "b"
        painter.drawText(
            QPointF(ox + bx + r * 0.5, oy + by), prefix
        )
        painter.drawText(QPointF(ox + r * 0.5, oy), "O")

    # ── handle positions ──────────────────────────────────────────────────────

    def _handle_positions(self, grid: LatticeGrid) -> dict[int, tuple[float, float]]:
        ox, oy = grid.origin_px
        ax, ay = grid.a_px
        bx, by = grid.b_px
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        avg_l = (la + lb) * 0.5 or 1.0

        # Rotation handle: perpendicular to a, at distance ~avg_l * 0.6 from origin
        if la > 1e-6:
            # Perpendicular to a, pointing "outward"
            perp_ax = -ay / la
            perp_ay =  ax / la
            rot_dist = avg_l * 0.65
        else:
            perp_ax, perp_ay = 0.0, 1.0
            rot_dist = avg_l * 0.65

        rx = ox + perp_ax * rot_dist
        ry = oy + perp_ay * rot_dist

        # Scale handle: at 0.55 * (a + b) from origin
        sx = ox + (ax + bx) * 0.55
        sy = oy + (ay + by) * 0.55

        return {
            _HANDLE_ORIGIN: (ox, oy),
            _HANDLE_A:      (ox + ax, oy + ay),
            _HANDLE_B:      (ox + bx, oy + by),
            _HANDLE_ROT:    (rx, ry),
            _HANDLE_SCALE:  (sx, sy),
        }

    def _hit_handle(self, scene_pos: QPointF) -> int:
        """Return the handle id hit by scene_pos, or _HANDLE_NONE."""
        if not self._grid.show_handles:
            return _HANDLE_NONE

        # Convert handle radius from screen to scene
        zoom = self.scene().views()[0].transform().m11() if self.scene() and self.scene().views() else 1.0
        r_scene = _HANDLE_R * 1.5 / max(zoom, 0.01)

        px, py = scene_pos.x(), scene_pos.y()
        handles = self._handle_positions(self._grid)
        for hid in (_HANDLE_ORIGIN, _HANDLE_A, _HANDLE_B, _HANDLE_ROT, _HANDLE_SCALE):
            hx, hy = handles[hid]
            if math.hypot(px - hx, py - hy) <= r_scene:
                return hid
        return _HANDLE_NONE

    # ── mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        hid = self._hit_handle(event.scenePos())
        if hid == _HANDLE_NONE:
            super().mousePressEvent(event)
            return
        self._active_handle = hid
        self._drag_start_scene = QPointF(event.scenePos())
        self._drag_grid_start = self._grid
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._active_handle == _HANDLE_NONE or self._drag_start_scene is None:
            super().mouseMoveEvent(event)
            return

        sp = event.scenePos()
        dx = sp.x() - self._drag_start_scene.x()
        dy = sp.y() - self._drag_start_scene.y()
        g0 = self._drag_grid_start

        if self._active_handle == _HANDLE_ORIGIN:
            self._grid = g0.translate(dx, dy)

        elif self._active_handle == _HANDLE_A:
            # New a endpoint is at origin + new_a
            ox, oy = g0.origin_px
            new_ax = g0.a_px[0] + dx
            new_ay = g0.a_px[1] + dy
            self._grid = g0.with_a_vector((new_ax, new_ay))

        elif self._active_handle == _HANDLE_B:
            new_bx = g0.b_px[0] + dx
            new_by = g0.b_px[1] + dy
            self._grid = g0.with_b_vector((new_bx, new_by))

        elif self._active_handle == _HANDLE_ROT:
            # Angle from origin to current mouse position vs drag start
            ox, oy = g0.origin_px
            angle_start = math.degrees(math.atan2(
                self._drag_start_scene.y() - oy,
                self._drag_start_scene.x() - ox,
            ))
            angle_now = math.degrees(math.atan2(sp.y() - oy, sp.x() - ox))
            delta = angle_now - angle_start
            self._grid = g0.rotate(delta)

        elif self._active_handle == _HANDLE_SCALE:
            # Distance from origin to drag-start vs current
            ox, oy = g0.origin_px
            d0 = math.hypot(
                self._drag_start_scene.x() - ox,
                self._drag_start_scene.y() - oy,
            )
            d1 = math.hypot(sp.x() - ox, sp.y() - oy)
            if d0 > 1e-6:
                factor = d1 / d0
                self._grid = g0.scale(factor)

        self.update()
        self.grid_changed.emit(self._grid)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._active_handle != _HANDLE_NONE:
            self._active_handle = _HANDLE_NONE
            self._drag_start_scene = None
            self._drag_grid_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event) -> None:
        hid = self._hit_handle(event.scenePos())
        if hid != _HANDLE_NONE:
            self.setCursor(Qt.SizeAllCursor if hid == _HANDLE_ORIGIN else Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        super().hoverMoveEvent(event)


# ── measurement panel ─────────────────────────────────────────────────────────

class LatticeGridPanel(QWidget):
    """
    Control and measurement panel for the lattice grid tool.

    Works for both real-space (RealSpaceCalibration) and reciprocal-space
    (ReciprocalCalibration) grids.
    """

    def __init__(
        self,
        grid_item: LatticeGridItem,
        calibration,                # RealSpaceCalibration | ReciprocalCalibration
        image_w: int,
        image_h: int,
        parent=None,
    ):
        super().__init__(parent)
        self._item = grid_item
        self._cal = calibration
        self._image_w = image_w
        self._image_h = image_h

        self._build()
        self._item.grid_changed.connect(self._on_grid_changed)
        self._refresh_measurements()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        # ── lattice type ──────────────────────────────────────────────────────
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Square", "Rectangular", "Hexagonal"])
        self._type_combo.setFont(QFont("Helvetica", 9))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_row.addWidget(self._type_combo, 1)
        lay.addLayout(type_row)

        # ── visibility toggles ────────────────────────────────────────────────
        self._show_grid_cb    = QCheckBox("Show grid")
        self._show_grid_cb.setChecked(True)
        self._show_grid_cb.toggled.connect(self._on_visibility_changed)

        self._show_handles_cb = QCheckBox("Show handles")
        self._show_handles_cb.setChecked(True)
        self._show_handles_cb.toggled.connect(self._on_visibility_changed)

        self._show_labels_cb  = QCheckBox("Show labels")
        self._show_labels_cb.setChecked(True)
        self._show_labels_cb.toggled.connect(self._on_visibility_changed)

        for cb in (self._show_grid_cb, self._show_handles_cb, self._show_labels_cb):
            cb.setFont(QFont("Helvetica", 9))
            lay.addWidget(cb)

        # ── action buttons ────────────────────────────────────────────────────
        reset_btn = QPushButton("Reset origin to centre")
        reset_btn.setFont(QFont("Helvetica", 9))
        reset_btn.setFixedHeight(24)
        reset_btn.clicked.connect(self._on_reset_origin)
        lay.addWidget(reset_btn)

        # ── measurements display ──────────────────────────────────────────────
        self._meas_lbl = QLabel("")
        self._meas_lbl.setFont(QFont("Courier", 8))
        self._meas_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._meas_lbl.setWordWrap(True)
        self._meas_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self._meas_lbl)

        # ── export buttons ────────────────────────────────────────────────────
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

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_grid_changed(self, grid: LatticeGrid) -> None:
        self._refresh_measurements()

    def _on_type_changed(self, idx: int) -> None:
        kinds: list[LatticeKind] = ["square", "rectangular", "hexagonal"]
        new_kind = kinds[idx]
        g = self._item.grid()
        la = g.a_length_px()
        lb = g.b_length_px()
        angle_a = g.a_angle_deg()

        if new_kind == "square":
            new_g = g.with_a_vector((la * math.cos(math.radians(angle_a)),
                                     la * math.sin(math.radians(angle_a))))
        elif new_kind == "rectangular":
            # Keep lengths but enforce orthogonality
            new_g = LatticeGrid(
                kind="rectangular", space=g.space,
                origin_px=g.origin_px,
                a_px=g.a_px,
                b_px=(lb * math.cos(math.radians(angle_a + 90)),
                      lb * math.sin(math.radians(angle_a + 90))),
                visible=g.visible, show_labels=g.show_labels, show_handles=g.show_handles,
            )
        else:  # hexagonal
            avg_l = (la + lb) * 0.5
            new_g = LatticeGrid.make_hexagonal(
                *g.origin_px, avg_l, angle_deg=angle_a, space=g.space,
            )
            new_g = LatticeGrid(
                kind="hexagonal", space=g.space,
                origin_px=new_g.origin_px,
                a_px=new_g.a_px, b_px=new_g.b_px,
                visible=g.visible, show_labels=g.show_labels, show_handles=g.show_handles,
            )

        from dataclasses import replace
        new_g = replace(new_g, kind=new_kind)
        self._item.set_grid(new_g)
        self._refresh_measurements()

    def _on_visibility_changed(self) -> None:
        from dataclasses import replace
        g = self._item.grid()
        self._item.set_grid(replace(
            g,
            visible=self._show_grid_cb.isChecked(),
            show_handles=self._show_handles_cb.isChecked(),
            show_labels=self._show_labels_cb.isChecked(),
        ))

    def _on_reset_origin(self) -> None:
        from dataclasses import replace
        g = self._item.grid()
        self._item.set_grid(g.reset_origin(self._image_w / 2.0, self._image_h / 2.0))
        self._refresh_measurements()

    def _on_export_with_grid(self) -> None:
        self._export(include_grid=True)

    def _on_export_grid_only(self) -> None:
        self._export(include_grid=False, grid_only=True)

    def _export(self, include_grid: bool = True, grid_only: bool = False) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "",
            "PNG image (*.png);;PDF document (*.pdf)",
        )
        if not path:
            return
        from probeflow.gui.lattice_export import export_grid
        export_grid(self._item, path, include_grid=include_grid, grid_only=grid_only)

    # ── measurement display ────────────────────────────────────────────────────

    def _refresh_measurements(self) -> None:
        grid = self._item.grid()
        try:
            if isinstance(self._cal, RealSpaceCalibration):
                d = format_real_space_measurements(grid, self._cal)
                lines = [
                    f"Type:    {d['kind']}",
                    f"Space:   real",
                    f"Origin:  {d['origin_px']}",
                    f"         {d['origin_phys']}",
                    f"a:       {d['a_px']}",
                    f"b:       {d['b_px']}",
                    f"|a|:     {d['a_length']}",
                    f"|b|:     {d['b_length']}",
                    f"angle:   {d['angle']}",
                    f"area:    {d['area']}",
                ]
            elif isinstance(self._cal, ReciprocalCalibration):
                d = format_reciprocal_measurements(grid, self._cal)
                lines = [
                    f"Type:    {d['kind']}",
                    f"Space:   reciprocal",
                    f"Origin:  {d['origin_px']}",
                    f"         {d['origin_q']}",
                    f"|g1|:    {d['g1']}",
                    f"|g2|:    {d['g2']}",
                    f"angle:   {d['angle']}",
                    f"area:    {d['area_q']}",
                ]
            else:
                lines = ["No calibration"]
        except Exception as exc:
            lines = [f"(error: {exc})"]
        self._meas_lbl.setText("\n".join(lines))

    def set_calibration(self, cal) -> None:
        self._cal = cal
        self._refresh_measurements()


# ── FFT matplotlib overlay ────────────────────────────────────────────────────

class FFTLatticeOverlay:
    """
    Manages a lattice grid overlay drawn on a matplotlib FFT axes.

    Intended for use with FFTViewerDialog (and subclasses) where the FFT is
    displayed using matplotlib.
    """

    def __init__(self, ax, canvas, qx_axis, qy_axis, image_w: int, image_h: int):
        self._ax = ax
        self._canvas = canvas
        self._qx = qx_axis
        self._qy = qy_axis
        self._image_w = image_w
        self._image_h = image_h
        self._artists: list = []
        self._grid: Optional[LatticeGrid] = None
        self._drag_handle: int = _HANDLE_NONE
        self._drag_start: Optional[tuple[float, float]] = None
        self._drag_grid_start: Optional[LatticeGrid] = None
        self._on_change_cb = None   # callable(LatticeGrid)

        self._cid_press   = canvas.mpl_connect("button_press_event",   self._on_press)
        self._cid_release = canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion  = canvas.mpl_connect("motion_notify_event",  self._on_motion)

    def set_grid(self, grid: LatticeGrid) -> None:
        self._grid = grid
        self.redraw()

    def grid(self) -> Optional[LatticeGrid]:
        return self._grid

    def set_on_change(self, cb) -> None:
        self._on_change_cb = cb

    def disconnect(self) -> None:
        for cid in (self._cid_press, self._cid_release, self._cid_motion):
            try:
                self._canvas.mpl_disconnect(cid)
            except Exception:
                pass

    # ── calibration helpers ───────────────────────────────────────────────────

    def _q_to_px(self, qx: float, qy: float) -> tuple[float, float]:
        """Convert q-space point to FFT pixel indices."""
        Nx = self._image_w
        Ny = self._image_h
        dqx = (float(self._qx[-1]) - float(self._qx[0])) / max(1, Nx - 1)
        dqy = (float(self._qy[-1]) - float(self._qy[0])) / max(1, Ny - 1)
        ix = (qx - float(self._qx[0])) / dqx
        iy = (qy - float(self._qy[0])) / dqy
        return ix, iy

    def _px_to_q(self, ix: float, iy: float) -> tuple[float, float]:
        Nx = self._image_w
        Ny = self._image_h
        dqx = (float(self._qx[-1]) - float(self._qx[0])) / max(1, Nx - 1)
        dqy = (float(self._qy[-1]) - float(self._qy[0])) / max(1, Ny - 1)
        qx = float(self._qx[0]) + ix * dqx
        qy = float(self._qy[0]) + iy * dqy
        return qx, qy

    def _grid_handles_q(self, grid: LatticeGrid) -> dict[int, tuple[float, float]]:
        """Return handle positions in q-space for a pixel-coordinate grid."""
        item_handles = {}
        # Compute as pixel positions then convert to q
        ox_px, oy_px = grid.origin_px
        ax_px, ay_px = grid.a_px
        bx_px, by_px = grid.b_px
        la = math.hypot(ax_px, ay_px)
        lb = math.hypot(bx_px, by_px)
        avg_l = (la + lb) * 0.5 or 1.0
        if la > 1e-6:
            perp_ax = -ay_px / la
            perp_ay =  ax_px / la
        else:
            perp_ax, perp_ay = 0.0, 1.0

        handle_px = {
            _HANDLE_ORIGIN: (ox_px, oy_px),
            _HANDLE_A:      (ox_px + ax_px, oy_px + ay_px),
            _HANDLE_B:      (ox_px + bx_px, oy_px + by_px),
            _HANDLE_ROT:    (ox_px + perp_ax * avg_l * 0.65,
                             oy_px + perp_ay * avg_l * 0.65),
            _HANDLE_SCALE:  (ox_px + (ax_px + bx_px) * 0.55,
                             oy_px + (ay_px + by_px) * 0.55),
        }
        for hid, (px, py) in handle_px.items():
            item_handles[hid] = self._px_to_q(px, py)
        return item_handles

    def _hit_handle_q(self, qx: float, qy: float) -> int:
        if self._grid is None:
            return _HANDLE_NONE
        handles_q = self._grid_handles_q(self._grid)
        ax_range = self._ax.get_xlim()
        ax_width = abs(ax_range[1] - ax_range[0])
        tol = ax_width * 0.03  # 3% of visible range
        for hid in (_HANDLE_ORIGIN, _HANDLE_A, _HANDLE_B, _HANDLE_ROT, _HANDLE_SCALE):
            hqx, hqy = handles_q[hid]
            if math.hypot(qx - hqx, qy - hqy) <= tol:
                return hid
        return _HANDLE_NONE

    # ── matplotlib events ─────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax or self._grid is None:
            return
        hid = self._hit_handle_q(event.xdata, event.ydata)
        if hid == _HANDLE_NONE:
            return
        self._drag_handle = hid
        self._drag_start = (event.xdata, event.ydata)
        self._drag_grid_start = self._grid

    def _on_release(self, event) -> None:
        self._drag_handle = _HANDLE_NONE
        self._drag_start = None
        self._drag_grid_start = None

    def _on_motion(self, event) -> None:
        if (self._drag_handle == _HANDLE_NONE
                or event.inaxes is not self._ax
                or self._drag_start is None
                or self._drag_grid_start is None):
            return

        # Delta in q-space → convert to pixel-space delta
        dqx = event.xdata - self._drag_start[0]
        dqy = event.ydata - self._drag_start[1]
        Nx, Ny = self._image_w, self._image_h
        q_range_x = float(self._qx[-1]) - float(self._qx[0])
        q_range_y = float(self._qy[-1]) - float(self._qy[0])
        dpx = dqx / q_range_x * Nx if q_range_x != 0 else 0.0
        dpy = dqy / q_range_y * Ny if q_range_y != 0 else 0.0

        g0 = self._drag_grid_start
        hid = self._drag_handle

        if hid == _HANDLE_ORIGIN:
            self._grid = g0.translate(dpx, dpy)
        elif hid == _HANDLE_A:
            self._grid = g0.with_a_vector((g0.a_px[0] + dpx, g0.a_px[1] + dpy))
        elif hid == _HANDLE_B:
            self._grid = g0.with_b_vector((g0.b_px[0] + dpx, g0.b_px[1] + dpy))
        elif hid == _HANDLE_ROT:
            ox, oy = g0.origin_px
            s_qx, s_qy = self._drag_start
            angle_start = math.degrees(math.atan2(s_qy - self._px_to_q(ox, 0)[0],
                                                   s_qx - self._px_to_q(0, oy)[1]))
            # Simpler: use q-space angles
            s_ox, s_oy = self._px_to_q(ox, oy)
            angle_start = math.degrees(math.atan2(s_qy - s_oy, s_qx - s_ox))
            angle_now   = math.degrees(math.atan2(event.ydata - s_oy, event.xdata - s_ox))
            self._grid = g0.rotate(angle_now - angle_start)
        elif hid == _HANDLE_SCALE:
            ox, oy = g0.origin_px
            s_ox, s_oy = self._px_to_q(ox, oy)
            d0 = math.hypot(self._drag_start[0] - s_ox, self._drag_start[1] - s_oy)
            d1 = math.hypot(event.xdata - s_ox, event.ydata - s_oy)
            if d0 > 1e-9:
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
        Nx, Ny = self._image_w, self._image_h

        def px2q(ix, iy):
            return self._px_to_q(ix, iy)

        # Determine range
        n_min, n_max, m_min, m_max = _grid_range(
            grid.origin_px, grid.a_px, grid.b_px, Nx, Ny
        )

        # Grid lines — parallel to b
        for n in range(n_min, n_max + 1):
            sx_px = ox_px + n * ax_px + m_min * bx_px
            sy_px = oy_px + n * ay_px + m_min * by_px
            ex_px = ox_px + n * ax_px + m_max * bx_px
            ey_px = oy_px + n * ay_px + m_max * by_px
            sqx, sqy = px2q(sx_px, sy_px)
            eqx, eqy = px2q(ex_px, ey_px)
            art, = self._ax.plot(
                [sqx, eqx], [sqy, eqy], color="#89b4fa", lw=0.8, alpha=0.7, zorder=5
            )
            self._artists.append(art)

        # Grid lines — parallel to a
        for m in range(m_min, m_max + 1):
            sx_px = ox_px + n_min * ax_px + m * bx_px
            sy_px = oy_px + n_min * ay_px + m * by_px
            ex_px = ox_px + n_max * ax_px + m * bx_px
            ey_px = oy_px + n_max * ay_px + m * by_px
            sqx, sqy = px2q(sx_px, sy_px)
            eqx, eqy = px2q(ex_px, ey_px)
            art, = self._ax.plot(
                [sqx, eqx], [sqy, eqy], color="#89b4fa", lw=0.8, alpha=0.7, zorder=5
            )
            self._artists.append(art)

        # Basis vectors
        oqx, oqy = px2q(ox_px, oy_px)
        aqx, aqy = px2q(ox_px + ax_px, oy_px + ay_px)
        bqx, bqy = px2q(ox_px + bx_px, oy_px + by_px)

        art, = self._ax.plot([oqx, aqx], [oqy, aqy], color="#a6e3a1", lw=1.8, zorder=6)
        self._artists.append(art)
        art, = self._ax.plot([oqx, bqx], [oqy, bqy], color="#fab387", lw=1.8, zorder=6)
        self._artists.append(art)

        # Handles
        handles_q = self._grid_handles_q(grid)
        handle_cols = {
            _HANDLE_ORIGIN: "#f38ba8",
            _HANDLE_A:      "#a6e3a1",
            _HANDLE_B:      "#fab387",
            _HANDLE_ROT:    "#cba6f7",
            _HANDLE_SCALE:  "#f9e2af",
        }
        if grid.show_handles:
            for hid, (hqx, hqy) in handles_q.items():
                art, = self._ax.plot(
                    hqx, hqy, "o", color=handle_cols[hid],
                    markersize=8, markeredgewidth=1.0, zorder=8,
                )
                self._artists.append(art)

        # Labels
        if grid.show_labels:
            for text, (hqx, hqy) in [
                ("g1" if grid.space == "reciprocal" else "a", handles_q[_HANDLE_A]),
                ("g2" if grid.space == "reciprocal" else "b", handles_q[_HANDLE_B]),
                ("O",  handles_q[_HANDLE_ORIGIN]),
            ]:
                art = self._ax.text(
                    hqx, hqy, f" {text}", color="#cdd6f4",
                    fontsize=8, va="center", zorder=9,
                )
                self._artists.append(art)

        self._canvas.draw_idle()


# ── export helper ─────────────────────────────────────────────────────────────

def _save_image_via_scene(item: LatticeGridItem, path: str,
                           include_grid: bool, grid_only: bool) -> None:
    """Render the grid overlay (and optionally the scene image) to a file."""
    from PySide6.QtGui import QImage, QPainter as _QPainter, QColor as _QColor
    from PySide6.QtCore import QRectF as _QRectF, Qt as _Qt
    import os

    scene = item.scene()
    if scene is None:
        raise RuntimeError("Item has no scene")

    Nx = item._image_w
    Ny = item._image_h

    img = QImage(Nx, Ny, QImage.Format_ARGB32)
    bg = QColor(0, 0, 0, 0) if grid_only else QColor("black")
    img.fill(bg)

    p = _QPainter(img)
    p.setRenderHint(_QPainter.Antialiasing)
    src_rect = _QRectF(0, 0, Nx, Ny)

    if not grid_only:
        # Render background (pixmap item)
        for scene_item in scene.items():
            from PySide6.QtWidgets import QGraphicsPixmapItem
            if isinstance(scene_item, QGraphicsPixmapItem):
                pixmap = scene_item.pixmap()
                p.drawPixmap(0, 0, Nx, Ny, pixmap)
                break

    if include_grid:
        from PySide6.QtWidgets import QStyleOptionGraphicsItem
        opt = QStyleOptionGraphicsItem()
        item.paint(p, opt)

    p.end()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from PySide6.QtGui import QPdfWriter, QPageSize, QMarginsF
        from PySide6.QtCore import QSizeF
        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QSizeF(Nx / 96.0, Ny / 96.0), QPageSize.Inch))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        pp = _QPainter(writer)
        pp.drawImage(0, 0, img)
        pp.end()
    else:
        img.save(path)


# ── public entry points ───────────────────────────────────────────────────────

def open_real_space_tool(
    canvas,               # ImageCanvas
    scan_range_m: tuple,
    image_shape: tuple,   # (Ny, Nx)
    parent=None,
) -> tuple["LatticeGridItem", "LatticeGridPanel"]:
    """
    Create and attach a lattice grid overlay to an ImageCanvas.

    Returns (item, panel). Caller is responsible for showing the panel
    (e.g. as a dock widget) and connecting panel signals.
    """
    Ny, Nx = image_shape
    cx, cy = Nx / 2.0, Ny / 2.0
    size = min(Nx, Ny) * 0.15

    cal = RealSpaceCalibration.from_scan_range(scan_range_m, Nx, Ny)
    grid = LatticeGrid.make_square(cx, cy, size, space="real")

    item = LatticeGridItem(grid, Nx, Ny)
    canvas.scene().addItem(item)

    panel = LatticeGridPanel(item, cal, Nx, Ny, parent=parent)
    return item, panel


def open_fft_tool(
    ax,
    canvas,
    qx_axis,
    qy_axis,
    image_shape: tuple,   # (Ny, Nx)
    parent=None,
) -> tuple["FFTLatticeOverlay", "LatticeGridPanel"]:
    """
    Create and attach a lattice grid overlay to an FFT matplotlib axes.

    Returns (overlay, panel). Caller shows the panel.
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

    # Wrap overlay in a fake "item" interface for LatticeGridPanel
    class _FFTItemAdapter:
        def __init__(self, ov, img_w, img_h):
            self._ov = ov
            self._img_w = img_w
            self._img_h = img_h
            self._cbs: list = []

        def grid(self):
            return self._ov.grid()

        def set_grid(self, g):
            self._ov.set_grid(g)
            for cb in self._cbs:
                cb(g)

        def scene(self):
            return None

        grid_changed = None  # panel connects differently below

        def connect_changed(self, cb):
            self._cbs.append(cb)
            self._ov.set_on_change(lambda g: [c(g) for c in self._cbs])

    adapter = _FFTItemAdapter(overlay, Nx, Ny)

    # Build a panel that works with the adapter via a duck-typed interface
    panel = _FFTPanelWrapper(adapter, cal, Nx, Ny, parent=parent)
    adapter.connect_changed(panel._on_grid_changed)
    return overlay, panel


class _FFTPanelWrapper(LatticeGridPanel):
    """LatticeGridPanel subclass that works with FFTLatticeOverlay via adapter."""

    def __init__(self, adapter, cal, image_w, image_h, parent=None):
        # We need to bypass the grid_changed signal wiring in __init__
        # because the adapter duck-types the item interface.
        self._adapter = adapter
        QWidget.__init__(self, parent)
        self._item = adapter          # type: ignore[assignment]
        self._cal = cal
        self._image_w = image_w
        self._image_h = image_h
        self._build()
        self._refresh_measurements()

    def _on_grid_changed(self, grid: LatticeGrid) -> None:
        self._refresh_measurements()

    def _on_visibility_changed(self) -> None:
        from dataclasses import replace
        g = self._item.grid()
        self._item.set_grid(replace(
            g,
            visible=self._show_grid_cb.isChecked(),
            show_handles=self._show_handles_cb.isChecked(),
            show_labels=self._show_labels_cb.isChecked(),
        ))

    def _on_type_changed(self, idx: int) -> None:
        kinds: list[LatticeKind] = ["square", "rectangular", "hexagonal"]
        new_kind = kinds[idx]
        g = self._item.grid()
        la = g.a_length_px()
        lb = g.b_length_px()
        angle_a = g.a_angle_deg()
        if new_kind == "square":
            new_g = g.with_a_vector((la * math.cos(math.radians(angle_a)),
                                     la * math.sin(math.radians(angle_a))))
        elif new_kind == "rectangular":
            from dataclasses import replace
            new_g = LatticeGrid(
                kind="rectangular", space=g.space,
                origin_px=g.origin_px, a_px=g.a_px,
                b_px=(lb * math.cos(math.radians(angle_a + 90)),
                      lb * math.sin(math.radians(angle_a + 90))),
                visible=g.visible, show_labels=g.show_labels,
                show_handles=g.show_handles,
            )
        else:
            avg_l = (la + lb) * 0.5
            new_g = LatticeGrid.make_hexagonal(
                *g.origin_px, avg_l, angle_deg=angle_a, space=g.space,
            )
            from dataclasses import replace
            new_g = replace(new_g, kind=new_kind, visible=g.visible,
                           show_labels=g.show_labels, show_handles=g.show_handles)
        self._item.set_grid(new_g)
        self._refresh_measurements()

    def _on_reset_origin(self) -> None:
        from dataclasses import replace
        g = self._item.grid()
        self._item.set_grid(g.reset_origin(self._image_w / 2.0, self._image_h / 2.0))
        self._refresh_measurements()

    def _on_export_with_grid(self) -> None:
        # FFT export not implemented in first version
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Export", "FFT grid export not yet implemented.")

    def _on_export_grid_only(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Export", "FFT grid export not yet implemented.")
