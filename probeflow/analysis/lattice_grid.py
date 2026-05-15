"""
Lattice grid measurement model for ProbeFlow.

GUI-independent.  All geometry is stored in pixel coordinates.
Physical units are produced by passing a calibration object into the
measurement helper functions at the bottom of this module.

Coordinate convention:
  x → right (column index)
  y → down  (row index)
  Angles are measured counter-clockwise from the +x axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

# ── type aliases ──────────────────────────────────────────────────────────────

LatticeKind = Literal["square", "rectangular", "hexagonal"]
GridSpace    = Literal["real", "reciprocal"]

Vec2 = tuple[float, float]

# ── data model ────────────────────────────────────────────────────────────────


@dataclass
class LatticeGrid:
    """
    Minimal, GUI-free representation of a 2-D lattice/grid overlay.

    All positions and vectors are in *image pixel* coordinates:
      origin_px  — position of the grid origin
      a_px       — first basis vector  (tip at origin_px + a_px)
      b_px       — second basis vector (tip at origin_px + b_px)

    The model is immutable-style: every operation returns a new instance.
    """

    kind:        LatticeKind
    space:       GridSpace
    origin_px:   Vec2
    a_px:        Vec2
    b_px:        Vec2
    visible:     bool = True
    show_labels: bool = True
    show_handles: bool = True

    # ── factories ──────────────────────────────────────────────────────────────

    @staticmethod
    def make_square(
        cx: float, cy: float, size_px: float, space: GridSpace = "real",
    ) -> "LatticeGrid":
        """Square lattice with horizontal a-vector."""
        return LatticeGrid(
            kind="square", space=space,
            origin_px=(cx, cy),
            a_px=(size_px, 0.0),
            b_px=(0.0, size_px),
        )

    @staticmethod
    def make_rectangular(
        cx: float, cy: float, ax_px: float, by_px: float, space: GridSpace = "real",
    ) -> "LatticeGrid":
        """Rectangular lattice with axes aligned to image axes."""
        return LatticeGrid(
            kind="rectangular", space=space,
            origin_px=(cx, cy),
            a_px=(ax_px, 0.0),
            b_px=(0.0, by_px),
        )

    @staticmethod
    def make_hexagonal(
        cx: float, cy: float, size_px: float,
        angle_deg: float = 0.0, space: GridSpace = "real",
    ) -> "LatticeGrid":
        """
        Hexagonal lattice.

        a is at angle_deg from +x; b is at angle_deg + 60°.
        Both vectors have length size_px.
        """
        a_rad = math.radians(angle_deg)
        b_rad = math.radians(angle_deg + 60.0)
        return LatticeGrid(
            kind="hexagonal", space=space,
            origin_px=(cx, cy),
            a_px=(size_px * math.cos(a_rad), size_px * math.sin(a_rad)),
            b_px=(size_px * math.cos(b_rad), size_px * math.sin(b_rad)),
        )

    # ── rigid-body operations ─────────────────────────────────────────────────

    def translate(self, dx: float, dy: float) -> "LatticeGrid":
        """Translate the grid origin; basis vectors unchanged."""
        ox, oy = self.origin_px
        return replace(self, origin_px=(ox + dx, oy + dy))

    def rotate(self, angle_deg: float) -> "LatticeGrid":
        """Rotate both basis vectors about the origin."""
        c = math.cos(math.radians(angle_deg))
        s = math.sin(math.radians(angle_deg))
        ax, ay = self.a_px
        bx, by = self.b_px
        return replace(
            self,
            a_px=(c * ax - s * ay, s * ax + c * ay),
            b_px=(c * bx - s * by, s * bx + c * by),
        )

    def scale(self, factor: float) -> "LatticeGrid":
        """Uniform scale of both basis vectors about the origin."""
        ax, ay = self.a_px
        bx, by = self.b_px
        return replace(
            self,
            a_px=(ax * factor, ay * factor),
            b_px=(bx * factor, by * factor),
        )

    # ── constrained handle moves ───────────────────────────────────────────────

    def with_a_vector(self, new_a: Vec2) -> "LatticeGrid":
        """
        Replace the a vector and re-enforce the lattice constraint.

        For square:      b is set perpendicular to a with the same length.
        For rectangular: b direction is updated (stay perpendicular to new a);
                         b length is preserved.
        For hexagonal:   both lengths equal; b is 60° ahead of a.
        """
        ax, ay = new_a
        la = math.hypot(ax, ay)
        if la < 1e-9:
            return self
        angle_a = math.atan2(ay, ax)

        if self.kind == "square":
            return replace(
                self,
                a_px=(la * math.cos(angle_a), la * math.sin(angle_a)),
                b_px=(la * math.cos(angle_a + math.pi / 2),
                      la * math.sin(angle_a + math.pi / 2)),
            )
        elif self.kind == "rectangular":
            lb = math.hypot(*self.b_px)
            if lb < 1e-9:
                lb = la
            return replace(
                self,
                a_px=new_a,
                b_px=(lb * math.cos(angle_a + math.pi / 2),
                      lb * math.sin(angle_a + math.pi / 2)),
            )
        elif self.kind == "hexagonal":
            return replace(
                self,
                a_px=(la * math.cos(angle_a), la * math.sin(angle_a)),
                b_px=(la * math.cos(angle_a + math.pi / 3),
                      la * math.sin(angle_a + math.pi / 3)),
            )
        return replace(self, a_px=new_a)

    def with_b_vector(self, new_b: Vec2) -> "LatticeGrid":
        """
        Replace the b vector and re-enforce the lattice constraint.

        For square:      a is set perpendicular to b (b − 90°) with same length.
        For rectangular: a direction updated (stay perpendicular to new b);
                         a length preserved.
        For hexagonal:   both lengths equal; a is 60° behind b.
        """
        bx, by = new_b
        lb = math.hypot(bx, by)
        if lb < 1e-9:
            return self
        angle_b = math.atan2(by, bx)

        if self.kind == "square":
            return replace(
                self,
                a_px=(lb * math.cos(angle_b - math.pi / 2),
                      lb * math.sin(angle_b - math.pi / 2)),
                b_px=(lb * math.cos(angle_b), lb * math.sin(angle_b)),
            )
        elif self.kind == "rectangular":
            la = math.hypot(*self.a_px)
            if la < 1e-9:
                la = lb
            return replace(
                self,
                a_px=(la * math.cos(angle_b - math.pi / 2),
                      la * math.sin(angle_b - math.pi / 2)),
                b_px=new_b,
            )
        elif self.kind == "hexagonal":
            return replace(
                self,
                a_px=(lb * math.cos(angle_b - math.pi / 3),
                      lb * math.sin(angle_b - math.pi / 3)),
                b_px=(lb * math.cos(angle_b), lb * math.sin(angle_b)),
            )
        return replace(self, b_px=new_b)

    # ── measurements ──────────────────────────────────────────────────────────

    def a_length_px(self) -> float:
        return math.hypot(*self.a_px)

    def b_length_px(self) -> float:
        return math.hypot(*self.b_px)

    def angle_deg(self) -> float:
        """Angle between a and b vectors in degrees (0–180)."""
        ax, ay = self.a_px
        bx, by = self.b_px
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la < 1e-9 or lb < 1e-9:
            return 0.0
        dot = ax * bx + ay * by
        cos_theta = max(-1.0, min(1.0, dot / (la * lb)))
        return math.degrees(math.acos(cos_theta))

    def area_px2(self) -> float:
        """Unit-cell area in pixels² (= |a × b|)."""
        ax, ay = self.a_px
        bx, by = self.b_px
        return abs(ax * by - ay * bx)

    def a_angle_deg(self) -> float:
        """Angle of a-vector from +x axis (−180 to 180)."""
        return math.degrees(math.atan2(self.a_px[1], self.a_px[0]))

    def b_angle_deg(self) -> float:
        """Angle of b-vector from +x axis (−180 to 180)."""
        return math.degrees(math.atan2(self.b_px[1], self.b_px[0]))

    def reset_origin(self, cx: float, cy: float) -> "LatticeGrid":
        """Move origin to (cx, cy) without changing basis vectors."""
        return replace(self, origin_px=(cx, cy))

    def set_a_length_px(self, length_px: float) -> "LatticeGrid":
        """
        Set |a| in pixels, preserving a-vector direction and lattice constraints.

        Square/hexagonal: scales both vectors (preserves b/a relationship).
        Rectangular: scales only a-vector; b is unchanged.
        """
        la = self.a_length_px()
        if la < 1e-9 or length_px < 1e-9:
            return self
        factor = length_px / la
        if self.kind in ("square", "hexagonal"):
            return self.scale(factor)
        ax, ay = self.a_px
        return replace(self, a_px=(ax * factor, ay * factor))

    def set_b_length_px(self, length_px: float) -> "LatticeGrid":
        """
        Set |b| in pixels, preserving b-vector direction and lattice constraints.

        Square/hexagonal: scales both vectors.
        Rectangular: scales only b-vector; a is unchanged.
        """
        lb = self.b_length_px()
        if lb < 1e-9 or length_px < 1e-9:
            return self
        factor = length_px / lb
        if self.kind in ("square", "hexagonal"):
            return self.scale(factor)
        bx, by = self.b_px
        return replace(self, b_px=(bx * factor, by * factor))

    def set_rotation_deg(self, angle_deg: float) -> "LatticeGrid":
        """Rotate the grid so the a-vector points at angle_deg from +x."""
        delta = angle_deg - self.a_angle_deg()
        return self.rotate(delta)


# ── calibration adapters ──────────────────────────────────────────────────────


@dataclass
class RealSpaceCalibration:
    """
    Pixel → physical unit conversion for a real-space image.

    px_size_x, px_size_y: metres per pixel (from scan_range_m / image shape).
    """

    px_size_x: float   # m / px
    px_size_y: float   # m / px
    image_width:  int
    image_height: int

    @staticmethod
    def from_scan_range(
        scan_range_m: tuple[float, float],
        image_width: int,
        image_height: int,
    ) -> "RealSpaceCalibration":
        return RealSpaceCalibration(
            px_size_x=float(scan_range_m[0]) / image_width,
            px_size_y=float(scan_range_m[1]) / image_height,
            image_width=image_width,
            image_height=image_height,
        )

    def vector_length_m(self, vec_px: Vec2) -> float:
        """Physical length of a pixel-space vector in metres."""
        vx, vy = vec_px
        wx = vx * self.px_size_x
        wy = vy * self.px_size_y
        return math.hypot(wx, wy)

    def origin_m(self, origin_px: Vec2) -> Vec2:
        """Origin position in metres."""
        return (origin_px[0] * self.px_size_x, origin_px[1] * self.px_size_y)


@dataclass
class ReciprocalCalibration:
    """
    FFT pixel → reciprocal-space conversion.

    qx_axis, qy_axis: 1-D arrays of q values in nm⁻¹ (shape [Nx] and [Ny]).
    centre_px: (cx_px, cy_px) position of DC/zero in FFT array indices.
    """

    qx_axis: np.ndarray   # nm⁻¹, length Nx
    qy_axis: np.ndarray   # nm⁻¹, length Ny
    image_width:  int
    image_height: int

    @property
    def centre_px(self) -> Vec2:
        return (self.image_width / 2.0, self.image_height / 2.0)

    def px_to_q(self, ix: float, iy: float) -> Vec2:
        """Convert FFT pixel indices (float) to q-space (nm⁻¹)."""
        dqx = (self.qx_axis[-1] - self.qx_axis[0]) / max(1, self.image_width  - 1)
        dqy = (self.qy_axis[-1] - self.qy_axis[0]) / max(1, self.image_height - 1)
        return (float(self.qx_axis[0]) + ix * dqx,
                float(self.qy_axis[0]) + iy * dqy)

    def vec_px_to_q(self, vec_px: Vec2) -> Vec2:
        """
        Convert a pixel-space displacement vector to q-space displacement.

        (Does not offset by centre; for differences only.)
        """
        dqx = (self.qx_axis[-1] - self.qx_axis[0]) / max(1, self.image_width  - 1)
        dqy = (self.qy_axis[-1] - self.qy_axis[0]) / max(1, self.image_height - 1)
        vx, vy = vec_px
        return (vx * dqx, vy * dqy)

    def vec_length_q(self, vec_px: Vec2) -> float:
        """Length of a pixel-space vector in q-space (nm⁻¹)."""
        qvx, qvy = self.vec_px_to_q(vec_px)
        return math.hypot(qvx, qvy)


# ── measurement formatting ────────────────────────────────────────────────────


def _fmt(value_m: float, unit: str, decimals: int = 3) -> str:
    return f"{value_m:.{decimals}g} {unit}"


def _choose_unit(value_m: float) -> tuple[float, str]:
    """Return (scale_factor, unit_string) for the most readable representation."""
    value_nm = value_m * 1e9
    if value_nm < 0.1:
        return (1e10, "Å")
    elif value_nm < 100.0:
        return (1e10, "Å") if value_nm < 1.0 else (1e9, "nm")
    else:
        return (1e9, "nm")


def format_real_space_measurements(
    grid: LatticeGrid,
    cal: RealSpaceCalibration,
) -> dict[str, str]:
    """
    Return a dict of formatted measurement strings for a real-space grid.

    Keys: kind, space, origin_px, origin_phys, a_px, b_px,
          a_length, b_length, angle, area.
    """
    ox, oy = grid.origin_px
    ax, ay = grid.a_px
    bx, by = grid.b_px

    la_m = cal.vector_length_m(grid.a_px)
    lb_m = cal.vector_length_m(grid.b_px)
    area_m2 = (
        abs((ax * cal.px_size_x) * (by * cal.px_size_y)
            - (ay * cal.px_size_y) * (bx * cal.px_size_x))
    )
    angle = grid.angle_deg()

    # Choose unit based on a-vector
    scale, unit = _choose_unit(la_m)
    area_unit = unit + "²"
    area_scale = scale ** 2

    return {
        "kind":       grid.kind,
        "space":      grid.space,
        "origin_px":  f"({ox:.1f}, {oy:.1f}) px",
        "origin_phys": (
            f"({ox * cal.px_size_x * scale:.3g}, "
            f"{oy * cal.px_size_y * scale:.3g}) {unit}"
        ),
        "a_px":       f"({ax:.2f}, {ay:.2f}) px",
        "b_px":       f"({bx:.2f}, {by:.2f}) px",
        "a_length":   f"{la_m * scale:.4g} {unit}",
        "b_length":   f"{lb_m * scale:.4g} {unit}",
        "angle":      f"{angle:.2g}°",
        "area":       f"{area_m2 * area_scale:.4g} {area_unit}",
    }


def format_reciprocal_measurements(
    grid: LatticeGrid,
    cal: ReciprocalCalibration,
) -> dict[str, str]:
    """
    Return a dict of formatted measurement strings for an FFT/reciprocal grid.
    """
    ox, oy = grid.origin_px
    qvec_a = cal.vec_px_to_q(grid.a_px)
    qvec_b = cal.vec_px_to_q(grid.b_px)
    g1 = cal.vec_length_q(grid.a_px)
    g2 = cal.vec_length_q(grid.b_px)

    def _period_str(g_inv_nm: float) -> str:
        if g_inv_nm < 1e-9:
            return "∞"
        d_nm = 1.0 / g_inv_nm
        if d_nm >= 1.0:
            return f"{d_nm:.3g} nm"
        return f"{d_nm * 10:.3g} Å"

    angle = grid.angle_deg()
    qax, qay = qvec_a
    qbx, qby = qvec_b
    area_q = abs(qax * qby - qay * qbx)

    qox, qoy = cal.px_to_q(ox, oy)

    return {
        "kind":        grid.kind,
        "space":       grid.space,
        "origin_px":   f"({ox:.1f}, {oy:.1f}) px",
        "origin_q":    f"({qox:.3g}, {qoy:.3g}) nm⁻¹",
        "g1_vec":      f"({qax:.3g}, {qay:.3g}) nm⁻¹",
        "g2_vec":      f"({qbx:.3g}, {qby:.3g}) nm⁻¹",
        "g1":          f"{g1:.4g} nm⁻¹  (d = {_period_str(g1)})",
        "g2":          f"{g2:.4g} nm⁻¹  (d = {_period_str(g2)})",
        "angle":       f"{angle:.2g}°",
        "area_q":      f"{area_q:.4g} nm⁻²",
    }
