"""ROI data model: geometry, mask rasterisation, and ROISet container.

Coordinate convention
---------------------
All pixel coordinates use (x, y) = (column, row) with the origin at the
top-left of the image.  Positive x goes right; positive y goes down.
This matches NumPy's (row, col) indexing via x=col, y=row.

ROI-under-transformation rules (Phase 0 convention, implemented here)
-----------------------------------------------------------------------
* flip_horizontal, flip_vertical, rot90_cw, rot180, rot270_cw:
    ROI geometry is transformed exactly to the new pixel-coordinate system.
    These are lossless; ROI.transform() always returns a valid ROI.

* rotate_arbitrary:
    Existing ROIs are invalidated.  ROI.transform() returns None.
    The caller (ROISet.transform_all / apply_geometric_op_to_scan) warns
    and removes invalidated ROIs.

* crop:
    ROI coordinates are shifted by (-x0, -y0).  ROIs entirely outside the
    crop region are dropped (transform returns None).  ROIs partially
    outside are clipped to the new image bounds.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _transform_point(
    x: float, y: float,
    operation: str,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    """Map pixel (x, y) through a geometric operation.

    image_shape is (Ny, Nx) – height then width – of the *pre-transform* image.
    Returns (new_x, new_y) in the post-transform image's coordinate system.
    """
    Ny, Nx = image_shape
    if operation == "flip_horizontal":
        return Nx - 1 - x, y
    if operation == "flip_vertical":
        return x, Ny - 1 - y
    if operation == "rot90_cw":
        return Ny - 1 - y, x
    if operation == "rot180":
        return Nx - 1 - x, Ny - 1 - y
    if operation == "rot270_cw":
        return y, Nx - 1 - x
    raise ValueError(f"_transform_point: unknown operation {operation!r}")


def _post_transform_shape(operation: str, image_shape: tuple[int, int]) -> tuple[int, int]:
    """Return the image shape after a geometric operation."""
    Ny, Nx = image_shape
    if operation in ("rot90_cw", "rot270_cw"):
        return Nx, Ny   # rows and cols swap
    return Ny, Nx       # shape unchanged for 180°, flips


def _bresenham_line(
    x0: int, y0: int, x1: int, y1: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return a boolean mask with a 1-pixel Bresenham line set to True."""
    mask = np.zeros(shape, dtype=bool)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask[y, x] = True
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return mask


# ── ROI ───────────────────────────────────────────────────────────────────────

@dataclass
class ROI:
    """A single region-of-interest with geometry, identity, and transform support.

    Geometry dict formats by kind
    -----------------------------
    rectangle    : {"x": float, "y": float, "width": float, "height": float}
    ellipse      : {"cx": float, "cy": float, "rx": float, "ry": float}
    polygon      : {"vertices": [[x, y], ...]}   (closed implicitly)
    freehand     : same as polygon
    line         : {"x1": float, "y1": float, "x2": float, "y2": float}
    point        : {"x": float, "y": float}
    multipolygon : {
        "components": [
            {"exterior": [[x, y], ...], "holes": [[[x, y], ...], ...]},
            ...
        ]
    }
    Each component is one polygon with an exterior ring and optional interior
    holes.  Produced by :func:`invert` and :func:`combine` when the result is
    non-simply-connected or multi-part.

    All coordinates are pixel-space (x=column, y=row).
    """

    id: str
    name: str
    kind: Literal["rectangle", "ellipse", "polygon", "freehand", "line", "point",
                  "multipolygon"]
    geometry: dict[str, Any]
    coord_system: Literal["pixel", "physical"] = "pixel"
    linked_file: str | None = None

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        kind: str,
        geometry: dict[str, Any],
        *,
        name: str | None = None,
        linked_file: str | None = None,
    ) -> "ROI":
        """Create a new ROI with a freshly generated UUID."""
        roi_id = str(uuid.uuid4())
        roi_name = name if name is not None else f"{kind}_{roi_id[:8]}"
        return cls(
            id=roi_id,
            name=roi_name,
            kind=kind,
            geometry=geometry,
            linked_file=linked_file,
        )

    # ── Rasterisation ─────────────────────────────────────────────────────────

    def to_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Return a boolean array of *shape* (Ny, Nx) with True inside this ROI."""
        Ny, Nx = shape
        g = self.geometry

        if self.kind == "rectangle":
            x = int(round(float(g["x"])))
            y = int(round(float(g["y"])))
            w = max(1, int(round(float(g["width"]))))
            h = max(1, int(round(float(g["height"]))))
            mask = np.zeros(shape, dtype=bool)
            y0 = max(0, y)
            y1 = min(Ny, y + h)
            x0 = max(0, x)
            x1 = min(Nx, x + w)
            if y1 > y0 and x1 > x0:
                mask[y0:y1, x0:x1] = True
            return mask

        if self.kind == "ellipse":
            cx = float(g["cx"])
            cy = float(g["cy"])
            rx = max(0.5, float(g["rx"]))
            ry = max(0.5, float(g["ry"]))
            yy, xx = np.mgrid[:Ny, :Nx]
            return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

        if self.kind in ("polygon", "freehand"):
            verts = g.get("vertices", [])
            if len(verts) < 3:
                return np.zeros(shape, dtype=bool)
            from matplotlib.path import Path as _MPath
            path = _MPath(verts)
            yy, xx = np.mgrid[:Ny, :Nx]
            points = np.column_stack([xx.ravel().astype(float) + 0.5,
                                       yy.ravel().astype(float) + 0.5])
            return path.contains_points(points).reshape(shape)

        if self.kind == "line":
            x0 = int(round(float(g["x1"])))
            y0 = int(round(float(g["y1"])))
            x1 = int(round(float(g["x2"])))
            y1 = int(round(float(g["y2"])))
            return _bresenham_line(x0, y0, x1, y1, shape)

        if self.kind == "point":
            x = int(round(float(g["x"])))
            y = int(round(float(g["y"])))
            mask = np.zeros(shape, dtype=bool)
            if 0 <= y < Ny and 0 <= x < Nx:
                mask[y, x] = True
            return mask

        if self.kind == "multipolygon":
            from matplotlib.path import Path as _MPath
            result = np.zeros(shape, dtype=bool)
            yy, xx = np.mgrid[:Ny, :Nx]
            points = np.column_stack([xx.ravel().astype(float) + 0.5,
                                       yy.ravel().astype(float) + 0.5])
            for comp in g.get("components", []):
                ext = comp.get("exterior", [])
                if len(ext) < 3:
                    continue
                comp_mask = _MPath(ext).contains_points(points).reshape(shape)
                for hole in comp.get("holes", []):
                    if len(hole) >= 3:
                        comp_mask &= ~_MPath(hole).contains_points(points).reshape(shape)
                result |= comp_mask
            return result

        return np.zeros(shape, dtype=bool)

    # ── Geometry accessors ────────────────────────────────────────────────────

    def bounds(self, shape: tuple[int, int]) -> tuple[int, int, int, int]:
        """Return (row_min, row_max, col_min, col_max) inclusive, clipped to shape.

        Returns (0, 0, 0, 0) if the mask is empty.
        """
        mask = self.to_mask(shape)
        if not mask.any():
            return (0, 0, 0, 0)
        rows, cols = np.nonzero(mask)
        return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())

    def crop(self, array: np.ndarray) -> np.ndarray:
        """Return the rectangular crop of *array* covering this ROI's bounding box."""
        shape = array.shape[:2]
        r0, r1, c0, c1 = self.bounds(shape)
        return array[r0:r1 + 1, c0:c1 + 1]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict."""
        return {
            "id":           self.id,
            "name":         self.name,
            "kind":         self.kind,
            "geometry":     _geometry_to_serialisable(self.geometry),
            "coord_system": self.coord_system,
            "linked_file":  self.linked_file,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ROI":
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        return cls(
            id=str(d["id"]),
            name=str(d["name"]),
            kind=str(d["kind"]),   # type: ignore[arg-type]
            geometry=_geometry_from_serialisable(dict(d["geometry"])),
            coord_system=str(d.get("coord_system", "pixel")),   # type: ignore[arg-type]
            linked_file=d.get("linked_file"),
        )

    # ── Geometric transformation ───────────────────────────────────────────────

    def transform(
        self,
        operation: str,
        params: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> "ROI | None":
        """Return a new ROI with geometry transformed for *operation*, or None.

        Returns None when the operation invalidates this ROI (rotate_arbitrary)
        or when the ROI falls entirely outside the post-transform image
        (crop out-of-bounds).

        Supported operations
        --------------------
        flip_horizontal, flip_vertical, rot90_cw, rot180, rot270_cw
            Exact pixel-coordinate transforms.  Always returns a valid ROI.
        crop
            params must contain {"x0": int, "y0": int, "x1": int, "y1": int}
            (inclusive bounds of the crop region in the original image).
        rotate_arbitrary
            Always returns None.
        """
        if operation == "rotate_arbitrary":
            return None

        if operation == "crop":
            return self._transform_crop(params, image_shape)

        if operation in ("flip_horizontal", "flip_vertical",
                         "rot90_cw", "rot180", "rot270_cw"):
            return self._transform_lossless(operation, image_shape)

        raise ValueError(f"ROI.transform: unknown operation {operation!r}")

    # ── Internal transform helpers ────────────────────────────────────────────

    def _transform_lossless(
        self, operation: str, image_shape: tuple[int, int],
    ) -> "ROI":
        """Apply a lossless geometric transform; always returns a valid ROI."""
        Ny, Nx = image_shape
        g = self.geometry

        def tp(x, y):
            return _transform_point(x, y, operation, image_shape)

        new_kind = self.kind  # kind is preserved by all lossless ops

        if self.kind == "rectangle":
            x, y, w, h = float(g["x"]), float(g["y"]), float(g["width"]), float(g["height"])
            if operation == "flip_horizontal":
                new_geom = {"x": Nx - x - w, "y": y, "width": w, "height": h}
            elif operation == "flip_vertical":
                new_geom = {"x": x, "y": Ny - y - h, "width": w, "height": h}
            elif operation == "rot90_cw":
                new_geom = {"x": Ny - y - h, "y": x, "width": h, "height": w}
            elif operation == "rot180":
                new_geom = {"x": Nx - x - w, "y": Ny - y - h, "width": w, "height": h}
            else:  # rot270_cw
                new_geom = {"x": y, "y": Nx - x - w, "width": h, "height": w}

        elif self.kind == "ellipse":
            cx, cy = float(g["cx"]), float(g["cy"])
            rx, ry = float(g["rx"]), float(g["ry"])
            new_cx, new_cy = tp(cx, cy)
            if operation in ("rot90_cw", "rot270_cw"):
                new_rx, new_ry = ry, rx
            else:
                new_rx, new_ry = rx, ry
            new_geom = {"cx": new_cx, "cy": new_cy, "rx": new_rx, "ry": new_ry}

        elif self.kind in ("polygon", "freehand"):
            verts = g.get("vertices", [])
            new_verts = [list(tp(v[0], v[1])) for v in verts]
            new_geom = {"vertices": new_verts}

        elif self.kind == "line":
            nx1, ny1 = tp(float(g["x1"]), float(g["y1"]))
            nx2, ny2 = tp(float(g["x2"]), float(g["y2"]))
            new_geom = {"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2}

        elif self.kind == "point":
            nx_, ny_ = tp(float(g["x"]), float(g["y"]))
            new_geom = {"x": nx_, "y": ny_}

        elif self.kind == "multipolygon":
            def _transform_ring(ring):
                return [list(tp(v[0], v[1])) for v in ring]
            new_components = []
            for comp in g.get("components", []):
                new_ext = _transform_ring(comp.get("exterior", []))
                new_holes = [_transform_ring(h) for h in comp.get("holes", [])]
                new_components.append({"exterior": new_ext, "holes": new_holes})
            new_geom = {"components": new_components}

        else:
            new_geom = dict(g)

        return ROI(
            id=self.id,
            name=self.name,
            kind=new_kind,
            geometry=new_geom,
            coord_system=self.coord_system,
            linked_file=self.linked_file,
        )

    def _transform_crop(
        self, params: dict[str, Any], image_shape: tuple[int, int],
    ) -> "ROI | None":
        """Shift ROI coordinates by crop offset; return None if outside."""
        x0 = int(params["x0"])
        y0 = int(params["y0"])
        x1 = int(params["x1"])
        y1 = int(params["y1"])
        crop_w = x1 - x0
        crop_h = y1 - y0
        if crop_w <= 0 or crop_h <= 0:
            return None

        g = self.geometry

        def shift(x, y):
            return x - x0, y - y0

        if self.kind == "rectangle":
            rx, ry = float(g["x"]) - x0, float(g["y"]) - y0
            rw, rh = float(g["width"]), float(g["height"])
            nx_ = max(0.0, rx)
            ny_ = max(0.0, ry)
            nw = min(rx + rw, float(crop_w)) - nx_
            nh = min(ry + rh, float(crop_h)) - ny_
            if nw <= 0 or nh <= 0:
                return None
            new_geom = {"x": nx_, "y": ny_, "width": nw, "height": nh}

        elif self.kind == "ellipse":
            cx, cy = float(g["cx"]) - x0, float(g["cy"]) - y0
            if not (0 <= cx < crop_w and 0 <= cy < crop_h):
                return None
            new_geom = {"cx": cx, "cy": cy, "rx": float(g["rx"]), "ry": float(g["ry"])}

        elif self.kind in ("polygon", "freehand"):
            verts = [[v[0] - x0, v[1] - y0] for v in g.get("vertices", [])]
            if not verts:
                return None
            new_geom = {"vertices": verts}

        elif self.kind == "line":
            lx1, ly1 = float(g["x1"]) - x0, float(g["y1"]) - y0
            lx2, ly2 = float(g["x2"]) - x0, float(g["y2"]) - y0
            new_geom = {"x1": lx1, "y1": ly1, "x2": lx2, "y2": ly2}

        elif self.kind == "point":
            px, py = float(g["x"]) - x0, float(g["y"]) - y0
            if not (0 <= px < crop_w and 0 <= py < crop_h):
                return None
            new_geom = {"x": px, "y": py}

        elif self.kind == "multipolygon":
            def _shift_ring(ring):
                return [[v[0] - x0, v[1] - y0] for v in ring]
            new_components = []
            for comp in g.get("components", []):
                new_ext = _shift_ring(comp.get("exterior", []))
                new_holes = [_shift_ring(h) for h in comp.get("holes", [])]
                new_components.append({"exterior": new_ext, "holes": new_holes})
            new_geom = {"components": new_components}

        else:
            new_geom = dict(g)

        return ROI(
            id=self.id,
            name=self.name,
            kind=self.kind,
            geometry=new_geom,
            coord_system=self.coord_system,
            linked_file=self.linked_file,
        )


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _geometry_to_serialisable(geometry: dict[str, Any]) -> dict[str, Any]:
    """Ensure geometry dict values are JSON-serialisable."""
    out: dict[str, Any] = {}
    for k, v in geometry.items():
        if isinstance(v, list):
            out[k] = [[float(c) for c in item] if hasattr(item, "__iter__")
                       else float(item) for item in v]
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _geometry_from_serialisable(geometry: dict[str, Any]) -> dict[str, Any]:
    """Restore native types from a JSON-decoded geometry dict."""
    out: dict[str, Any] = {}
    for k, v in geometry.items():
        if k == "vertices" and isinstance(v, list):
            out[k] = [[float(c) for c in item] for item in v]
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = [float(item) for item in v]
        else:
            out[k] = v
    return out


# ── Legacy geometry compat ────────────────────────────────────────────────────

def roi_from_legacy_geometry_dict(
    shape: tuple[int, int],
    geometry: dict[str, Any],
) -> "ROI | None":
    """Convert a legacy ProcessingStep geometry dict to an ROI.

    Legacy format (from processing/state.py pre-Phase-1)
    ----------------------------------------------------
    rectangle : {"kind": "rectangle", "rect_px": (x0, y0, x1, y1)}
                OR {"kind": "rectangle", "bounds_frac": (x0f, y0f, x1f, y1f)}
    ellipse   : {"kind": "ellipse",   "rect_px": (x0, y0, x1, y1)}
                (bounding box; cx/cy/rx/ry derived)
    polygon   : {"kind": "polygon",   "points_px": [(x, y), ...]}
                OR {"kind": "polygon", "points_frac": [(x, y), ...]}

    Returns None if the geometry cannot be interpreted.
    """
    if not isinstance(geometry, dict):
        return None
    kind = str(geometry.get("kind", ""))
    Ny, Nx = shape

    if kind == "rectangle":
        rect = _legacy_rect(shape, geometry)
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        w = max(1, x1 - x0 + 1)
        h = max(1, y1 - y0 + 1)
        return ROI.new("rectangle", {"x": float(x0), "y": float(y0),
                                      "width": float(w), "height": float(h)})

    if kind == "ellipse":
        rect = _legacy_rect(shape, geometry)
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max(0.5, (x1 - x0 + 1) / 2.0)
        ry = max(0.5, (y1 - y0 + 1) / 2.0)
        return ROI.new("ellipse", {"cx": cx, "cy": cy, "rx": rx, "ry": ry})

    if kind == "polygon":
        points = _legacy_points(shape, geometry)
        if len(points) < 3:
            return None
        verts = [[float(p[0]), float(p[1])] for p in points]
        return ROI.new("polygon", {"vertices": verts})

    return None


def _legacy_rect(
    shape: tuple[int, int],
    geometry: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Extract and clamp a legacy (x0, y0, x1, y1) inclusive rect."""
    Ny, Nx = shape
    for key in ("rect_px", "bounds_px", "rect"):
        rect = geometry.get(key)
        if rect is not None:
            try:
                if len(rect) == 4:
                    x0, y0, x1, y1 = [int(round(float(v))) for v in rect]
                    return _clamp_rect(Ny, Nx, x0, y0, x1, y1)
            except (TypeError, ValueError):
                pass
    bounds_frac = geometry.get("bounds_frac")
    if bounds_frac is not None:
        try:
            x0f, y0f, x1f, y1f = [float(v) for v in bounds_frac]
        except (TypeError, ValueError):
            return None
        x0 = int(round(min(x0f, x1f) * (Nx - 1)))
        y0 = int(round(min(y0f, y1f) * (Ny - 1)))
        x1 = int(round(max(x0f, x1f) * (Nx - 1)))
        y1 = int(round(max(y0f, y1f) * (Ny - 1)))
        return _clamp_rect(Ny, Nx, x0, y0, x1, y1)
    return None


def _legacy_points(
    shape: tuple[int, int],
    geometry: dict[str, Any],
) -> list[tuple[float, float]]:
    """Extract vertex list from a legacy polygon geometry dict."""
    Ny, Nx = shape
    raw = geometry.get("points_px") or geometry.get("points")
    if raw is not None:
        pts = []
        for item in raw:
            try:
                pts.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return pts
    frac = geometry.get("points_frac")
    if frac is not None:
        pts = []
        for item in frac:
            try:
                pts.append((float(item[0]) * (Nx - 1), float(item[1]) * (Ny - 1)))
            except (TypeError, ValueError, IndexError):
                continue
        return pts
    return []


def _clamp_rect(
    Ny: int, Nx: int,
    x0: int, y0: int, x1: int, y1: int,
) -> tuple[int, int, int, int] | None:
    """Clamp a rect to image bounds; return None if result is empty."""
    x0 = max(0, min(Nx - 1, x0))
    x1 = max(0, min(Nx - 1, x1))
    y0 = max(0, min(Ny - 1, y0))
    y1 = max(0, min(Ny - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


# ── Mask helpers ──────────────────────────────────────────────────────────────

def combine_masks(
    masks: list[np.ndarray],
    mode: Literal["union", "intersection", "difference", "xor"],
) -> np.ndarray:
    """Combine a list of boolean masks using the given set operation.

    For "difference", the first mask is the base and all subsequent masks are
    subtracted from it.  Requires at least one mask.
    """
    if not masks:
        raise ValueError("combine_masks requires at least one mask")
    result = masks[0].copy()
    if len(masks) == 1:
        return result
    for other in masks[1:]:
        if mode == "union":
            result |= other
        elif mode == "intersection":
            result &= other
        elif mode == "difference":
            result &= ~other
        elif mode == "xor":
            result ^= other
        else:
            raise ValueError(f"Unknown combine mode {mode!r}")
    return result


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Return the logical complement of a boolean mask."""
    return ~mask


# ── Shapely-backed geometry algebra ───────────────────────────────────────────

_AREA_KINDS: frozenset[str] = frozenset(
    {"rectangle", "ellipse", "polygon", "freehand", "multipolygon"}
)


def _roi_to_shapely(roi: "ROI"):
    """Convert an ROI to a Shapely geometry.

    Supports area kinds only (rectangle, ellipse, polygon, freehand,
    multipolygon). Raises ``ValueError`` for line and point ROIs.
    """
    import math as _math
    from shapely.geometry import box as _box, Polygon as _Poly, MultiPolygon as _MPoly

    g = roi.geometry

    if roi.kind == "rectangle":
        x, y = float(g["x"]), float(g["y"])
        w, h = float(g["width"]), float(g["height"])
        return _box(x, y, x + w, y + h)

    if roi.kind == "ellipse":
        cx, cy = float(g["cx"]), float(g["cy"])
        rx, ry = float(g["rx"]), float(g["ry"])
        # Approximate with 64-point polygon
        angles = [2.0 * _math.pi * i / 64 for i in range(64)]
        pts = [(cx + rx * _math.cos(a), cy + ry * _math.sin(a)) for a in angles]
        return _Poly(pts)

    if roi.kind in ("polygon", "freehand"):
        verts = g.get("vertices", [])
        if len(verts) < 3:
            raise ValueError(f"ROI {roi.name!r}: polygon needs at least 3 vertices")
        return _Poly([(v[0], v[1]) for v in verts])

    if roi.kind == "multipolygon":
        polys = []
        for comp in g.get("components", []):
            ext = [(v[0], v[1]) for v in comp.get("exterior", [])]
            holes = [[(v[0], v[1]) for v in h] for h in comp.get("holes", [])]
            if len(ext) >= 3:
                polys.append(_Poly(ext, holes))
        if not polys:
            raise ValueError(f"ROI {roi.name!r}: multipolygon has no valid components")
        return _MPoly(polys) if len(polys) > 1 else polys[0]

    raise ValueError(
        f"Cannot convert ROI kind {roi.kind!r} to Shapely geometry. "
        f"Supported: {sorted(_AREA_KINDS)}"
    )


def _shapely_to_roi(geom, *, name: str, coord_system: str = "pixel") -> "ROI":
    """Convert a Shapely geometry to an ROI.

    Returns a ``"polygon"`` ROI for simple polygons (no holes), or a
    ``"multipolygon"`` ROI for polygons with holes or multi-part geometries.
    Raises ``ValueError`` for empty or unsupported geometry types.
    """
    from shapely.geometry import Polygon as _Poly, MultiPolygon as _MPoly

    if geom.is_empty:
        raise ValueError("Shapely geometry is empty — cannot convert to ROI")

    def _ring_to_list(ring) -> list[list[float]]:
        coords = list(ring.coords)
        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]  # drop repeated closing vertex
        return [[float(c[0]), float(c[1])] for c in coords]

    if isinstance(geom, _Poly):
        if not list(geom.interiors):
            # Simple polygon — use existing kind
            return ROI.new("polygon",
                           {"vertices": _ring_to_list(geom.exterior)},
                           name=name)
        # Polygon with holes → multipolygon
        comp = {
            "exterior": _ring_to_list(geom.exterior),
            "holes": [_ring_to_list(h) for h in geom.interiors],
        }
        return ROI.new("multipolygon", {"components": [comp]}, name=name)

    if isinstance(geom, _MPoly):
        components = []
        for poly in geom.geoms:
            components.append({
                "exterior": _ring_to_list(poly.exterior),
                "holes": [_ring_to_list(h) for h in poly.interiors],
            })
        return ROI.new("multipolygon", {"components": components}, name=name)

    raise ValueError(
        f"Unexpected Shapely geometry type: {type(geom).__name__}. "
        "Expected Polygon or MultiPolygon."
    )


def invert(roi: "ROI", image_shape: tuple[int, int]) -> "ROI":
    """Return a new ROI representing the complement of *roi* within *image_shape*.

    The result is the image bounding rectangle minus the input ROI's geometry,
    computed via Shapely.  For a simply-connected ROI fully inside the image,
    this produces a ``"multipolygon"`` ROI with one component (exterior =
    image boundary, hole = ROI interior).

    Parameters
    ----------
    roi
        The ROI to invert.  Must be an area kind (rectangle, ellipse, polygon,
        freehand, multipolygon).  ``line`` and ``point`` raise ``ValueError``.
    image_shape
        ``(Ny, Nx)`` of the image this ROI belongs to.

    Returns
    -------
    ROI
        New ROI with a generated UUID and name ``"not_<roi.name>"``.  The
        ``coord_system`` of the input is preserved.
    """
    if roi.kind not in _AREA_KINDS:
        raise ValueError(
            f"invert does not support ROI kind {roi.kind!r}. "
            f"Supported: {sorted(_AREA_KINDS)}"
        )
    from shapely.geometry import box as _box
    Ny, Nx = image_shape
    image_box = _box(0, 0, Nx, Ny)
    roi_geom = _roi_to_shapely(roi)
    result = image_box.difference(roi_geom)
    return _shapely_to_roi(
        result,
        name=f"not_{roi.name}",
    )


def combine(
    rois: "list[ROI]",
    mode: "Literal['union', 'intersection', 'difference', 'xor']",
) -> "ROI":
    """Return a new ROI representing the geometric combination of *rois*.

    All inputs must be area-kind ROIs (rectangle, ellipse, polygon, freehand,
    multipolygon).  Mixing area ROIs with line or point ROIs raises
    ``ValueError``.

    Parameters
    ----------
    rois
        List of ROIs to combine.  Must contain at least one element.
    mode
        ``"union"``        — Shapely ``unary_union`` of all geometries.
        ``"intersection"`` — Pairwise intersection from left to right.
        ``"difference"``   — First ROI minus all subsequent ROIs in order.
                             Order matters: ``combine([A, B], "difference")``
                             returns ``A - B``, not ``B - A``.
        ``"xor"``          — Symmetric difference (pairwise from left).

    Returns
    -------
    ROI
        New ROI with a generated UUID.  Kind is ``"polygon"`` for simple
        results, ``"multipolygon"`` for multi-part or holed results.

    Raises
    ------
    ValueError
        If *rois* is empty, any ROI has a non-area kind, or the operation
        produces an empty geometry.
    """
    if not rois:
        raise ValueError("combine requires at least one ROI")

    for r in rois:
        if r.kind not in _AREA_KINDS:
            raise ValueError(
                f"combine does not support ROI kind {r.kind!r} ({r.name!r}). "
                f"Only area kinds are supported: {sorted(_AREA_KINDS)}"
            )

    geometries = [_roi_to_shapely(r) for r in rois]
    _mode = str(mode)

    if _mode == "union":
        from shapely.ops import unary_union as _unary_union
        result = _unary_union(geometries)
    elif _mode == "intersection":
        result = geometries[0]
        for g in geometries[1:]:
            result = result.intersection(g)
    elif _mode == "difference":
        result = geometries[0]
        for g in geometries[1:]:
            result = result.difference(g)
    elif _mode == "xor":
        result = geometries[0]
        for g in geometries[1:]:
            result = result.symmetric_difference(g)
    else:
        raise ValueError(
            f"Unknown combine mode {mode!r}. "
            "Supported: 'union', 'intersection', 'difference', 'xor'."
        )

    if result.is_empty:
        raise ValueError(
            f"combine({mode!r}) produced an empty geometry for the given ROIs."
        )

    names = "_".join(r.name for r in rois[:3])
    if len(rois) > 3:
        names += f"_and_{len(rois) - 3}_more"
    return _shapely_to_roi(result, name=f"{mode}_{names}")


def translate(roi: "ROI", dx: float, dy: float) -> "ROI":
    """Return a copy of *roi* with all coordinates shifted by (dx, dy) pixels."""
    g = roi.geometry
    k = roi.kind
    if k == "rectangle":
        new_g = {**g, "x": g["x"] + dx, "y": g["y"] + dy}
    elif k == "ellipse":
        new_g = {**g, "cx": g["cx"] + dx, "cy": g["cy"] + dy}
    elif k in ("polygon", "freehand"):
        new_g = {"vertices": [[v[0] + dx, v[1] + dy] for v in g.get("vertices", [])]}
    elif k == "line":
        new_g = {**g, "x1": g["x1"] + dx, "y1": g["y1"] + dy,
                 "x2": g["x2"] + dx, "y2": g["y2"] + dy}
    elif k == "point":
        new_g = {**g, "x": g["x"] + dx, "y": g["y"] + dy}
    elif k == "multipolygon":
        new_comps = []
        for comp in g.get("components", []):
            ext = [[v[0] + dx, v[1] + dy] for v in comp.get("exterior", [])]
            holes = [[[v[0] + dx, v[1] + dy] for v in h] for h in comp.get("holes", [])]
            new_comps.append({"exterior": ext, "holes": holes})
        new_g = {"components": new_comps}
    else:
        new_g = dict(g)
    return ROI(id=roi.id, name=roi.name, kind=roi.kind, geometry=new_g,
               coord_system=roi.coord_system, linked_file=roi.linked_file)


# ── ROISet ────────────────────────────────────────────────────────────────────

@dataclass
class ROISet:
    """An ordered collection of ROIs belonging to one image.

    *image_id* ties this set to a scan / ProbeFlowItem; its value is opaque
    to ROISet itself (typically the file stem or a UUID).
    """

    image_id: str
    rois: list[ROI] = field(default_factory=list)
    active_roi_id: str | None = None

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, roi: ROI) -> None:
        """Append *roi* to the set."""
        self.rois.append(roi)

    def remove(self, roi_id: str) -> None:
        """Remove the ROI with *roi_id*; silently ignore if not found."""
        self.rois = [r for r in self.rois if r.id != roi_id]
        if self.active_roi_id == roi_id:
            self.active_roi_id = None

    def get(self, roi_id: str) -> ROI | None:
        """Return the ROI with *roi_id*, or None."""
        for r in self.rois:
            if r.id == roi_id:
                return r
        return None

    def get_by_name(self, name: str) -> ROI | None:
        """Return the first ROI whose *name* matches, or None."""
        for r in self.rois:
            if r.name == name:
                return r
        return None

    def set_active(self, roi_id: str | None) -> None:
        """Set the active ROI by ID; pass None to clear the selection."""
        if roi_id is not None and self.get(roi_id) is None:
            raise ValueError(f"ROI {roi_id!r} not in this ROISet")
        self.active_roi_id = roi_id

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict."""
        return {
            "image_id":      self.image_id,
            "rois":          [r.to_dict() for r in self.rois],
            "active_roi_id": self.active_roi_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ROISet":
        """Reconstruct from the dict produced by :meth:`to_dict`."""
        roi_set = cls(
            image_id=str(d["image_id"]),
            active_roi_id=d.get("active_roi_id"),
        )
        for roi_dict in d.get("rois", []):
            try:
                roi_set.rois.append(ROI.from_dict(roi_dict))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to reconstruct ROI from dict: {exc!r} — "
                    f"offending dict: {roi_dict!r}"
                ) from exc
        return roi_set

    # ── Geometric transformation ───────────────────────────────────────────────

    def transform_all(
        self,
        operation: str,
        params: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> list[str]:
        """Apply *operation* to every ROI; return list of invalidated ROI IDs.

        Invalidated ROIs are NOT removed from this set — that policy decision
        belongs to the caller (typically apply_geometric_op_to_scan).
        """
        invalidated: list[str] = []
        new_rois: list[ROI] = []
        for roi in self.rois:
            transformed = roi.transform(operation, params, image_shape)
            if transformed is None:
                invalidated.append(roi.id)
                new_rois.append(roi)   # keep for now; caller decides
            else:
                new_rois.append(transformed)
        self.rois = new_rois
        return invalidated
