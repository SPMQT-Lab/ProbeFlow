"""Geometric image operations: profiling, zero-setting, distortion correction, transforms."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ._image_utils import _finite_median


# ═════════════════════════════════════════════════════════════════════════════
# 13.  line_profile  — z values along a straight segment, with physical x-axis
# ═════════════════════════════════════════════════════════════════════════════

def line_profile(
    arr: np.ndarray,
    p0_px: "tuple[float, float] | None" = None,
    p1_px: "tuple[float, float] | None" = None,
    *,
    roi: "Any | None" = None,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    n_samples: Optional[int] = None,
    width_px: float = 1.0,
    interp: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``arr`` along a line segment.

    Parameters
    ----------
    arr
        2-D scan plane (any units).
    p0_px, p1_px
        Endpoint pixel coordinates ``(x, y)``. May be sub-pixel. Required
        when *roi* is ``None``.
    roi
        Optional :class:`probeflow.core.roi.ROI` with ``kind='line'``.  When
        provided, *p0_px* and *p1_px* are derived from the ROI geometry and
        must not be supplied separately.  Any other kind raises ``ValueError``.
    pixel_size_x_m, pixel_size_y_m
        Physical pixel spacing in metres along x and y. Used to express the
        sample axis in metres (handles non-square pixels correctly).
    n_samples
        Number of samples along the line. Default = ``ceil(geometric pixel
        length) + 1``.
    width_px
        Half-thickness of a perpendicular averaging band in pixels. ``1.0``
        samples the line itself; larger values average across a swath, which
        is useful for noisy traces.
    interp
        ``"linear"`` (default; via ``scipy.ndimage.map_coordinates`` order 1)
        or ``"nearest"`` (order 0).

    Returns
    -------
    s_m, z
        ``s_m`` — distance along the line in metres (length ``n_samples``).
        ``z`` — sampled values, one per ``s_m`` entry.
    """
    if roi is not None:
        if roi.kind != "line":
            raise ValueError(
                f"line_profile requires roi.kind='line', got {roi.kind!r}"
            )
        if p0_px is not None or p1_px is not None:
            raise ValueError(
                "Provide either roi or p0_px/p1_px, not both"
            )
        g = roi.geometry
        p0_px = (float(g["x1"]), float(g["y1"]))
        p1_px = (float(g["x2"]), float(g["y2"]))
    else:
        if p0_px is None or p1_px is None:
            raise ValueError("Either roi or both p0_px and p1_px must be provided")

    if arr.ndim != 2:
        raise ValueError("line_profile expects a 2-D array")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("pixel_size_*_m must be > 0")
    if width_px < 1.0:
        raise ValueError("width_px must be >= 1.0")
    if interp not in ("linear", "nearest"):
        raise ValueError(f"Unknown interp {interp!r}")

    from scipy.ndimage import map_coordinates

    x0, y0 = float(p0_px[0]), float(p0_px[1])
    x1, y1 = float(p1_px[0]), float(p1_px[1])
    dx_px, dy_px = x1 - x0, y1 - y0
    px_len = float(np.hypot(dx_px, dy_px))
    if px_len < 1e-9:
        raise ValueError("p0 and p1 are the same point")

    if n_samples is None:
        n_samples = int(math.ceil(px_len)) + 1
    if n_samples < 2:
        n_samples = 2

    ts = np.linspace(0.0, 1.0, n_samples)
    xs = x0 + ts * dx_px
    ys = y0 + ts * dy_px

    order = 1 if interp == "linear" else 0

    if width_px <= 1.0:
        # ``map_coordinates`` takes (row, col) = (y, x).
        z = map_coordinates(arr, np.vstack([ys, xs]), order=order, mode="reflect")
    else:
        # Physical-space perpendicular mapped back to pixel coordinates.
        # Line direction in physical space: (dx_px*psx, dy_px*psy).
        # Physical perpendicular: (-dy_px*psy, dx_px*psx), then divide by
        # pixel sizes to convert back to pixel-space offsets.
        # For square pixels this reduces to (-dy_px, dx_px) / px_len.
        nx_un = -dy_px * (pixel_size_y_m / pixel_size_x_m)
        ny_un =  dx_px * (pixel_size_x_m / pixel_size_y_m)
        perp_px_len = math.sqrt(nx_un ** 2 + ny_un ** 2)
        if perp_px_len > 1e-12:
            nx, ny = nx_un / perp_px_len, ny_un / perp_px_len
        else:
            nx, ny = -dy_px / px_len, dx_px / px_len
        n_perp = int(round(width_px))
        offsets = np.linspace(-(width_px - 1) / 2.0,
                              (width_px - 1) / 2.0, n_perp)
        accum = np.zeros(n_samples, dtype=np.float64)
        for off in offsets:
            ys_o = ys + off * ny
            xs_o = xs + off * nx
            accum += map_coordinates(arr, np.vstack([ys_o, xs_o]),
                                     order=order, mode="reflect")
        z = accum / n_perp

    # Physical distance: scale x and y components by their respective pixel sizes.
    dx_m = dx_px * pixel_size_x_m
    dy_m = dy_px * pixel_size_y_m
    seg_len_m = float(np.hypot(dx_m, dy_m))
    s_m = ts * seg_len_m
    return s_m, z.astype(arr.dtype, copy=False)


# ═════════════════════════════════════════════════════════════════════════════
# 14.  set_zero_point  (Gwyddion-style "Set Z=0 here")
# ═════════════════════════════════════════════════════════════════════════════

def set_zero_point(
    arr: np.ndarray,
    y_px: int,
    x_px: int,
    *,
    patch: int = 1,
) -> np.ndarray:
    """Subtract the mean of a small patch around ``(y_px, x_px)`` from the image.

    Parameters
    ----------
    arr
        2-D scan plane.
    y_px, x_px
        Pixel coordinates of the click. Coordinates outside the array are
        clipped to the nearest edge pixel.
    patch
        Half-size of the averaging window in pixels. ``patch=1`` averages a
        3×3 region (the default; matches Gwyddion's "Z=0 at pixel"). Use 0 to
        sample a single pixel.
    """
    if arr.ndim != 2:
        raise ValueError("set_zero_point expects a 2-D array")
    Ny, Nx = arr.shape
    y = max(0, min(int(y_px), Ny - 1))
    x = max(0, min(int(x_px), Nx - 1))
    p = max(0, int(patch))
    y0, y1 = max(0, y - p), min(Ny, y + p + 1)
    x0, x1 = max(0, x - p), min(Nx, x + p + 1)
    region = arr[y0:y1, x0:x1]
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return arr.astype(np.float64, copy=True)
    ref = float(finite.mean())
    return arr.astype(np.float64, copy=True) - ref


def set_zero_plane(
    arr: np.ndarray,
    points_px: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    patch: int = 1,
) -> np.ndarray:
    """Subtract the plane defined by three clicked reference points.

    ``points_px`` contains ``(x_px, y_px)`` coordinates.  The height at each
    point is estimated from a small finite-valued patch, then a plane
    ``z = ax + by + c`` is fitted through those three references and subtracted
    from the whole image.  This is a manual zero-plane operation, distinct from
    automatic polynomial/background subtraction.
    """
    if arr.ndim != 2:
        raise ValueError("set_zero_plane expects a 2-D array")
    if len(points_px) < 3:
        return arr.astype(np.float64, copy=True)

    a = arr.astype(np.float64, copy=True)
    Ny, Nx = a.shape
    p = max(0, int(patch))
    samples = []
    for point in points_px[:3]:
        try:
            x_px, y_px = int(point[0]), int(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        x_px = max(0, min(Nx - 1, x_px))
        y_px = max(0, min(Ny - 1, y_px))
        y0, y1 = max(0, y_px - p), min(Ny, y_px + p + 1)
        x0, x1 = max(0, x_px - p), min(Nx, x_px + p + 1)
        vals = a[y0:y1, x0:x1]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        samples.append((float(x_px), float(y_px), float(np.mean(finite))))

    if len(samples) < 3:
        return a

    # Review image-proc #6 (fixed 2026-05-28): normalise pixel
    # coordinates to [-1, 1] before solving so the 3x3 design matrix is
    # well-conditioned for large images (raw pixel indices give a
    # condition number ~Nx, e.g. ~1024 on a 1024×1024 scan).  An
    # additional triangle-area guard catches sub-pixel-close click
    # pairs that would slip past the rank check.
    sx = 2.0 / max(Nx - 1, 1)
    sy = 2.0 / max(Ny - 1, 1)
    cx = 0.5 * (Nx - 1)
    cy = 0.5 * (Ny - 1)

    # Normalised coords for the fit
    A_norm = np.array(
        [[(x - cx) * sx, (y - cy) * sy, 1.0] for x, y, _z in samples],
        dtype=np.float64,
    )
    if np.linalg.matrix_rank(A_norm) < 3:
        raise ValueError(
            "set_zero_plane: the three reference points are collinear or coincident; "
            "choose points that form a non-degenerate triangle."
        )
    # Signed triangle area in NORMALISED units — guards against three
    # points that are technically non-collinear but cluster within a
    # sub-pixel patch (where the rank-check passes but the resulting
    # plane has a wildly amplified slope).
    (xn0, yn0, _), (xn1, yn1, _), (xn2, yn2, _) = (
        (A_norm[i, 0], A_norm[i, 1], A_norm[i, 2]) for i in range(3)
    )
    tri_area = 0.5 * abs(
        (xn1 - xn0) * (yn2 - yn0) - (xn2 - xn0) * (yn1 - yn0)
    )
    # In normalised coords each pixel spans sx (or sy).  Require the
    # triangle area to be at least 2*sx*sy — i.e. roughly 4 pixels² —
    # so a 3-click triangle on a tight cluster of pixels is rejected
    # with a clear error rather than producing an unphysical plane.
    if tri_area < 2.0 * sx * sy:
        raise ValueError(
            "set_zero_plane: the three reference points enclose too small a "
            "triangle (sub-pixel cluster).  Choose points spread across the "
            "image."
        )
    z = np.array([z for _x, _y, z in samples], dtype=np.float64)
    coeffs_norm = np.linalg.solve(A_norm, z)
    # Evaluate the plane on the original pixel grid by mapping every
    # pixel through the same normalisation used for the fit.
    yy, xx = np.mgrid[:Ny, :Nx]
    plane = (
        coeffs_norm[0] * ((xx - cx) * sx)
        + coeffs_norm[1] * ((yy - cy) * sy)
        + coeffs_norm[2]
    )
    out = a - plane
    out[~np.isfinite(a)] = np.nan
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 17.  linear_undistort  (ImageJ Linear_Undistort port)
# ═════════════════════════════════════════════════════════════════════════════

def linear_undistort(
    arr: np.ndarray,
    *,
    shear_x: float = 0.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Apply an affine drift/creep correction to a scan plane.

    The forward map shifts column ``c`` by ``shear_x * (row / (Ny - 1))``
    pixels and rescales the row coordinate by ``scale_y``. Inverse-mapped via
    ``scipy.ndimage.map_coordinates`` so every output pixel comes from one
    bilinearly-interpolated location in the input.

    NaN and inf pixels are filled with the finite mean before mapping and
    restored to NaN in the output, so the missing-data mask is preserved.
    """
    from scipy.ndimage import map_coordinates

    if arr.ndim != 2:
        raise ValueError("linear_undistort expects a 2-D array")
    if scale_y <= 0:
        raise ValueError("scale_y must be > 0")
    a = arr.astype(np.float64, copy=True)
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        a[nan_mask] = float(np.nanmean(a))
    Ny, Nx = a.shape
    yy, xx = np.indices(a.shape).astype(np.float64)
    src_y = yy / max(scale_y, 1e-9)
    src_x = xx - shear_x * (yy / max(Ny - 1, 1))
    out = map_coordinates(
        a, np.vstack([src_y.ravel(), src_x.ravel()]),
        order=1, mode="reflect",
    ).reshape(Ny, Nx)
    if nan_mask.any():
        out[nan_mask] = np.nan
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 18.  affine_lattice_correction
# ═════════════════════════════════════════════════════════════════════════════

def affine_lattice_correction(
    arr: np.ndarray,
    matrix: np.ndarray,
    *,
    expand_canvas: bool = True,
    interpolation: str = "bilinear",
    fill_mode: str = "nan",
    fill_value: float | None = None,
) -> np.ndarray:
    """Apply a 2×2 affine lattice correction to a scan plane.

    ``matrix`` is the forward pixel-space transform: it maps a point in the
    measured (distorted) image to the corresponding point in the ideal
    (corrected) image, with both expressed relative to the image centre.

    For a correction computed in physical nm space, convert to pixel space
    first::

        S = np.diag([1 / px_nm_x, 1 / px_nm_y])
        T_px = S @ T_nm @ np.linalg.inv(S)

    For square pixels (px_nm_x == px_nm_y) T_px == T_nm, so no conversion
    is needed.

    Parameters
    ----------
    arr : 2-D float ndarray
    matrix : (2, 2) ndarray
        Pixel-space forward transform (measured → ideal) around image centre.
    expand_canvas : bool
        If True, enlarge the output canvas so no transformed corner is cropped.
    interpolation : {'nearest', 'bilinear', 'bicubic'}
        scipy.ndimage order: 0, 1, 3.
    fill_mode : {'nan', 'background', 'zero'}
        Fill for regions outside the input image extent.
    fill_value : float or None
        Explicit fill for ``fill_mode='background'``; defaults to
        ``nanmedian(arr)``.
    """
    import math as _math
    from scipy.ndimage import map_coordinates

    if arr.ndim != 2:
        raise ValueError("affine_lattice_correction expects a 2-D array")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError(f"matrix must be shape (2, 2), got {matrix.shape}")
    cond = np.linalg.cond(matrix)
    if not np.isfinite(cond) or cond > 1e10:
        raise ValueError("Correction matrix is singular or near-singular.")

    order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
    if interpolation not in order_map:
        raise ValueError(
            f"interpolation must be one of {sorted(order_map)!r}, "
            f"got {interpolation!r}"
        )
    if fill_mode not in {"nan", "background", "zero"}:
        raise ValueError(
            f"fill_mode must be 'nan', 'background', or 'zero', got {fill_mode!r}"
        )

    Ny, Nx = arr.shape
    a = arr.astype(np.float64, copy=True)

    # Replace NaN/inf with finite mean for interpolation; restore below
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        finite_vals = a[~nan_mask]
        temp_fill = float(finite_vals.mean()) if finite_vals.size else 0.0
        a[nan_mask] = temp_fill

    T_inv = np.linalg.inv(matrix)

    # Transform origin = image centre
    cx_in = (Nx - 1) / 2.0
    cy_in = (Ny - 1) / 2.0

    if expand_canvas:
        # Forward-transform the 4 input corners to find output bounds.
        # Corners relative to image centre: (±cx_in, ±cy_in) in (x=col, y=row).
        hw, hh = cx_in, cy_in
        corners = np.array([[-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]])
        corners_out = (matrix @ corners.T).T  # shape (4, 2): (dx_col, dy_row)
        c_min = corners_out[:, 0].min()
        c_max = corners_out[:, 0].max()
        r_min = corners_out[:, 1].min()
        r_max = corners_out[:, 1].max()
        Nx_out = int(_math.ceil(c_max - c_min)) + 1
        Ny_out = int(_math.ceil(r_max - r_min)) + 1
        cx_out = -c_min   # column in output image where the origin (input centre) lands
        cy_out = -r_min
    else:
        Nx_out, Ny_out = Nx, Ny
        cx_out, cy_out = cx_in, cy_in

    # Build output pixel grid and compute source coords via inverse mapping.
    rows_out, cols_out = np.indices((Ny_out, Nx_out), dtype=np.float64)
    dc_out = cols_out - cx_out
    dr_out = rows_out - cy_out
    coords_out = np.vstack([dc_out.ravel(), dr_out.ravel()])  # (2, N)
    coords_in_rel = T_inv @ coords_out                         # (2, N)
    col_src = coords_in_rel[0] + cx_in
    row_src = coords_in_rel[1] + cy_in

    if fill_mode == "background":
        cval = fill_value if fill_value is not None else float(np.nanmedian(arr))
    else:
        cval = 0.0

    out = map_coordinates(
        a,
        np.vstack([row_src, col_src]),
        order=order_map[interpolation],
        mode="constant",
        cval=cval,
    ).reshape(Ny_out, Nx_out)

    if fill_mode == "nan":
        oob = (row_src < 0) | (row_src > Ny - 1) | (col_src < 0) | (col_src > Nx - 1)
        out[oob.reshape(Ny_out, Nx_out)] = np.nan

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 19.  blend_forward_backward  (ImageJ Blend_Images port)
# ═════════════════════════════════════════════════════════════════════════════

def blend_forward_backward(
    fwd: np.ndarray,
    bwd: np.ndarray,
    *,
    weight: float = 0.5,
) -> np.ndarray:
    """Blend a forward-scan plane with a horizontally-mirrored backward plane.

    Parameters
    ----------
    fwd, bwd
        2-D arrays of the same shape. ``bwd`` is left-right flipped before
        blending so the same physical location overlaps in both planes.
    weight
        Weight of the forward plane in [0, 1]. ``0.5`` is a symmetric mean.
    """
    if fwd.shape != bwd.shape:
        raise ValueError("fwd and bwd must have the same shape")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    f = fwd.astype(np.float64, copy=True)
    b = np.fliplr(bwd.astype(np.float64, copy=True))
    out = weight * f + (1.0 - weight) * b
    nan_mask = ~np.isfinite(f) | ~np.isfinite(b)
    if nan_mask.any():
        out[nan_mask] = np.where(np.isfinite(f), f, b)[nan_mask]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 19.  Geometric transforms — flip and rotation
#
# ROI-under-transformation conventions (for Phase 1 reference):
#
# - flip_horizontal, flip_vertical, rotate_90_cw, rotate_180, rotate_270_cw:
#     ROI geometry is transformed to the new pixel-coordinate system. These are
#     exact, lossless transformations. Phase 1 should update ROI pixel
#     coordinates when these ops are applied.
#
# - rotate_arbitrary:
#     Existing ROIs are INVALIDATED and must be removed or marked invalid.
#     Reason: floating-point geometry + bilinear-interpolated pixels make
#     round-tripping unreliable. The caller (apply_processing_state) warns and
#     removes any roi steps when rotate_arbitrary is encountered in the state.
#
# - crop (future):
#     ROI geometry is transformed to the new coordinate system. ROIs entirely
#     outside the crop are dropped; ROIs partially outside are clipped.
# ═════════════════════════════════════════════════════════════════════════════

def flip_horizontal(arr: np.ndarray) -> np.ndarray:
    """Flip the scan left-to-right (mirror about the vertical axis)."""
    return np.fliplr(arr.astype(np.float64, copy=True))


def flip_vertical(arr: np.ndarray) -> np.ndarray:
    """Flip the scan top-to-bottom (mirror about the horizontal axis)."""
    return np.flipud(arr.astype(np.float64, copy=True))


def rotate_90_cw(arr: np.ndarray) -> np.ndarray:
    """Rotate the scan 90° clockwise. Swaps width and height."""
    return np.rot90(arr.astype(np.float64, copy=True), k=3)


def rotate_180(arr: np.ndarray) -> np.ndarray:
    """Rotate the scan 180°. Preserves width and height."""
    return np.rot90(arr.astype(np.float64, copy=True), k=2)


def rotate_270_cw(arr: np.ndarray) -> np.ndarray:
    """Rotate the scan 270° clockwise (= 90° counter-clockwise). Swaps width and height."""
    return np.rot90(arr.astype(np.float64, copy=True), k=1)


def rotate_arbitrary(
    arr: np.ndarray,
    angle_degrees: float,
    *,
    order: int = 1,
) -> np.ndarray:
    """Rotate the scan by an arbitrary angle with canvas expansion.

    Positive angles are counter-clockwise (standard mathematical convention).
    The output canvas is enlarged to contain the entire rotated image; newly
    introduced pixels are zero. No input pixels are lost.

    Uses bilinear interpolation (``order=1``) by default, which is appropriate
    for STM topography data: it preserves smooth gradients without the ringing
    that cubic introduces near step edges.

    Parameters
    ----------
    arr
        2-D scan plane.
    angle_degrees
        Rotation angle in degrees. Positive = counter-clockwise.
    order
        Interpolation order: 0=nearest, 1=bilinear (default), 2=quadratic,
        3=bicubic. Order 1 is recommended for STM topography.
    """
    if not isinstance(order, int) or order < 0 or order > 3:
        raise ValueError(f"order must be 0–3, got {order!r}")
    from scipy.ndimage import rotate as _ndimage_rotate
    a = arr.astype(np.float64, copy=True)
    Ny, Nx = a.shape

    # Review physics #2 (fixed 2026-05-28): the previous implementation
    # passed ``cval=np.nan`` directly to scipy.ndimage.rotate.  With
    # order>=1 (bilinear), every output pixel within one pixel of the
    # boundary samples one or more NaN neighbours and a single NaN
    # contribution propagates as NaN — the rotated image lost a halo of
    # valid data ~1 px deep all around its boundary, and oblique angles
    # made the loss worse.  Now follow the same pattern as
    # ``affine_lattice_correction``: temp-fill NaN with the finite mean
    # so interpolation stays well-defined, then explicitly mark
    # out-of-bounds pixels as NaN by inverse-mapping a "valid" mask.
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        finite_vals = a[~nan_mask]
        temp_fill = float(finite_vals.mean()) if finite_vals.size else 0.0
        a[nan_mask] = temp_fill
    else:
        temp_fill = 0.0

    out = _ndimage_rotate(
        a, float(angle_degrees), reshape=True, order=order,
        mode='constant', cval=temp_fill,
    )

    # Build a 0/1 "valid input pixel" mask, rotate it with the same
    # parameters, and mark output pixels NaN wherever the rotated mask
    # dropped below 1 (i.e. the bilinear stencil partly sampled outside
    # the original valid area).  This correctly NaN-marks both the
    # canvas-expanded corners and the interior holes that the original
    # nan_mask carved out.
    valid_in = (~nan_mask).astype(np.float64) if nan_mask.any() else np.ones_like(a)
    valid_out = _ndimage_rotate(
        valid_in, float(angle_degrees), reshape=True,
        order=max(order, 1),  # >=1 so the mask shrinks by the interpolation footprint
        mode='constant', cval=0.0,
    )
    out[valid_out < (1.0 - 1e-6)] = np.nan
    return out


def shear(
    arr: np.ndarray,
    *,
    shear_x: float = 0.0,
    shear_y: float = 0.0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Apply a 2-component shear correction to a scan plane.

    Applies the shear matrix ``[[1, shear_x], [shear_y, 1]]`` via
    :func:`affine_lattice_correction` with canvas expansion.  Positive
    ``shear_x`` tilts columns to the right; positive ``shear_y`` tilts rows
    downward.

    Parameters
    ----------
    arr
        2-D scan plane.
    shear_x
        Off-diagonal element in the column direction (horizontal shear).
    shear_y
        Off-diagonal element in the row direction (vertical shear).
    interpolation
        ``'nearest'``, ``'bilinear'`` (default), or ``'bicubic'``.
    """
    matrix = np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=np.float64)
    return affine_lattice_correction(
        arr, matrix,
        expand_canvas=True,
        interpolation=interpolation,
        fill_mode="nan",
    )


def scale_image(
    arr: np.ndarray,
    new_height: int,
    new_width: int,
    *,
    order: int = 1,
) -> np.ndarray:
    """Resample *arr* to ``(new_height, new_width)`` pixels.

    The physical scan range (nm²) is unchanged; only pixel density differs.
    Uses ``scipy.ndimage.zoom`` with the requested interpolation order.

    Parameters
    ----------
    arr
        2-D scan plane.
    new_height, new_width
        Target pixel dimensions in (row, column) order.
    order
        Interpolation order: 0 = nearest, 1 = bilinear (default), 3 = bicubic.
    """
    from scipy.ndimage import zoom as _zoom
    if arr.ndim != 2:
        raise ValueError("scale_image expects a 2-D array")
    if new_height < 1 or new_width < 1:
        raise ValueError("Target dimensions must be >= 1")
    a = arr.astype(np.float64, copy=True)
    Ny, Nx = a.shape
    zoom_y = new_height / Ny
    zoom_x = new_width / Nx
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        fill = float(np.nanmean(a)) if np.isfinite(a).any() else 0.0
        a[nan_mask] = fill
    out = _zoom(a, (zoom_y, zoom_x), order=order, mode="reflect")
    if nan_mask.any():
        scaled_nan = _zoom(
            nan_mask.astype(np.float64), (zoom_y, zoom_x), order=0, mode="reflect"
        ) > 0.5
        out[scaled_nan] = np.nan
    return out


def threshold_image(
    arr: np.ndarray,
    *,
    lower: "float | None" = None,
    upper: "float | None" = None,
    mode: str = "clip",
) -> np.ndarray:
    """Apply a threshold to a scan plane.

    Parameters
    ----------
    arr
        2-D scan plane.
    lower, upper
        Threshold bounds in the same units as *arr*.  ``None`` means unbounded
        on that side.
    mode
        ``'clip'``: values outside ``[lower, upper]`` become NaN, preserving
        the original data range inside the band.
        ``'binarize'``: pixels *inside* ``[lower, upper]`` become 1.0, outside 0.0.
    """
    if mode not in ("clip", "binarize"):
        raise ValueError(f"mode must be 'clip' or 'binarize', got {mode!r}")
    a = arr.astype(np.float64, copy=True)
    if mode == "clip":
        finite = np.isfinite(a)
        if lower is not None:
            a[finite & (a < lower)] = np.nan
        if upper is not None:
            a[finite & (a > upper)] = np.nan
    else:  # binarize
        mask = np.ones(a.shape, dtype=bool)
        if lower is not None:
            mask &= a >= lower
        if upper is not None:
            mask &= a <= upper
        a = mask.astype(np.float64)
    return a


def quantize_bit_depth(arr: np.ndarray, bits: int) -> np.ndarray:
    """Reduce the effective precision of *arr* to *bits* integer levels.

    Maps the finite value range ``[vmin, vmax]`` to ``2**bits`` evenly-spaced
    levels (round-trip through integer quantization), then maps back to the
    original physical range.  The result is still a float64 array but contains
    only ``2**bits`` distinct values.  NaN pixels are preserved unchanged.

    Parameters
    ----------
    arr:
        2-D float64 input array (SI values).
    bits:
        Target bit depth.  Typical values: ``8`` (256 levels) or
        ``16`` (65 536 levels).

    Returns
    -------
    np.ndarray
        float64 array with quantized values.
    """
    if bits < 1 or bits > 32:
        raise ValueError(f"bits must be in [1, 32], got {bits}")
    n_levels = 2 ** bits
    finite = np.isfinite(arr)
    if not finite.any():
        return arr.astype(np.float64, copy=True)
    vmin = float(arr[finite].min())
    vmax = float(arr[finite].max())
    if vmax == vmin:
        return arr.astype(np.float64, copy=True)
    result = arr.astype(np.float64, copy=True)
    # Normalise finite pixels to [0, n_levels-1], round, then back to SI
    scale = (n_levels - 1) / (vmax - vmin)
    result[finite] = (
        np.round((arr[finite] - vmin) * scale) / scale + vmin
    )
    return result
