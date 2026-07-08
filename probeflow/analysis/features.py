"""
ProbeFlow — feature detection, counting, and classification.

Layered on top of :mod:`probeflow.processing` to extract *discrete objects*
from an STM scan plane: particles / molecules, atoms (template matching), and
classifications. All functions return SI-unit dataclasses that serialise
cleanly to JSON via :mod:`probeflow.io.writers.json`.

Design notes
------------
* Everything operates on a single 2-D float array plus physical pixel sizes.
  The historical ``pixel_size_m`` scalar is still accepted for square-pixel
  callers; newer callers should pass ``pixel_size_x_m`` and ``pixel_size_y_m``
  so coordinates remain correct on rectangular scans.
* OpenCV and scikit-learn are optional at import time — they are imported
  lazily inside the functions that need them. A ``features`` extra in
  pyproject.toml pulls them in on demand.
* No Qt / no PySide6 import here — this module must stay usable from worker
  threads and batch scripts.
* Ported loosely from UniMR (particle segmentation + classification) and
  AiSurf (template-matching atom counter). Both are PNG-only; here they
  become first-class STM operations with physical units preserved.

Placement note for future maintainers / AI coding agents
--------------------------------------------------------
These routines are the numerical kernels for the GUI Features tab, not Browse
thumbnail logic and not the standard Viewer processing panel. Keep this module
GUI-free and import it lazily from ``probeflow.gui.features`` or the CLI. That
keeps optional analysis dependencies such as OpenCV and scikit-learn out of the
core browse/convert path and prevents specialized particle/counting tools from
entangling basic image manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _pixel_scales(
    pixel_size_m: float,
    pixel_size_x_m: Optional[float] = None,
    pixel_size_y_m: Optional[float] = None,
) -> tuple[float, float, float]:
    """Return ``(dx, dy, sqrt(dx*dy))`` in metres."""
    dx = float(pixel_size_x_m if pixel_size_x_m is not None else pixel_size_m)
    dy = float(pixel_size_y_m if pixel_size_y_m is not None else pixel_size_m)
    if dx <= 0 or dy <= 0:
        raise ValueError("pixel sizes must be > 0")
    return dx, dy, float(np.sqrt(dx * dy))

from probeflow.analysis.helpers import (
    cv2_module,
    missing_extra_message as _missing_extra_message,
    to_uint8_for_cv as _to_uint8,
)


# ─── Lazy imports ────────────────────────────────────────────────────────────

def _cv():
    return cv2_module("feature detection")


def _sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - deps guard
        raise ImportError(
            _missing_extra_message("scikit-learn", "sklearn", "feature detection")
        ) from exc
    import sklearn
    return sklearn


# ═════════════════════════════════════════════════════════════════════════════
# Scale-invariant classification crop pipeline
# ═════════════════════════════════════════════════════════════════════════════
# The classifier fingerprints a crop around each particle. The legacy crop is a
# fixed *pixel* window (``_crop_particle``), so its physical field of view (FOV)
# = crop_size_px × pixel_size varies per scan and the same molecule embeds
# differently at different resolutions — which made the cross-scan feature bank
# scale-blind. The physical-FOV path below crops a fixed FOV in *nanometres* and
# resamples to a fixed grid, so every embedding is computed at the same nm/px by
# construction, and optionally blends the background toward the crop median via
# the segmentation mask so shape (not shared background) drives the similarity.
#
# EMBED_VERSION identifies this crop/mask/resample pipeline. The feature bank
# stores it per entry and refuses to compare embeddings across versions — so any
# change to the crop, mask, or resample logic below MUST bump EMBED_VERSION.
DEFAULT_CROP_FOV_NM = 15.0   # physical field of view of each classifier crop, nm
DEFAULT_OUT_PX = 96          # crops resampled to this fixed grid before encoding
MIN_CROP_PX = 24             # below this the source scan is too coarse for the FOV
EMBED_VERSION = "physfov-mask-v1"


def _particle_center_px(particle: "Particle") -> tuple[int, int]:
    cx = (particle.bbox_px[0] + particle.bbox_px[2]) // 2
    cy = (particle.bbox_px[1] + particle.bbox_px[3]) // 2
    return int(cx), int(cy)


def _extract_window(arr2d: np.ndarray, cx: int, cy: int,
                    win_x: int, win_y: int) -> np.ndarray:
    """Reflect-padded ``(win_y, win_x)`` crop centred on ``(cx, cy)``.

    Shared geometry so an image and its mask crop identically and stay aligned.
    """
    Ny, Nx = arr2d.shape
    hx, hy = win_x // 2, win_y // 2
    x0, x1 = cx - hx, cx - hx + win_x
    y0, y1 = cy - hy, cy - hy + win_y
    pad_l, pad_r = max(0, -x0), max(0, x1 - Nx)
    pad_t, pad_b = max(0, -y0), max(0, y1 - Ny)
    x0c, x1c = max(0, x0), min(Nx, x1)
    y0c, y1c = max(0, y0), min(Ny, y1)
    crop = arr2d[y0c:y1c, x0c:x1c]
    if pad_l or pad_r or pad_t or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r)), mode="reflect")
    return crop


def _physical_window_px(pixel_size_x_m: float, pixel_size_y_m: float,
                        fov_nm: float) -> tuple[int, int]:
    """Pixel side lengths that span ``fov_nm`` on each axis (rectangular-safe)."""
    fov_m = float(fov_nm) * 1e-9
    win_x = int(round(fov_m / float(pixel_size_x_m)))
    win_y = int(round(fov_m / float(pixel_size_y_m)))
    return max(1, win_x), max(1, win_y)


def _particle_mask_full(shape: tuple[int, int], particle: "Particle",
                        pixel_size_x_m: float, pixel_size_y_m: float) -> np.ndarray:
    """Rasterise a particle's contour into a full-image uint8 mask (1 = inside).

    Falls back to the bounding box when no usable contour is stored.
    """
    Ny, Nx = shape
    mask = np.zeros((Ny, Nx), dtype=np.uint8)
    contour = getattr(particle, "contour_xy_m", None)
    if contour:
        pts = np.array(
            [[int(round(x / pixel_size_x_m)), int(round(y / pixel_size_y_m))]
             for x, y in contour],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            _cv().fillPoly(mask, [pts], 1)
            return mask
    x0, y0, x1, y1 = particle.bbox_px
    mask[max(0, int(y0)):min(Ny, int(y1)), max(0, int(x0)):min(Nx, int(x1))] = 1
    return mask


def _crop_particle_physical(
    arr: np.ndarray,
    particle: "Particle",
    *,
    fov_nm: float,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    out_px: int = DEFAULT_OUT_PX,
    apply_mask: bool = True,
) -> np.ndarray:
    """Fixed-physical-FOV, fixed-grid crop for scale-invariant embedding.

    Crops a ``fov_nm`` window around the centroid (reflect-padded at edges),
    optionally blends the background toward the crop median through the feathered
    segmentation mask, then resamples to ``out_px × out_px``. Raises ValueError
    when the source scan is too coarse (window < ``MIN_CROP_PX``) to represent
    the FOV without upsampling detail that isn't there.
    """
    cv2 = _cv()
    win_x, win_y = _physical_window_px(pixel_size_x_m, pixel_size_y_m, fov_nm)
    if min(win_x, win_y) < MIN_CROP_PX:
        raise ValueError(
            f"scan too coarse for a {fov_nm:.1f} nm crop: the FOV spans only "
            f"{min(win_x, win_y)} px (< {MIN_CROP_PX} px minimum). Use a finer "
            "scan or a larger FOV."
        )
    cx, cy = _particle_center_px(particle)
    crop = _extract_window(np.asarray(arr, dtype=np.float64), cx, cy, win_x, win_y)
    if apply_mask:
        mask_full = _particle_mask_full(arr.shape, particle,
                                        pixel_size_x_m, pixel_size_y_m)
        mcrop = _extract_window(mask_full.astype(np.float64), cx, cy, win_x, win_y)
        # Feather so the blend has no hard edge (hard edges create CLIP
        # artefacts), then pull background toward the crop median.
        sigma = max(win_x, win_y) / 24.0
        if sigma > 0:
            mcrop = cv2.GaussianBlur(mcrop, (0, 0), sigmaX=sigma)
        mcrop = np.clip(mcrop, 0.0, 1.0)
        bg = float(np.median(crop))
        crop = bg + (crop - bg) * mcrop
    interp = cv2.INTER_AREA if out_px < min(win_x, win_y) else cv2.INTER_LINEAR
    resized = cv2.resize(crop.astype(np.float32), (out_px, out_px),
                         interpolation=interp)
    return resized.astype(np.float64)


def _stack_crops(
    arr: np.ndarray,
    particles,
    *,
    crop_size_px: int,
    pixel_size_x_m,
    pixel_size_y_m,
    fov_nm: float,
    out_px: int,
    apply_mask: bool,
) -> np.ndarray:
    """Stack crops for a list of particles via the physical or legacy path.

    Physical path (``pixel_size_*`` given): scale-invariant fixed-FOV crops.
    Legacy path (``pixel_size_*`` None): the historical fixed-pixel crops.
    Returns an empty ``(0, H, W)`` stack when ``particles`` is empty.
    """
    physical = pixel_size_x_m is not None and pixel_size_y_m is not None
    if not particles:
        side = out_px if physical else crop_size_px
        return np.empty((0, side, side), dtype=np.float64)
    if physical:
        return np.stack([
            _crop_particle_physical(
                arr, p, fov_nm=fov_nm,
                pixel_size_x_m=pixel_size_x_m, pixel_size_y_m=pixel_size_y_m,
                out_px=out_px, apply_mask=apply_mask,
            )
            for p in particles
        ], axis=0)
    return np.stack([_crop_particle(arr, p, crop_size_px) for p in particles],
                    axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Particle segmentation   (UniMR-style Otsu + contour pipeline, in SI units)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Particle:
    """A segmented particle / molecule / island on an STM scan.

    All coordinates are in metres relative to the scan origin (top-left).
    Heights are in the z-unit of the source plane (usually metres for Z planes,
    amperes for current planes).
    """
    index: int
    centroid_x_m: float
    centroid_y_m: float
    area_m2: float
    area_nm2: float
    bbox_m: Tuple[float, float, float, float]  # x0, y0, x1, y1
    bbox_px: Tuple[int, int, int, int]         # x0, y0, x1, y1
    mean_height: float
    max_height: float
    min_height: float
    n_pixels: int
    contour_xy_m: List[Tuple[float, float]] = field(default_factory=list)
    orientation_deg: float = 0.0
    """Major-axis angle in degrees, 0–180°.  Computed via PCA on the rasterised
    particle mask.  0° = pointing right (East); 90° = pointing down (South in
    image coordinates).  Two particles with the same shape but mirrored
    directions have the same orientation (headless vector)."""
    sharpness: float = 0.0
    """Variance of the Laplacian within the particle bounding-box crop (uint8
    scale).  Higher = sharper / more well-defined edges.  Useful to separate
    fuzzy molecules from sharp ones when the overall shape looks similar."""

    def to_dict(self) -> dict:
        return asdict(self)


def segment_particles(
    arr: np.ndarray,
    pixel_size_m: float,
    *,
    pixel_size_x_m: Optional[float] = None,
    pixel_size_y_m: Optional[float] = None,
    threshold: str = "otsu",
    manual_value: Optional[float] = None,
    invert: bool = False,
    min_area_nm2: float = 0.5,
    max_area_nm2: Optional[float] = None,
    size_sigma_clip: Optional[float] = 2.0,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    exclude_mask: Optional[np.ndarray] = None,
    max_exclude_overlap: float = 0.25,
) -> List[Particle]:
    """Segment bright features on a scan plane into a list of Particles.

    Parameters
    ----------
    arr
        2-D float array (one scan plane).
    pixel_size_m
        Backward-compatible scalar physical pixel size, metres.
    pixel_size_x_m, pixel_size_y_m
        Optional physical pixel width and height. Use these for rectangular
        scans; areas use ``dx * dy`` and coordinates use the corresponding
        axis scale.
    threshold
        ``"otsu"`` (default) uses Otsu's automatic threshold on the percentile-
        normalised uint8 view. ``"manual"`` uses ``manual_value`` in 0-255
        bytes. ``"adaptive"`` uses cv2.adaptiveThreshold with a mean block.
    manual_value
        Required when ``threshold="manual"`` — the 0-255 byte cutoff.
    invert
        If True, segment *dark* features (depressions) instead of bright ones.
    min_area_nm2, max_area_nm2
        Absolute physical area filters in nm². ``max_area_nm2=None`` disables
        the upper bound.
    size_sigma_clip
        Drop particles whose area is more than this many σ from the mean area
        of surviving particles (set to None to disable). Catches salt-and-
        pepper tiny blobs and full-image artefacts that Otsu can let through.
    clip_low, clip_high
        Percentile range for the float→uint8 rescale.
    exclude_mask
        Optional boolean array, same shape as ``arr``. A segmented particle is
        rejected wholesale when more than ``max_exclude_overlap`` of its pixels
        fall inside the mask. Use it to drop molecules sitting on a step edge
        (see :func:`probeflow.analysis.step_edges.step_edge_mask`) or in any
        other algorithmically-defined exclusion zone — cleaner than zeroing
        pixels, which can split a molecule across the boundary.
    max_exclude_overlap
        Overlap fraction (0–1) above which a particle is rejected when
        ``exclude_mask`` is supplied. Default 0.25: a particle whose edge merely
        grazes the zone is kept; one sitting squarely on it is dropped.

    Returns
    -------
    list[Particle]
        In arbitrary order (sort by ``.area_nm2`` for a canonical ordering).
    """
    if arr.ndim != 2:
        raise ValueError("segment_particles expects a 2-D array")
    dx_m, dy_m, _ = _pixel_scales(pixel_size_m, pixel_size_x_m, pixel_size_y_m)

    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape != arr.shape:
            raise ValueError("exclude_mask must have the same shape as arr")

    # Constant or empty planes carry no features. OpenCV's Otsu threshold on
    # a flat image is undefined (newer versions return a full-image foreground
    # mask, which surfaces as a phantom whole-image particle).
    finite = arr[np.isfinite(arr)] if arr.size else arr
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        return []

    cv2 = _cv()
    Ny, Nx = arr.shape
    u8 = _to_uint8(arr, clip_low=clip_low, clip_high=clip_high)

    if threshold == "otsu":
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(u8, 0, 255, flag + cv2.THRESH_OTSU)
    elif threshold == "manual":
        if manual_value is None:
            raise ValueError("threshold='manual' requires manual_value")
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(u8, float(manual_value), 255, flag)
    elif threshold == "adaptive":
        block = max(11, (min(Nx, Ny) // 16) | 1)
        mask = cv2.adaptiveThreshold(
            u8, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            block, 2,
        )
    else:
        raise ValueError(f"Unknown threshold method {threshold!r}")

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    px_area = dx_m * dy_m
    particles: List[Particle] = []
    for i, cnt in enumerate(contours):
        # Rasterised particle mask (robust for holes and thin shapes).
        p_mask = np.zeros((Ny, Nx), dtype=np.uint8)
        cv2.drawContours(p_mask, [cnt], -1, color=1, thickness=-1)
        n_pix = int(p_mask.sum())
        if n_pix == 0:
            continue

        # Whole-particle exclusion: drop a particle that sits substantially in
        # the exclusion zone (e.g. on a step edge). Done here, before the heavier
        # PCA/Laplacian work, so it short-circuits cheaply.
        if exclude_mask is not None:
            overlap = int((p_mask.astype(bool) & exclude_mask).sum()) / n_pix
            if overlap > max_exclude_overlap:
                continue

        area_m2 = n_pix * px_area
        area_nm2 = area_m2 * 1e18
        if area_nm2 < min_area_nm2:
            continue
        if max_area_nm2 is not None and area_nm2 > max_area_nm2:
            continue

        ys, xs = np.where(p_mask > 0)
        cx_px = float(xs.mean())
        cy_px = float(ys.mean())
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        heights = arr[ys, xs]
        finite = heights[np.isfinite(heights)]
        if finite.size == 0:
            mean_h = max_h = min_h = float("nan")
        else:
            mean_h = float(finite.mean())
            max_h = float(finite.max())
            min_h = float(finite.min())

        contour_xy_m = [
            (float(pt[0][0]) * dx_m,
             float(pt[0][1]) * dy_m)
            for pt in cnt
        ]

        # ── Major-axis orientation via PCA on rasterised pixel positions ─────
        # Pixel x-axis corresponds to scan columns; orientation is 0–180° so
        # it's a *headless* direction (East=0°, South=90° in image coords).
        if n_pix >= 2:
            pts_f = np.column_stack(
                [xs - float(xs.mean()), ys - float(ys.mean())]
            ).astype(np.float64)
            cov = pts_f.T @ pts_f / pts_f.shape[0]
            _, eigvecs = np.linalg.eigh(cov)
            major_vec = eigvecs[:, -1]   # eigenvector with largest eigenvalue
            orient = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
            if orient < 0.0:
                orient += 180.0
        else:
            orient = 0.0

        # ── Sharpness: variance of Laplacian on the bbox crop ────────────────
        # cv2.Laplacian on a uint8 crop gives a measure of edge strength.
        # High variance → sharp, well-defined edges.
        # Low variance → fuzzy / blurred particle.
        try:
            crop_u8 = u8[y0: y1 + 1, x0: x1 + 1]
            if crop_u8.size > 0:
                lap = cv2.Laplacian(crop_u8.astype(float), cv2.CV_64F)
                sharpness = float(lap.var())
            else:
                sharpness = 0.0
        except Exception:
            sharpness = 0.0

        particles.append(Particle(
            index=len(particles),
            centroid_x_m=cx_px * dx_m,
            centroid_y_m=cy_px * dy_m,
            area_m2=area_m2,
            area_nm2=area_nm2,
            bbox_m=(x0 * dx_m, y0 * dy_m,
                    (x1 + 1) * dx_m, (y1 + 1) * dy_m),
            bbox_px=(x0, y0, x1 + 1, y1 + 1),
            mean_height=mean_h,
            max_height=max_h,
            min_height=min_h,
            n_pixels=n_pix,
            contour_xy_m=contour_xy_m,
            orientation_deg=orient,
            sharpness=sharpness,
        ))

    if size_sigma_clip is not None and len(particles) > 3:
        areas = np.array([p.area_nm2 for p in particles])
        mean_a = float(areas.mean())
        std_a = float(areas.std())
        if std_a > 0:
            lo = max(min_area_nm2, mean_a - size_sigma_clip * std_a)
            hi = mean_a + size_sigma_clip * std_a
            if max_area_nm2 is not None:
                hi = min(hi, max_area_nm2)
            particles = [p for p in particles if lo <= p.area_nm2 <= hi]
            # Re-index after clipping.
            for k, p in enumerate(particles):
                p.index = k

    return particles


# ═════════════════════════════════════════════════════════════════════════════
# 2. Template-match counting  (AiSurf atom_counting algorithm, ported)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """One template-match detection (typically an atom or repeated motif)."""
    index: int
    x_m: float
    y_m: float
    x_px: int
    y_px: int
    correlation: float
    local_height: float

    def to_dict(self) -> dict:
        return asdict(self)


def _peak_local_max(response: np.ndarray, min_distance_px: int,
                    threshold_abs: float) -> np.ndarray:
    """Pure-numpy local-maximum peak finder.

    Returns an (N, 2) array of (row, col) indices, sorted by descending
    response. Equivalent to skimage.feature.peak_local_max with the chosen
    distance and absolute threshold — we roll our own to avoid an extra dep.
    """
    resp = response.copy()
    resp[resp < threshold_abs] = -np.inf
    peaks: List[Tuple[int, int]] = []
    # Flat-sort indices by response (descending).
    flat_order = np.argsort(resp.ravel())[::-1]
    Ny, Nx = resp.shape
    taken = np.zeros_like(resp, dtype=bool)
    for idx in flat_order:
        val = resp.ravel()[idx]
        if not np.isfinite(val) or val == -np.inf:
            break
        r, c = divmod(int(idx), Nx)
        if taken[r, c]:
            continue
        peaks.append((r, c))
        r0 = max(0, r - min_distance_px)
        r1 = min(Ny, r + min_distance_px + 1)
        c0 = max(0, c - min_distance_px)
        c1 = min(Nx, c + min_distance_px + 1)
        taken[r0:r1, c0:c1] = True
    return np.array(peaks, dtype=int) if peaks else np.zeros((0, 2), dtype=int)


def count_features(
    arr: np.ndarray,
    template: np.ndarray,
    pixel_size_m: float,
    *,
    pixel_size_x_m: Optional[float] = None,
    pixel_size_y_m: Optional[float] = None,
    min_correlation: float = 0.5,
    min_distance_m: Optional[float] = None,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> List[Detection]:
    """Count repeated features using normalised cross-correlation.

    Parameters
    ----------
    arr
        2-D float scan plane.
    template
        2-D float array — a crop of the repeating motif. Must be smaller than
        ``arr`` in both dimensions.
    pixel_size_m
        Backward-compatible scalar physical pixel size, metres.
    pixel_size_x_m, pixel_size_y_m
        Optional physical pixel width and height. Detected coordinates use the
        corresponding axis scale; the scalar geometric mean is used only for
        the square-pixel suppression window.
    min_correlation
        Reject peaks whose normalised cross-correlation is below this value.
        AiSurf's recommended range is 0.4 – 0.6.
    min_distance_m
        Physical exclusion radius for non-maximum suppression. Default: half
        the geometric mean of the template side lengths.

    Returns
    -------
    list[Detection]
        Positions, correlations, and local heights of detected features.
    """
    if arr.ndim != 2 or template.ndim != 2:
        raise ValueError("arr and template must be 2-D arrays")
    if template.shape[0] >= arr.shape[0] or template.shape[1] >= arr.shape[1]:
        raise ValueError("template must be smaller than arr in both dims")
    dx_m, dy_m, geom_px_m = _pixel_scales(
        pixel_size_m, pixel_size_x_m, pixel_size_y_m
    )

    cv2 = _cv()

    img_u8 = _to_uint8(arr, clip_low=clip_low, clip_high=clip_high)
    tmpl_u8 = _to_uint8(template, clip_low=clip_low, clip_high=clip_high)

    response = cv2.matchTemplate(img_u8, tmpl_u8, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape
    oy, ox = th // 2, tw // 2  # offset to recover whole-image coordinates

    if min_distance_m is None:
        min_distance_m = 0.5 * float(np.sqrt(th * tw)) * geom_px_m
    min_distance_px = max(1, int(round(min_distance_m / geom_px_m)))

    peaks = _peak_local_max(response,
                            min_distance_px=min_distance_px,
                            threshold_abs=float(min_correlation))

    detections: List[Detection] = []
    for i, (r, c) in enumerate(peaks):
        x_px = int(c + ox)
        y_px = int(r + oy)
        if not (0 <= x_px < arr.shape[1] and 0 <= y_px < arr.shape[0]):
            continue
        h = float(arr[y_px, x_px]) if np.isfinite(arr[y_px, x_px]) else float("nan")
        detections.append(Detection(
            index=i,
            x_m=x_px * dx_m,
            y_m=y_px * dy_m,
            x_px=x_px,
            y_px=y_px,
            correlation=float(response[r, c]),
            local_height=h,
        ))
    # Re-index after filtering.
    for k, d in enumerate(detections):
        d.index = k
    return detections


# ═════════════════════════════════════════════════════════════════════════════
# 3. Few-shot classification  (UniMR-style, CLIP-free path)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Classification:
    """Result of classifying one particle against a set of labelled samples."""
    particle_index: int
    class_name: str
    similarity: float

    def to_dict(self) -> dict:
        return asdict(self)


def _crop_particle(arr: np.ndarray, particle: "Particle",
                   crop_size_px: int) -> np.ndarray:
    """Centre-crop a square patch around ``particle`` of side crop_size_px.

    Reflect-pads near the edges so downstream encoders see a fixed shape.
    """
    Ny, Nx = arr.shape
    cx = (particle.bbox_px[0] + particle.bbox_px[2]) // 2
    cy = (particle.bbox_px[1] + particle.bbox_px[3]) // 2

    half = crop_size_px // 2
    x0, x1 = cx - half, cx - half + crop_size_px
    y0, y1 = cy - half, cy - half + crop_size_px

    pad_l = max(0, -x0)
    pad_r = max(0, x1 - Nx)
    pad_t = max(0, -y0)
    pad_b = max(0, y1 - Ny)
    x0c, x1c = max(0, x0), min(Nx, x1)
    y0c, y1c = max(0, y0), min(Ny, y1)
    crop = arr[y0c:y1c, x0c:x1c]
    if pad_l or pad_r or pad_t or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r)), mode="reflect")
    return crop


def _embed_raw(crops: np.ndarray) -> np.ndarray:
    """Flattened, per-crop z-score-normalised embedding (N, D).

    FUTURE OPPORTUNITY (biggest classify win): the crop is a fixed square that,
    for a small particle on a flat terrace, is mostly background.  Flattening
    the whole crop lets that shared background dominate the vector, so cosine
    similarities between *any* two particles crush toward ~1.0 and barely
    spread — which is why outlier rejection is unreliable and the auto-threshold
    in ``_threshold_similarities`` has to fall back to argmax-only (see the note
    there).  A particle-region-weighted / masked embedding — e.g. weight pixels
    by the segmentation mask, or crop tightly to the bounding box and resample
    to a common size — would let the actual shape drive the similarity and make
    genuine 'other' detection work.  See memory ``project_unimr_review``.
    """
    flat = crops.reshape(crops.shape[0], -1).astype(np.float64)
    # Per-crop normalisation to be brightness-invariant.
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True) + 1e-8
    return (flat - mu) / sd


def _embed_pca_kmeans(crops: np.ndarray, *, n_components: int = 16) -> np.ndarray:
    """PCA-reduced embedding. Returns an (N, n_components) array."""
    _sklearn()
    from sklearn.decomposition import PCA

    flat = _embed_raw(crops)
    n_components = min(n_components, flat.shape[0] - 1, flat.shape[1])
    if n_components <= 0:
        return flat
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(flat)


# ── CLIP encoder (UniMR-faithful, optional dependency) ────────────────────────
# UniMR (the upstream feature-counting tool) vectorises each molecule crop with
# OpenAI CLIP ViT-B/32 and matches by cosine similarity. ProbeFlow's port
# shipped only the pixel encoders to stay torch-free; this restores CLIP as a
# *selectable* encoder. torch + the openai-clip package are optional — install
# ``probeflow[clip]`` — and imported lazily so the core app never requires them.

_CLIP_CACHE: dict = {}


def clip_available() -> bool:
    """True when the optional CLIP stack (``clip`` + ``torch``) actually imports.

    Uses a real (cached) import rather than ``find_spec`` so a broken or partial
    install — e.g. a torch wheel that failed to extract — reports ``False``
    instead of enabling an encoder that would crash on first use.
    """
    if "ok" not in _CLIP_CACHE:
        try:
            import clip  # noqa: F401
            import torch  # noqa: F401
            _CLIP_CACHE["ok"] = True
        except Exception:
            _CLIP_CACHE["ok"] = False
    return _CLIP_CACHE["ok"]


def _load_clip():
    """Load and cache CLIP ViT-B/32 (model, preprocess, device). UniMR-style."""
    if "model" not in _CLIP_CACHE:
        try:
            import clip
            import torch
        except ImportError as exc:  # pragma: no cover - deps guard
            raise ImportError(
                _missing_extra_message("CLIP", "clip", "the CLIP classify encoder")
                + "\n  (also needs: torch, torchvision)"
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        _CLIP_CACHE.update(model=model, preprocess=preprocess, device=device)
    return _CLIP_CACHE["model"], _CLIP_CACHE["preprocess"], _CLIP_CACHE["device"]


def _normalise_brightness_u8(crop_u8: np.ndarray, target: float = 120.0) -> np.ndarray:
    """Scale a uint8 crop so its mean luminance is ~``target`` (UniMR step)."""
    mean = float(crop_u8.mean())
    if mean <= 1e-6:
        return crop_u8
    scaled = crop_u8.astype(np.float64) * (target / mean)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def embed_particles_clip(
    arr: np.ndarray,
    particles: Sequence["Particle"],
    *,
    crop_size_px: int = 48,
    pixel_size_x_m: Optional[float] = None,
    pixel_size_y_m: Optional[float] = None,
    fov_nm: float = DEFAULT_CROP_FOV_NM,
    out_px: int = DEFAULT_OUT_PX,
    apply_mask: bool = True,
) -> np.ndarray:
    """CLIP-embed a set of particles cropped from ``arr``. Returns an (N, 512) array.

    Public helper (used by the feature-bank GUI) so callers get the exact same
    crop + embedding pipeline the classifier uses, without reaching into the
    private ``_crop_particle`` / ``_embed_clip`` internals. Returns an empty
    ``(0, 512)`` array when given no particles.

    When ``pixel_size_x_m`` / ``pixel_size_y_m`` are given the scale-invariant
    physical-FOV crop is used (:data:`EMBED_VERSION`), which is what the feature
    bank requires so cross-scan embeddings are comparable. Without them the
    legacy fixed-pixel crop is used (back-compat for callers that don't classify
    across scans).
    """
    particles = list(particles)
    if not particles:
        return np.empty((0, 512), dtype=np.float64)
    crops = _stack_crops(
        arr, particles,
        crop_size_px=crop_size_px,
        pixel_size_x_m=pixel_size_x_m, pixel_size_y_m=pixel_size_y_m,
        fov_nm=fov_nm, out_px=out_px, apply_mask=apply_mask,
    )
    return _embed_clip(crops)


def _embed_clip(crops: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    """CLIP image embeddings for crops, mirroring UniMR's pipeline.

    Each float crop is percentile-clipped to uint8, brightness-normalised to a
    mean of 120, expanded to 3-channel RGB, run through CLIP's own ``preprocess``
    (resize/centre-crop to 224 + CLIP normalisation), and encoded with
    ``model.encode_image``.  Returns an ``(N, 512)`` float64 array.
    """
    import torch
    from PIL import Image

    model, preprocess, device = _load_clip()
    tensors = []
    for crop in crops:
        u8 = _normalise_brightness_u8(_to_uint8(crop))
        rgb = np.repeat(u8[:, :, None], 3, axis=2)   # grayscale → RGB
        tensors.append(preprocess(Image.fromarray(rgb, mode="RGB")))

    feats: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size]).to(device)
            feats.append(model.encode_image(batch).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


def _rotate_crop(crop: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 2-D float array by ``angle_deg`` degrees (OpenCV, reflect-padded).

    Uses ``cv2.warpAffine`` with ``BORDER_REFLECT`` so the corners are filled
    with plausible image data rather than zeros.  Falls back to returning the
    original crop if OpenCV is unavailable.
    """
    try:
        cv2 = _cv()
        h, w = crop.shape
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
        rotated = cv2.warpAffine(
            crop.astype(np.float32), M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return rotated.astype(crop.dtype)
    except Exception:
        return crop


# Minimum best-similarity spread (cosine units) that can plausibly hold two
# distinct populations — a cluster of genuine matches and a cluster of
# non-matches ('other').  Below this the scene has no real outliers and any
# data-driven cutoff would land inside a single tight cluster, wrongly
# rejecting real matches.  Genuine 'other' molecules sit far below same-class
# matches, so a true bimodal distribution spreads well past this floor.
_MIN_SIMILARITY_SPREAD = 0.10


def _threshold_similarities(sims: np.ndarray, method: str) -> float:
    """Return the similarity cutoff above which a particle is 'a match'.

    The data-driven methods (``gmm`` / ``otsu`` / ``distribution``) all assume
    the best-match similarities are *bimodal*.  When the scene contains only
    particles the user cares about (no genuine outliers), the similarities form
    a single tight cluster and any cutoff placed inside it rejects most real
    matches as ``"other"``.  Guard against that: when the spread is too small to
    hold two populations — or when the GMM's two components are not genuinely
    separated — return a *permissive* cutoff (the minimum similarity) so every
    particle is classified by its nearest labelled sample (argmax only).

    Note: the raw pixel embedding is dominated by flat background, so even
    genuine outliers can sit inside this tight band; reliable outlier rejection
    needs a particle-region-weighted embedding (a separate, deeper change).
    """
    if sims.size == 0:
        return 0.0

    permissive = float(sims.min())   # cutoff ≤ every sim ⇒ classify all by argmax
    if float(np.ptp(sims)) < _MIN_SIMILARITY_SPREAD:
        return permissive

    if method == "gmm":
        _sklearn()
        from sklearn.mixture import GaussianMixture
        if sims.size < 4:
            return permissive
        gmm = GaussianMixture(n_components=2, random_state=0).fit(sims.reshape(-1, 1))
        order = np.argsort(gmm.means_.ravel())
        mus = gmm.means_.ravel()[order]
        sds = np.sqrt(gmm.covariances_.ravel()[order])
        # Trust the split only if the components are genuinely separated:
        # their means differ by more than the sum of their standard deviations.
        if (mus[1] - mus[0]) < (sds[0] + sds[1]):
            return permissive
        return float((mus[0] + mus[1]) / 2.0)
    if method == "otsu":
        # Otsu on a 256-bin histogram of the similarities.
        lo, hi = float(sims.min()), float(sims.max())
        if hi <= lo:
            return lo
        hist, edges = np.histogram(sims, bins=256, range=(lo, hi))
        p = hist / max(1, hist.sum())
        omega = np.cumsum(p)
        mu = np.cumsum(p * (edges[:-1] + np.diff(edges) / 2.0))
        mu_t = mu[-1]
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
        idx = int(np.nanargmax(sigma_b))
        return float(edges[idx])
    if method == "distribution":
        return float(sims.mean() + sims.std())
    raise ValueError(f"Unknown threshold method {method!r}")


def classify_particles(
    arr: np.ndarray,
    particles: Sequence["Particle"],
    samples: Sequence[Tuple[str, "Particle"]],
    *,
    encoder: str = "raw",
    threshold_method: str = "gmm",
    manual_threshold: float = 0.5,
    crop_size_px: int = 48,
    use_sharpness: bool = False,
    sharpness_weight: float = 3.0,
    rotate_augment: bool = False,
    bank_samples: Optional[Sequence[Tuple[str, Sequence[float]]]] = None,
    pixel_size_x_m: Optional[float] = None,
    pixel_size_y_m: Optional[float] = None,
    fov_nm: float = DEFAULT_CROP_FOV_NM,
    out_px: int = DEFAULT_OUT_PX,
    apply_mask: bool = True,
    bank_areas: Optional[Sequence[float]] = None,
    area_gate_ratio: Optional[float] = None,
) -> List[Classification]:
    """Classify each particle against labelled sample particles.

    Parameters
    ----------
    arr
        The scan plane the particles were detected on.
    particles
        Sequence of Particles to classify.
    samples
        List of ``(class_name, sample_particle)`` pairs.  Multiple particles
        may share the same class_name; their similarities are max-pooled.
    encoder
        ``"raw"`` — flattened z-normalised pixel vectors.
        ``"pca_kmeans"`` — PCA-reduced pixel vectors.
        ``"clip"`` — OpenAI CLIP ViT-B/32 image embeddings (UniMR-faithful;
        needs the optional ``probeflow[clip]`` extra). Far more discriminative
        than the pixel encoders, which the upstream UniMR tool relies on.
    threshold_method
        How to pick the "match" cutoff: ``"gmm"`` (default), ``"otsu"``,
        ``"distribution"``, or ``"manual"`` (uses ``manual_threshold``).
    manual_threshold
        Fixed similarity cutoff used when ``threshold_method="manual"``.
        Range 0–1; particles whose best similarity ≥ this value are classified.
    crop_size_px
        Side length of the square crop centred on each particle centroid.
    use_sharpness
        When True, append a normalised sharpness dimension (variance of
        Laplacian, stored in ``Particle.sharpness``) to each embedding vector.
        Scaled by ``sharpness_weight`` so it contributes meaningfully to the
        cosine similarity.  Use this when you have two molecule types that look
        the same in shape but one is fuzzy and one is sharp.
    sharpness_weight
        How strongly the sharpness dimension pulls the embedding apart.
        Default 3.0 works well when the shape similarity is high and blur is
        the main discriminating feature.
    rotate_augment
        When True, generate 36 rotated copies (0–350°, step 10°) of each
        labelled sample crop before embedding.  The best-match similarity
        across all rotations is used, making classification rotation-invariant.
        Mirrors the UniMR rotation-augmentation approach.
    bank_samples
        Optional ``(class_name, embedding)`` pairs from the persistent feature
        bank (:mod:`probeflow.analysis.feature_bank`) — pre-computed CLIP
        vectors of samples labelled on *other* scans.  They join the in-image
        samples as extra match candidates (same max-pool-by-class semantics),
        so classification can draw on examples gathered across many images.
        Only valid with ``encoder="clip"``: bank vectors are CLIP embeddings
        and are not comparable to the per-image raw/PCA pixel vectors.  With a
        non-empty bank, ``samples`` may be empty — a new scan can be classified
        without labelling anything in it.
    pixel_size_x_m, pixel_size_y_m
        Physical pixel sizes of ``arr`` (metres). When both are given, crops use
        the scale-invariant physical-FOV path (see :data:`EMBED_VERSION`): a
        fixed ``fov_nm`` window resampled to ``out_px``, so embeddings are
        comparable across scan resolutions. **Required whenever ``bank_samples``
        is used** — the bank holds physical-FOV embeddings that are only
        comparable to physical-FOV crops. Omit both for the legacy fixed-pixel
        crop (back-compat for in-image-only, non-bank classification).
    fov_nm, out_px, apply_mask
        Physical-FOV crop parameters (used only when pixel sizes are given):
        field of view in nm, output grid size, and whether to blend background
        toward the crop median via the feathered segmentation mask.
    bank_areas, area_gate_ratio
        Optional per-bank-entry particle areas (nm²) and a size gate. When
        ``area_gate_ratio`` is set (e.g. 2.0), a bank sample is excluded for a
        given particle unless their areas agree within that factor — cheap
        rejection of same-shape-different-size confusions (monomer vs dimer).

    Returns
    -------
    list[Classification]
        One entry per input particle; particles below threshold are labelled
        ``"other"``.
    """
    if not particles:
        return []
    bank = list(bank_samples) if bank_samples else []
    if bank and encoder != "clip":
        raise ValueError(
            "bank_samples holds CLIP embeddings and requires encoder='clip' "
            f"(got encoder={encoder!r})"
        )
    physical = pixel_size_x_m is not None and pixel_size_y_m is not None
    if bank and not physical:
        raise ValueError(
            "bank_samples requires the physical-FOV crop path — pass "
            "pixel_size_x_m and pixel_size_y_m so bank and in-image embeddings "
            "are computed at the same scale"
        )
    if not samples and not bank:
        return [Classification(p.index, "other", 0.0) for p in particles]

    all_particles = list(particles)
    sample_particles = [sp for _, sp in samples]

    _crop_kw = dict(
        crop_size_px=crop_size_px,
        pixel_size_x_m=pixel_size_x_m, pixel_size_y_m=pixel_size_y_m,
        fov_nm=fov_nm, out_px=out_px, apply_mask=apply_mask,
    )
    pcrops = _stack_crops(arr, all_particles, **_crop_kw)

    # Build sample crops — optionally augmented with 36 rotations.  With a
    # bank-only run (no in-image samples) this degenerates to an empty stack;
    # every match candidate then comes from the bank.
    raw_scrops = list(_stack_crops(arr, sample_particles, **_crop_kw))
    if not raw_scrops:
        scrops = np.empty((0,) + pcrops.shape[1:], dtype=pcrops.dtype)
        sample_names: List[str] = []
    elif rotate_augment:
        aug_scrops: List[np.ndarray] = []
        aug_names:  List[str]        = []
        for (name, _), scrop in zip(samples, raw_scrops):
            for angle in range(0, 360, 10):   # 0, 10, 20, …, 350  (36 angles)
                aug_scrops.append(_rotate_crop(scrop, float(angle)))
                aug_names.append(name)
        scrops = np.stack(aug_scrops, axis=0)
        sample_names = aug_names
    else:
        scrops = np.stack(raw_scrops, axis=0)
        sample_names = [name for name, _ in samples]

    if encoder == "raw":
        p_emb = _embed_raw(pcrops)
        s_emb = _embed_raw(scrops)
    elif encoder == "pca_kmeans":
        combined = np.concatenate([pcrops, scrops], axis=0)
        emb = _embed_pca_kmeans(combined)
        p_emb = emb[:len(pcrops)]
        s_emb = emb[len(pcrops):]
    elif encoder == "clip":
        # Embed particles and samples together so they share one forward pass.
        combined = np.concatenate([pcrops, scrops], axis=0)
        emb = _embed_clip(combined)
        p_emb = emb[:len(pcrops)]
        s_emb = emb[len(pcrops):]
    else:
        raise ValueError(f"Unknown encoder {encoder!r}")

    # ── Optional sharpness dimension ──────────────────────────────────────────
    # Appends a single z-normalised sharpness value (variance of Laplacian) to
    # each embedding vector, scaled by sharpness_weight so it contributes
    # meaningfully to the cosine similarity.  This pulls fuzzy and sharp
    # molecules apart even when their pixel patterns look nearly identical.
    #
    # Bug-fix: when rotate_augment=True each original sample becomes 36 rotated
    # copies in scrops/s_emb.  Repeat each sample's sharpness value to match.
    if use_sharpness:
        p_sharp = np.array([getattr(p,  "sharpness", 0.0) for p in all_particles],
                           dtype=np.float64)
        s_sharp = np.array([getattr(sp, "sharpness", 0.0) for sp in sample_particles],
                           dtype=np.float64)
        n_aug = scrops.shape[0] // max(1, len(sample_particles))
        if n_aug > 1:
            s_sharp = np.repeat(s_sharp, n_aug)
        all_sharp = np.concatenate([p_sharp, s_sharp])
        sharp_mean = float(all_sharp.mean())
        sharp_std  = float(all_sharp.std()) + 1e-8
        p_sharp_n = ((p_sharp - sharp_mean) / sharp_std * sharpness_weight
                     ).reshape(-1, 1)
        s_sharp_n = ((s_sharp - sharp_mean) / sharp_std * sharpness_weight
                     ).reshape(-1, 1)
        p_emb = np.concatenate([p_emb, p_sharp_n], axis=1)
        s_emb = np.concatenate([s_emb, s_sharp_n], axis=1)

    # ── Banked samples (cross-scan CLIP embeddings) ───────────────────────────
    # Appended after the sharpness dimension: bank entries carry no sharpness,
    # so they get 0.0 — the z-normalised mean — as a neutral value.
    if bank:
        bank_names = [str(name) for name, _ in bank]
        bank_emb = np.asarray(
            [np.asarray(v, dtype=np.float64).ravel() for _, v in bank],
            dtype=np.float64,
        )
        expected = p_emb.shape[1] - (1 if use_sharpness else 0)
        if bank_emb.ndim != 2 or bank_emb.shape[1] != expected:
            raise ValueError(
                f"bank embeddings have dimension "
                f"{bank_emb.shape[1] if bank_emb.ndim == 2 else '?'} but the "
                f"CLIP encoder produces {expected}; the bank file may be "
                "corrupt or from an incompatible encoder"
            )
        if use_sharpness:
            bank_emb = np.concatenate(
                [bank_emb, np.zeros((bank_emb.shape[0], 1))], axis=1)
        s_emb = np.concatenate([s_emb, bank_emb], axis=0)
        sample_names = list(sample_names) + bank_names

    # Cosine similarity: normalise rows, dot.
    def _unit(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + 1e-12)

    p_u = _unit(p_emb)
    s_u = _unit(s_emb)
    sim_matrix = p_u @ s_u.T    # (Np, Ns)

    # ── Optional physical-size gate on bank candidates ────────────────────────
    # A bank sample and a particle that share a shape but differ in physical
    # size (e.g. monomer vs dimer) should not match. When an area gate is set,
    # blank out (−inf) the similarity to any bank column whose area disagrees
    # with the particle's by more than ``area_gate_ratio``×. Only bank columns
    # are gated — in-image samples share this scan and need no size check.
    if bank and area_gate_ratio and area_gate_ratio > 1.0 and bank_areas is not None:
        n_bank = len(bank)
        first_bank_col = sim_matrix.shape[1] - n_bank
        b_area = list(bank_areas)
        p_area = [float(getattr(p, "area_nm2", 0.0) or 0.0) for p in all_particles]
        for j in range(n_bank):
            bj = float(b_area[j]) if j < len(b_area) and b_area[j] else 0.0
            if bj <= 0.0:
                continue  # unknown bank area — can't judge, leave it in
            col = first_bank_col + j
            for i, ai in enumerate(p_area):
                if ai <= 0.0:
                    continue
                ratio = max(ai / bj, bj / ai)
                if ratio > area_gate_ratio:
                    sim_matrix[i, col] = -np.inf

    # Per-particle: max similarity across all samples (or augmented rotations)
    # plus the class of the argmax.
    best_sim = sim_matrix.max(axis=1)
    best_idx = sim_matrix.argmax(axis=1)

    if threshold_method == "manual":
        cutoff = float(manual_threshold)
    else:
        # The area gate can set a row entirely to −inf (every bank candidate
        # rejected on size); those must not poison the data-driven cutoff.
        finite = best_sim[np.isfinite(best_sim)]
        cutoff = _threshold_similarities(
            finite if finite.size else best_sim, threshold_method)

    out: List[Classification] = []
    for i, (p, sim, argmax) in enumerate(zip(all_particles, best_sim, best_idx)):
        matched = np.isfinite(sim) and sim >= cutoff
        label = sample_names[int(argmax)] if matched else "other"
        out.append(Classification(
            particle_index=p.index,
            class_name=label,
            similarity=float(sim) if np.isfinite(sim) else 0.0,
        ))
    return out
