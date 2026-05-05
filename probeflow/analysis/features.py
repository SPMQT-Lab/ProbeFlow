"""
ProbeFlow — feature detection, counting, and classification.

Layered on top of :mod:`probeflow.processing` to extract *discrete objects*
from an STM scan plane: particles / molecules, atoms (template matching), and
classifications. All functions return SI-unit dataclasses that serialise
cleanly to JSON via :mod:`probeflow.io.writers.json`.

Design notes
------------
* Everything operates on a single 2-D float array plus ``pixel_size_m``.
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

    def to_dict(self) -> dict:
        return asdict(self)


def segment_particles(
    arr: np.ndarray,
    pixel_size_m: float,
    *,
    threshold: str = "otsu",
    manual_value: Optional[float] = None,
    invert: bool = False,
    min_area_nm2: float = 0.5,
    max_area_nm2: Optional[float] = None,
    size_sigma_clip: Optional[float] = 2.0,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> List[Particle]:
    """Segment bright features on a scan plane into a list of Particles.

    Parameters
    ----------
    arr
        2-D float array (one scan plane).
    pixel_size_m
        Physical pixel size, metres. When pixels are non-square this should be
        the geometric mean (``sqrt(dx * dy)``).
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

    Returns
    -------
    list[Particle]
        In arbitrary order (sort by ``.area_nm2`` for a canonical ordering).
    """
    if arr.ndim != 2:
        raise ValueError("segment_particles expects a 2-D array")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be > 0")

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

    px_area = pixel_size_m * pixel_size_m
    particles: List[Particle] = []
    for i, cnt in enumerate(contours):
        # Rasterised particle mask (robust for holes and thin shapes).
        p_mask = np.zeros((Ny, Nx), dtype=np.uint8)
        cv2.drawContours(p_mask, [cnt], -1, color=1, thickness=-1)
        n_pix = int(p_mask.sum())
        if n_pix == 0:
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
            (float(pt[0][0]) * pixel_size_m,
             float(pt[0][1]) * pixel_size_m)
            for pt in cnt
        ]

        particles.append(Particle(
            index=len(particles),
            centroid_x_m=cx_px * pixel_size_m,
            centroid_y_m=cy_px * pixel_size_m,
            area_m2=area_m2,
            area_nm2=area_nm2,
            bbox_m=(x0 * pixel_size_m, y0 * pixel_size_m,
                    (x1 + 1) * pixel_size_m, (y1 + 1) * pixel_size_m),
            bbox_px=(x0, y0, x1 + 1, y1 + 1),
            mean_height=mean_h,
            max_height=max_h,
            min_height=min_h,
            n_pixels=n_pix,
            contour_xy_m=contour_xy_m,
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
        Physical pixel size.
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
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be > 0")

    cv2 = _cv()

    img_u8 = _to_uint8(arr, clip_low=clip_low, clip_high=clip_high)
    tmpl_u8 = _to_uint8(template, clip_low=clip_low, clip_high=clip_high)

    response = cv2.matchTemplate(img_u8, tmpl_u8, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape
    oy, ox = th // 2, tw // 2  # offset to recover whole-image coordinates

    if min_distance_m is None:
        min_distance_m = 0.5 * float(np.sqrt(th * tw)) * pixel_size_m
    min_distance_px = max(1, int(round(min_distance_m / pixel_size_m)))

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
            x_m=x_px * pixel_size_m,
            y_m=y_px * pixel_size_m,
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
    cx = int(round(particle.centroid_x_m
                   / (particle.bbox_m[2] - particle.bbox_m[0])
                   * (particle.bbox_px[2] - particle.bbox_px[0]))) \
        if (particle.bbox_m[2] - particle.bbox_m[0]) > 0 else particle.bbox_px[0]
    # Simpler: recover centroid pixel from bbox midpoint.
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
    """Flattened, per-crop z-score-normalised embedding (N, D)."""
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


def _threshold_similarities(sims: np.ndarray, method: str) -> float:
    """Return the similarity cutoff above which a particle is 'a match'."""
    if sims.size == 0:
        return 0.0
    if method == "gmm":
        _sklearn()
        from sklearn.mixture import GaussianMixture
        if sims.size < 4 or np.ptp(sims) < 1e-9:
            return float(sims.mean())
        gmm = GaussianMixture(n_components=2, random_state=0).fit(sims.reshape(-1, 1))
        mus = np.sort(gmm.means_.ravel())
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
    crop_size_px: int = 48,
) -> List[Classification]:
    """Classify each particle against labelled sample particles.

    Parameters
    ----------
    arr
        The scan plane the particles were detected on.
    particles
        Sequence of Particles to classify.
    samples
        List of ``(class_name, sample_particle)`` pairs. Multiple particles may
        share the same class_name; their similarities are max-pooled.
    encoder
        ``"raw"`` — flattened z-normalised pixel vectors.
        ``"pca_kmeans"`` — PCA-reduced pixel vectors.
    threshold_method
        How to pick the "match" cutoff: ``"gmm"`` (default), ``"otsu"``,
        ``"distribution"``.

    Returns
    -------
    list[Classification]
        One entry per input particle; particles below threshold are labelled
        ``"other"``.
    """
    if not particles:
        return []
    if not samples:
        return [Classification(p.index, "other", 0.0) for p in particles]

    all_particles = list(particles)
    sample_particles = [sp for _, sp in samples]

    pcrops = np.stack([_crop_particle(arr, p, crop_size_px)
                       for p in all_particles], axis=0)
    scrops = np.stack([_crop_particle(arr, sp, crop_size_px)
                       for sp in sample_particles], axis=0)

    if encoder == "raw":
        p_emb = _embed_raw(pcrops)
        s_emb = _embed_raw(scrops)
    elif encoder == "pca_kmeans":
        combined = np.concatenate([pcrops, scrops], axis=0)
        emb = _embed_pca_kmeans(combined)
        p_emb = emb[:len(pcrops)]
        s_emb = emb[len(pcrops):]
    else:
        raise ValueError(f"Unknown encoder {encoder!r}")

    # Cosine similarity: normalise rows, dot.
    def _unit(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (n + 1e-12)

    p_u = _unit(p_emb)
    s_u = _unit(s_emb)
    sim_matrix = p_u @ s_u.T    # (Np, Ns)

    # Per-particle: max similarity across all samples + the class of the argmax.
    best_sim = sim_matrix.max(axis=1)
    best_idx = sim_matrix.argmax(axis=1)
    sample_names = [name for name, _ in samples]

    cutoff = _threshold_similarities(best_sim, threshold_method)
    out: List[Classification] = []
    for i, (p, sim, argmax) in enumerate(zip(all_particles, best_sim, best_idx)):
        label = sample_names[int(argmax)] if sim >= cutoff else "other"
        out.append(Classification(
            particle_index=p.index,
            class_name=label,
            similarity=float(sim),
        ))
    return out
