"""
ProbeFlow — lattice extraction for atomically-resolved STM images.

Ported from AiSurf's lattice_extraction notebook (SIFT keypoints → silhouette
clustering → kNN vector clustering → primitive lattice vectors). The physics
is identical; the API is a single ``extract_lattice()`` call that takes a
Scan plane plus a ``pixel_size_m`` and returns a :class:`LatticeResult`.

Optional dependency: this module needs OpenCV (for SIFT) and scikit-learn
(for silhouette-scored clustering). Install via ``pip install probeflow[features]``.

Keep heavy imports lazy so ``import probeflow`` never pays the cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

from probeflow.analysis.helpers import cv2_module, to_uint8_for_cv as _to_uint8

# ─── Lazy imports ────────────────────────────────────────────────────────────

def _cv():
    return cv2_module("lattice extraction")


def _sk_cluster():
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for lattice extraction. "
            "Install with `pip install probeflow[features]`."
        ) from exc
    return AgglomerativeClustering, silhouette_score


# ─── Parameters and result dataclasses ──────────────────────────────────────

@dataclass
class LatticeParams:
    """Tunable knobs — defaults match AiSurf's published recommendations."""
    # SIFT
    contrast_threshold: float = 0.003
    sigma: float = 4.0
    n_octave_layers: int = 8
    # Keypoint filtering
    size_threshold: float = 2.0
    edge_threshold: float = 1.0
    # Keypoint clustering
    cluster_kp_low: int = 2
    cluster_kp_high: int = 12
    cluster_choice: int = 1           # 1 = most populated cluster
    # Nearest-neighbour clustering
    cluster_kNN_low: int = 6
    cluster_kNN_high: int = 24
    clustersize_threshold: float = 0.3
    # Image prep
    clip_low: float = 1.0
    clip_high: float = 99.0


@dataclass
class LatticeResult:
    """Extracted lattice description in physical units."""
    a_vector_m: Tuple[float, float]      # (ax, ay) in metres
    b_vector_m: Tuple[float, float]
    a_length_m: float
    b_length_m: float
    gamma_deg: float                     # angle between a and b
    a_vector_px: Tuple[float, float]     # same vectors in pixels
    b_vector_px: Tuple[float, float]
    n_keypoints: int
    n_keypoints_used: int                # after cluster selection
    keypoints_xy_px: List[Tuple[float, float]] = field(default_factory=list)
    cluster_labels: List[int] = field(default_factory=list)
    primary_cluster: int = -1
    pixel_size_m: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    uu = u / (np.linalg.norm(u) + 1e-12)
    vv = v / (np.linalg.norm(v) + 1e-12)
    cos_th = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_th)))


def _best_clustering(descriptors: np.ndarray, span: range,
                     **agg_kwargs) -> np.ndarray:
    """Pick the # of clusters within ``span`` that maximises silhouette.

    Mirrors AiSurf's ``find_best_clustering``. Returns an integer label array.
    """
    AgglomerativeClustering, silhouette_score = _sk_cluster()
    if descriptors.shape[0] < max(span) + 1:
        # Not enough samples to try all n_clusters — cap the span.
        span = range(2, min(descriptors.shape[0], max(span)))
        if len(list(span)) == 0:
            return np.zeros(descriptors.shape[0], dtype=int)

    best_labels = None
    best_score = -1.0
    for n in span:
        if n >= descriptors.shape[0]:
            break
        labels = AgglomerativeClustering(n_clusters=n, **agg_kwargs) \
            .fit_predict(descriptors)
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(descriptors, labels)
        except Exception:
            continue
        if s > best_score:
            best_score = s
            best_labels = labels
    if best_labels is None:
        return np.zeros(descriptors.shape[0], dtype=int)
    return best_labels


def _kNN_displacements(positions: np.ndarray, k: int) -> np.ndarray:
    """For each point return its k nearest neighbours' displacement vectors.

    Parameters
    ----------
    positions
        (N, 2) array of (x, y) positions.
    k
        Number of neighbours to take (self excluded).

    Returns
    -------
    (N*k, 2) array of displacement vectors, each row = neighbour − point.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    _, idxs = tree.query(positions, k=k + 1)  # +1 to include self
    out = []
    for i, neighbours in enumerate(idxs):
        for j in neighbours[1:]:  # skip self
            out.append(positions[j] - positions[i])
    return np.asarray(out, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# extract_lattice
# ═════════════════════════════════════════════════════════════════════════════

def extract_lattice(
    arr: np.ndarray,
    pixel_size_m: float,
    params: Optional[LatticeParams] = None,
) -> LatticeResult:
    """Extract primitive lattice vectors from an STM image.

    Parameters
    ----------
    arr
        2-D scan plane (one channel).
    pixel_size_m
        Physical pixel size (metres). Must be > 0.
    params
        Tunable parameters. Defaults are AiSurf's recommended settings.

    Returns
    -------
    LatticeResult
        Primitive lattice vectors (a, b) and diagnostic cluster labels.

    Notes
    -----
    Algorithm (after AiSurf):

    1. SIFT keypoints on the percentile-normalised uint8 view.
    2. Drop keypoints that are too large, too small, or too near the border.
    3. Cluster the 128-dim SIFT descriptors; pick the most populated cluster
       (``cluster_choice``) to isolate one sublattice.
    4. For each keypoint in that cluster, take its ``cluster_kNN_low``
       nearest neighbours → displacement vectors.
    5. Cluster those displacements; the average of each well-populated
       cluster is a candidate lattice vector.
    6. Sort candidates by |v|. Pick the two smallest whose
       ``|cos θ| < 0.95`` → the primitive lattice vectors a, b.
    """
    if arr.ndim != 2:
        raise ValueError("extract_lattice expects a 2-D array")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be > 0")
    params = params or LatticeParams()

    cv2 = _cv()
    u8 = _to_uint8(arr, clip_low=params.clip_low, clip_high=params.clip_high)

    sift = cv2.SIFT_create(
        contrastThreshold=params.contrast_threshold,
        sigma=params.sigma,
        nOctaveLayers=params.n_octave_layers,
    )
    kps, descs = sift.detectAndCompute(u8, None)
    if kps is None or len(kps) < 10 or descs is None:
        raise RuntimeError(
            f"SIFT found only {0 if kps is None else len(kps)} keypoints — "
            "image too flat or SIFT parameters too strict."
        )

    sizes = np.array([kp.size for kp in kps])
    xys = np.array([kp.pt for kp in kps], dtype=np.float64)  # (N, 2) in px

    median_size = float(np.median(sizes))
    keep = (sizes >= median_size / params.size_threshold) & \
           (sizes <= median_size * params.size_threshold)
    Ny, Nx = arr.shape
    edge = median_size * params.edge_threshold
    keep &= ((xys[:, 0] > edge) & (xys[:, 0] < Nx - edge) &
             (xys[:, 1] > edge) & (xys[:, 1] < Ny - edge))

    kps_f = xys[keep]
    descs_f = descs[keep]
    if kps_f.shape[0] < max(6, params.cluster_kNN_low + 1):
        raise RuntimeError(
            f"After filtering only {kps_f.shape[0]} keypoints remain — "
            "cannot extract lattice."
        )

    kp_labels = _best_clustering(
        descs_f,
        span=range(params.cluster_kp_low, params.cluster_kp_high + 1),
    )

    counts = np.bincount(kp_labels)
    order = np.argsort(counts)[::-1]   # most populated first
    # cluster_choice is 1-indexed (1 = first/largest)
    c_idx = min(max(params.cluster_choice, 1), len(order)) - 1
    primary = int(order[c_idx])

    kps_primary = kps_f[kp_labels == primary]
    if kps_primary.shape[0] < params.cluster_kNN_low + 1:
        # Fall back to the largest cluster even if it wasn't the requested one
        primary = int(order[0])
        kps_primary = kps_f[kp_labels == primary]

    disp = _kNN_displacements(kps_primary, k=params.cluster_kNN_low)
    # Keep only "positive half" to collapse symmetric pairs (x > 0, or x == 0 and y > 0).
    keep_half = (disp[:, 0] > 0) | ((disp[:, 0] == 0) & (disp[:, 1] > 0))
    disp_pos = disp[keep_half]
    if disp_pos.shape[0] < params.cluster_kNN_low:
        disp_pos = disp  # fallback

    nnv_labels = _best_clustering(
        disp_pos,
        span=range(params.cluster_kNN_low, params.cluster_kNN_high + 1),
    )

    # Average displacement per cluster; filter small clusters.
    cluster_ids = sorted(set(nnv_labels))
    cluster_sizes = np.array([(nnv_labels == c).sum() for c in cluster_ids])
    max_size = int(cluster_sizes.max())
    keep_mask = cluster_sizes >= params.clustersize_threshold * max_size
    cluster_ids = [c for c, k in zip(cluster_ids, keep_mask) if k]
    candidates = np.array([
        disp_pos[nnv_labels == c].mean(axis=0) for c in cluster_ids
    ])
    norms = np.linalg.norm(candidates, axis=1)
    order = np.argsort(norms)
    candidates = candidates[order]

    # Pick the two smallest sufficiently non-colinear vectors.
    a_vec = b_vec = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            u, v = candidates[i], candidates[j]
            if abs(np.cos(np.radians(_angle_between_deg(u, v)))) < 0.95:
                a_vec, b_vec = u, v
                break
        if a_vec is not None:
            break
    if a_vec is None or b_vec is None:
        raise RuntimeError("Could not find two non-colinear lattice vectors.")

    a_m = (float(a_vec[0]) * pixel_size_m, float(a_vec[1]) * pixel_size_m)
    b_m = (float(b_vec[0]) * pixel_size_m, float(b_vec[1]) * pixel_size_m)
    a_len = float(np.linalg.norm(a_vec)) * pixel_size_m
    b_len = float(np.linalg.norm(b_vec)) * pixel_size_m
    gamma = _angle_between_deg(a_vec, b_vec)

    return LatticeResult(
        a_vector_m=a_m,
        b_vector_m=b_m,
        a_length_m=a_len,
        b_length_m=b_len,
        gamma_deg=gamma,
        a_vector_px=(float(a_vec[0]), float(a_vec[1])),
        b_vector_px=(float(b_vec[0]), float(b_vec[1])),
        n_keypoints=int(kps_f.shape[0]),
        n_keypoints_used=int(kps_primary.shape[0]),
        keypoints_xy_px=[(float(x), float(y)) for (x, y) in kps_f.tolist()],
        cluster_labels=[int(lbl) for lbl in kp_labels.tolist()],
        primary_cluster=primary,
        pixel_size_m=float(pixel_size_m),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PDF report
# ═════════════════════════════════════════════════════════════════════════════

def write_lattice_pdf(
    scan,
    lattice: LatticeResult,
    out_path,
    *,
    plane_idx: int = 0,
    colormap: str = "gray",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> None:
    """Render a 2×2 lattice summary PDF (matplotlib).

    Panels:

    * Top-left  — the scan with keypoints coloured by descriptor cluster.
    * Top-right — the scan with the primitive vectors a, b drawn from the
      image centre, and a reference unit cell overlaid.
    * Bottom-left  — histogram of kNN displacement magnitudes.
    * Bottom-right — text summary of |a|, |b|, γ, keypoint counts.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from pathlib import Path

    if plane_idx < 0 or plane_idx >= scan.n_planes:
        raise ValueError(f"plane_idx={plane_idx} out of range")
    arr = scan.planes[plane_idx]
    finite = arr[np.isfinite(arr)]
    vmin = float(np.percentile(finite, clip_low))
    vmax = float(np.percentile(finite, clip_high))

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=150)
    ax_kp, ax_ab, ax_hist, ax_txt = axes.ravel()

    # Top-left: keypoints coloured by cluster
    ax_kp.imshow(arr, origin="upper", cmap=colormap, vmin=vmin, vmax=vmax)
    if lattice.keypoints_xy_px:
        kp_arr = np.array(lattice.keypoints_xy_px)
        labels = np.array(lattice.cluster_labels)
        for c in set(labels.tolist()):
            m = labels == c
            ax_kp.scatter(kp_arr[m, 0], kp_arr[m, 1],
                          s=6, alpha=0.6, label=f"cluster {c}")
        ax_kp.legend(loc="upper right", fontsize=7)
    ax_kp.set_title("Keypoints (by descriptor cluster)")
    ax_kp.set_xticks([]); ax_kp.set_yticks([])

    # Top-right: primitive vectors + unit cell
    ax_ab.imshow(arr, origin="upper", cmap=colormap, vmin=vmin, vmax=vmax)
    cy, cx = arr.shape[0] / 2.0, arr.shape[1] / 2.0
    ax, ay = lattice.a_vector_px
    bx, by = lattice.b_vector_px
    ax_ab.arrow(cx, cy, ax, ay, color="red", width=1.0, head_width=5, length_includes_head=True)
    ax_ab.arrow(cx, cy, bx, by, color="yellow", width=1.0, head_width=5, length_includes_head=True)
    cell = Polygon(
        [(cx, cy), (cx + ax, cy + ay),
         (cx + ax + bx, cy + ay + by), (cx + bx, cy + by)],
        closed=True, fill=False, edgecolor="cyan", linewidth=1.5,
    )
    ax_ab.add_patch(cell)
    ax_ab.set_title("Primitive vectors a (red), b (yellow)")
    ax_ab.set_xticks([]); ax_ab.set_yticks([])

    # Bottom-left: displacement-magnitude histogram
    if lattice.keypoints_xy_px and lattice.n_keypoints_used > 2:
        primary_kps = [
            kp for kp, lbl in zip(lattice.keypoints_xy_px, lattice.cluster_labels)
            if lbl == lattice.primary_cluster
        ]
        if len(primary_kps) > 1:
            pts = np.array(primary_kps)
            diffs = pts[:, None, :] - pts[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1)).ravel()
            dists = dists[dists > 0]
            if dists.size > 0:
                ax_hist.hist(dists * lattice.pixel_size_m * 1e9, bins=40,
                             color="steelblue", edgecolor="black")
                ax_hist.axvline(lattice.a_length_m * 1e9, color="red",
                                linestyle="--", label="|a|")
                ax_hist.axvline(lattice.b_length_m * 1e9, color="orange",
                                linestyle="--", label="|b|")
                ax_hist.set_xlabel("Pairwise distance (nm)")
                ax_hist.set_ylabel("Counts")
                ax_hist.legend()
    ax_hist.set_title("Primary-cluster pairwise distances")

    # Bottom-right: text summary
    ax_txt.axis("off")
    lines = [
        f"Source:         {scan.source_path.name}",
        f"Plane:          {plane_idx}  ({scan.plane_names[plane_idx]})",
        "",
        f"|a| = {lattice.a_length_m * 1e9:7.3f} nm",
        f"|b| = {lattice.b_length_m * 1e9:7.3f} nm",
        f" γ  = {lattice.gamma_deg:7.2f} °",
        "",
        f"a = ({lattice.a_vector_m[0] * 1e9:+.3f},"
        f" {lattice.a_vector_m[1] * 1e9:+.3f}) nm",
        f"b = ({lattice.b_vector_m[0] * 1e9:+.3f},"
        f" {lattice.b_vector_m[1] * 1e9:+.3f}) nm",
        "",
        f"Total keypoints:   {lattice.n_keypoints}",
        f"Primary cluster:   {lattice.n_keypoints_used}",
    ]
    ax_txt.text(0.02, 0.98, "\n".join(lines), family="monospace",
                fontsize=10, va="top", ha="left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Unit-cell averaging
#   Given primitive vectors a, b, walk a regular (i, j) grid covering the
#   image, sample one unit cell at each lattice site, and average them. This
#   yields a clean, low-noise picture of the canonical motif — the AiSurf
#   ``average_cell`` workflow.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class UnitCellResult:
    """Output of :func:`average_unit_cell`."""
    avg_cell: np.ndarray            # (h, w) averaged unit-cell image
    n_cells: int                    # how many cells contributed
    cell_size_px: Tuple[int, int]   # (h, w) of avg_cell
    cell_size_m: Tuple[float, float]  # height_m, width_m (along b, then along a)
    a_vector_px: Tuple[float, float]
    b_vector_px: Tuple[float, float]


def average_unit_cell(
    arr: np.ndarray,
    lattice: "LatticeResult",
    *,
    oversample: float = 1.5,
    border_margin_px: int = 4,
) -> UnitCellResult:
    """Average all unit cells in ``arr`` defined by ``lattice``.

    For each lattice site ``r_ij = origin + i·a + j·b`` lying fully inside the
    image (with a ``border_margin_px`` safety strip), sample a parallelogram
    of side ``a × b`` resampled onto a regular grid and accumulate it. The
    final average is the canonical motif.

    Parameters
    ----------
    arr
        2-D scan plane the lattice was extracted from.
    lattice
        Result of :func:`extract_lattice`.
    oversample
        Pixel grid for the averaged cell, in units of ``max(|a|, |b|)``. The
        default 1.5 oversamples slightly so the output stays sharp even when
        the lattice vectors are not axis-aligned.
    border_margin_px
        Skip lattice sites whose unit cell would touch within this many pixels
        of the image edge — avoids reflection artefacts near the boundary.

    Returns
    -------
    UnitCellResult
    """
    if arr.ndim != 2:
        raise ValueError("average_unit_cell expects a 2-D array")
    if oversample <= 0:
        raise ValueError("oversample must be > 0")

    from scipy.ndimage import map_coordinates

    a_px = np.array(lattice.a_vector_px, dtype=np.float64)
    b_px = np.array(lattice.b_vector_px, dtype=np.float64)
    a_len = float(np.linalg.norm(a_px))
    b_len = float(np.linalg.norm(b_px))
    if a_len < 1e-9 or b_len < 1e-9:
        raise RuntimeError("Lattice vectors are degenerate.")

    # Output grid: dimension along a (width) and along b (height).
    side = max(a_len, b_len) * oversample
    Wc = max(8, int(round(side)))
    Hc = max(8, int(round(side)))

    Ny, Nx = arr.shape
    # Origin: the centroid of the primary cluster's keypoints (more robust
    # than the image centre for asymmetric crops). Falls back to image centre
    # if keypoint info is unavailable.
    if lattice.keypoints_xy_px:
        kp = np.array(lattice.keypoints_xy_px)
        labels = np.array(lattice.cluster_labels)
        primary = kp[labels == lattice.primary_cluster] if labels.size else kp
        if primary.size:
            origin = primary.mean(axis=0)
        else:
            origin = np.array([Nx / 2.0, Ny / 2.0])
    else:
        origin = np.array([Nx / 2.0, Ny / 2.0])

    # Choose an integer (i, j) range generous enough to tile the whole image.
    # Solve for the (i, j) that map to each image corner, then expand by 1.
    M = np.column_stack([a_px, b_px])
    if abs(np.linalg.det(M)) < 1e-9:
        raise RuntimeError("Lattice vectors are colinear.")
    Minv = np.linalg.inv(M)
    corners = np.array([
        [0, 0], [Nx, 0], [0, Ny], [Nx, Ny],
    ], dtype=np.float64) - origin
    ij_corners = corners @ Minv.T
    i_min = int(math.floor(ij_corners[:, 0].min())) - 1
    i_max = int(math.ceil(ij_corners[:, 0].max())) + 1
    j_min = int(math.floor(ij_corners[:, 1].min())) - 1
    j_max = int(math.ceil(ij_corners[:, 1].max())) + 1

    # Local sampling grid for one unit cell, in (u, v) ∈ [0, 1) × [0, 1).
    us = (np.arange(Wc) + 0.5) / Wc
    vs = (np.arange(Hc) + 0.5) / Hc
    UU, VV = np.meshgrid(us, vs)         # shape (Hc, Wc)
    UU = UU.ravel()
    VV = VV.ravel()

    accum = np.zeros(Hc * Wc, dtype=np.float64)
    n_cells = 0

    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            site = origin + i * a_px + j * b_px
            # Image-space coords of every sample in this cell.
            xs = site[0] + UU * a_px[0] + VV * b_px[0]
            ys = site[1] + UU * a_px[1] + VV * b_px[1]
            # Reject cells touching the border.
            if (xs.min() < border_margin_px or xs.max() > Nx - 1 - border_margin_px or
                    ys.min() < border_margin_px or ys.max() > Ny - 1 - border_margin_px):
                continue
            samples = map_coordinates(arr, np.vstack([ys, xs]),
                                      order=1, mode="reflect")
            # Replace NaNs with the cell mean to keep the running average finite.
            if np.isnan(samples).any():
                m = np.isfinite(samples)
                if not m.any():
                    continue
                samples = np.where(m, samples, samples[m].mean())
            accum += samples
            n_cells += 1

    if n_cells == 0:
        raise RuntimeError(
            "No interior lattice cells found — check the lattice vectors or "
            "lower border_margin_px."
        )

    avg = (accum / n_cells).reshape(Hc, Wc).astype(arr.dtype, copy=False)

    # Physical extent of the averaged cell.
    cell_w_m = float(np.hypot(a_px[0], a_px[1])) * lattice.pixel_size_m
    cell_h_m = float(np.hypot(b_px[0], b_px[1])) * lattice.pixel_size_m

    return UnitCellResult(
        avg_cell=avg,
        n_cells=n_cells,
        cell_size_px=(Hc, Wc),
        cell_size_m=(cell_h_m, cell_w_m),
        a_vector_px=lattice.a_vector_px,
        b_vector_px=lattice.b_vector_px,
    )
