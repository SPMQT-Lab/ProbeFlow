"""QuickSeg terrace segmentation backend for Dataset Builder.

This module is intentionally Qt-free.  It owns the pure preprocessing,
watershed, overlay, and persistence helpers used by the Dataset Builder task
router.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import convolve, gaussian_filter, median_filter
from skimage import exposure
from skimage.draw import line as draw_line
from skimage.filters import apply_hysteresis_threshold, scharr, sobel
from skimage.measure import label as measure_label, regionprops
from skimage.morphology import closing, dilation, disk, remove_small_objects
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.segmentation import watershed

from probeflow.dataset_builder.models import DatasetTaskConfig, ReviewRecord
from probeflow.dataset_builder.sidecar_state import load_review_record, save_review_record, utc_now
from probeflow.processing.alignment import facet_level
from probeflow.processing.background import subtract_background


EDGE_SCALE_PRESETS: dict[str, tuple[float, float, float]] = {
    "fine": (0.4, 0.8, 1.4),
    "balanced": (0.6, 1.2, 2.4),
    "broad": (1.0, 2.0, 3.5),
}

ADVANCED_PARAMETERS: dict[str, Any] = {
    "version": 1,
    "background_mode": "subtract_background_order1",
    "plane_percentile_low": 1.0,
    "plane_percentile_high": 99.0,
    "plane_facet_threshold_deg": 3.0,
    "tv_iterations": 300,
    "tv_eps": 2.0e-5,
    "median_size": 0,
    "bilateral_sigma_color": 0.035,
    "bilateral_sigma_spatial": 2.0,
    "gradient_operator": "scharr",
    "gradient_combine": "max",
    "gradient_clip_percentiles": (2.0, 99.6),
    "clahe_clip_limit": 0.015,
    "elevation_gamma": 0.65,
    "hysteresis_high_percentile": 96.8,
    "line_angles_deg": (0, 30, 60, 90, 120, 150),
    "line_lengths": (7, 11, 15),
    "close_radius": 1,
    "dilate_radius": 1,
    "barrier_blur_sigma": 0.8,
    "horizontal_artifact_min_width_fraction": 0.65,
    "horizontal_artifact_max_height_fraction": 0.04,
    "horizontal_artifact_min_aspect_ratio": 8.0,
    "horizontal_artifact_max_angle_deg": 8.0,
    "horizontal_artifact_score_percentile": 94.0,
    "horizontal_artifact_run_close_px": 9,
    "horizontal_artifact_row_expand_px": 1,
    "horizontal_artifact_blur_sigma": 1.0,
    "compactness_watershed": 0.0,
    "watershed_connectivity": 1,
    "watershed_line": False,
}


def _finite(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)[np.isfinite(arr)]


def _clip_to_percentiles(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        return out
    values = out[finite]
    p_lo, p_hi = np.percentile(values, [float(lo), float(hi)])
    if not np.isfinite(p_lo):
        p_lo = float(np.nanmin(values))
    if not np.isfinite(p_hi):
        p_hi = float(np.nanmax(values))
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        return out
    out[finite] = np.clip(out[finite], p_lo, p_hi)
    return out


def _plane_fit_background(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if finite.sum() < 3:
        return a.copy()
    clipped = _clip_to_percentiles(a, lo, hi)
    fit_mask = np.isfinite(clipped)
    if fit_mask.sum() < 3:
        fit_mask = finite
    ys, xs = np.nonzero(fit_mask)
    zs = a[fit_mask]
    if zs.size < 3:
        return a.copy()
    A = np.column_stack([
        xs.astype(np.float64),
        ys.astype(np.float64),
        np.ones(xs.size, dtype=np.float64),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
    yy, xx = np.indices(a.shape)
    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    return a - plane


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class QuickSegParams:
    """User-facing QuickSeg tuning parameters.

    The preprocessing stack has many fixed implementation parameters, but the
    labelling UI deliberately exposes only these eight knobs plus overlay
    opacity.  Older sidecars/configs with the previous verbose parameter names
    are migrated in :meth:`from_dict`.
    """

    denoise_strength: float = 0.04
    smooth_along_scan: float = 1.2
    smooth_across_scan: float = 0.7
    edge_scale: str = "balanced"
    edge_sensitivity: float = 84.0
    min_edge_size: int = 40
    edge_connect_strength: float = 0.45
    barrier_strength: float = 0.18
    horizontal_defect_suppression: float = 0.0
    overlay_opacity: float = 0.55

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QuickSegParams":
        data = dict(data or {})
        return cls(
            denoise_strength=float(data.get("denoise_strength", data.get("tv_weight", cls.denoise_strength))),
            smooth_along_scan=float(data.get("smooth_along_scan", data.get("gaussian_sigma", cls.smooth_along_scan))),
            smooth_across_scan=float(data.get("smooth_across_scan", cls.smooth_across_scan)),
            edge_scale=str(data.get("edge_scale") or cls.edge_scale)
            if str(data.get("edge_scale") or cls.edge_scale) in EDGE_SCALE_PRESETS
            else cls.edge_scale,
            edge_sensitivity=float(data.get("edge_sensitivity", cls.edge_sensitivity)),
            min_edge_size=int(data.get("min_edge_size", cls.min_edge_size)),
            edge_connect_strength=float(data.get("edge_connect_strength", cls.edge_connect_strength)),
            barrier_strength=float(data.get("barrier_strength", cls.barrier_strength)),
            horizontal_defect_suppression=float(
                data.get("horizontal_defect_suppression", cls.horizontal_defect_suppression)
            ),
            overlay_opacity=float(data.get("overlay_opacity", cls.overlay_opacity)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["advanced_parameters_version"] = int(ADVANCED_PARAMETERS["version"])
        return data

    def normalized_background_mode(self) -> str:
        key = _normalize_key(ADVANCED_PARAMETERS["background_mode"])
        aliases = {
            "none": "none",
            "off": "none",
            "subtract_background_order1": "subtract_background_order1",
            "subtract_background": "subtract_background_order1",
            "order1": "subtract_background_order1",
            "old_quickseg_plane_fit": "plane_fit",
            "plane_fit": "plane_fit",
            "legacy_plane_fit": "plane_fit",
            "facet_level": "facet_level",
        }
        return aliases.get(key, key)


@dataclass(frozen=True)
class QuickSegSeed:
    x: int
    y: int
    terrace_label_id: int
    order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "terrace_label_id": int(self.terrace_label_id),
            "order": int(self.order),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuickSegSeed":
        return cls(
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            terrace_label_id=int(data.get("terrace_label_id", 0)),
            order=int(data.get("order", 0)),
        )


@dataclass
class QuickSegPrepared:
    raw: np.ndarray
    corrected: np.ndarray
    equalized: np.ndarray
    denoised: np.ndarray
    gaussian: np.ndarray
    gradient: np.ndarray
    flat_display: np.ndarray | None = None
    anisotropic_blur: np.ndarray | None = None
    gradient_contrast: np.ndarray | None = None
    connected_edge_mask: np.ndarray | None = None
    horizontal_artifact_mask: np.ndarray | None = None
    watershed_elevation_unsuppressed: np.ndarray | None = None
    watershed_elevation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.flat_display is None:
            self.flat_display = self.equalized
        if self.anisotropic_blur is None:
            self.anisotropic_blur = self.gaussian
        if self.gradient_contrast is None:
            self.gradient_contrast = self.gradient
        if self.connected_edge_mask is None:
            self.connected_edge_mask = np.zeros_like(self.raw, dtype=bool)
        if self.horizontal_artifact_mask is None:
            self.horizontal_artifact_mask = np.zeros_like(self.raw, dtype=bool)
        if self.watershed_elevation_unsuppressed is None:
            self.watershed_elevation_unsuppressed = self.gradient
        if self.watershed_elevation is None:
            self.watershed_elevation = self.gradient


@dataclass
class QuickSegState:
    seeds: list[QuickSegSeed] = field(default_factory=list)
    current_label: int = 1
    next_order: int = 1
    params: QuickSegParams = field(default_factory=QuickSegParams)
    result: np.ndarray | None = None
    result_path: str | None = None

    def to_task_data(self) -> dict[str, Any]:
        return {
            "quickseg": {
                "version": 1,
                "current_label": int(self.current_label),
                "next_order": int(self.next_order),
                "params": self.params.to_dict(),
                "seeds": [seed.to_dict() for seed in self.seeds],
                "result_path": self.result_path,
                "result_shape": list(self.result.shape) if self.result is not None else None,
                "result_dtype": str(self.result.dtype) if self.result is not None else None,
            }
        }

    @classmethod
    def from_task_data(cls, data: dict[str, Any] | None) -> "QuickSegState":
        payload = dict((data or {}).get("quickseg") or {})
        params = QuickSegParams.from_dict(payload.get("params"))
        seeds = [QuickSegSeed.from_dict(item) for item in payload.get("seeds", [])]
        state = cls(
            seeds=seeds,
            current_label=max(
                int(payload.get("current_label", 1)),
                max((seed.terrace_label_id for seed in seeds), default=0),
            ),
            next_order=max(
                int(payload.get("next_order", 1)),
                max((seed.order for seed in seeds), default=0) + 1,
            ),
            params=params,
            result=None,
            result_path=payload.get("result_path"),
        )
        return state

    def refresh_counters(self) -> None:
        self.current_label = max(self.current_label, max((s.terrace_label_id for s in self.seeds), default=0) or 1)
        self.next_order = max(self.next_order, max((s.order for s in self.seeds), default=0) + 1)


def connectivity(number: int) -> np.ndarray:
    if number == 1:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    if number == 2:
        return np.ones((3, 3), dtype=bool)
    if number == 3:
        return np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=bool,
        )
    if number == 4:
        return np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=bool,
        )
    if number == 5:
        return np.ones((3, 3), dtype=bool)
    raise ValueError('Value error for function "connectivity", must be an integer in [1,5]')


def equalise(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return arr.copy()
    out = arr.copy()
    out[finite] -= float(np.nanmin(out[finite]))
    return out


def finite_valid_mask(img: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(img, dtype=np.float64))


def robust_rescale(
    img: np.ndarray,
    percentiles: tuple[float, float] = (1.0, 99.0),
    mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    valid = np.isfinite(arr) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(arr))
    out = np.zeros(arr.shape, dtype=np.float64)
    if not valid.any():
        return out
    lo, hi = np.percentile(arr[valid], [float(percentiles[0]), float(percentiles[1])])
    if not np.isfinite(lo):
        lo = float(np.nanmin(arr[valid]))
    if not np.isfinite(hi):
        hi = float(np.nanmax(arr[valid]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[valid] = 0.0
        return out
    out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def quickseg_gradient_sigmas(params: QuickSegParams) -> tuple[float, float, float]:
    return EDGE_SCALE_PRESETS.get(str(params.edge_scale), EDGE_SCALE_PRESETS["balanced"])


def _line_footprint(length: int, angle_deg: float) -> np.ndarray:
    length = max(1, int(length))
    size = length if length % 2 else length + 1
    center = size // 2
    theta = math.radians(float(angle_deg))
    dx = math.cos(theta) * (length - 1) / 2.0
    dy = math.sin(theta) * (length - 1) / 2.0
    r0 = int(round(center - dy))
    c0 = int(round(center - dx))
    r1 = int(round(center + dy))
    c1 = int(round(center + dx))
    footprint = np.zeros((size, size), dtype=bool)
    rr, cc = draw_line(r0, c0, r1, c1)
    rr = np.clip(rr, 0, size - 1)
    cc = np.clip(cc, 0, size - 1)
    footprint[rr, cc] = True
    return footprint


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    fp = _line_footprint(length, angle_deg).astype(np.float64)
    total = float(fp.sum())
    return fp / total if total > 0 else fp


def oriented_line_smooth(
    edge: np.ndarray,
    *,
    angles: tuple[int, ...] | tuple[float, ...],
    lengths: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(edge, dtype=np.float64)
    responses: list[np.ndarray] = []
    for length in lengths:
        for angle in angles:
            responses.append(convolve(arr, _line_kernel(int(length), float(angle)), mode="reflect"))
    if not responses:
        return arr.copy()
    return np.max(np.stack(responses, axis=0), axis=0)


def oriented_binary_connect(
    mask: np.ndarray,
    *,
    angles: tuple[int, ...] | tuple[float, ...],
    lengths: tuple[int, ...],
) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for length in lengths:
        for angle in angles:
            out = closing(out, footprint=_line_footprint(int(length), float(angle)))
    return out


def remove_small_edge_objects(mask: np.ndarray, threshold: int) -> np.ndarray:
    min_size = max(1, int(threshold))
    return remove_small_objects(np.asarray(mask, dtype=bool), min_size=min_size)


def _component_horizontal_angle_deg(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 90.0
    xy = np.column_stack([coords[:, 1].astype(np.float64), coords[:, 0].astype(np.float64)])
    xy -= np.mean(xy, axis=0, keepdims=True)
    cov = np.cov(xy, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return 90.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    angle = abs(math.degrees(math.atan2(float(vec[1]), float(vec[0]))))
    return min(angle, abs(180.0 - angle))


def _horizontal_run_artifact_mask(mask: np.ndarray) -> np.ndarray:
    candidate = np.asarray(mask, dtype=bool)
    if candidate.ndim != 2 or not candidate.any():
        return np.zeros(candidate.shape, dtype=bool)
    height, width = candidate.shape
    close_px = max(1, int(ADVANCED_PARAMETERS["horizontal_artifact_run_close_px"]))
    if close_px > 1:
        candidate = closing(candidate, footprint=np.ones((1, close_px), dtype=bool))
    min_width = int(math.ceil(float(ADVANCED_PARAMETERS["horizontal_artifact_min_width_fraction"]) * max(width, 1)))
    expand = max(0, int(ADVANCED_PARAMETERS["horizontal_artifact_row_expand_px"]))
    artifact = np.zeros(candidate.shape, dtype=bool)
    for row_idx in range(height):
        row = candidate[row_idx]
        if int(row.sum()) < min_width:
            continue
        padded = np.concatenate(([False], row, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts = changes[0::2]
        ends = changes[1::2]
        for start, end in zip(starts, ends):
            if int(end - start) < min_width:
                continue
            row0 = max(0, row_idx - expand)
            row1 = min(height, row_idx + expand + 1)
            artifact[row0:row1, int(start):int(end)] = True
    return artifact


def detect_horizontal_artifact_mask(edge_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(edge_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    min_width = float(ADVANCED_PARAMETERS["horizontal_artifact_min_width_fraction"]) * max(width, 1)
    max_height = float(ADVANCED_PARAMETERS["horizontal_artifact_max_height_fraction"]) * max(height, 1)
    min_aspect = float(ADVANCED_PARAMETERS["horizontal_artifact_min_aspect_ratio"])
    max_angle = float(ADVANCED_PARAMETERS["horizontal_artifact_max_angle_deg"])
    labels = measure_label(mask, connectivity=2)
    artifact = _horizontal_run_artifact_mask(mask)
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        comp_width = max_col - min_col
        comp_height = max_row - min_row
        if comp_width < min_width:
            continue
        if comp_height > max(1.0, max_height):
            continue
        if comp_width / max(comp_height, 1) < min_aspect:
            continue
        if _component_horizontal_angle_deg(region.coords) > max_angle:
            continue
        coords = region.coords
        artifact[coords[:, 0], coords[:, 1]] = True
    return artifact


def quickseg_stage(prepared: QuickSegPrepared, stage: str) -> np.ndarray:
    key = str(stage or "watershed_elevation")
    mapping = {
        "flat_display": prepared.flat_display,
        "denoised": prepared.denoised,
        "anisotropic_blur": prepared.anisotropic_blur,
        "gradient_contrast": prepared.gradient_contrast,
        "connected_edge_mask": prepared.connected_edge_mask,
        "horizontal_artifact_mask": prepared.horizontal_artifact_mask,
        "watershed_elevation_unsuppressed": prepared.watershed_elevation_unsuppressed,
        "watershed_elevation": prepared.watershed_elevation,
    }
    arr = mapping.get(key, prepared.watershed_elevation)
    if arr is None:
        arr = prepared.gradient
    return np.asarray(arr)


def reorder_labels_area(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    new_labels = np.zeros_like(labels, dtype=np.int32)
    regions = sorted((r for r in regionprops(labels) if r.label != 0), key=lambda r: r.area)
    for new_label, region in enumerate(regions, start=1):
        coords = region.coords
        new_labels[coords[:, 0], coords[:, 1]] = new_label
    return new_labels


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    max_label = int(labels.max()) if labels.size else 0
    colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    for i in range(1, max_label + 1):
        hue = (i * 0.61803398875) % 1.0
        colors[i] = hsv_to_rgb(hue, 0.85, 0.95)
    return colors[labels]


def _draw_seed_overlay(
    rgba: np.ndarray,
    seeds: list[QuickSegSeed],
    *,
    opacity: float,
    radius: int = 3,
) -> None:
    if not seeds:
        return
    h, w = rgba.shape[:2]
    for seed in seeds:
        y = int(seed.y)
        x = int(seed.x)
        if not (0 <= y < h and 0 <= x < w):
            continue
        color = hsv_to_rgb((seed.terrace_label_id * 0.23) % 1.0, 0.8, 1.0)
        yy, xx = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        disk = xx * xx + yy * yy <= radius * radius
        ys0 = max(0, y - radius)
        xs0 = max(0, x - radius)
        ys1 = min(h, y + radius + 1)
        xs1 = min(w, x + radius + 1)
        sub = disk[
            (ys0 - (y - radius)):(disk.shape[0] - ((y + radius + 1) - ys1)),
            (xs0 - (x - radius)):(disk.shape[1] - ((x + radius + 1) - xs1)),
        ]
        if not sub.any():
            continue
        region = rgba[ys0:ys1, xs0:xs1]
        region[sub, 0] = color[0]
        region[sub, 1] = color[1]
        region[sub, 2] = color[2]
        region[sub, 3] = np.maximum(region[sub, 3], int(255 * min(1.0, max(0.0, opacity))))


def quickseg_overlay_rgba(
    labels: np.ndarray | None,
    seeds: list[QuickSegSeed],
    *,
    show_seeds: bool = True,
    show_boundaries: bool = True,
    show_filled_regions: bool = True,
    opacity: float = 0.55,
) -> np.ndarray | None:
    if labels is None:
        return None
    lab = np.asarray(labels, dtype=np.int32)
    if lab.ndim != 2:
        raise ValueError("QuickSeg labels must be a 2-D array")
    h, w = lab.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if show_filled_regions and lab.size:
        rgb = colorize_labels(lab)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = np.where(lab > 0, int(255 * min(1.0, max(0.0, opacity))), 0).astype(np.uint8)
    if show_boundaries and lab.size:
        boundary = np.zeros_like(lab, dtype=bool)
        boundary[:-1, :] |= lab[:-1, :] != lab[1:, :]
        boundary[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        rgba[boundary, :3] = np.array([255, 255, 255], dtype=np.uint8)
        rgba[boundary, 3] = 255
    if show_seeds:
        _draw_seed_overlay(rgba, seeds, opacity=1.0)
    return rgba


def apply_quickseg_background(
    arr: np.ndarray,
    params: QuickSegParams,
    *,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
) -> np.ndarray:
    mode = params.normalized_background_mode()
    if mode == "none":
        return np.asarray(arr, dtype=np.float64).copy()
    if mode == "subtract_background_order1":
        return subtract_background(np.asarray(arr, dtype=np.float64), order=1)
    if mode == "plane_fit":
        return _plane_fit_background(
            arr,
            float(ADVANCED_PARAMETERS["plane_percentile_low"]),
            float(ADVANCED_PARAMETERS["plane_percentile_high"]),
        )
    if mode == "facet_level":
        return facet_level(
            np.asarray(arr, dtype=np.float64),
            threshold_deg=float(ADVANCED_PARAMETERS["plane_facet_threshold_deg"]),
            pixel_size_x_m=float(pixel_size_x_m),
            pixel_size_y_m=float(pixel_size_y_m),
        )
    raise ValueError(f"Unknown QuickSeg background mode {ADVANCED_PARAMETERS['background_mode']!r}")


def prepare_quickseg_inputs(
    arr: np.ndarray,
    params: QuickSegParams,
    *,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
) -> QuickSegPrepared:
    raw = np.asarray(arr, dtype=np.float64)
    valid = finite_valid_mask(raw)
    corrected = apply_quickseg_background(
        raw,
        params,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
    )
    flat_display = robust_rescale(corrected, (1.0, 99.0), valid)
    equalized = flat_display
    denoised = denoise_tv_chambolle(
        equalized,
        weight=float(params.denoise_strength),
        eps=float(ADVANCED_PARAMETERS["tv_eps"]),
        max_num_iter=int(ADVANCED_PARAMETERS["tv_iterations"]),
    ).astype(np.float64, copy=False)

    median_size = int(ADVANCED_PARAMETERS["median_size"])
    if median_size > 1:
        denoised = median_filter(denoised, size=median_size, mode="reflect").astype(np.float64, copy=False)

    sigma_color = float(ADVANCED_PARAMETERS["bilateral_sigma_color"])
    if sigma_color > 0:
        denoised = denoise_bilateral(
            denoised,
            sigma_color=sigma_color,
            sigma_spatial=float(ADVANCED_PARAMETERS["bilateral_sigma_spatial"]),
            channel_axis=None,
        ).astype(np.float64, copy=False)

    anisotropic_blur = gaussian_filter(
        denoised,
        sigma=(float(params.smooth_across_scan), float(params.smooth_along_scan)),
        mode="reflect",
    ).astype(np.float64, copy=False)

    grad_maps: list[np.ndarray] = []
    operator = str(ADVANCED_PARAMETERS["gradient_operator"]).lower()
    for sigma in quickseg_gradient_sigmas(params):
        work = anisotropic_blur
        if float(sigma) > 0:
            work = gaussian_filter(work, sigma=float(sigma), mode="reflect")
        grad = scharr(work) if operator == "scharr" else sobel(work)
        grad_maps.append(robust_rescale(grad, (1.0, 99.8), valid))
    if grad_maps:
        stack = np.stack(grad_maps, axis=0)
        if str(ADVANCED_PARAMETERS["gradient_combine"]).lower() == "mean":
            gradient_raw = np.mean(stack, axis=0)
        else:
            gradient_raw = np.max(stack, axis=0)
    else:
        gradient_raw = sobel(anisotropic_blur)

    gradient_contrast = robust_rescale(
        gradient_raw,
        tuple(ADVANCED_PARAMETERS["gradient_clip_percentiles"]),
        valid,
    )
    clahe_clip = float(ADVANCED_PARAMETERS["clahe_clip_limit"])
    if clahe_clip > 0:
        gradient_contrast = exposure.equalize_adapthist(
            np.clip(gradient_contrast, 0.0, 1.0),
            clip_limit=clahe_clip,
        ).astype(np.float64, copy=False)
        gradient_contrast = robust_rescale(gradient_contrast, (0.5, 99.5), valid)

    edge_values = gradient_contrast[valid & np.isfinite(gradient_contrast)]
    if edge_values.size:
        low = float(np.percentile(edge_values, float(params.edge_sensitivity)))
        high = float(np.percentile(edge_values, float(ADVANCED_PARAMETERS["hysteresis_high_percentile"])))
        if high <= low:
            high = low
        edge_mask = apply_hysteresis_threshold(gradient_contrast, low, high)
    else:
        edge_mask = np.zeros(raw.shape, dtype=bool)
    edge_mask &= valid
    edge_mask = remove_small_edge_objects(edge_mask, int(params.min_edge_size))

    close_radius = int(ADVANCED_PARAMETERS["close_radius"])
    if close_radius > 0:
        edge_mask = closing(edge_mask, footprint=disk(close_radius))
    edge_mask = oriented_binary_connect(
        edge_mask,
        angles=tuple(ADVANCED_PARAMETERS["line_angles_deg"]),
        lengths=tuple(ADVANCED_PARAMETERS["line_lengths"]),
    )
    dilate_radius = int(ADVANCED_PARAMETERS["dilate_radius"])
    if dilate_radius > 0:
        edge_mask = dilation(edge_mask, footprint=disk(dilate_radius))
    edge_mask &= valid

    line_smoothed = oriented_line_smooth(
        gradient_contrast,
        angles=tuple(ADVANCED_PARAMETERS["line_angles_deg"]),
        lengths=tuple(ADVANCED_PARAMETERS["line_lengths"]),
    )
    barrier_blur_sigma = float(ADVANCED_PARAMETERS["barrier_blur_sigma"])
    gate = gaussian_filter(edge_mask.astype(np.float64), sigma=max(0.1, barrier_blur_sigma), mode="reflect")
    connect_strength = float(np.clip(params.edge_connect_strength, 0.0, 1.0))
    connected_edge = (
        (1.0 - connect_strength) * gradient_contrast
        + connect_strength * (gate * line_smoothed + (1.0 - gate) * gradient_contrast)
    )
    connected_edge = robust_rescale(connected_edge, (0.5, 99.7), valid)
    artifact_candidate = edge_mask.copy()
    connected_values = connected_edge[valid & np.isfinite(connected_edge)]
    if connected_values.size:
        score_threshold = float(
            np.percentile(
                connected_values,
                float(ADVANCED_PARAMETERS["horizontal_artifact_score_percentile"]),
            )
        )
        artifact_candidate |= connected_edge >= score_threshold
    artifact_candidate &= valid
    horizontal_artifact_mask = detect_horizontal_artifact_mask(artifact_candidate)
    horizontal_artifact_mask &= valid

    suppression = float(np.clip(params.horizontal_defect_suppression, 0.0, 1.0))
    artifact_weight = np.zeros(raw.shape, dtype=np.float64)
    if suppression > 0 and horizontal_artifact_mask.any():
        blur_sigma = float(ADVANCED_PARAMETERS["horizontal_artifact_blur_sigma"])
        artifact_weight = gaussian_filter(
            horizontal_artifact_mask.astype(np.float64),
            sigma=max(0.0, blur_sigma),
            mode="reflect",
        )
        artifact_weight = np.maximum(artifact_weight, horizontal_artifact_mask.astype(np.float64))
        artifact_weight = np.clip(artifact_weight, 0.0, 1.0)

    elevation_base = np.clip(connected_edge, 0.0, 1.0) ** float(ADVANCED_PARAMETERS["elevation_gamma"])
    barrier = gaussian_filter(edge_mask.astype(np.float64), sigma=barrier_blur_sigma, mode="reflect")
    unsuppressed_elevation = robust_rescale(
        elevation_base + float(params.barrier_strength) * barrier,
        (0.5, 99.7),
        valid,
    )
    if suppression > 0 and horizontal_artifact_mask.any():
        barrier_source = edge_mask.astype(np.float64) * (1.0 - suppression * artifact_weight)
        suppressed_barrier = gaussian_filter(barrier_source, sigma=barrier_blur_sigma, mode="reflect")
        elevation = robust_rescale(
            elevation_base + float(params.barrier_strength) * suppressed_barrier,
            (0.5, 99.7),
            valid,
        )
        elevation = elevation * (1.0 - suppression * artifact_weight)
        elevation = np.clip(elevation, 0.0, 1.0)
    else:
        elevation = unsuppressed_elevation.copy()
    elevation[~valid] = 0.0
    unsuppressed_elevation[~valid] = 0.0

    return QuickSegPrepared(
        raw=raw,
        corrected=corrected,
        equalized=equalized,
        denoised=denoised,
        gaussian=anisotropic_blur,
        gradient=elevation,
        flat_display=flat_display,
        anisotropic_blur=anisotropic_blur,
        gradient_contrast=gradient_contrast,
        connected_edge_mask=edge_mask.astype(np.float64),
        horizontal_artifact_mask=horizontal_artifact_mask.astype(np.float64),
        watershed_elevation_unsuppressed=unsuppressed_elevation,
        watershed_elevation=elevation,
    )


def watershed_labels(
    prepared: QuickSegPrepared,
    seeds: list[QuickSegSeed],
    params: QuickSegParams,
) -> np.ndarray:
    elevation = quickseg_stage(prepared, "watershed_elevation").astype(np.float64, copy=False)
    if elevation.shape != prepared.raw.shape:
        raise ValueError("Prepared watershed elevation shape does not match raw image")
    markers = np.zeros(prepared.raw.shape, dtype=np.int32)
    for seed in seeds:
        x = int(seed.x)
        y = int(seed.y)
        if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
            markers[y, x] = int(seed.terrace_label_id)
    if int(markers.max()) < 1:
        return np.zeros(prepared.raw.shape, dtype=np.int32)
    labels = watershed(
        elevation,
        markers=markers,
        connectivity=connectivity(int(ADVANCED_PARAMETERS["watershed_connectivity"])),
        compactness=float(ADVANCED_PARAMETERS["compactness_watershed"]),
        watershed_line=bool(ADVANCED_PARAMETERS["watershed_line"]),
    ).astype(np.int32, copy=False)
    return reorder_labels_area(labels)


def quickseg_result_path(scan_path: str | Path) -> Path:
    path = Path(scan_path)
    return path.parent / f"{path.stem}.quickseg_terraces.npy"


def quickseg_record_payload(state: QuickSegState) -> dict[str, Any]:
    return state.to_task_data()["quickseg"]


def save_quickseg_state(
    scan_path: str | Path,
    state: QuickSegState,
    *,
    config: DatasetTaskConfig,
    status: str,
    notes: str = "",
) -> tuple[Path | None, Path]:
    scan_path = Path(scan_path)
    result_path = quickseg_result_path(scan_path)
    if state.result is not None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("wb") as fh:
            np.save(fh, np.asarray(state.result, dtype=np.int32), allow_pickle=False)
        state.result_path = str(result_path)
    else:
        state.result_path = None
    record = ReviewRecord(
        source_path=str(scan_path),
        plane_index=config.plane_index,
        task=config.task,
        label_type=config.label_type,
        label_name=config.label_name,
        status=status,
        annotator=config.annotator,
        notes=notes,
        proposal_method=config.proposal_method,
        proposal_parameters=dict(config.proposal_params),
        task_data=state.to_task_data(),
        updated_at=utc_now(),
    )
    state_path = save_review_record(scan_path, record)
    return (result_path if state.result is not None else None), state_path


def load_quickseg_state(
    scan_path: str | Path,
    *,
    config: DatasetTaskConfig,
) -> tuple[QuickSegState, ReviewRecord | None, Path]:
    record = load_review_record(
        scan_path,
        task=config.task,
        plane_index=config.plane_index,
        label_name=config.label_name,
    )
    if record is None:
        state = QuickSegState()
        return state, None, quickseg_result_path(scan_path)
    state = QuickSegState.from_task_data(record.task_data)
    payload = dict((record.task_data or {}).get("quickseg") or {})
    state.params = QuickSegParams.from_dict(
        payload.get("params") or record.proposal_parameters or state.params.to_dict()
    )
    result_path = Path(state.result_path) if state.result_path else quickseg_result_path(scan_path)
    if result_path.exists():
        try:
            state.result = np.load(result_path, allow_pickle=False)
            state.result_path = str(result_path)
        except Exception:
            state.result = None
    return state, record, result_path


def quickseg_review_record_exists(
    scan_path: str | Path,
    *,
    config: DatasetTaskConfig,
) -> bool:
    return load_review_record(
        scan_path,
        task=config.task,
        plane_index=config.plane_index,
        label_name=config.label_name,
    ) is not None


def quickseg_seed_from_point(
    x: int,
    y: int,
    terrace_label_id: int,
    order: int,
) -> QuickSegSeed:
    return QuickSegSeed(x=int(x), y=int(y), terrace_label_id=int(terrace_label_id), order=int(order))


def quickseg_state_to_review_payload(state: QuickSegState) -> dict[str, Any]:
    return state.to_task_data()
