"""QuickSeg terrace segmentation backend for Dataset Builder.

This module is intentionally Qt-free.  It owns the pure preprocessing,
watershed, overlay, and persistence helpers used by the Dataset Builder task
router.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.measure import regionprops
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import watershed

from probeflow.dataset_builder.models import DatasetTaskConfig, ReviewRecord
from probeflow.dataset_builder.sidecar_state import load_review_record, save_review_record, utc_now
from probeflow.processing.alignment import facet_level
from probeflow.processing.background import subtract_background


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
    """QuickSeg preprocessing and watershed parameters."""

    background_mode: str = "subtract_background_order1"
    plane_percentile_low: float = 1.0
    plane_percentile_high: float = 99.0
    plane_facet_threshold_deg: float = 3.0
    tv_weight: float = 0.25
    tv_iterations: int = 500
    tv_eps: float = 2.0e-5
    gaussian_sigma: float = 4.0
    gaussian_order: int = 0
    gaussian_mode: str = "reflect"
    gaussian_axes: tuple[int, ...] | None = (1,)
    watershed_connectivity: int = 1
    watershed_compactness: float = 0.0
    watershed_line: bool = False
    overlay_opacity: float = 0.55

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QuickSegParams":
        data = dict(data or {})
        axes = data.get("gaussian_axes", (1,))
        if axes is None:
            parsed_axes = None
        elif isinstance(axes, (list, tuple)):
            parsed_axes = tuple(int(v) for v in axes)
        else:
            parsed_axes = (int(axes),)
        return cls(
            background_mode=str(data.get("background_mode") or cls.background_mode),
            plane_percentile_low=float(data.get("plane_percentile_low", cls.plane_percentile_low)),
            plane_percentile_high=float(data.get("plane_percentile_high", cls.plane_percentile_high)),
            plane_facet_threshold_deg=float(
                data.get("plane_facet_threshold_deg", cls.plane_facet_threshold_deg)
            ),
            tv_weight=float(data.get("tv_weight", cls.tv_weight)),
            tv_iterations=int(data.get("tv_iterations", cls.tv_iterations)),
            tv_eps=float(data.get("tv_eps", cls.tv_eps)),
            gaussian_sigma=float(data.get("gaussian_sigma", cls.gaussian_sigma)),
            gaussian_order=int(data.get("gaussian_order", cls.gaussian_order)),
            gaussian_mode=str(data.get("gaussian_mode") or cls.gaussian_mode),
            gaussian_axes=parsed_axes,
            watershed_connectivity=int(data.get("watershed_connectivity", cls.watershed_connectivity)),
            watershed_compactness=float(data.get("watershed_compactness", cls.watershed_compactness)),
            watershed_line=bool(data.get("watershed_line", cls.watershed_line)),
            overlay_opacity=float(data.get("overlay_opacity", cls.overlay_opacity)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.gaussian_axes is None:
            data["gaussian_axes"] = None
        else:
            data["gaussian_axes"] = list(self.gaussian_axes)
        return data

    def normalized_background_mode(self) -> str:
        key = _normalize_key(self.background_mode)
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
            params.plane_percentile_low,
            params.plane_percentile_high,
        )
    if mode == "facet_level":
        return facet_level(
            np.asarray(arr, dtype=np.float64),
            threshold_deg=float(params.plane_facet_threshold_deg),
            pixel_size_x_m=float(pixel_size_x_m),
            pixel_size_y_m=float(pixel_size_y_m),
        )
    raise ValueError(f"Unknown QuickSeg background mode {params.background_mode!r}")


def prepare_quickseg_inputs(
    arr: np.ndarray,
    params: QuickSegParams,
    *,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
) -> QuickSegPrepared:
    raw = np.asarray(arr, dtype=np.float64)
    corrected = apply_quickseg_background(
        raw,
        params,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
    )
    equalized = equalise(corrected)
    denoised = denoise_tv_chambolle(
        equalized,
        weight=float(params.tv_weight),
        eps=float(params.tv_eps),
        max_num_iter=int(params.tv_iterations),
    ).astype(np.float64, copy=False)
    gaussian_kwargs: dict[str, Any] = {
        "sigma": float(params.gaussian_sigma),
        "order": int(params.gaussian_order),
        "mode": str(params.gaussian_mode),
        "cval": 0.05,
    }
    if params.gaussian_axes is not None:
        gaussian_kwargs["axes"] = tuple(int(v) for v in params.gaussian_axes)
    gaussian = gaussian_filter(denoised, **gaussian_kwargs).astype(np.float64, copy=False)
    gradient = sobel(gaussian).astype(np.float64, copy=False)
    return QuickSegPrepared(
        raw=raw,
        corrected=corrected,
        equalized=equalized,
        denoised=denoised,
        gaussian=gaussian,
        gradient=gradient,
    )


def watershed_labels(
    prepared: QuickSegPrepared,
    seeds: list[QuickSegSeed],
    params: QuickSegParams,
) -> np.ndarray:
    if prepared.gradient.shape != prepared.raw.shape:
        raise ValueError("Prepared gradient shape does not match raw image")
    markers = np.zeros(prepared.raw.shape, dtype=np.int32)
    for seed in seeds:
        x = int(seed.x)
        y = int(seed.y)
        if 0 <= y < markers.shape[0] and 0 <= x < markers.shape[1]:
            markers[y, x] = int(seed.terrace_label_id)
    if int(markers.max()) < 1:
        return np.zeros(prepared.raw.shape, dtype=np.int32)
    labels = watershed(
        prepared.gradient,
        markers=markers,
        connectivity=connectivity(int(params.watershed_connectivity)),
        compactness=float(params.watershed_compactness),
        watershed_line=bool(params.watershed_line),
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
