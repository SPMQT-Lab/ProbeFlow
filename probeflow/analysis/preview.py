"""Qt-free preview analysis pipeline for ScanFlow-backed previews.

This module stays entirely in the non-GUI layer. It loads an STM scan,
applies a lightweight background correction, finds candidate features, and
returns JSON-serialisable preview metadata plus coordinate-normalised feature
rows that higher-level tools can use for motion handoff.

The default path is segmentation-first:
1. build a foreground mask from the background-corrected plane,
2. extract connected bright regions as candidate features,
3. fall back to point-feature detection when segmentation yields nothing.

The result object carries both pixel-space and physical-space coordinates so
callers do not have to guess which conversion is appropriate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from probeflow.analysis.feature_finder import FeatureDetectionResult, find_image_features
from probeflow.analysis.features import Particle, segment_particles
from probeflow.core.metadata import ScanMetadata, metadata_from_scan
from probeflow.core.scan_loader import load_scan
from probeflow.processing.background import STMBackgroundParams, preview_stm_background

FeatureMode = Literal["segmentation_first", "segmentation_only", "points_only"]


class _ScanLike(Protocol):
    planes: list[np.ndarray]
    plane_names: list[str]
    plane_units: list[str]
    scan_range_m: tuple[float, float] | None
    source_path: Path
    source_format: str


@dataclass(frozen=True)
class PreviewAnalysisParams:
    """Inputs controlling preview generation."""

    plane_index: int = 0
    background_mode: str = "linear"
    background_strength: float = 1.0
    feature_mode: FeatureMode = "segmentation_first"
    threshold: str = "otsu"
    manual_threshold: float = 128.0
    invert: bool = False
    min_area_nm2: float = 0.5
    max_area_nm2: float | None = None
    size_sigma_clip: float | None = 2.0
    clip_low: float = 1.0
    clip_high: float = 99.0
    max_features: int = 64
    min_distance_px: float = 6.0
    smoothing_sigma_px: float = 1.0
    point_threshold_sigma: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreviewFeatureRow:
    """One detected preview target in both pixel and physical coordinates."""

    index: int
    source: str
    x_px: float
    y_px: float
    x_nm: float
    y_nm: float
    dx_nm: float
    dy_nm: float
    x_m: float
    y_m: float
    dx_m: float
    dy_m: float
    score: float
    bbox_px: tuple[int, int, int, int] | None = None
    label: str = ""
    area_nm2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class PreviewResult:
    """Full preview output for UI and provenance layers."""

    source_path: Path
    source_format: str
    scan_metadata: ScanMetadata
    params: PreviewAnalysisParams
    plane_index: int
    plane_name: str
    plane_unit: str
    scan_shape: tuple[int, int]
    scan_range_m: tuple[float, float] | None
    raw_plane: np.ndarray
    background_corrected_plane: np.ndarray
    background_image: np.ndarray
    feature_rows: tuple[PreviewFeatureRow, ...]
    warnings: tuple[str, ...] = ()
    analysis_status: str = "success"

    @property
    def overlay_coordinates(self) -> tuple[tuple[float, float], ...]:
        return tuple((row.x_px, row.y_px) for row in self.feature_rows)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload = {
            "source_path": str(self.source_path),
            "source_format": self.source_format,
            "scan_metadata": _json_safe(self.scan_metadata),
            "params": self.params.to_dict(),
            "plane_index": self.plane_index,
            "plane_name": self.plane_name,
            "plane_unit": self.plane_unit,
            "scan_shape": list(self.scan_shape),
            "scan_range_m": list(self.scan_range_m) if self.scan_range_m is not None else None,
            "feature_rows": [row.to_dict() for row in self.feature_rows],
            "warnings": list(self.warnings),
            "analysis_status": self.analysis_status,
            "overlay_coordinates": [list(pt) for pt in self.overlay_coordinates],
        }
        if include_arrays:
            payload["raw_plane"] = self.raw_plane.tolist()
            payload["background_corrected_plane"] = self.background_corrected_plane.tolist()
            payload["background_image"] = self.background_image.tolist()
        else:
            payload["raw_plane_shape"] = list(self.raw_plane.shape)
            payload["background_corrected_plane_shape"] = list(self.background_corrected_plane.shape)
            payload["background_image_shape"] = list(self.background_image.shape)
        return payload


def run_preview(source: Path | str | _ScanLike, params: PreviewAnalysisParams | None = None) -> PreviewResult:
    """Load a scan or use an existing Scan and build preview analysis output."""

    params = params or PreviewAnalysisParams()
    scan = _load_scan_like(source)
    metadata = metadata_from_scan(scan)
    source_path = Path(scan.source_path)
    plane_index = _resolve_plane_index(params.plane_index, len(scan.planes))

    raw = np.asarray(scan.planes[plane_index], dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("preview analysis requires a 2-D scan plane")
    scan_shape = (int(raw.shape[0]), int(raw.shape[1]))

    pixel_size_x_m, pixel_size_y_m = _pixel_sizes_m(scan.scan_range_m, scan_shape)
    bg_params = _background_params(params)
    bg_result = preview_stm_background(raw, bg_params)
    corrected = np.asarray(bg_result.corrected, dtype=np.float64)
    background_image = np.asarray(bg_result.background_image, dtype=np.float64)

    feature_rows, warnings = _detect_features(
        corrected,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        params=params,
    )

    plane_name = scan.plane_names[plane_index] if plane_index < len(scan.plane_names) else f"Plane {plane_index}"
    plane_unit = scan.plane_units[plane_index] if plane_index < len(scan.plane_units) else ""

    return PreviewResult(
        source_path=source_path,
        source_format=scan.source_format,
        scan_metadata=metadata,
        params=params,
        plane_index=plane_index,
        plane_name=plane_name,
        plane_unit=plane_unit,
        scan_shape=scan_shape,
        scan_range_m=_scan_range_tuple(scan.scan_range_m),
        raw_plane=raw,
        background_corrected_plane=corrected,
        background_image=background_image,
        feature_rows=tuple(feature_rows),
        warnings=tuple(warnings),
        analysis_status="success" if feature_rows else "empty",
    )


def apply_preview_background(
    raw_plane: np.ndarray,
    params: PreviewAnalysisParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the ProbeFlow background correction used by the previewer."""

    params = params or PreviewAnalysisParams()
    raw = np.asarray(raw_plane, dtype=np.float64)
    bg_result = preview_stm_background(raw, _background_params(params))
    corrected = np.asarray(bg_result.corrected, dtype=np.float64)
    background_image = np.asarray(bg_result.background_image, dtype=np.float64)
    return corrected, background_image


def detect_preview_features(
    corrected_plane: np.ndarray,
    *,
    scan_range_m: tuple[float, float] | None,
    params: PreviewAnalysisParams | None = None,
) -> tuple[list[PreviewFeatureRow], list[str]]:
    """Run the ProbeFlow preview feature detector against one plane."""

    params = params or PreviewAnalysisParams()
    corrected = np.asarray(corrected_plane, dtype=np.float64)
    pixel_size_x_m, pixel_size_y_m = _pixel_sizes_m(scan_range_m, corrected.shape)
    return _detect_features(
        corrected,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        params=params,
    )


def preview_record(result: PreviewResult) -> dict[str, Any]:
    """Return a JSON-safe provenance record without embedding raw arrays."""

    return result.to_dict(include_arrays=False)


def _load_scan_like(source: Path | str | _ScanLike) -> _ScanLike:
    if hasattr(source, "planes") and hasattr(source, "scan_range_m"):
        return cast(_ScanLike, source)
    return cast(_ScanLike, load_scan(source))


def _resolve_plane_index(index: int, n_planes: int) -> int:
    if n_planes <= 0:
        raise ValueError("scan has no planes")
    if index < 0 or index >= n_planes:
        raise IndexError(f"plane_index {index} out of range for {n_planes} plane(s)")
    return index


def _pixel_sizes_m(scan_range_m: tuple[float, float] | None, shape: tuple[int, int]) -> tuple[float, float]:
    ny, nx = int(shape[0]), int(shape[1])
    if scan_range_m is None:
        return (1.0, 1.0)
    sx = float(scan_range_m[0]) / max(nx, 1)
    sy = float(scan_range_m[1]) / max(ny, 1)
    return (sx, sy)


def _scan_range_tuple(scan_range_m: tuple[float, float] | None) -> tuple[float, float] | None:
    if scan_range_m is None:
        return None
    return (float(scan_range_m[0]), float(scan_range_m[1]))


def _background_params(params: PreviewAnalysisParams) -> STMBackgroundParams:
    blur = None
    if params.background_mode == "low_pass":
        blur = max(0.5, float(params.background_strength))
    return STMBackgroundParams(
        model=params.background_mode,
        blur_length=blur,
    )


def _detect_features(
    corrected: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    params: PreviewAnalysisParams,
) -> tuple[list[PreviewFeatureRow], list[str]]:
    warnings: list[str] = []
    if params.feature_mode != "points_only":
        segmentation = _segment_features(
            corrected,
            pixel_size_x_m=pixel_size_x_m,
            pixel_size_y_m=pixel_size_y_m,
            params=params,
            max_features=params.max_features,
        )
        if segmentation:
            return segmentation, warnings
        warnings.append("segmentation returned no features; falling back to point detection")
        if params.feature_mode == "segmentation_only":
            return [], warnings

    points = _detect_points(
        corrected,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        params=params,
    )
    if not points:
        warnings.append("point detection returned no features")
    else:
        warnings.append("point detection fallback used")
    return points, warnings


def _segment_features(
    image: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    params: PreviewAnalysisParams,
    max_features: int,
) -> list[PreviewFeatureRow]:
    rows: list[PreviewFeatureRow] = []
    particles: list[Particle] = segment_particles(
        np.asarray(image, dtype=np.float64),
        pixel_size_m=float((pixel_size_x_m * pixel_size_y_m) ** 0.5),
        pixel_size_x_m=float(pixel_size_x_m),
        pixel_size_y_m=float(pixel_size_y_m),
        threshold=str(params.threshold),
        manual_value=float(params.manual_threshold),
        invert=bool(params.invert),
        min_area_nm2=float(params.min_area_nm2),
        max_area_nm2=None if params.max_area_nm2 is None else float(params.max_area_nm2),
        size_sigma_clip=None if params.size_sigma_clip is None else float(params.size_sigma_clip),
        clip_low=float(params.clip_low),
        clip_high=float(params.clip_high),
    )
    particles.sort(key=lambda p: (-float(p.area_nm2), float(p.centroid_y_m), float(p.centroid_x_m)))

    ny, nx = np.asarray(image).shape
    for particle in particles[: int(max_features)]:
        x_m = float(particle.centroid_x_m)
        y_m = float(particle.centroid_y_m)
        rows.append(
            PreviewFeatureRow(
                index=len(rows),
                source="segmentation",
                x_px=x_m / pixel_size_x_m,
                y_px=y_m / pixel_size_y_m,
                x_nm=x_m * 1e9,
                y_nm=y_m * 1e9,
                dx_nm=(x_m - (nx / 2.0) * pixel_size_x_m) * 1e9,
                dy_nm=(y_m - (ny / 2.0) * pixel_size_y_m) * 1e9,
                x_m=x_m,
                y_m=y_m,
                dx_m=x_m - (nx / 2.0) * pixel_size_x_m,
                dy_m=y_m - (ny / 2.0) * pixel_size_y_m,
                score=float(particle.mean_height),
                bbox_px=tuple(int(v) for v in particle.bbox_px),
                label=f"region-{len(rows) + 1}",
                area_nm2=float(particle.area_nm2),
            )
        )
    return rows


def _detect_points(
    image: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    params: PreviewAnalysisParams,
) -> list[PreviewFeatureRow]:
    arr = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return []
    work = np.where(finite, arr, float(np.nanmedian(arr[finite])))
    sigma = max(0.0, float(params.smoothing_sigma_px))
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        work = gaussian_filter(work, sigma=sigma, mode="nearest")

    threshold = float(np.nanmedian(work) + float(params.point_threshold_sigma) * np.nanstd(work))
    detection: FeatureDetectionResult = find_image_features(
        work,
        mode="maxima",
        threshold_mode="above",
        threshold_low=threshold,
        min_distance_px=float(params.min_distance_px),
        smoothing_sigma_px=0.0,
    )
    rows: list[PreviewFeatureRow] = []
    ny, nx = arr.shape
    for point in detection.points[: int(params.max_features)]:
        x_px = float(point.x_px)
        y_px = float(point.y_px)
        x_m = x_px * pixel_size_x_m
        y_m = y_px * pixel_size_y_m
        rows.append(
            PreviewFeatureRow(
                index=len(rows),
                source="points",
                x_px=x_px,
                y_px=y_px,
                x_nm=x_m * 1e9,
                y_nm=y_m * 1e9,
                dx_nm=(x_px - (nx / 2.0)) * pixel_size_x_m * 1e9,
                dy_nm=(y_px - (ny / 2.0)) * pixel_size_y_m * 1e9,
                x_m=x_m,
                y_m=y_m,
                dx_m=(x_px - (nx / 2.0)) * pixel_size_x_m,
                dy_m=(y_px - (ny / 2.0)) * pixel_size_y_m,
                score=float(point.value),
                bbox_px=(int(round(x_px)), int(round(y_px)), int(round(x_px)) + 1, int(round(y_px)) + 1),
                label="point",
            )
        )
    return rows


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


__all__ = [
    "FeatureMode",
    "apply_preview_background",
    "detect_preview_features",
    "PreviewAnalysisParams",
    "PreviewFeatureRow",
    "PreviewResult",
    "preview_record",
    "run_preview",
]
