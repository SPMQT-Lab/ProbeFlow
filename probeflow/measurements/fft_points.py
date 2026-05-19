"""Point-mask and FFT helpers for detected feature points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from probeflow.measurements.models import FeaturePoint, MeasurementResult, Scalar


@dataclass(frozen=True)
class PointFFTResult:
    """FFT result computed from a binary point mask."""

    fft_magnitude: np.ndarray
    qx: np.ndarray | None
    qy: np.ndarray | None
    units: str | None
    n_points: int
    mask_shape: tuple[int, int]
    radius_px: int


def points_to_mask(
    points: Iterable[FeaturePoint | tuple[float, float]],
    shape: tuple[int, int],
    *,
    radius_px: int = 0,
    shape_mode: str = "disk",
) -> np.ndarray:
    """Rasterize feature points to a binary mask, optionally dilated."""
    mask = np.zeros(shape, dtype=bool)
    radius = int(max(0, radius_px))
    mode = (shape_mode or "disk").strip().lower()
    for point in points:
        x_px, y_px = _point_xy(point)
        cx = int(round(x_px))
        cy = int(round(y_px))
        if radius == 0:
            if 0 <= cy < shape[0] and 0 <= cx < shape[1]:
                mask[cy, cx] = True
            continue
        for row in range(max(0, cy - radius), min(shape[0], cy + radius + 1)):
            for col in range(max(0, cx - radius), min(shape[1], cx + radius + 1)):
                if mode == "square" or (row - cy) ** 2 + (col - cx) ** 2 <= radius ** 2:
                    mask[row, col] = True
    if mode not in {"disk", "square"}:
        raise ValueError("shape_mode must be 'disk' or 'square'")
    return mask


def fft_from_point_mask(
    mask: np.ndarray,
    *,
    pixel_size_x: float | None = None,
    pixel_size_y: float | None = None,
    spatial_unit: str | None = None,
    n_points: int | None = None,
    radius_px: int = 0,
) -> PointFFTResult:
    """Compute centered FFT magnitude from a binary point mask."""
    arr = np.asarray(mask, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("point mask must be 2-D")
    fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft2(arr)))
    ny, nx = arr.shape
    if pixel_size_x is None or pixel_size_y is None:
        qx = np.fft.fftshift(np.fft.fftfreq(nx, d=1.0))
        qy = np.fft.fftshift(np.fft.fftfreq(ny, d=1.0))
        units = "cycles/pixel"
    else:
        qx = np.fft.fftshift(np.fft.fftfreq(nx, d=float(pixel_size_x)))
        qy = np.fft.fftshift(np.fft.fftfreq(ny, d=float(pixel_size_y)))
        units = f"cycles/{spatial_unit}" if spatial_unit else "cycles/unit"
    return PointFFTResult(
        fft_magnitude=fft_magnitude,
        qx=qx,
        qy=qy,
        units=units,
        n_points=int(np.count_nonzero(arr)) if n_points is None else int(n_points),
        mask_shape=(int(ny), int(nx)),
        radius_px=int(radius_px),
    )


def point_mask_to_csv_text(
    mask: np.ndarray,
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    """Return a 0/1 CSV matrix for a binary point mask."""
    import csv
    import io

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("point mask must be 2-D")
    out = io.StringIO()
    writer = csv.writer(out)
    if metadata:
        for key, value in metadata.items():
            writer.writerow([f"# {key}", _metadata_value(value)])
    for row in arr:
        writer.writerow(["1" if value else "0" for value in row])
    return out.getvalue()


def point_fft_to_csv_text(
    result: PointFFTResult,
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    """Return long-form CSV text for a point-mask FFT result."""
    import csv
    import io

    mag = np.asarray(result.fft_magnitude, dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError("fft_magnitude must be 2-D")
    qx = result.qx if result.qx is not None else np.arange(mag.shape[1], dtype=float)
    qy = result.qy if result.qy is not None else np.arange(mag.shape[0], dtype=float)
    out = io.StringIO()
    writer = csv.writer(out)
    if metadata:
        for key, value in metadata.items():
            writer.writerow([f"# {key}", _metadata_value(value)])
    writer.writerow(["qx", "qy", "magnitude", "unit"])
    for row, qy_value in enumerate(qy):
        for col, qx_value in enumerate(qx):
            writer.writerow([
                f"{float(qx_value):.10g}",
                f"{float(qy_value):.10g}",
                f"{float(mag[row, col]):.10g}",
                result.units or "",
            ])
    return out.getvalue()


def point_fft_summary_result(
    result: PointFFTResult,
    *,
    measurement_id: str,
    source_label: str,
    source_path: str | None = None,
    channel: str | None = None,
    mask_pixels: int | None = None,
    shape_mode: str | None = None,
    data_basis: str = "binary_point_mask",
    notes: str = "",
) -> MeasurementResult:
    """Summarize a point-mask FFT as an exportable measurement row."""
    dominant_qx, dominant_qy, dominant_frequency, peak_magnitude = _dominant_fft_peak(result)
    values: dict[str, Scalar] = {
        "dominant_frequency": dominant_frequency,
        "dominant_qx": dominant_qx,
        "dominant_qy": dominant_qy,
        "peak_magnitude": peak_magnitude,
        "n_points": int(result.n_points),
    }
    if mask_pixels is not None:
        values["n_mask_pixels"] = int(mask_pixels)
    context: dict[str, Scalar] = {
        "data_basis": data_basis,
        "mask_shape_y": int(result.mask_shape[0]),
        "mask_shape_x": int(result.mask_shape[1]),
        "radius_px": int(result.radius_px),
        "shape_mode": shape_mode,
        "fft_units": result.units,
    }
    return MeasurementResult(
        measurement_id=measurement_id,
        kind="point_fft",
        source_label=source_label,
        source_path=source_path,
        channel=channel,
        x_unit=result.units,
        y_unit=None,
        values=values,
        context=context,
        notes=notes,
    )


def _dominant_fft_peak(result: PointFFTResult) -> tuple[float | None, float | None, float | None, float | None]:
    mag = np.asarray(result.fft_magnitude, dtype=np.float64)
    if mag.ndim != 2 or mag.size == 0:
        return None, None, None, None
    work = mag.copy()
    cy = mag.shape[0] // 2
    cx = mag.shape[1] // 2
    work[cy, cx] = -np.inf
    finite = np.isfinite(work)
    if not np.any(finite):
        return None, None, None, None
    row, col = np.unravel_index(int(np.nanargmax(work)), work.shape)
    qx = result.qx if result.qx is not None else np.arange(mag.shape[1], dtype=float)
    qy = result.qy if result.qy is not None else np.arange(mag.shape[0], dtype=float)
    qx_value = float(qx[col])
    qy_value = float(qy[row])
    return (
        qx_value,
        qy_value,
        float(np.hypot(qx_value, qy_value)),
        float(mag[row, col]),
    )


def _point_xy(point: FeaturePoint | tuple[float, float]) -> tuple[float, float]:
    if isinstance(point, FeaturePoint):
        return point.x_px, point.y_px
    return float(point[0]), float(point[1])


def _metadata_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)
