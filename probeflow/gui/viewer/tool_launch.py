"""GUI-free launch contexts for image viewer analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from probeflow.gui.roi_context import (
    PointSource,
    active_area_roi_area_m2,
    point_source_arrays_m,
    point_source_arrays_px,
    point_source_metadata,
)

POINT_SOURCE_REQUIRED_MESSAGE = "Run Feature finder or select point ROIs first."
LATTICE_REQUIRED_MESSAGE = "Open the Lattice/Grid tool first."
IMAGE_REQUIRED_MESSAGE = "No image loaded."


@dataclass(frozen=True)
class PairCorrelationLaunchContext:
    """Inputs needed to open the pair-correlation dialog."""

    sources_m: dict[str, np.ndarray]
    source_metadata: dict[str, dict[str, object]]
    roi_area_m2: float | None
    pixel_size_x_m: float
    pixel_size_y_m: float
    status_message: str | None = None

    @property
    def ready(self) -> bool:
        return self.status_message is None


@dataclass(frozen=True)
class FeatureLatticeLaunchContext:
    """Inputs needed to open the feature-to-lattice dialog."""

    sources_px: dict[str, np.ndarray]
    source_metadata: dict[str, dict[str, object]]
    lattice_origin_px: tuple[float, float] | None
    a_px: tuple[float, float] | None
    b_px: tuple[float, float] | None
    image_shape: tuple[int, int] | None
    pixel_size_x_m: float
    pixel_size_y_m: float
    status_message: str | None = None

    @property
    def ready(self) -> bool:
        return self.status_message is None


@dataclass(frozen=True)
class LatticeGridLaunchContext:
    """Inputs needed to open the real-space lattice/grid tool."""

    image_shape: tuple[int, int] | None
    scan_range_m: tuple[float, float] | None
    status_message: str | None = None

    @property
    def ready(self) -> bool:
        return self.status_message is None


def pair_correlation_launch_context(
    point_sources: list[PointSource],
    *,
    active_area_roi: Any = None,
    image_shape: tuple[int, int] | None = None,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> PairCorrelationLaunchContext:
    """Build a pair-correlation launch context from point sources and ROI state."""
    sources_m = point_source_arrays_m(point_sources)
    metadata = point_source_metadata(point_sources)
    if not sources_m:
        return PairCorrelationLaunchContext(
            sources_m={},
            source_metadata={},
            roi_area_m2=None,
            pixel_size_x_m=float(pixel_size_x_m),
            pixel_size_y_m=float(pixel_size_y_m),
            status_message=POINT_SOURCE_REQUIRED_MESSAGE,
        )

    roi_area_m2 = None
    if image_shape is not None:
        roi_area_m2 = active_area_roi_area_m2(
            active_area_roi,
            image_shape,
            pixel_size_x_m=pixel_size_x_m,
            pixel_size_y_m=pixel_size_y_m,
        )
    return PairCorrelationLaunchContext(
        sources_m=sources_m,
        source_metadata=metadata,
        roi_area_m2=roi_area_m2,
        pixel_size_x_m=float(pixel_size_x_m),
        pixel_size_y_m=float(pixel_size_y_m),
    )


def lattice_grid_launch_context(
    image: np.ndarray | None,
    *,
    scan_range_m: tuple[float, float] | None = None,
) -> LatticeGridLaunchContext:
    """Build a lattice-grid launch context from the current image state."""
    if image is None:
        return LatticeGridLaunchContext(
            image_shape=None,
            scan_range_m=None,
            status_message=IMAGE_REQUIRED_MESSAGE,
        )
    image_shape = tuple(int(v) for v in image.shape[:2])
    if scan_range_m is None:
        scan_range_m = (float(image_shape[1]) * 1e-9, float(image_shape[0]) * 1e-9)
    return LatticeGridLaunchContext(
        image_shape=image_shape,
        scan_range_m=(float(scan_range_m[0]), float(scan_range_m[1])),
    )


def feature_lattice_launch_context(
    point_sources: list[PointSource],
    *,
    lattice_grid: Any = None,
    image_shape: tuple[int, int] | None = None,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> FeatureLatticeLaunchContext:
    """Build a feature-lattice launch context from point sources and grid state."""
    sources_px = point_source_arrays_px(point_sources)
    metadata = point_source_metadata(point_sources)
    if not sources_px:
        return FeatureLatticeLaunchContext(
            sources_px={},
            source_metadata={},
            lattice_origin_px=None,
            a_px=None,
            b_px=None,
            image_shape=image_shape,
            pixel_size_x_m=float(pixel_size_x_m),
            pixel_size_y_m=float(pixel_size_y_m),
            status_message=POINT_SOURCE_REQUIRED_MESSAGE,
        )
    if lattice_grid is None:
        return FeatureLatticeLaunchContext(
            sources_px=sources_px,
            source_metadata=metadata,
            lattice_origin_px=None,
            a_px=None,
            b_px=None,
            image_shape=image_shape,
            pixel_size_x_m=float(pixel_size_x_m),
            pixel_size_y_m=float(pixel_size_y_m),
            status_message=LATTICE_REQUIRED_MESSAGE,
        )
    return FeatureLatticeLaunchContext(
        sources_px=sources_px,
        source_metadata=metadata,
        lattice_origin_px=lattice_grid.origin_px,
        a_px=lattice_grid.a_px,
        b_px=lattice_grid.b_px,
        image_shape=image_shape,
        pixel_size_x_m=float(pixel_size_x_m),
        pixel_size_y_m=float(pixel_size_y_m),
    )
