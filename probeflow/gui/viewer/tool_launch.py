"""GUI-free launch contexts for image viewer analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from probeflow.analysis.adstat_adapter import compare_point_source_view_spec
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
class AdStatWorkbenchLaunchContext:
    """Inputs produced by the direct ProbeFlow-to-AdStat analysis path."""

    view_spec: Any | None
    point_source_label: str | None
    status_message: str | None = None

    @property
    def ready(self) -> bool:
        return self.status_message is None


@dataclass(frozen=True)
class AdStatStatisticsRequest:
    """User-selected inputs for a real-data AdStat workbench run."""

    point_source_label: str | None
    region_mode: str
    roi_or_mask: Any = None
    models: tuple[str, ...] = ("poisson",)
    n_simulations: int = 100
    random_seed: int | None = 0
    include_ordering: bool = False


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


def adstat_workbench_launch_context(
    point_sources: list[PointSource],
    *,
    scan: Any = None,
    active_area_roi: Any = None,
    roi_or_mask: Any = None,
    image_shape: tuple[int, int] | None = None,
    point_source_label: str | None = None,
    models: tuple[str, ...] = ("poisson",),
    pair_bin_width_nm: float | None = None,
    pair_max_radius_nm: float | None = None,
    cluster_radius_nm: float | None = None,
    n_simulations: int = 19,
    random_seed: int | None = 0,
    include_ordering: bool = False,
    request: AdStatStatisticsRequest | None = None,
) -> AdStatWorkbenchLaunchContext:
    """Build a Qt-renderable AdStat workbench spec from current viewer state."""

    if request is not None:
        point_source_label = request.point_source_label
        roi_or_mask = request.roi_or_mask
        models = request.models
        n_simulations = request.n_simulations
        random_seed = request.random_seed
        include_ordering = request.include_ordering
    elif roi_or_mask is None:
        roi_or_mask = active_area_roi

    if scan is None:
        return AdStatWorkbenchLaunchContext(
            view_spec=None,
            point_source_label=None,
            status_message=IMAGE_REQUIRED_MESSAGE,
        )
    if not point_sources:
        return AdStatWorkbenchLaunchContext(
            view_spec=None,
            point_source_label=None,
            status_message=POINT_SOURCE_REQUIRED_MESSAGE,
        )

    source = _select_point_source(point_sources, point_source_label)
    try:
        view_spec = compare_point_source_view_spec(
            source,
            scan=scan,
            roi_or_mask=roi_or_mask,
            image_shape=image_shape,
            scan_id=_scan_id(scan),
            models=models,
            pair_bin_width_nm=pair_bin_width_nm,
            pair_max_radius_nm=pair_max_radius_nm,
            cluster_radius_nm=cluster_radius_nm,
            n_simulations=n_simulations,
            random_seed=random_seed,
            include_ordering=include_ordering,
        )
    except ImportError as exc:
        return AdStatWorkbenchLaunchContext(
            view_spec=None,
            point_source_label=source.label,
            status_message=str(exc),
        )
    except Exception as exc:
        return AdStatWorkbenchLaunchContext(
            view_spec=None,
            point_source_label=source.label,
            status_message=f"AdStat analysis failed: {exc}",
        )
    return AdStatWorkbenchLaunchContext(
        view_spec=view_spec,
        point_source_label=source.label,
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


def _select_point_source(
    point_sources: list[PointSource],
    label: str | None,
) -> PointSource:
    if label:
        for source in point_sources:
            if source.label == label:
                return source
    return point_sources[0]


def _scan_id(scan: Any) -> str | None:
    source_path = getattr(scan, "source_path", None)
    if source_path is not None:
        try:
            return source_path.stem
        except AttributeError:
            return str(source_path)
    return None


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
