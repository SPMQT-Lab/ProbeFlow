"""ProbeFlow-to-AdStat integration adapter.

The functions in this module are deliberately small conversion boundaries:
ProbeFlow owns scans, point detection, ROI/mask editing, and curation; AdStat
owns point-pattern statistics and result view specifications.  Imports from
AdStat are lazy so ProbeFlow remains importable without the optional analysis
engine installed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np


KEEP_STATUSES = frozenset({"accepted", "manual"})

# Local-order / lattice statistics. These answer a different question from the
# core randomness null (is there square/triangular local order?), are sensitive
# to the neighbour cutoff and edge effects, and AdStat itself excludes them from
# its core verdict rollup. ProbeFlow treats them as opt-in (off by default).
ORDERING_STATISTICS = frozenset(
    {"pair_correlation_g_r_theta", "bond_order_psi6", "bond_order_psi4"}
)

__all__ = [
    "AdStatPointSetRecord",
    "ORDERING_STATISTICS",
    "adstat_sandbox_context",
    "adstat_sandbox_preview",
    "adstat_sandbox_state",
    "adstat_sandbox_view_spec",
    "compare_particle_collection_view_spec",
    "compare_point_set_record_view_spec",
    "compare_point_set_records_view_spec",
    "compare_point_source_view_spec",
    "feature_counting_to_particle_table",
    "feature_layers_to_adstat",
    "point_set_record",
    "point_source_to_particle_table",
    "roi_to_region",
    "scan_calibration_to_adstat",
    "workbench_view_spec",
]


@dataclass(frozen=True)
class AdStatPointSetRecord:
    """One scan's point set and analysis region for a multi-image collection."""

    dataset_id: str | None
    table: Any
    region: Any
    calibration: Any
    series_value: float | None = None
    series_unit: str | None = None
    series_label: str | None = None
    source_metadata: dict[str, object] | None = None


def scan_calibration_to_adstat(scan: Any) -> Any:
    """Convert a ProbeFlow ``Scan`` into an AdStat ``ImageCalibration``."""

    adstat = _adstat()
    width_m, height_m = _scan_range_m(scan)
    width_px, height_px = _scan_dims(scan)
    if width_px <= 0 or height_px <= 0:
        raise ValueError("scan dimensions must be positive")
    return adstat.ImageCalibration(
        pixel_size_x_nm=float(width_m) * 1e9 / float(width_px),
        pixel_size_y_nm=float(height_m) * 1e9 / float(height_px),
        width_px=int(width_px),
        height_px=int(height_px),
        origin="upper_left",
        rotation_angle_deg=0.0,
    )


def point_source_to_particle_table(
    source: Any,
    *,
    scan_id: str | None = None,
    accepted_statuses: Iterable[str] = KEEP_STATUSES,
) -> Any:
    """Convert a ProbeFlow ``PointSource`` into an AdStat ``ParticleTable``.

    ``PointSource`` is already the common GUI-free bridge for Feature Finder
    maxima/minima, detected feature maxima, and point ROIs.  Coordinates in
    ``points_m`` are converted to AdStat's canonical nanometres while original
    pixel coordinates are preserved on each particle.
    """

    adstat = _adstat()
    points_px = _xy_array(_field(source, "points_px"), name="points_px")
    points_m = _xy_array(_field(source, "points_m"), name="points_m")
    if len(points_px) != len(points_m):
        raise ValueError("points_px and points_m must have the same length")
    if len(points_m) == 0:
        raise ValueError("point source must contain at least one point")

    label = _field(source, "label", None)
    source_type = _field(source, "source_type", None)
    metadata = dict(_field(source, "metadata", {}) or {})
    particles = tuple(
        adstat.Particle(
            id=_point_id(metadata, index),
            x_nm=float(points_m[index, 0]) * 1e9,
            y_nm=float(points_m[index, 1]) * 1e9,
            x_px=float(points_px[index, 0]),
            y_px=float(points_px[index, 1]),
            metadata={
                "probeflow_source": label,
                "probeflow_source_type": source_type,
                "probeflow_status": "accepted",
            },
        )
        for index in range(len(points_m))
    )
    return adstat.ParticleTable(
        particles=particles,
        metadata={
            "scan_id": scan_id,
            "probeflow_point_source": label,
            "probeflow_point_source_type": source_type,
            "accepted_statuses": tuple(sorted(str(s) for s in accepted_statuses)),
            **metadata,
        },
    )


def feature_counting_to_particle_table(
    items: Iterable[Any],
    *,
    scan_id: str | None = None,
    calibration: Any | None = None,
    accepted_statuses: Iterable[str] = KEEP_STATUSES,
) -> Any:
    """Convert Feature Counting particles or template detections to AdStat.

    Supported records are native ``probeflow.analysis.features.Particle`` and
    ``Detection`` instances, dictionaries from ProbeFlow JSON exports, or
    similarly shaped objects.
    """

    adstat = _adstat()
    keep = frozenset(str(status) for status in accepted_statuses)
    particles = []
    source_kinds: set[str] = set()
    for index, item in enumerate(items):
        status = _field(item, "status", "accepted")
        if status is not None and str(status) not in keep:
            continue
        source_kind = _feature_counting_kind(item)
        source_kinds.add(source_kind)
        x_nm, y_nm = _feature_counting_xy_nm(item)
        x_px, y_px = _feature_counting_xy_px(item, calibration=calibration)
        particles.append(
            adstat.Particle(
                id=_feature_counting_id(item, index),
                x_nm=x_nm,
                y_nm=y_nm,
                x_px=x_px,
                y_px=y_px,
                orientation_deg=_optional_float(_field(item, "orientation_deg", None)),
                area_nm2=_optional_float(_field(item, "area_nm2", None)),
                confidence=_feature_counting_confidence(item),
                label=_optional_str(_field(item, "class_name", None)),
                metadata=_feature_counting_metadata(item, source_kind, status),
            )
        )
    if not particles:
        raise ValueError("feature counting conversion produced no accepted points")
    return adstat.ParticleTable(
        particles=tuple(particles),
        metadata={
            "scan_id": scan_id,
            "probeflow_point_source": "Feature Counting",
            "probeflow_point_source_type": ",".join(sorted(source_kinds)),
            "accepted_statuses": tuple(sorted(keep)),
        },
    )


def roi_to_region(
    roi_or_mask: Any,
    *,
    scan: Any,
    image_shape: tuple[int, int] | None = None,
) -> Any:
    """Convert a ProbeFlow area ROI or mask into an AdStat analysis region.

    ``None`` and non-area ROIs fall back to the full image rectangle.  Boolean
    arrays and area ROIs become ``MaskRegion`` objects aligned to the scan
    calibration.
    """

    adstat = _adstat()
    calibration = scan_calibration_to_adstat(scan)
    if image_shape is None:
        width_px, height_px = _scan_dims(scan)
        image_shape = (height_px, width_px)

    mask = _mask_from_roi_or_mask(roi_or_mask, image_shape)
    if mask is None:
        return adstat.RectangularRegion(
            width_nm=calibration.width_nm,
            height_nm=calibration.height_nm,
        )
    return adstat.MaskRegion(mask, calibration=calibration, mask_path=_roi_label(roi_or_mask))


def feature_layers_to_adstat(
    layers: Iterable[Any],
    *,
    calibration: Any,
    require_independent: bool = True,
) -> tuple[Any, ...]:
    """Convert ProbeFlow-shaped independent feature layers to AdStat layers."""

    adstat = _adstat()
    converted = []
    for layer in layers:
        provenance = dict(_field(layer, "provenance", {}) or {})
        if require_independent and (
            provenance.get("measured_independently") is not True
            or provenance.get("derived_from_particles") is True
        ):
            raise ValueError(
                "feature layers used for AdStat comparison must be independent "
                "measurements, not maps derived from the tested particles"
            )
        kind = str(_field(layer, "kind", _field(layer, "type", "")))
        name = str(_field(layer, "name", _field(layer, "label", kind or "feature")))
        feature_type = str(_field(layer, "feature_type", kind or "feature"))
        source = provenance.get("source") or _field(layer, "source", None)
        metadata = {"provenance": provenance}
        if kind == "points":
            xy_nm = _points_layer_xy_nm(layer, calibration)
            converted.append(
                adstat.PointFeatureLayer(
                    name=name,
                    xy_nm=xy_nm,
                    feature_type=feature_type,
                    source=None if source is None else str(source),
                    metadata=metadata,
                )
            )
        elif kind == "lines":
            segments_nm = _line_layer_segments_nm(layer, calibration)
            converted.append(
                adstat.LineFeatureLayer(
                    name=name,
                    segments_nm=segments_nm,
                    feature_type=feature_type,
                    source=None if source is None else str(source),
                    metadata=metadata,
                )
            )
        else:
            raise ValueError(f"unsupported feature layer kind: {kind!r}")
    return tuple(converted)


def _filter_table_to_region(table: Any, region: Any) -> Any:
    """Restrict a particle table to the points inside the analysis region.

    Null models simulate the observed count uniformly inside the region, so
    observed points outside it would inflate the simulated density and bias
    every matched statistic toward a false "not random" verdict. Points outside
    are dropped; the drop is recorded on the table metadata so view specs can
    report it.
    """

    adstat = _adstat()
    n_total = len(table)
    if n_total == 0:
        return table
    xy_nm = np.asarray(table.xy_nm, dtype=float)
    keep = np.asarray(region.contains(xy_nm), dtype=bool)
    n_kept = int(keep.sum())
    if n_kept == n_total:
        return table
    if n_kept == 0:
        raise ValueError(
            f"none of the {n_total} points lie inside the analysis region"
        )
    if n_kept < 2:
        raise ValueError(
            f"only {n_kept} of {n_total} points lies inside the analysis "
            "region; comparison requires at least two points inside it"
        )
    particles = tuple(
        particle for particle, kept in zip(table.particles, keep) if kept
    )
    metadata = dict(table.metadata)
    metadata["region_filtered"] = True
    metadata["n_points_total"] = n_total
    metadata["n_points_in_region"] = n_kept
    return adstat.ParticleTable(particles=particles, metadata=metadata)


def _region_filter_status_lines(table: Any) -> tuple[str, ...]:
    metadata = getattr(table, "metadata", {}) or {}
    if not metadata.get("region_filtered"):
        return ()
    kept = metadata.get("n_points_in_region")
    total = metadata.get("n_points_total")
    return (
        f"{kept} of {total} points lie inside the analysis region; the rest "
        "were excluded from the statistics and the null models.",
    )


def point_set_record(
    *,
    dataset_id: str | None,
    scan: Any,
    point_source: Any | None = None,
    feature_counting_items: Iterable[Any] | None = None,
    roi_or_mask: Any = None,
    image_shape: tuple[int, int] | None = None,
    series_value: float | None = None,
    series_unit: str | None = None,
    series_label: str | None = None,
) -> AdStatPointSetRecord:
    """Build one manifest-like in-memory record for a ProbeFlow scan.

    The table is restricted to the points inside the analysis region so the
    observed statistics and the simulated nulls describe the same window.
    """

    calibration = scan_calibration_to_adstat(scan)
    if point_source is not None:
        table = point_source_to_particle_table(point_source, scan_id=dataset_id)
        source_metadata = dict(_field(point_source, "metadata", {}) or {})
    elif feature_counting_items is not None:
        table = feature_counting_to_particle_table(
            feature_counting_items,
            scan_id=dataset_id,
            calibration=calibration,
        )
        source_metadata = None
    else:
        raise ValueError("point_set_record requires a point_source or feature_counting_items")
    region = roi_to_region(roi_or_mask, scan=scan, image_shape=image_shape)
    table = _filter_table_to_region(table, region)
    if source_metadata is None:
        source_metadata = dict(table.metadata)
    return AdStatPointSetRecord(
        dataset_id=dataset_id,
        table=table,
        region=region,
        calibration=calibration,
        series_value=series_value,
        series_unit=series_unit,
        series_label=series_label,
        source_metadata=source_metadata,
    )


def workbench_view_spec(
    *,
    table: Any,
    region: Any,
    summary: Any,
    comparisons: Iterable[Any] = (),
    active_model: str | None = None,
    **kwargs: Any,
) -> Any:
    """Return AdStat's GUI-free workbench view spec for ProbeFlow Qt rendering."""

    viz_spec = _adstat_viz_spec()
    return viz_spec.view_spec_for(
        viz_spec.WorkbenchViewInput(
            table=table,
            region=region,
            summary=summary,
            comparisons=tuple(comparisons),
            active_model=active_model,
            **kwargs,
        )
    )


def _pixel_resolution_floor_nm(calibration: Any | None) -> float | None:
    if calibration is None:
        return None
    values = [
        _optional_float(_field(calibration, "pixel_size_x_nm", None)),
        _optional_float(_field(calibration, "pixel_size_y_nm", None)),
    ]
    finite = [float(value) for value in values if value is not None and value > 0.0]
    return max(finite) if finite else None


def _region_area_nm2(region: Any) -> float:
    """Best-effort analysis-region area in nm^2 (mask area or rectangle area)."""

    area = float(getattr(region, "area_nm2", 0.0) or 0.0)
    if area > 0.0:
        return area
    width = float(getattr(region, "width_nm", 0.0) or 0.0)
    height = float(getattr(region, "height_nm", 0.0) or 0.0)
    return width * height


def _default_statistic_scales(
    region: Any,
    n_points: int = 0,
    *,
    resolution_floor_nm: float | None = None,
) -> SimpleNamespace:
    """Derive pair / nearest-neighbor / cluster scales from the analysis region.

    AdStat requires at least one comparison statistic to be configured; passing
    ``None`` for every scale raises ``"at least one comparison statistic must be
    configured"``.  Real-data callers (the viewer) do not ask the user for these
    nanometre scales, so we derive teaching-quality defaults from the region size,
    matching the generated sandbox (a 100 nm field gives a 20 nm radius, a
    1 nm bin, a 2 nm cluster radius, and a 1.5 nm hard-core radius).
    """

    area = float(getattr(region, "area_nm2", 0.0) or 0.0)
    char = math.sqrt(area) if area > 0.0 else 0.0
    width = float(getattr(region, "width_nm", 0.0) or 0.0)
    height = float(getattr(region, "height_nm", 0.0) or 0.0)
    if char <= 0.0:
        char = (
            math.sqrt(width * height)
            if width > 0.0 and height > 0.0
            else max(width, height)
        )
    if char <= 0.0:
        char = 100.0  # last-resort fallback for degenerate regions
    max_radius = 0.2 * char
    # Distances can only be measured out to roughly half the field; bound the
    # nearest-neighbor cap below to this so it never runs past the analysis area.
    distance_cap = 0.5 * char
    if width > 0.0 and height > 0.0:
        # Keep radii inside thin regions so edge effects stay bounded.
        max_radius = min(max_radius, 0.45 * min(width, height))
        distance_cap = min(distance_cap, 0.45 * min(width, height))
    floor = (
        float(resolution_floor_nm)
        if resolution_floor_nm is not None and resolution_floor_nm > 0.0
        else None
    )
    raw_bin_width = max_radius / 20.0
    bin_width = max(raw_bin_width, floor) if floor is not None else raw_bin_width

    mean_spacing = (
        math.sqrt(area / float(n_points)) if n_points > 0 and area > 0.0 else 0.0
    )
    # The nearest-neighbor distribution is set by point density, not by the pair
    # radius: in sparse fields the typical NN distance exceeds 0.2*char, so a
    # pair-radius cap would clip the histogram exactly where the signal lives.
    # Extend the NN range to ~1.5x the mean spacing (covering the bulk of the
    # Poisson NN tail), never below the pair radius and never past the field.
    nn_max_distance = max_radius
    if mean_spacing > 0.0:
        nn_max_distance = min(max(max_radius, 1.5 * mean_spacing), distance_cap)
    raw_nn_bin_width = nn_max_distance / 20.0
    nn_bin_width = max(raw_nn_bin_width, floor) if floor is not None else raw_nn_bin_width

    # Hard-core radius mirrors the sandbox ratio (1.5 nm in a 100 nm field) but is
    # capped below the mean point spacing so the null model can always place the
    # same number of points without exhausting its attempt limit.
    hard_core_radius = 0.015 * char
    if mean_spacing > 0.0 and n_points > 1:
        hard_core_radius = min(hard_core_radius, 0.5 * mean_spacing)
    return SimpleNamespace(
        pair_bin_width_nm=bin_width,
        pair_max_radius_nm=max_radius,
        nn_bin_width_nm=nn_bin_width,
        nn_max_distance_nm=nn_max_distance,
        cluster_radius_nm=max_radius * 0.1,
        hard_core_radius_nm=hard_core_radius,
        # Feature-conditioned Poisson smooths the measured feature layer with this
        # kernel; mirror the sandbox ratio (4 nm in a 100 nm field).
        feature_kernel_sigma_nm=0.04 * char,
        resolution_floor_nm=floor,
        raw_pair_bin_width_nm=raw_bin_width,
        bin_width_resolution_limited=(
            bool(floor is not None and raw_bin_width < floor)
        ),
    )


def _bond_order_neighbor_radius_nm(
    table: Any,
    *,
    resolution_floor_nm: float | None = None,
) -> float | None:
    try:
        xy_nm = np.asarray(table.xy_nm, dtype=float)
    except Exception:  # noqa: BLE001 - optional stat should not block analysis
        return None
    if xy_nm.ndim != 2 or xy_nm.shape[1] != 2 or len(xy_nm) < 2:
        return None
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(xy_nm).query(xy_nm, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if len(nearest) == 0:
        return None
    radius = 1.35 * float(np.median(nearest))
    if resolution_floor_nm is not None and resolution_floor_nm > 0.0:
        radius = max(radius, float(resolution_floor_nm))
    return radius


def _resolution_status_lines(
    scales: SimpleNamespace,
    *,
    explicit_pair_bin_width_nm: float | None = None,
    explicit_nn_bin_width_nm: float | None = None,
) -> tuple[str, ...]:
    floor = getattr(scales, "resolution_floor_nm", None)
    if floor is None or float(floor) <= 0.0:
        return ()
    lines: list[str] = []
    if _automatic_bins_resolution_limited(
        scales,
        explicit_pair_bin_width_nm=explicit_pair_bin_width_nm,
        explicit_nn_bin_width_nm=explicit_nn_bin_width_nm,
    ):
        lines.append(
            f"Pixel size is {float(floor):g} nm; automatic distance bins were "
            "clamped to this resolution floor."
        )
    for label, value in (
        ("pair", explicit_pair_bin_width_nm),
        ("nearest-neighbor", explicit_nn_bin_width_nm),
    ):
        if value is not None and float(value) < float(floor):
            lines.append(
                f"Warning: explicit {label} bin width {float(value):g} nm is "
                f"below the pixel-size floor ({float(floor):g} nm)."
            )
    return tuple(lines)


def _automatic_bins_resolution_limited(
    scales: SimpleNamespace,
    *,
    explicit_pair_bin_width_nm: float | None = None,
    explicit_nn_bin_width_nm: float | None = None,
) -> bool:
    if not getattr(scales, "bin_width_resolution_limited", False):
        return False
    return explicit_pair_bin_width_nm is None or explicit_nn_bin_width_nm is None


def _with_resolution_metadata(
    spec: Any,
    *,
    scales: SimpleNamespace,
    status_lines: tuple[str, ...],
    resolution_limited: bool | None = None,
) -> Any:
    if not status_lines and getattr(scales, "resolution_floor_nm", None) is None:
        return spec
    metadata = dict(getattr(spec, "metadata", {}) or {})
    floor = getattr(scales, "resolution_floor_nm", None)
    if floor is not None:
        metadata["pixel_resolution_floor_nm"] = float(floor)
    if resolution_limited is None:
        resolution_limited = bool(
            getattr(scales, "bin_width_resolution_limited", False)
        )
    metadata["bin_width_resolution_limited"] = bool(
        resolution_limited
    )
    return replace(
        spec,
        status_lines=tuple(getattr(spec, "status_lines", ()) or ()) + status_lines,
        metadata=metadata,
    )


def _filter_ordering_statistics(spec: Any) -> Any:
    """Drop local-order panels and verdict rows from a view spec.

    Used when ordering statistics are not requested, so the display never shows
    ψ4/ψ6/g(r,θ) as co-equal verdicts in a plain randomness analysis. Verdict
    rows carry the statistic id at index 1.
    """

    panels = tuple(getattr(spec, "panels", ()) or ())
    verdict_rows = tuple(getattr(spec, "verdict_rows", ()) or ())
    kept_panels = tuple(
        p for p in panels if str(getattr(p, "statistic", "")) not in ORDERING_STATISTICS
    )
    kept_rows = tuple(
        row for row in verdict_rows if not (len(row) > 1 and str(row[1]) in ORDERING_STATISTICS)
    )
    if len(kept_panels) == len(panels) and len(kept_rows) == len(verdict_rows):
        return spec
    try:
        return replace(spec, panels=kept_panels, verdict_rows=kept_rows)
    except TypeError:  # pragma: no cover - non-dataclass spec
        return spec


def compare_particle_collection_view_spec(
    *,
    scan: Any,
    point_source: Any | None = None,
    feature_counting_items: Iterable[Any] | None = None,
    roi_or_mask: Any = None,
    image_shape: tuple[int, int] | None = None,
    scan_id: str | None = None,
    feature_layers: Iterable[Any] = (),
    models: Iterable[str] = ("poisson",),
    pair_bin_width_nm: float | None = None,
    pair_max_radius_nm: float | None = None,
    nn_bin_width_nm: float | None = None,
    nn_max_distance_nm: float | None = None,
    cluster_radius_nm: float | None = None,
    n_simulations: int = 19,
    random_seed: int | None = 0,
    include_ordering: bool = False,
) -> Any:
    """Run AdStat for one ProbeFlow point collection.

    This is the ProbeFlow-side contract wrapper for any v1 point population:
    Feature Finder/ROI ``PointSource`` records or raw Feature Counting/template
    records.  The current viewer UI supplies ``point_source``; Feature Counting
    and series callers can use the same API once they can hand their records to
    the viewer layer.

    ``include_ordering`` controls the opt-in local-order statistics (ψ4/ψ6 and
    the angular pair map). When ``False`` (the default) they are neither computed
    nor shown, so a plain randomness analysis is not cluttered with lattice
    verdicts that answer a different question.
    """

    adstat_analysis = _adstat_analysis()
    record = point_set_record(
        dataset_id=scan_id,
        scan=scan,
        point_source=point_source,
        feature_counting_items=feature_counting_items,
        roi_or_mask=roi_or_mask,
        image_shape=image_shape,
    )
    table = record.table
    region = record.region
    try:
        n_points = len(table)
    except TypeError:
        n_points = 0
    resolution_floor_nm = _pixel_resolution_floor_nm(record.calibration)
    explicit_pair_bin_width_nm = pair_bin_width_nm
    explicit_nn_bin_width_nm = nn_bin_width_nm
    scales = _default_statistic_scales(
        region,
        n_points=n_points,
        resolution_floor_nm=resolution_floor_nm,
    )
    if pair_bin_width_nm is None:
        pair_bin_width_nm = scales.pair_bin_width_nm
    if pair_max_radius_nm is None:
        pair_max_radius_nm = scales.pair_max_radius_nm
    if nn_bin_width_nm is None:
        nn_bin_width_nm = scales.nn_bin_width_nm
    if nn_max_distance_nm is None:
        nn_max_distance_nm = scales.nn_max_distance_nm
    if cluster_radius_nm is None:
        cluster_radius_nm = scales.cluster_radius_nm
    converted_layers = feature_layers_to_adstat(
        feature_layers,
        calibration=record.calibration,
    )
    comparison_feature_layer = converted_layers[0] if converted_layers else None
    pair_angle_bin_width_deg = 15.0 if include_ordering else None
    bond_order_neighbor_radius_nm = (
        _bond_order_neighbor_radius_nm(table, resolution_floor_nm=resolution_floor_nm)
        if include_ordering
        else None
    )
    summary = adstat_analysis.summarize_particle_table(
        table,
        region=region,
        feature_layers=converted_layers,
        cluster_radius_nm=cluster_radius_nm,
        pair_bin_width_nm=pair_bin_width_nm,
        pair_max_radius_nm=pair_max_radius_nm,
        pair_angle_bin_width_deg=pair_angle_bin_width_deg,
        pair_reference_angle_deg=0.0,
    )
    comparisons = adstat_analysis.compare_particle_table(
        table,
        region=region,
        models=tuple(models),
        pair_bin_width_nm=pair_bin_width_nm,
        pair_max_radius_nm=pair_max_radius_nm,
        pair_angle_bin_width_deg=pair_angle_bin_width_deg,
        pair_reference_angle_deg=0.0,
        nn_bin_width_nm=nn_bin_width_nm,
        nn_max_distance_nm=nn_max_distance_nm,
        cluster_radius_nm=cluster_radius_nm,
        bond_order_neighbor_radius_nm=bond_order_neighbor_radius_nm,
        hard_core_radius_nm=scales.hard_core_radius_nm,
        poisson_seed=random_seed,
        hard_core_seed=random_seed,
        feature_layer=comparison_feature_layer,
        feature_kernel_sigma_nm=scales.feature_kernel_sigma_nm,
        feature_seed=random_seed,
        n_simulations=int(n_simulations),
    )
    active_model = comparisons[0].model_name if comparisons else next(iter(models), None)
    spec = workbench_view_spec(
        table=table,
        region=region,
        summary=summary,
        comparisons=comparisons,
        active_model=active_model,
        feature_xy_nm=_first_point_feature_xy_nm(converted_layers),
    )
    spec = _with_resolution_metadata(
        spec,
        scales=scales,
        status_lines=_resolution_status_lines(
            scales,
            explicit_pair_bin_width_nm=explicit_pair_bin_width_nm,
            explicit_nn_bin_width_nm=explicit_nn_bin_width_nm,
        )
        + _region_filter_status_lines(table),
        resolution_limited=_automatic_bins_resolution_limited(
            scales,
            explicit_pair_bin_width_nm=explicit_pair_bin_width_nm,
            explicit_nn_bin_width_nm=explicit_nn_bin_width_nm,
        ),
    )
    return spec if include_ordering else _filter_ordering_statistics(spec)


def compare_point_source_view_spec(
    point_source: Any,
    *,
    scan: Any,
    roi_or_mask: Any = None,
    image_shape: tuple[int, int] | None = None,
    scan_id: str | None = None,
    feature_layers: Iterable[Any] = (),
    models: Iterable[str] = ("poisson",),
    pair_bin_width_nm: float | None = None,
    pair_max_radius_nm: float | None = None,
    nn_bin_width_nm: float | None = None,
    nn_max_distance_nm: float | None = None,
    cluster_radius_nm: float | None = None,
    n_simulations: int = 19,
    random_seed: int | None = 0,
    include_ordering: bool = False,
) -> Any:
    """Run the direct AdStat PointSource path and return a Qt-renderable view spec."""

    return compare_particle_collection_view_spec(
        scan=scan,
        point_source=point_source,
        roi_or_mask=roi_or_mask,
        image_shape=image_shape,
        scan_id=scan_id,
        feature_layers=feature_layers,
        models=models,
        pair_bin_width_nm=pair_bin_width_nm,
        pair_max_radius_nm=pair_max_radius_nm,
        nn_bin_width_nm=nn_bin_width_nm,
        nn_max_distance_nm=nn_max_distance_nm,
        cluster_radius_nm=cluster_radius_nm,
        n_simulations=n_simulations,
        random_seed=random_seed,
        include_ordering=include_ordering,
    )


def compare_point_set_record_view_spec(
    record: Any,
    *,
    models: Iterable[str] = ("poisson",),
    feature_layers: Iterable[Any] = (),
    n_simulations: int = 60,
    random_seed: int | None = 0,
    include_ordering: bool = False,
) -> Any:
    """Single-image workbench view spec from a prebuilt point-set record.

    Uses the record's own table/region/calibration (so a saved set is analysed
    with the image it came from, not the current viewer image), deriving statistic
    scales from its region like the live single-image path. ``feature_layers`` are
    independently-measured layers used by the measured-feature Poisson model.
    ``include_ordering`` adds the opt-in ψ4/ψ6/g(r,θ) local-order statistics.
    """

    adstat_analysis = _adstat_analysis()
    table = record.table
    region = record.region
    try:
        n_points = len(table)
    except TypeError:
        n_points = 0
    resolution_floor_nm = _pixel_resolution_floor_nm(record.calibration)
    scales = _default_statistic_scales(
        region,
        n_points=n_points,
        resolution_floor_nm=resolution_floor_nm,
    )
    converted_layers = feature_layers_to_adstat(
        feature_layers, calibration=record.calibration
    )
    comparison_feature_layer = converted_layers[0] if converted_layers else None
    pair_angle_bin_width_deg = 15.0 if include_ordering else None
    bond_order_neighbor_radius_nm = (
        _bond_order_neighbor_radius_nm(table, resolution_floor_nm=resolution_floor_nm)
        if include_ordering
        else None
    )
    summary = adstat_analysis.summarize_particle_table(
        table,
        region=region,
        feature_layers=converted_layers,
        cluster_radius_nm=scales.cluster_radius_nm,
        pair_bin_width_nm=scales.pair_bin_width_nm,
        pair_max_radius_nm=scales.pair_max_radius_nm,
        pair_angle_bin_width_deg=pair_angle_bin_width_deg,
        pair_reference_angle_deg=0.0,
    )
    comparisons = adstat_analysis.compare_particle_table(
        table,
        region=region,
        models=tuple(models),
        pair_bin_width_nm=scales.pair_bin_width_nm,
        pair_max_radius_nm=scales.pair_max_radius_nm,
        pair_angle_bin_width_deg=pair_angle_bin_width_deg,
        pair_reference_angle_deg=0.0,
        nn_bin_width_nm=scales.nn_bin_width_nm,
        nn_max_distance_nm=scales.nn_max_distance_nm,
        cluster_radius_nm=scales.cluster_radius_nm,
        bond_order_neighbor_radius_nm=bond_order_neighbor_radius_nm,
        hard_core_radius_nm=scales.hard_core_radius_nm,
        hard_core_seed=random_seed,
        feature_layer=comparison_feature_layer,
        feature_kernel_sigma_nm=scales.feature_kernel_sigma_nm,
        poisson_seed=random_seed,
        feature_seed=random_seed,
        n_simulations=int(n_simulations),
    )
    active_model = comparisons[0].model_name if comparisons else next(iter(models), None)
    spec = workbench_view_spec(
        table=table,
        region=region,
        summary=summary,
        comparisons=comparisons,
        active_model=active_model,
        feature_xy_nm=_first_point_feature_xy_nm(converted_layers),
    )
    spec = _with_resolution_metadata(
        spec,
        scales=scales,
        status_lines=_resolution_status_lines(scales) + _region_filter_status_lines(table),
    )
    return spec if include_ordering else _filter_ordering_statistics(spec)


def compare_point_set_records_view_spec(
    records: Iterable[Any],
    *,
    models: Iterable[str] = ("poisson",),
    n_simulations: int = 60,
    random_seed: int | None = 0,
) -> Any:
    """Pool several saved point-set records as replicates → one combined view spec.

    Each ``record`` is an :class:`AdStatPointSetRecord` (e.g. from
    ``FeatureSet.to_point_set_record``). Statistic scales are derived from the
    first record's region (shared across all images so AdStat can pool them).
    Distance ranges use the smallest point count (the sparsest image has the
    largest spacings to cover) while the hard-core radius cap uses the largest
    point count (the densest image constrains what the null can still place).
    Renders through AdStat's coverage-series view spec, which exposes
    per-statistic pooled panels plus the combined verdict rows.
    """

    record_list = list(records)
    if not record_list:
        raise ValueError("compare_point_set_records_view_spec requires at least one record")
    adstat_analysis = _adstat_analysis()
    viz_spec = _adstat_viz_spec()

    counts = [len(rec.table) for rec in record_list]
    floors = [
        _pixel_resolution_floor_nm(rec.calibration)
        for rec in record_list
    ]
    finite_floors = [floor for floor in floors if floor is not None and floor > 0.0]
    resolution_floor_nm = max(finite_floors) if finite_floors else None
    scales = _default_statistic_scales(
        record_list[0].region,
        n_points=min(counts),
        resolution_floor_nm=resolution_floor_nm,
    )
    # The hard-core cap must stay placeable for the densest image (smallest
    # mean spacing), so it is derived from the largest count — unlike the
    # NN distance range above, which must cover the sparsest image.
    dense_scales = _default_statistic_scales(
        record_list[0].region,
        n_points=max(counts),
        resolution_floor_nm=resolution_floor_nm,
    )
    # Statistic scales come from the first record's region; warn if the pooled
    # images are heterogeneous enough that the shared scales may not suit them.
    pool_status_lines: list[str] = []
    n_filtered_images = 0
    n_filtered_points = 0
    for rec in record_list:
        metadata = getattr(rec.table, "metadata", {}) or {}
        if metadata.get("region_filtered"):
            n_filtered_images += 1
            n_filtered_points += int(metadata.get("n_points_total", 0)) - int(
                metadata.get("n_points_in_region", 0)
            )
    if n_filtered_points > 0:
        pool_status_lines.append(
            f"{n_filtered_points} points outside their analysis regions were "
            f"excluded across {n_filtered_images} image(s)."
        )
    areas = [_region_area_nm2(rec.region) for rec in record_list]
    finite_areas = [area for area in areas if area > 0.0]
    if len(finite_areas) >= 2 and max(finite_areas) > 1.5 * min(finite_areas):
        pool_status_lines.append(
            "Pooled images span different analysis-region areas "
            f"({min(finite_areas):.0f}-{max(finite_areas):.0f} nm^2); statistic "
            "scales are taken from the first image and may not suit all of them."
        )
    if len(finite_floors) >= 2 and max(finite_floors) > 1.5 * min(finite_floors):
        pool_status_lines.append(
            "Pooled images have different pixel sizes; distance bins use the "
            f"coarsest resolution floor ({max(finite_floors):g} nm)."
        )
    overrides = adstat_analysis.SeriesAnalysisOverrides(
        pair_bin_width_nm=scales.pair_bin_width_nm,
        pair_max_radius_nm=scales.pair_max_radius_nm,
        nn_bin_width_nm=scales.nn_bin_width_nm,
        nn_max_distance_nm=scales.nn_max_distance_nm,
        cluster_radius_nm=scales.cluster_radius_nm,
        hard_core_radius_nm=dense_scales.hard_core_radius_nm,
    )
    items = [
        (rec.table, rec.region, rec.dataset_id or f"image_{index + 1}")
        for index, rec in enumerate(record_list)
    ]
    result = adstat_analysis.pool_particle_tables(
        items,
        models=tuple(models),
        n_simulations=int(n_simulations),
        random_seed=random_seed,
        overrides=overrides,
    )
    spec = viz_spec.view_spec_for(result)
    return _with_resolution_metadata(
        spec,
        scales=scales,
        status_lines=_resolution_status_lines(scales) + tuple(pool_status_lines),
    )


def adstat_sandbox_context() -> Any:
    """Return AdStat's GUI-free sandbox symbols for ProbeFlow Qt shells."""

    sandbox = _adstat_sandbox()
    return SimpleNamespace(
        SandboxConfig=sandbox.SandboxConfig,
        SandboxState=sandbox.SandboxState,
        SANDBOX_PATTERNS=tuple(sandbox.SANDBOX_PATTERNS),
        SANDBOX_MODELS=tuple(sandbox.SANDBOX_MODELS),
        ORDERED_ISLAND_LATTICES=tuple(
            getattr(sandbox, "ORDERED_ISLAND_LATTICES", ("triangular", "square"))
        ),
        ORDERED_ISLAND_BACKGROUNDS=tuple(
            getattr(
                sandbox,
                "ORDERED_ISLAND_BACKGROUNDS",
                ("none", "random", "clustered"),
            )
        ),
    )


def adstat_sandbox_preview(config: Any, *, active_model: str | None = None) -> Any:
    """Return lightweight generated-data preview points without full comparison."""

    sandbox = _adstat_sandbox()
    region = sandbox.RectangularRegion(
        width_nm=float(config.width_nm),
        height_nm=float(config.height_nm),
    )
    feature_layer = sandbox.synthetic_feature_layer(config)
    table = sandbox.generate_demo_pattern(
        config,
        region=region,
        feature_layer=feature_layer,
        rng=np.random.default_rng(int(config.seed)),
    )
    simulated = None
    if active_model and hasattr(sandbox, "_simulate_overlay"):
        try:
            simulated = sandbox._simulate_overlay(  # noqa: SLF001 - adapter shields GUI from AdStat internals
                config,
                region,
                feature_layer,
                active_model,
                len(table),
            )
        except Exception:  # noqa: BLE001 - preview overlay is best-effort, never block the preview
            simulated = None
    return SimpleNamespace(
        xy_nm=np.asarray(table.xy_nm, dtype=float),
        width_nm=float(config.width_nm),
        height_nm=float(config.height_nm),
        feature_xy_nm=np.asarray(feature_layer.xy_nm, dtype=float),
        simulated_xy_nm=(
            None if simulated is None else np.asarray(simulated.xy_nm, dtype=float)
        ),
    )


def adstat_sandbox_state(config: Any | None = None) -> Any:
    """Create a default AdStat sandbox state lazily."""

    context = adstat_sandbox_context()
    return context.SandboxState(config)


def adstat_sandbox_view_spec(state: Any, *, include_ordering: bool = False) -> Any:
    """Return a ProbeFlow-renderable sandbox view spec.

    AdStat's native sandbox view spec carries the status, verdict rows, and
    active-model panels. ProbeFlow prepends a real-space panel so the teaching
    dialog always has a visible point pattern once a sandbox run exists.

    The sandbox backend always computes every statistic, so when
    ``include_ordering`` is ``False`` (the default) the ψ4/ψ6/g(r,θ) panels and
    verdict rows are filtered out for display, matching the opt-in behaviour of
    the real-data path.
    """

    viz_spec = _adstat_viz_spec()
    spec = viz_spec.view_spec_for(state)
    if not include_ordering:
        spec = _filter_ordering_statistics(spec)
    result = getattr(state, "result", None)
    if result is None:
        return spec

    feature_layer = getattr(result, "feature_layer", None)
    feature_xy_nm = getattr(feature_layer, "xy_nm", None)
    table = getattr(result, "table", None)
    region = getattr(result, "region", None)
    observed = getattr(table, "xy_nm", None)
    panel = viz_spec.PanelSpec(
        statistic="sandbox_realspace",
        title="generated point pattern",
        kind="realspace",
        x_label="x (nm)",
        y_label="y (nm)",
        reference_line=None,
        glossary_term="roi_mask",
        verdict_label="",
        verdict_color=viz_spec.verdict_color(""),
        global_p=float("nan"),
        observed=observed,
        caption_lines=(
            "TEST MODE - GENERATED DATA",
            f"pattern: {getattr(getattr(state, 'config', None), 'pattern', '')}",
            f"seed: {getattr(getattr(state, 'config', None), 'seed', '')}",
            f"simulations: {getattr(getattr(state, 'config', None), 'n_simulations', '')}",
        ),
        metadata={
            "data_mode": "sandbox",
            "table": table,
            "region": region,
            "simulated": getattr(result, "simulated", None),
            "feature_xy_nm": feature_xy_nm,
            "overlays": ("simulated", "features"),
            "particle_count": len(table) if table is not None else 0,
        },
    )
    return replace(spec, panels=(panel, *tuple(spec.panels)))


def _first_point_feature_xy_nm(layers: Iterable[Any]) -> np.ndarray | None:
    for layer in layers:
        xy_nm = _field(layer, "xy_nm", None)
        if xy_nm is not None:
            return _xy_array(xy_nm, name="feature_xy_nm")
    return None


def _adstat() -> Any:
    try:
        import adstat
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "ProbeFlow's AdStat adapter requires the optional 'adstat' package. "
            "Install AdStat or run ProbeFlow with the AdStat source tree on PYTHONPATH."
        ) from exc
    return adstat


def _adstat_analysis() -> Any:
    try:
        from adstat import analysis
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "ProbeFlow's AdStat comparison path requires adstat.analysis."
        ) from exc
    return analysis


def _adstat_viz_spec() -> Any:
    try:
        from adstat import viz_spec
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "ProbeFlow's AdStat view-spec path requires adstat.viz_spec."
        ) from exc
    return viz_spec


def _adstat_sandbox() -> Any:
    try:
        from adstat.analysis import sandbox
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "ProbeFlow's AdStat sandbox requires adstat.analysis.sandbox."
        ) from exc
    return sandbox


def _scan_dims(scan: Any) -> tuple[int, int]:
    dims = _field(scan, "dims", None)
    if callable(dims):
        dims = dims()
    if dims is not None:
        return int(dims[0]), int(dims[1])
    planes = _field(scan, "planes", None)
    if planes:
        height_px, width_px = np.asarray(planes[0]).shape[:2]
        return int(width_px), int(height_px)
    raise ValueError("scan must provide dims or at least one plane")


def _scan_range_m(scan: Any) -> tuple[float, float]:
    scan_range = _field(scan, "scan_range_m", None)
    if scan_range is None:
        raise ValueError("scan must provide scan_range_m")
    return float(scan_range[0]), float(scan_range[1])


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _xy_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    return arr


def _point_id(metadata: Mapping[str, object], index: int) -> str | None:
    ids = metadata.get("point_ids")
    if ids is None:
        return None
    try:
        return str(list(ids)[index])
    except Exception:
        return None


def _feature_counting_kind(item: Any) -> str:
    if _field(item, "centroid_x_m", None) is not None:
        return "particles"
    if _field(item, "x_m", None) is not None:
        return "detections"
    return str(_field(item, "kind", "feature_counting"))


def _feature_counting_xy_nm(item: Any) -> tuple[float, float]:
    if _field(item, "centroid_x_m", None) is not None:
        return (
            float(_field(item, "centroid_x_m")) * 1e9,
            float(_field(item, "centroid_y_m")) * 1e9,
        )
    if _field(item, "x_m", None) is not None:
        return float(_field(item, "x_m")) * 1e9, float(_field(item, "y_m")) * 1e9
    if _field(item, "x_nm", None) is not None:
        return float(_field(item, "x_nm")), float(_field(item, "y_nm"))
    raise ValueError("feature counting item needs metre or nanometre coordinates")


def _feature_counting_xy_px(
    item: Any,
    *,
    calibration: Any | None,
) -> tuple[float | None, float | None]:
    x_px = _optional_float(_field(item, "x_px", None))
    y_px = _optional_float(_field(item, "y_px", None))
    if x_px is not None and y_px is not None:
        return x_px, y_px
    if calibration is None:
        return None, None
    x_nm, y_nm = _feature_counting_xy_nm(item)
    xy_px = calibration.nm_to_pixel(np.asarray([[x_nm, y_nm]], dtype=float))[0]
    return float(xy_px[0]), float(xy_px[1])


def _feature_counting_id(item: Any, fallback_index: int) -> str:
    for field in ("id", "point_id", "particle_id"):
        value = _field(item, field, None)
        if value not in (None, ""):
            return str(value)
    value = _field(item, "index", fallback_index)
    return str(value)


def _feature_counting_confidence(item: Any) -> float | None:
    for field in ("confidence", "correlation", "similarity"):
        value = _optional_float(_field(item, field, None))
        if value is not None:
            return value
    return None


def _feature_counting_metadata(
    item: Any,
    source_kind: str,
    status: object,
) -> dict[str, object]:
    metadata = {
        "probeflow_source": "feature_counting",
        "probeflow_source_type": source_kind,
        "probeflow_status": status,
    }
    for field in (
        "index",
        "bbox_m",
        "bbox_px",
        "mean_height",
        "max_height",
        "min_height",
        "n_pixels",
        "sharpness",
        "correlation",
        "local_height",
    ):
        value = _field(item, field, None)
        if value is not None:
            metadata[field] = value
    return metadata


def _mask_from_roi_or_mask(
    roi_or_mask: Any,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    if roi_or_mask is None:
        return None
    if isinstance(roi_or_mask, np.ndarray) or _looks_like_mask(roi_or_mask):
        mask = np.asarray(roi_or_mask, dtype=bool)
        if mask.shape != tuple(image_shape):
            raise ValueError(
                f"mask shape must match image_shape {tuple(image_shape)}, got {mask.shape}"
            )
        return mask
    if _field(roi_or_mask, "kind", None) not in {
        "rectangle",
        "ellipse",
        "polygon",
        "freehand",
        "multipolygon",
    }:
        return None
    try:
        mask = np.asarray(roi_or_mask.to_mask(tuple(image_shape)), dtype=bool)
    except Exception as exc:
        raise ValueError("could not rasterize ProbeFlow ROI for AdStat") from exc
    if not mask.any():
        raise ValueError("analysis ROI mask must contain at least one allowed pixel")
    return mask


def _looks_like_mask(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and np.asarray(value).ndim == 2


def _roi_label(roi_or_mask: Any) -> str | None:
    if roi_or_mask is None or isinstance(roi_or_mask, np.ndarray) or _looks_like_mask(roi_or_mask):
        return None
    return _optional_str(_field(roi_or_mask, "name", _field(roi_or_mask, "id", None)))


def _points_layer_xy_nm(layer: Any, calibration: Any) -> np.ndarray:
    if _field(layer, "xy_nm", None) is not None:
        return _xy_array(_field(layer, "xy_nm"), name="xy_nm")
    if _field(layer, "points_nm", None) is not None:
        return _xy_array(_field(layer, "points_nm"), name="points_nm")
    points = _field(layer, "points_px", None)
    if points is None:
        points = _field(layer, "points", None)
    if points is None:
        raise ValueError("point feature layer requires points_px or xy_nm")
    xy_px = np.asarray(
        [[_field(point, "x_px"), _field(point, "y_px")] for point in points],
        dtype=float,
    )
    return calibration.pixel_to_nm(xy_px)


def _line_layer_segments_nm(layer: Any, calibration: Any) -> np.ndarray:
    if _field(layer, "segments_nm", None) is not None:
        return np.asarray(_field(layer, "segments_nm"), dtype=float)
    segments = _field(layer, "segments_px", None)
    if segments is None:
        segments = _field(layer, "segments", None)
    if segments is None:
        raise ValueError("line feature layer requires segments_px or segments_nm")
    converted = []
    for segment in segments:
        endpoints_px = np.asarray(
            [
                [_field(segment, "x1_px"), _field(segment, "y1_px")],
                [_field(segment, "x2_px"), _field(segment, "y2_px")],
            ],
            dtype=float,
        )
        converted.append(calibration.pixel_to_nm(endpoints_px))
    return np.asarray(converted, dtype=float)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
