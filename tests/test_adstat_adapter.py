"""Tests for the ProbeFlow-to-AdStat adapter seam."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("adstat")

from probeflow.analysis.adstat_adapter import (  # noqa: E402
    adstat_sandbox_context,
    adstat_sandbox_preview,
    adstat_sandbox_state,
    adstat_sandbox_view_spec,
    compare_particle_collection_view_spec,
    compare_point_set_records_view_spec,
    compare_point_source_view_spec,
    feature_counting_to_particle_table,
    feature_layers_to_adstat,
    point_set_record,
    point_source_to_particle_table,
    roi_to_region,
    scan_calibration_to_adstat,
)
from probeflow.analysis.features import Detection, Particle as FeatureParticle  # noqa: E402
from probeflow.analysis.pair_correlation import compute_pair_correlation  # noqa: E402
from probeflow.core.roi import ROI  # noqa: E402
from probeflow.gui.roi_context import PointSource  # noqa: E402
from probeflow.gui.viewer.tool_launch import adstat_workbench_launch_context  # noqa: E402


class _Scan:
    scan_range_m = (6e-9, 10e-9)
    dims = (12, 8)


class _SquareScan:
    scan_range_m = (20e-9, 20e-9)
    dims = (10, 10)


class _CoarseAnisotropicScan:
    scan_range_m = (40e-9, 80e-9)
    dims = (20, 20)


def _point_source() -> PointSource:
    points_px = np.array(
        [[1.0, 1.0], [4.0, 1.0], [9.0, 5.0], [2.0, 5.0]],
        dtype=float,
    )
    return PointSource(
        label="Feature result",
        source_type="feature_finder",
        points_px=points_px,
        points_m=points_px * np.array([0.5e-9, 1.25e-9]),
        metadata={
            "detection_mode": "minima",
            "threshold_mode": "below",
            "point_count": 4,
        },
    )


def _feature_counting_items():
    return [
        FeatureParticle(
            index=10,
            centroid_x_m=2.0e-9,
            centroid_y_m=2.5e-9,
            area_m2=2.0e-18,
            area_nm2=2.0,
            bbox_m=(1.0e-9, 2.0e-9, 3.0e-9, 4.0e-9),
            bbox_px=(2, 1, 6, 3),
            mean_height=1.0,
            max_height=2.0,
            min_height=0.5,
            n_pixels=4,
            orientation_deg=30.0,
            sharpness=12.0,
        ),
        Detection(
            index=11,
            x_m=3.0e-9,
            y_m=5.0e-9,
            x_px=6,
            y_px=4,
            correlation=0.82,
            local_height=1.5,
        ),
    ]


def test_scan_calibration_preserves_anisotropic_probe_flow_pixels() -> None:
    calibration = scan_calibration_to_adstat(_Scan())

    assert calibration.pixel_size_x_nm == pytest.approx(0.5)
    assert calibration.pixel_size_y_nm == pytest.approx(1.25)
    assert calibration.width_px == 12
    assert calibration.height_px == 8


def test_point_source_to_particle_table_converts_feature_finder_points() -> None:
    table = point_source_to_particle_table(_point_source(), scan_id="scan-a")

    assert len(table) == 4
    np.testing.assert_allclose(
        table.xy_nm,
        [[0.5, 1.25], [2.0, 1.25], [4.5, 6.25], [1.0, 6.25]],
    )
    assert [particle.x_px for particle in table.particles] == [1.0, 4.0, 9.0, 2.0]
    assert table.metadata["scan_id"] == "scan-a"
    assert table.metadata["probeflow_point_source_type"] == "feature_finder"
    assert table.metadata["detection_mode"] == "minima"


def test_feature_counting_particles_and_template_detections_convert_to_table() -> None:
    calibration = scan_calibration_to_adstat(_Scan())

    table = feature_counting_to_particle_table(
        _feature_counting_items(),
        scan_id="scan-a",
        calibration=calibration,
    )

    assert len(table) == 2
    np.testing.assert_allclose(table.xy_nm, [[2.0, 2.5], [3.0, 5.0]])
    assert table.particles[0].x_px == pytest.approx(4.0)
    assert table.particles[0].y_px == pytest.approx(2.0)
    assert table.particles[0].area_nm2 == pytest.approx(2.0)
    assert table.particles[0].orientation_deg == pytest.approx(30.0)
    assert table.particles[1].x_px == pytest.approx(6.0)
    assert table.particles[1].confidence == pytest.approx(0.82)
    assert table.metadata["probeflow_point_source_type"] == "detections,particles"


def test_roi_to_region_uses_area_roi_mask_or_full_image_fallback() -> None:
    roi = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 3})
    masked = roi_to_region(roi, scan=_Scan(), image_shape=(8, 12))
    full = roi_to_region(None, scan=_Scan(), image_shape=(8, 12))

    assert masked.boundary_condition == "irregular_mask"
    assert masked.area_nm2 == pytest.approx(6 * 0.5 * 1.25)
    assert full.boundary_condition == "finite_hard_boundary"
    assert full.area_nm2 == pytest.approx(60.0)


def test_feature_layers_to_adstat_converts_independent_points_and_lines() -> None:
    calibration = scan_calibration_to_adstat(_Scan())
    layers = [
        {
            "name": "defects",
            "kind": "points",
            "feature_type": "defect",
            "provenance": {
                "source": "manual_defect_marks",
                "measured_independently": True,
                "derived_from_particles": False,
            },
            "points_px": [
                {"id": "d1", "x_px": 3.0, "y_px": 2.0},
                {"id": "d2", "x_px": 10.0, "y_px": 5.0},
            ],
        },
        {
            "name": "steps",
            "kind": "lines",
            "feature_type": "step",
            "provenance": {
                "source": "manual_step_trace",
                "measured_independently": True,
                "derived_from_particles": False,
            },
            "segments_px": [
                {"id": "s1", "x1_px": 0.0, "y1_px": 3.0, "x2_px": 11.0, "y2_px": 3.0},
            ],
        },
    ]

    converted = feature_layers_to_adstat(layers, calibration=calibration)

    assert [layer.name for layer in converted] == ["defects", "steps"]
    np.testing.assert_allclose(converted[0].xy_nm, [[1.5, 2.5], [5.0, 6.25]])
    np.testing.assert_allclose(
        converted[1].segments_nm,
        [[[0.0, 3.75], [5.5, 3.75]]],
    )


def test_feature_layers_to_adstat_rejects_particle_derived_layers() -> None:
    calibration = scan_calibration_to_adstat(_Scan())
    layer = {
        "name": "bad",
        "kind": "points",
        "feature_type": "derived",
        "provenance": {
            "source": "centroids",
            "measured_independently": True,
            "derived_from_particles": True,
        },
        "points_px": [{"x_px": 1.0, "y_px": 2.0}],
    }

    with pytest.raises(ValueError, match="independent"):
        feature_layers_to_adstat([layer], calibration=calibration)


def test_point_set_record_keeps_series_metadata_and_region() -> None:
    record = point_set_record(
        dataset_id="cov_0p1_img01",
        scan=_Scan(),
        point_source=_point_source(),
        roi_or_mask=np.ones((8, 12), dtype=bool),
        image_shape=(8, 12),
        series_value=0.1,
        series_unit="ML",
        series_label="0.1 ML",
    )

    assert record.dataset_id == "cov_0p1_img01"
    assert record.series_value == pytest.approx(0.1)
    assert record.region.boundary_condition == "irregular_mask"
    assert len(record.table) == 4


def test_sandbox_preview_overlay_is_independent_from_generated_points() -> None:
    context = adstat_sandbox_context()
    config = context.SandboxConfig(pattern="random", n=40, n_simulations=6, seed=3)

    preview = adstat_sandbox_preview(config, active_model="homogeneous_poisson")

    assert preview.simulated_xy_nm is not None
    assert preview.xy_nm.shape == preview.simulated_xy_nm.shape
    assert not np.array_equal(preview.xy_nm, preview.simulated_xy_nm)


def test_compare_point_source_view_spec_returns_qt_renderable_panels() -> None:
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_Scan(),
        roi_or_mask=None,
        image_shape=(8, 12),
        scan_id="scan-a",
        pair_bin_width_nm=1.0,
        pair_max_radius_nm=4.0,
        cluster_radius_nm=2.0,
        n_simulations=4,
        random_seed=17,
    )

    panel_kinds = [panel.kind for panel in spec.panels]
    assert panel_kinds[0] == "realspace"
    assert "curve" in panel_kinds
    assert spec.verdict_rows
    assert spec.metadata["active_model"] == "homogeneous_poisson"


def test_compare_point_source_view_spec_derives_scales_when_unset() -> None:
    # The viewer does not ask the user for nm scales, so it passes None for every
    # radius/bin. AdStat 0.2 rejects an all-None configuration; the adapter must
    # derive scales from the region and still produce the core statistic panels.
    # Local-order statistics are opt-in and excluded by default.
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_Scan(),
        image_shape=(8, 12),
        n_simulations=6,
        random_seed=0,
    )

    statistics = {
        panel.statistic
        for panel in spec.panels
        if getattr(panel, "statistic", None) and panel.statistic != "realspace"
    }
    assert {
        "pair_correlation_g_r",
        "nearest_neighbor_distribution",
        "ripley_l_function",
        "cluster_size_counts",
    } <= statistics
    # ψ4/ψ6/g(r,θ) are opt-in: not present unless include_ordering=True.
    assert statistics.isdisjoint(
        {"pair_correlation_g_r_theta", "bond_order_psi6", "bond_order_psi4"}
    )
    assert spec.verdict_rows


def test_compare_point_source_view_spec_excludes_ordering_by_default() -> None:
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_Scan(),
        image_shape=(8, 12),
        n_simulations=4,
        random_seed=0,
    )
    panels = {panel.statistic for panel in spec.panels}
    assert {"pair_correlation_g_r_theta", "bond_order_psi6", "bond_order_psi4"}.isdisjoint(
        panels
    )
    assert all(
        not (len(row) > 1 and str(row[1]) in {"bond_order_psi6", "bond_order_psi4"})
        for row in (spec.verdict_rows or ())
    )


def test_compare_point_source_view_spec_includes_ordering_when_opted_in() -> None:
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_Scan(),
        image_shape=(8, 12),
        n_simulations=4,
        random_seed=0,
        include_ordering=True,
    )
    panels = {panel.statistic: panel for panel in spec.panels}

    assert panels["pair_correlation_g_r_theta"].kind == "heatmap"
    assert panels["bond_order_psi6"].metadata["neighbor_rule"] == "fixed_radius"
    assert panels["bond_order_psi4"].metadata["neighbor_radius_nm"] >= 1.25


def test_compare_point_source_auto_bins_clamp_to_square_pixel_resolution() -> None:
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_SquareScan(),
        image_shape=(10, 10),
        n_simulations=4,
        random_seed=0,
    )
    pair_panel = next(
        panel for panel in spec.panels if panel.statistic == "pair_correlation_g_r"
    )

    assert spec.metadata["pixel_resolution_floor_nm"] == pytest.approx(2.0)
    assert spec.metadata["bin_width_resolution_limited"] is True
    assert any("Pixel size is 2" in line for line in spec.status_lines)
    assert np.diff(pair_panel.x).min() == pytest.approx(2.0)


def test_compare_point_source_auto_bins_use_larger_non_square_pixel() -> None:
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_CoarseAnisotropicScan(),
        image_shape=(20, 20),
        n_simulations=4,
        random_seed=0,
    )
    pair_panel = next(
        panel for panel in spec.panels if panel.statistic == "pair_correlation_g_r"
    )

    assert spec.metadata["pixel_resolution_floor_nm"] == pytest.approx(4.0)
    assert spec.metadata["bin_width_resolution_limited"] is True
    assert any("Pixel size is 4" in line for line in spec.status_lines)
    assert np.diff(pair_panel.x).min() == pytest.approx(4.0)


def test_compare_point_source_explicit_small_bins_warn_but_are_preserved() -> None:
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_CoarseAnisotropicScan(),
        image_shape=(20, 20),
        pair_bin_width_nm=0.5,
        pair_max_radius_nm=5.0,
        nn_bin_width_nm=0.5,
        nn_max_distance_nm=5.0,
        n_simulations=4,
        random_seed=0,
    )
    pair_panel = next(
        panel for panel in spec.panels if panel.statistic == "pair_correlation_g_r"
    )

    assert spec.metadata["bin_width_resolution_limited"] is False
    assert not any("automatic distance bins" in line for line in spec.status_lines)
    assert any("explicit pair bin width 0.5 nm" in line for line in spec.status_lines)
    assert any(
        "explicit nearest-neighbor bin width 0.5 nm" in line
        for line in spec.status_lines
    )
    assert np.diff(pair_panel.x).min() == pytest.approx(0.5)


def test_compare_point_set_records_use_max_pixel_resolution_floor() -> None:
    fine = point_set_record(
        dataset_id="fine",
        scan=_SquareScan(),
        point_source=_point_source(),
        image_shape=(10, 10),
    )
    coarse = point_set_record(
        dataset_id="coarse",
        scan=_CoarseAnisotropicScan(),
        point_source=_point_source(),
        image_shape=(20, 20),
    )

    spec = compare_point_set_records_view_spec(
        [fine, coarse],
        n_simulations=4,
        random_seed=0,
    )

    assert spec.metadata["pixel_resolution_floor_nm"] == pytest.approx(4.0)
    assert spec.metadata["bin_width_resolution_limited"] is True
    assert any("Pixel size is 4" in line for line in spec.status_lines)
    assert not any(
        panel.statistic
        in {"pair_correlation_g_r_theta", "bond_order_psi6", "bond_order_psi4"}
        for panel in spec.panels
    )


def test_compare_point_source_view_spec_supports_hard_core_model() -> None:
    # Hard-core needs a hard_core_radius_nm; the adapter must derive one so the
    # lesson's hard-core model transfers to real data without manual tuning.
    spec = compare_point_source_view_spec(
        _point_source(),
        scan=_Scan(),
        image_shape=(8, 12),
        models=("hard_core_random",),
        n_simulations=6,
        random_seed=7,
    )

    assert spec.metadata["active_model"] == "hard_core_random"
    assert {
        comparison.ensemble.base_seed
        for comparison in spec.metadata["comparison_results"]
    } == {7}
    assert spec.verdict_rows


def test_compare_particle_collection_view_spec_accepts_feature_counting_records() -> None:
    spec = compare_particle_collection_view_spec(
        scan=_Scan(),
        feature_counting_items=_feature_counting_items(),
        image_shape=(8, 12),
        scan_id="feature-counting-scan",
        feature_layers=[
            {
                "name": "manual defects",
                "kind": "points",
                "feature_type": "defect",
                "provenance": {
                    "source": "manual marks",
                    "measured_independently": True,
                    "derived_from_particles": False,
                },
                "points_px": [{"x_px": 1.0, "y_px": 2.0}],
            }
        ],
        pair_bin_width_nm=1.0,
        pair_max_radius_nm=4.0,
        n_simulations=2,
        random_seed=17,
    )

    realspace = spec.panels[0]
    assert realspace.kind == "realspace"
    assert realspace.metadata["particle_count"] == 2
    np.testing.assert_allclose(realspace.metadata["feature_xy_nm"], [[0.5, 2.5]])
    assert spec.verdict_rows


def test_adstat_workbench_launch_context_uses_existing_point_source_path() -> None:
    context = adstat_workbench_launch_context(
        [_point_source()],
        scan=_Scan(),
        image_shape=(8, 12),
        point_source_label="Feature result",
        pair_bin_width_nm=1.0,
        pair_max_radius_nm=4.0,
        cluster_radius_nm=2.0,
        n_simulations=4,
        random_seed=17,
    )

    assert context.ready
    assert context.point_source_label == "Feature result"
    assert context.view_spec is not None
    assert context.view_spec.panels[0].kind == "realspace"


def test_pair_correlation_migration_documents_intentional_engine_difference() -> None:
    source = _point_source()
    old = compute_pair_correlation(
        source.points_m,
        roi_area_m2=60e-18,
        r_max_m=4e-9,
        bin_width_m=1e-9,
    )

    spec = compare_point_source_view_spec(
        source,
        scan=_Scan(),
        image_shape=(8, 12),
        pair_bin_width_nm=1.0,
        pair_max_radius_nm=4.0,
        n_simulations=4,
        random_seed=17,
    )
    pair_panel = next(
        panel for panel in spec.panels if panel.statistic == "pair_correlation_g_r"
    )

    assert old.edge_correction == "square_window_translational"
    assert pair_panel.reference_line == 1.0
    assert pair_panel.metadata["n_simulations"] == 4
    assert pair_panel.band_low is not None
    assert pair_panel.band_high is not None


def test_adstat_sandbox_state_and_view_spec_are_probe_flow_renderable() -> None:
    context = adstat_sandbox_context()
    state = adstat_sandbox_state(
        context.SandboxConfig(n=8, n_simulations=2, seed=3)
    )

    ready = adstat_sandbox_view_spec(state)
    assert ready.panels == ()
    assert ready.metadata["has_result"] is False

    state.run()
    spec = adstat_sandbox_view_spec(state)

    assert spec.panels[0].kind == "realspace"
    assert spec.panels[0].metadata["data_mode"] == "sandbox"
    assert spec.panels[0].metadata["particle_count"] == 8
    assert spec.panels[0].metadata["simulated"] is not None
    assert spec.panels[0].metadata["feature_xy_nm"] is not None
    assert spec.verdict_rows
    assert spec.metadata["has_result"] is True
