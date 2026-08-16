"""Contracts for the kernel-free processing-operation catalog."""

from __future__ import annotations

import pytest

from probeflow.core.operation_specs import BUILTIN_OPERATIONS
from probeflow.core.operations import OperationCatalog, OperationSpec
from probeflow.core.operations.frequency import FREQUENCY_OPERATION_SPECS
from probeflow.core.operations.geometry import GEOMETRY_OPERATION_SPECS
from probeflow.core.operations.scoped import SCOPED_OPERATION_SPECS
from probeflow.core.operations.spatial import SPATIAL_OPERATION_SPECS


def test_operation_spec_copies_defaults_and_preserves_unknown_parameters():
    defaults = {"sigma": 1.0, "points": [1, 2]}
    spec = OperationSpec("smooth", "spatial", default_params=defaults)
    defaults["points"].append(3)

    first = spec.params_with_defaults({"future_parameter": True})
    first["points"].append(4)
    second = spec.params_with_defaults(None)

    assert second == {"sigma": 1.0, "points": [1, 2]}
    assert first["future_parameter"] is True


def test_catalog_keeps_canonical_lookup_separate_from_alias_resolution():
    rotate = OperationSpec(
        "rotate_90_cw",
        "geometry",
        aliases=frozenset({"rot90_cw"}),
        shape_policy="swap_axes",
        range_policy="swap_axes",
    )
    catalog = OperationCatalog((rotate,))

    assert catalog.by_id("rot90_cw") is None
    assert catalog.resolve("rot90_cw") is rotate
    assert catalog.resolve("rotate_90_cw") is rotate


def test_catalog_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="globally unique"):
        OperationCatalog(
            (
                OperationSpec("first", "spatial", aliases=frozenset({"old"})),
                OperationSpec("second", "spatial", aliases=frozenset({"old"})),
            )
        )


def test_spatial_specs_capture_existing_defaults_and_scope_rules():
    catalog = OperationCatalog(SPATIAL_OPERATION_SPECS)

    assert catalog.operation_ids == {
        "remove_bad_lines",
        "align_rows",
        "plane_bg",
        "stm_line_bg",
        "stm_background",
        "facet_level",
        "smooth",
        "median_smooth",
        "gaussian_high_pass",
        "edge_detect",
        "remove_spots_auto",
        "image_threshold",
        "quantize_bit_depth",
    }
    assert catalog.roi_eligible_ids == {
        "smooth",
        "median_smooth",
        "gaussian_high_pass",
        "edge_detect",
    }
    assert catalog.by_id("plane_bg").default_params == {
        "order": 1,
        "step_tolerance": False,
    }
    assert catalog.by_id("facet_level").calibration_inputs == {
        "pixel_size_x_m",
        "pixel_size_y_m",
    }


def test_frequency_specs_capture_existing_defaults_and_scope_rules():
    catalog = OperationCatalog(FREQUENCY_OPERATION_SPECS)

    assert catalog.operation_ids == {
        "fourier_filter",
        "fft_soft_border",
        "periodic_notch_filter",
        "mains_pickup_suppression",
        "inverse_fft_filter",
        "symmetrize_fft",
    }
    assert catalog.roi_eligible_ids == {"fourier_filter", "fft_soft_border"}
    assert catalog.by_id("inverse_fft_filter").params_with_defaults(None) == {
        "selections": [],
        "mode": "remove_selected",
        "conjugate_symmetric": True,
        "soft_px": 0.0,
    }


def test_scoped_specs_capture_wrappers_and_multi_input_rules():
    catalog = OperationCatalog(SCOPED_OPERATION_SPECS)

    assert catalog.operation_ids == {
        "arithmetic",
        "set_zero_point",
        "set_zero_plane",
        "roi",
        "mask",
        "interpolate_masked",
    }
    assert catalog.roi_eligible_ids == {"arithmetic"}
    assert catalog.by_id("roi").handler_key == "scope_roi"
    assert catalog.by_id("mask").handler_key == "scope_mask"
    assert catalog.by_id("arithmetic").default_params["operation"] == "add"


def test_geometry_specs_capture_aliases_shape_and_range_rules():
    catalog = OperationCatalog(GEOMETRY_OPERATION_SPECS)

    assert catalog.operation_ids == {
        "linear_undistort",
        "affine_lattice_correction",
        "flip_horizontal",
        "flip_vertical",
        "rotate_90_cw",
        "rotate_180",
        "rotate_270_cw",
        "rotate_arbitrary",
        "shear",
        "scale_image",
        "crop",
    }
    assert catalog.resolve("rot90_cw").operation_id == "rotate_90_cw"
    assert catalog.by_id("rotate_90_cw").shape_policy == "swap_axes"
    assert catalog.by_id("rotate_90_cw").range_policy == "swap_axes"
    assert catalog.by_id("scale_image").range_policy == "preserve"
    assert catalog.by_id("crop").range_policy == "scale_with_shape"


def test_builtin_catalog_matches_the_existing_processing_vocabulary():
    from probeflow.core.processing_state import _ROI_ELIGIBLE_OPS, _SUPPORTED_OPS

    assert BUILTIN_OPERATIONS.operation_ids == _SUPPORTED_OPS
    assert BUILTIN_OPERATIONS.roi_eligible_ids == _ROI_ELIGIBLE_OPS
    assert len(BUILTIN_OPERATIONS.specs) == 36


def test_builtin_contract_contains_no_numerical_handlers():
    assert all(isinstance(spec.handler_key, str) for spec in BUILTIN_OPERATIONS.specs)
    assert not any(
        callable(value)
        for spec in BUILTIN_OPERATIONS.specs
        for value in spec.default_params.values()
    )
