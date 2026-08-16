"""Contracts for the kernel-free processing-operation catalog."""

from __future__ import annotations

import pytest

from probeflow.core.operations import OperationCatalog, OperationSpec
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
