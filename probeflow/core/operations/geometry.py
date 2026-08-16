"""Geometry, resampling, and calibration-range operation contracts."""

from __future__ import annotations

from probeflow.core.operations.model import OperationSpec


GEOMETRY_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        "linear_undistort",
        "geometry",
        display_name="Linear undistort",
        default_params={"shear_x": 0.0, "scale_y": 1.0},
    ),
    OperationSpec(
        "affine_lattice_correction",
        "geometry",
        default_params={
            "expand_canvas": True,
            "interpolation": "bilinear",
            "fill_mode": "nan",
        },
        required_params=frozenset({"matrix"}),
        optional_params=frozenset(
            {
                "fill_value",
                "full_matrix",
                "preserve_orientation",
                "polar_rotation_deg",
                "ideal_a_nm",
                "ideal_b_nm",
                "ideal_angle_deg",
                "measured_a_nm",
                "measured_b_nm",
                "known_structure",
            }
        ),
        shape_policy="resize",
        range_policy="scale_with_shape",
    ),
    OperationSpec("flip_horizontal", "geometry"),
    OperationSpec("flip_vertical", "geometry"),
    OperationSpec(
        "rotate_90_cw",
        "geometry",
        aliases=frozenset({"rot90_cw"}),
        shape_policy="swap_axes",
        range_policy="swap_axes",
    ),
    OperationSpec(
        "rotate_180",
        "geometry",
        aliases=frozenset({"rot180"}),
    ),
    OperationSpec(
        "rotate_270_cw",
        "geometry",
        aliases=frozenset({"rot270_cw"}),
        shape_policy="swap_axes",
        range_policy="swap_axes",
    ),
    OperationSpec(
        "rotate_arbitrary",
        "geometry",
        default_params={"angle_degrees": 0.0, "order": 1},
        shape_policy="resize",
        range_policy="scale_with_shape",
    ),
    OperationSpec(
        "shear",
        "geometry",
        default_params={
            "shear_x": 0.0,
            "shear_y": 0.0,
            "interpolation": "bilinear",
        },
        shape_policy="resize",
        range_policy="scale_with_shape",
    ),
    OperationSpec(
        "scale_image",
        "geometry",
        default_params={"order": 1},
        required_params=frozenset({"new_height", "new_width"}),
        shape_policy="resize",
    ),
    OperationSpec(
        "crop",
        "geometry",
        required_params=frozenset({"x0", "y0", "x1", "y1"}),
        shape_policy="resize",
        range_policy="scale_with_shape",
    ),
)


__all__ = ["GEOMETRY_OPERATION_SPECS"]
