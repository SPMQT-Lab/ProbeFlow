"""Shape-preserving spatial and background operation contracts."""

from __future__ import annotations

from probeflow.core.operations.model import OperationSpec


_LOCAL_SCOPES = frozenset({"global", "roi", "mask"})
_PIXEL_CALIBRATION = frozenset({"pixel_size_x_m", "pixel_size_y_m"})


SPATIAL_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        "remove_bad_lines",
        "spatial",
        display_name="Bad-line removal",
        default_params={
            "threshold_mad": 5.0,
            "method": "mad",
            "polarity": "bright",
            "min_segment_length_px": 2,
            "max_adjacent_bad_lines": 1,
        },
    ),
    OperationSpec(
        "align_rows",
        "spatial",
        display_name="Row alignment",
        default_params={"method": "median"},
    ),
    OperationSpec(
        "plane_bg",
        "spatial",
        display_name="Background subtraction",
        default_params={"order": 1, "step_tolerance": False},
        optional_params=frozenset(
            {
                "fit_rect",
                "fit_roi_id",
                "fit_roi_expr",
                "apply_roi_id",
                "apply_roi_expr",
                "exclude_roi_id",
                "exclude_roi_expr",
            }
        ),
        calibration_inputs=_PIXEL_CALIBRATION,
    ),
    OperationSpec(
        "stm_line_bg",
        "spatial",
        display_name="Background subtraction",
        default_params={"mode": "step_tolerant"},
    ),
    OperationSpec(
        "stm_background",
        "spatial",
        display_name="Background subtraction",
        default_params={
            "fit_region": "whole_image",
            "line_statistic": "median",
            "model": "linear",
            "linear_x_first": False,
            "preserve_level": "median",
        },
        optional_params=frozenset({"blur_length", "jump_threshold", "fit_roi_id"}),
    ),
    OperationSpec(
        "facet_level",
        "spatial",
        display_name="Background subtraction",
        default_params={"threshold_deg": 3.0},
        calibration_inputs=_PIXEL_CALIBRATION,
    ),
    OperationSpec(
        "smooth",
        "spatial",
        display_name="Gaussian blur/smoothing",
        default_params={"sigma_px": 1.0},
        allowed_scopes=_LOCAL_SCOPES,
    ),
    OperationSpec(
        "median_smooth",
        "spatial",
        default_params={"size_px": 3},
        allowed_scopes=_LOCAL_SCOPES,
    ),
    OperationSpec(
        "gaussian_high_pass",
        "spatial",
        display_name="Gaussian high-pass filter",
        default_params={"sigma_px": 8.0},
        allowed_scopes=_LOCAL_SCOPES,
    ),
    OperationSpec(
        "edge_detect",
        "spatial",
        display_name="Edge detection",
        default_params={"method": "laplacian", "sigma": 1.0, "sigma2": 2.0},
        allowed_scopes=_LOCAL_SCOPES,
    ),
    OperationSpec(
        "remove_spots_auto",
        "spatial",
        default_params={"threshold_mad": 6.0, "window_px": 5},
    ),
    OperationSpec(
        "image_threshold",
        "spatial",
        default_params={"mode": "clip"},
        optional_params=frozenset({"lower", "upper"}),
    ),
    OperationSpec(
        "quantize_bit_depth",
        "spatial",
        required_params=frozenset({"bits"}),
        optional_params=frozenset({"vmin", "vmax"}),
    ),
)


__all__ = ["SPATIAL_OPERATION_SPECS"]
