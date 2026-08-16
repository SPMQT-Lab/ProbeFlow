"""Scoped, zeroing, interpolation, and multi-input operation contracts."""

from __future__ import annotations

from probeflow.core.operations.model import OperationSpec


SCOPED_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        "arithmetic",
        "scoped",
        default_params={
            "operand_type": "constant",
            "operation": "add",
            "pattern": "checkerboard",
            "amplitude_si": 0.0,
            "period_px": 16,
            "seed": 1,
        },
        optional_params=frozenset(
            {
                "value_si",
                "factor",
                "source_path",
                "plane_idx",
                "source_fingerprint",
                "source_channel",
            }
        ),
        allowed_scopes=frozenset({"global", "roi", "mask"}),
    ),
    OperationSpec(
        "set_zero_point",
        "scoped",
        display_name="Set zero point",
        default_params={"x_px": 0, "y_px": 0, "patch": 1},
    ),
    OperationSpec(
        "set_zero_plane",
        "scoped",
        display_name="Set zero plane",
        default_params={"points_px": (), "patch": 1},
    ),
    OperationSpec(
        "roi",
        "scoped",
        default_params={"step": {}},
        optional_params=frozenset(
            {"roi_id", "frozen_geometry", "scope_semantics", "frame_shape"}
        ),
        handler_key="scope_roi",
    ),
    OperationSpec(
        "mask",
        "scoped",
        default_params={"step": {}},
        optional_params=frozenset(
            {"mask_id", "frozen_mask", "scope_semantics", "frame_shape"}
        ),
        handler_key="scope_mask",
    ),
    OperationSpec(
        "interpolate_masked",
        "scoped",
        optional_params=frozenset({"frozen_mask", "frozen_geometry"}),
    ),
)


__all__ = ["SCOPED_OPERATION_SPECS"]
