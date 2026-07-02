"""Automatic label proposals for Dataset Builder."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from probeflow.dataset_builder.models import DatasetTaskConfig, ProposalResult


def generate_proposal(
    arr: np.ndarray,
    *,
    px_x_m: float,
    px_y_m: float,
    config: DatasetTaskConfig,
    source_channel: str | None = None,
) -> ProposalResult:
    """Generate a first-pass label proposal for a Dataset Builder task."""

    method = config.proposal_method
    params = dict(config.proposal_params)

    if config.task == "step_edge_mask" or method == "step_edge":
        from probeflow.analysis.step_edges import step_edge_mask

        molecule_size_nm = float(params.get("molecule_size_nm", 1.0))
        margin_nm = float(params.get("margin_nm", 0.3))
        min_height_nm = float(params.get("min_step_height_nm", 0.0))
        mask = step_edge_mask(
            arr,
            pixel_size_x_m=px_x_m,
            pixel_size_y_m=px_y_m,
            molecule_diameter_m=molecule_size_nm * 1e-9,
            threshold_deg=float(params.get("threshold_deg", 20.0)),
            dilate_m=margin_nm * 1e-9,
            min_step_height_m=min_height_nm * 1e-9 if min_height_nm > 0 else None,
            suppress_dark=bool(params.get("suppress_dark", False)),
        )
        out_params: dict[str, Any] = {
            "molecule_size_nm": molecule_size_nm,
            "threshold_deg": float(params.get("threshold_deg", 20.0)),
            "margin_nm": margin_nm,
            "min_step_height_nm": min_height_nm,
            "suppress_dark": bool(params.get("suppress_dark", False)),
            "source_channel": source_channel,
        }
        return ProposalResult(
            label_type="mask",
            method="step_edge",
            parameters=out_params,
            mask=mask,
        )

    if method == "canny":
        from probeflow.processing.edge_detection import canny_edges

        result = canny_edges(
            arr,
            sigma=float(params.get("sigma", 2.0)),
            threshold_mode=str(params.get("threshold_mode", "percentile")),
            low=float(params.get("low", 60.0)),
            high=float(params.get("high", 85.0)),
            pixel_size_x_nm=px_x_m * 1e9,
            pixel_size_y_nm=px_y_m * 1e9,
            source_channel=source_channel,
        )
        return ProposalResult(
            label_type="mask",
            method="canny",
            parameters=dict(result.parameters),
            mask=result.edge_mask,
        )

    if config.task == "point_labels" or method == "feature_points":
        from probeflow.analysis.feature_finder import find_image_features

        result = find_image_features(
            arr,
            mode=str(params.get("mode", "maxima")),
            threshold_mode=str(params.get("threshold_mode", "above")),
            threshold_low=params.get("threshold_low"),
            threshold_high=params.get("threshold_high"),
            min_distance_px=float(params.get("min_distance_px", 3.0)),
            smoothing_sigma_px=float(params.get("smoothing_sigma_px", 0.0)),
        )
        return ProposalResult(
            label_type="points",
            method="feature_points",
            parameters={
                "result": asdict(result),
                "source_channel": source_channel,
            },
            points=tuple(result.points),
        )

    raise ValueError(f"No proposal implementation for method {method!r}")

