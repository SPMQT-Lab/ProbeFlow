"""Frequency-domain processing-operation contracts."""

from __future__ import annotations

from probeflow.core.operations.model import OperationSpec


_LOCAL_SCOPES = frozenset({"global", "roi", "mask"})


FREQUENCY_OPERATION_SPECS: tuple[OperationSpec, ...] = (
    OperationSpec(
        "fourier_filter",
        "frequency",
        display_name="FFT filtering",
        default_params={"mode": "low_pass", "cutoff": 0.10, "window": "hanning"},
        allowed_scopes=_LOCAL_SCOPES,
    ),
    OperationSpec(
        "fft_soft_border",
        "frequency",
        display_name="FFT filtering",
        default_params={"mode": "low_pass", "cutoff": 0.10, "border_frac": 0.12},
        allowed_scopes=_LOCAL_SCOPES,
    ),
    OperationSpec(
        "periodic_notch_filter",
        "frequency",
        display_name="FFT filtering",
        default_params={"peaks": (), "radius_px": 3.0},
    ),
    OperationSpec(
        "mains_pickup_suppression",
        "frequency",
        default_params={
            "mains_frequency_hz": 50.0,
            "harmonics": 3,
            "notch_radius_px": 3.0,
            "fast_axis": "x",
            "snap_window_px": 2,
            "notch_shape": "spot",
            "min_q_nm_inv": 0.0,
            "notch_fill": "zero",
        },
        optional_params=frozenset(
            {"scan_speed_m_per_s", "scan_range_m", "extra_streaks_px"}
        ),
        calibration_inputs=frozenset({"pixel_size_x_m", "pixel_size_y_m"}),
    ),
    OperationSpec(
        "inverse_fft_filter",
        "frequency",
        default_params={
            "selections": [],
            "mode": "remove_selected",
            "conjugate_symmetric": True,
            "soft_px": 0.0,
        },
    ),
    OperationSpec(
        "symmetrize_fft",
        "frequency",
        default_params={
            "n_fold": 1,
            "mirror": False,
            "mirror_axis_deg": 0.0,
            "register": True,
            "interpolation": "linear",
            "strict_coverage": False,
        },
    ),
)


__all__ = ["FREQUENCY_OPERATION_SPECS"]
