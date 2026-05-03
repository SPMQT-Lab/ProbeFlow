"""Reader for Createc ``.dat`` scan files — returns a :class:`probeflow.core.scan_model.Scan`.

This loads a Createc raw scan directly into the Scan abstraction without the
lossy percentile clipping that ``probeflow.io.converters.createc_dat_to_sxm.process_dat`` applies.  The
returned planes carry true physical units (metres for Z, amperes for current)
so processing pipelines can operate on them the same way they operate on a
``.sxm``-sourced Scan.

Createc channel layout:
  * Canonical STM 4-plane files are reordered from native
    [Z fwd, I fwd, Z bwd, I bwd] to public [Z fwd, Z bwd, I fwd, I bwd].
  * Legacy STM 2-plane files with only [Z fwd, I fwd] synthesise backward
    planes and flag them in ``scan.plane_synthetic``.
  * Selected-channel and auxiliary layouts are returned in native order with
    the best-known names/units from the decode report.

Orientation:
  * Createc stores backward scan rows in left-to-right display order already
    (unlike Nanonis .sxm, where backward rows are stored right-to-left).  No
    horizontal flip is applied here; the planes are returned as-is in display
    orientation.
  * Vertical origin is kept as Createc stores it (Y flip is not applied
    here).  This matches the current ``dat→sxm`` conversion convention so
    that ``load_scan(dat)`` and ``load_scan(sxm_from_dat)`` give identical
    arrays — a future release can reconsider this if the display convention
    is unified across formats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.io.createc_interpretation import createc_dat_experiment_metadata
from probeflow.io.readers.createc_dat import (
    has_canonical_stm_four_channel_layout,
    has_legacy_stm_two_channel_layout,
    read_createc_dat_report,
    scale_channels_for_scan,
    scan_range_m_from_header,
)
from probeflow.core.scan_model import Scan


def read_dat(path) -> Scan:
    """Load a Createc ``.dat`` into a Scan (display-oriented, SI units)."""
    path = Path(path)
    report = read_createc_dat_report(path, include_raw=True)
    hdr = dict(report.header)
    scaled = scale_channels_for_scan(report)
    num_chan = report.detected_channel_count

    # Createc native STM → canonical [Z fwd, Z bwd, I fwd, I bwd] ordering.
    # Non-STM selected-channel layouts stay in native report order so their
    # labels/units do not inherit STM positional assumptions.
    if has_canonical_stm_four_channel_layout(report):
        FT, FC, BT, BC = scaled[0], scaled[1], scaled[2], scaled[3]
        synthetic = [False, False, False, False]
        plane_names = ["Z forward", "Z backward", "Current forward", "Current backward"]
        plane_units = ["m", "m", "A", "A"]
        planes = [FT, BT, FC, BC]
    elif has_legacy_stm_two_channel_layout(report):
        FT, FC = scaled[0], scaled[1]
        BT = FT.copy()
        BC = FC.copy()
        synthetic = [False, True, False, True]
        plane_names = ["Z forward", "Z backward", "Current forward", "Current backward"]
        plane_units = ["m", "m", "A", "A"]
        planes = [FT, BT, FC, BC]
    else:
        synthetic = [False] * num_chan
        plane_names = [info.name for info in report.channel_info]
        plane_units = [info.unit for info in report.channel_info]
        planes = scaled

    def _as_f64(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64)

    oriented_planes: list[np.ndarray] = [_as_f64(arr) for arr in planes]

    return Scan(
        planes=oriented_planes,
        plane_names=plane_names,
        plane_units=plane_units,
        plane_synthetic=synthetic,
        header=hdr,
        scan_range_m=scan_range_m_from_header(hdr),
        source_path=path,
        source_format="dat",
        experiment_metadata=createc_dat_experiment_metadata(hdr),
    )


def read_dat_metadata(path):
    """Return :class:`~probeflow.core.metadata.ScanMetadata` for a Createc ``.dat``."""
    from probeflow.core.metadata import metadata_from_createc_dat_report

    return metadata_from_createc_dat_report(
        read_createc_dat_report(path, include_raw=False)
    )
