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
  * Createc stores backward scan rows in **left-to-right display order**
    (unlike Nanonis .sxm, where backward rows are stored right-to-left in
    acquisition order and must be flipped by the reader).  This was verified
    empirically on real 4-channel data: loading the .dat directly and loading
    an SXM converted from the same .dat via ``createc_dat_to_sxm.process_dat``
    produce byte-identical backward planes.  No horizontal flip is applied
    here; the planes are returned as-is.

    Round-trip invariant: the converter applies ``np.fliplr`` to the raw
    backward planes before writing SXM (so they are stored in Nanonis
    acquisition order), and the SXM reader flips them back.  The net effect
    is identity, preserving the display-order orientation that Createc natively
    uses.

  * Vertical origin is kept as Createc stores it (Y flip is not applied
    here).  This matches the ``dat→sxm`` conversion convention so that
    ``load_scan(dat)`` and ``load_scan(sxm_from_dat)`` give identical arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CreatecPublicPlane:
    """One public-facing Createc plane after ProbeFlow layout resolution."""

    public_index: int
    public_name: str
    public_unit: str
    source_native_index: int | None
    source_name: str
    source_unit: str
    scale_factor: float
    semantic: str | None
    direction: str | None
    synthetic: bool
    array: np.ndarray


def createc_public_planes_from_report(
    report,
    arrays: list[np.ndarray],
) -> list[CreatecPublicPlane]:
    """Return planes in ProbeFlow's public order with Createc metadata attached.

    ``arrays`` must be ordered in Createc native channel order as decoded by
    :func:`probeflow.io.readers.createc_dat.scale_channels_for_scan` or the
    equivalent raw decode stack.
    """

    if len(arrays) != report.detected_channel_count:
        raise ValueError(
            f"expected {report.detected_channel_count} array(s), got {len(arrays)}"
        )

    channel_info = list(report.channel_info)

    def _build(
        public_index: int,
        source_native_index: int | None,
        *,
        public_name: str,
        public_unit: str,
        source_name: str,
        source_unit: str,
        scale_factor: float,
        semantic: str | None,
        direction: str | None,
        synthetic: bool,
        array: np.ndarray,
    ) -> CreatecPublicPlane:
        return CreatecPublicPlane(
            public_index=public_index,
            public_name=public_name,
            public_unit=public_unit,
            source_native_index=source_native_index,
            source_name=source_name,
            source_unit=source_unit,
            scale_factor=float(scale_factor),
            semantic=semantic,
            direction=direction,
            synthetic=synthetic,
            array=array,
        )

    if has_canonical_stm_four_channel_layout(report):
        # Native Createc order: [Z fwd, I fwd, Z bwd, I bwd]
        # Public ProbeFlow order: [Z fwd, Z bwd, I fwd, I bwd]
        z_fwd, i_fwd, z_bwd, i_bwd = channel_info
        return [
            _build(
                0, 0,
                public_name="Z forward",
                public_unit="m",
                source_name=z_fwd.name,
                source_unit=z_fwd.unit,
                scale_factor=z_fwd.scale_factor,
                semantic=z_fwd.semantic,
                direction=z_fwd.direction,
                synthetic=False,
                array=arrays[0],
            ),
            _build(
                1, 2,
                public_name="Z backward",
                public_unit="m",
                source_name=z_bwd.name,
                source_unit=z_bwd.unit,
                scale_factor=z_bwd.scale_factor,
                semantic=z_bwd.semantic,
                direction=z_bwd.direction,
                synthetic=False,
                array=arrays[2],
            ),
            _build(
                2, 1,
                public_name="Current forward",
                public_unit="A",
                source_name=i_fwd.name,
                source_unit=i_fwd.unit,
                scale_factor=i_fwd.scale_factor,
                semantic=i_fwd.semantic,
                direction=i_fwd.direction,
                synthetic=False,
                array=arrays[1],
            ),
            _build(
                3, 3,
                public_name="Current backward",
                public_unit="A",
                source_name=i_bwd.name,
                source_unit=i_bwd.unit,
                scale_factor=i_bwd.scale_factor,
                semantic=i_bwd.semantic,
                direction=i_bwd.direction,
                synthetic=False,
                array=arrays[3],
            ),
        ]

    if has_legacy_stm_two_channel_layout(report):
        # Public layout keeps the canonical four-plane shape by mirroring the
        # forward planes into synthetic backward planes.
        z_fwd, i_fwd = channel_info
        return [
            _build(
                0, 0,
                public_name="Z forward",
                public_unit="m",
                source_name=z_fwd.name,
                source_unit=z_fwd.unit,
                scale_factor=z_fwd.scale_factor,
                semantic=z_fwd.semantic,
                direction=z_fwd.direction,
                synthetic=False,
                array=arrays[0],
            ),
            _build(
                1, 0,
                public_name="Z backward",
                public_unit="m",
                source_name=z_fwd.name,
                source_unit=z_fwd.unit,
                scale_factor=z_fwd.scale_factor,
                semantic=z_fwd.semantic,
                direction="backward",
                synthetic=True,
                array=np.array(arrays[0], copy=True),
            ),
            _build(
                2, 1,
                public_name="Current forward",
                public_unit="A",
                source_name=i_fwd.name,
                source_unit=i_fwd.unit,
                scale_factor=i_fwd.scale_factor,
                semantic=i_fwd.semantic,
                direction=i_fwd.direction,
                synthetic=False,
                array=arrays[1],
            ),
            _build(
                3, 1,
                public_name="Current backward",
                public_unit="A",
                source_name=i_fwd.name,
                source_unit=i_fwd.unit,
                scale_factor=i_fwd.scale_factor,
                semantic=i_fwd.semantic,
                direction="backward",
                synthetic=True,
                array=np.array(arrays[1], copy=True),
            ),
        ]

    public_planes: list[CreatecPublicPlane] = []
    for info in channel_info:
        public_planes.append(
            _build(
                info.native_index,
                info.native_index,
                public_name=info.name,
                public_unit=info.unit,
                source_name=info.name,
                source_unit=info.unit,
                scale_factor=info.scale_factor,
                semantic=info.semantic,
                direction=info.direction,
                synthetic=False,
                array=arrays[info.native_index],
            )
        )
    return public_planes


def read_dat(path) -> Scan:
    """Load a Createc ``.dat`` into a Scan (display-oriented, SI units)."""
    path = Path(path)
    report = read_createc_dat_report(path, include_raw=True)
    hdr = dict(report.header)
    scaled = scale_channels_for_scan(report)
    public_planes = createc_public_planes_from_report(report, scaled)

    synthetic = [plane.synthetic for plane in public_planes]
    plane_names = [plane.public_name for plane in public_planes]
    plane_units = [plane.public_unit for plane in public_planes]
    planes = [plane.array for plane in public_planes]

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
        warnings=report.warnings,
    )


def read_dat_metadata(path):
    """Return :class:`~probeflow.core.metadata.ScanMetadata` for a Createc ``.dat``."""
    from probeflow.core.metadata import metadata_from_createc_dat_report

    return metadata_from_createc_dat_report(
        read_createc_dat_report(path, include_raw=False)
    )
