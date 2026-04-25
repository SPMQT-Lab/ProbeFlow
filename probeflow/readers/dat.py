"""Reader for Createc ``.dat`` scan files — returns a :class:`probeflow.scan.Scan`.

This loads a Createc raw scan directly into the Scan abstraction without the
lossy percentile clipping that ``probeflow.dat_sxm.process_dat`` applies.  The
returned planes carry true physical units (metres for Z, amperes for current)
so processing pipelines can operate on them the same way they operate on a
``.sxm``-sourced Scan.

Createc channel layout:
  * 4-channel files contain [FT, FC, BT, BC] = [Z fwd, I fwd, Z bwd, I bwd].
  * 2-channel files contain only [FT, FC]; the backward planes are
    synthesised as copies of the forward planes and flagged in
    ``scan.plane_synthetic``.

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

import zlib
from pathlib import Path
from typing import List

import numpy as np

from probeflow.common import (
    _f,
    _i,
    detect_channels,
    find_hdr,
    get_dac_bits,
    i_scale_a_per_dac,
    parse_header,
    trim_stack,
    v_per_dac,
    z_scale_m_per_dac,
)
from probeflow.scan import Scan


def read_dat(path) -> Scan:
    """Load a Createc ``.dat`` into a Scan (display-oriented, SI units)."""
    path = Path(path)
    raw = path.read_bytes()

    if b"DATA" not in raw:
        raise ValueError(
            f"{path.name}: missing DATA marker — not a valid Createc .dat file"
        )

    hb, comp = raw.split(b"DATA", 1)
    hdr = parse_header(hb)

    Nx = _i(find_hdr(hdr, "Num.X", 0), 0)
    Ny = _i(find_hdr(hdr, "Num.Y", 0), 0)
    if Nx <= 0 or Ny <= 0:
        raise ValueError(f"{path.name}: invalid dimensions Nx={Nx}, Ny={Ny}")

    if _i(find_hdr(hdr, "ScanmodeSine", 0), 0) != 0:
        raise NotImplementedError(
            f"{path.name}: sine scan mode is not supported"
        )

    try:
        payload = zlib.decompress(comp)
    except zlib.error as exc:
        raise ValueError(
            f"{path.name}: zlib decompression failed — {exc}"
        ) from exc

    stack, num_chan = detect_channels(payload, Ny, Nx)
    stack, Ny = trim_stack(stack)
    # Keep the header consistent with any trim we just applied so downstream
    # writers (e.g. dat→sxm construction) see the corrected pixel count.
    hdr["Num.Y"] = str(Ny)

    bits = get_dac_bits(hdr)
    vpd = v_per_dac(bits)
    zs = z_scale_m_per_dac(hdr, vpd)
    is_ = i_scale_a_per_dac(hdr, vpd)

    # Apply physical scaling in-place: even indices = Z (metres), odd = I (amps).
    for k in range(num_chan):
        stack[k] = (stack[k] * (zs if k % 2 == 0 else is_)).astype(np.float32)

    # Createc native → canonical [Z fwd, Z bwd, I fwd, I bwd] ordering
    if num_chan == 4:
        FT, FC, BT, BC = stack[0], stack[1], stack[2], stack[3]
        synthetic = [False, False, False, False]
    elif num_chan == 2:
        FT, FC = stack[0], stack[1]
        BT = FT.copy()
        BC = FC.copy()
        synthetic = [False, True, False, True]
    else:
        raise ValueError(
            f"{path.name}: unexpected channel count {num_chan} (expected 2 or 4)"
        )

    def _orient(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64)

    planes: List[np.ndarray] = [
        _orient(FT),
        _orient(BT),
        _orient(FC),
        _orient(BC),
    ]

    lx_a = _f(hdr.get("Length x[A]", "0"), 0.0)
    ly_a = _f(hdr.get("Length y[A]", "0"), 0.0)
    scan_range_m = (lx_a * 1e-10, ly_a * 1e-10)

    return Scan(
        planes=planes,
        plane_names=["Z forward", "Z backward", "Current forward", "Current backward"],
        plane_units=["m", "m", "A", "A"],
        plane_synthetic=synthetic,
        header=hdr,
        scan_range_m=scan_range_m,
        source_path=path,
        source_format="dat",
    )
