"""Shared utilities for Nanonis file conversion tools."""

import re
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

log = logging.getLogger(__name__)

# Hardware constants — unified across both tools.
# Nanonis DAQ uses a ±10 V reference over 2^DAC_BITS counts.
DAC_BITS_DEFAULT = 20
DAC_VOLTAGE_REF = 10.0  # V


def v_per_dac(bits: int = DAC_BITS_DEFAULT) -> float:
    """Return volts per DAC count: V_ref / 2^bits."""
    return DAC_VOLTAGE_REF / (2 ** bits)


def parse_header(hb: bytes) -> dict:
    """Parse key=value lines from a Nanonis / Createc header block.

    Decodes with ASCII first; falls back to Latin-1 on any byte the
    ASCII codec cannot represent.  Createc headers contain Latin-1
    characters (e.g. ``Å`` in ``Dacto[Å]z``) and the previous
    ``errors="ignore"`` policy silently dropped them, leaving the key
    as ``Dacto[]z`` — invisible to downstream lookups.  Production
    Createc reader code already uses a Latin-1-aware parser; this
    shared helper now matches that behaviour so any caller that
    happens to wire through it gets the same result.  Review IO #6
    (fixed 2026-05-28).
    """
    def _decode(b: bytes) -> str:
        try:
            return b.decode("ascii")
        except UnicodeDecodeError:
            return b.decode("latin-1")

    hdr: dict = {}
    for line in hb.splitlines():
        if b"=" in line:
            k, v = line.split(b"=", 1)
            key = _decode(k).split("/")[-1].strip()
            val = _decode(v).strip()
            hdr[key] = val
    return hdr


def find_hdr(hdr: dict, hint: str, default=None):
    """Case-insensitive substring search across header keys."""
    for k in hdr:
        if hint.lower() in k.lower():
            return hdr[k]
    return default


def get_dac_bits(hdr: dict, default: int = DAC_BITS_DEFAULT) -> int:
    """Extract DAC resolution in bits from the header; falls back to default."""
    raw = find_hdr(hdr, "DAC-Type", None)
    if raw is None:
        return default
    m = re.search(r"\d+", str(raw).lower().strip())
    if m:
        try:
            return int(m.group())
        except ValueError:
            pass
    return default


def sanitize(name: str) -> str:
    """Make a string safe for use as a filename component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def z_scale_m_per_dac(hdr: dict, vpd: float) -> float:
    """
    Return metres per DAC count for the Z channel.
    Prefers the explicit Createc Dacto[A]z header field.

    Despite the historical ``[A]`` label, Createc's Dacto fields behave as
    nm/DAC in the image files ProbeFlow supports. The lateral calibration gives
    the sanity check: ``Delta X [Dac] * Dacto[A]xy`` matches ``Length x[A] /
    Num.X`` only after converting the Dacto value as nm, not as Å. Keep this
    comment close to the conversion so future cleanup does not reintroduce a
    factor-of-10 Z-height error.

    Falls back to 2 * ZPiezoconst (nm/V) * V/DAC for older headers that lack
    Dacto[A]z.  The factor of 2 matches the empirical relationship
    ``Dacto[A]z ≈ 2 * ZPiezoconst * vpd`` observed across all known Createc
    fixtures: ZPiezoconst already captures the full piezo + HV-amplifier
    sensitivity (nm per volt of DAC output), and the bipolar ±V_ref DAC gives
    2*V_ref / 2^bits volts per count.

    After testing on some additional old createc files, GainZ value can change to be not 10.
    One file had it as 3, and analysis of that image gave a 3* larger step height, suggesting scaling
    was a factor of 3 larger than expected from the Dacto[A]z formula.

    the model now is:
    ZPiezoconst / Dacto[A]z = piezo displacement per DAC output count at a reference Z gain (with GainZ_ref = 10)
    GainZ = user-selected Z high-voltage amplifier gain
    raw_z_counts = feedback/controller counts saved before, or not fully including, that gain

    With this, we obtain the physical height to be:

    z = raw_z_counts * base_dac_to_z * actual_gain / reference_gain

    with reference gain as 10 this leaves the GainZ=10 files unchanged and scales GainZ=3 files by 0.3

    """
    gain_z = _f(find_hdr(hdr, "GainZ", 10), 10)
    gain_multiplier = gain_z / 10.0 # 10.0 is the reference gain for this scaling model.
    dz = _f(find_hdr(hdr, "Dacto[A]z", None))
    if dz is not None:
        return dz * gain_multiplier * 1e-9  # Createc Dacto field: nm/DAC → m/DAC

    zp = _f(find_hdr(hdr, "ZPiezoconst", 19.2), 19.2)  # nm/V in Createc files
    # 2 * (V/DAC) * (nm/V) = nm/DAC → × 1e-9 → m/DAC
    return 2.0 * vpd * zp * gain_multiplier * 1e-9


def i_scale_a_per_dac(hdr: dict, vpd: float, negative: bool = True) -> float:
    """
    Return amperes per DAC count for the current channel.
    sign convention: negative=True matches the typical Nanonis polarity.
    """
    gain_pow = _f(
        find_hdr(hdr, "GainPre", _f(find_hdr(hdr, "GainPre 10^", 9), 9)),
        9.0,
    )
    preamp = 10.0 ** gain_pow  # V/A
    sign = -1.0 if negative else 1.0
    return sign * vpd / preamp  # (V/DAC) / (V/A) = A/DAC


def detect_channels(payload: bytes, Ny: int, Nx: int) -> Tuple[np.ndarray, int]:
    """
    Decode the zlib payload as a (numChan, Ny, Nx) float32 stack.
    Tries 4-channel first, then 2-channel.
    Raises ValueError with a clear message if neither fits.
    """
    for n in (4, 2):
        needed = n * Ny * Nx
        if len(payload) // 4 >= needed:
            arr = np.frombuffer(payload, dtype="<f4", count=needed).copy()
            log.debug("Detected %d channels (%d floats)", n, needed)
            return arr.reshape((n, Ny, Nx)), n
    raise ValueError(
        f"Payload too small for 2- or 4-channel data "
        f"(Ny={Ny}, Nx={Nx}, payload floats={len(payload) // 4})"
    )


def trim_stack(stack: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Remove trailing scan lines that did not reach the final column.

    A zero in one channel is a valid DAC value and cannot by itself mark an
    incomplete acquisition.  The legacy Createc fallback therefore requires a
    completed scan line to reach the final column in at least one stored
    channel. Callers with vendor completion metadata should use that first.
    Returns (trimmed_stack, new_Ny).
    """
    Ny = stack.shape[1]
    final_column = stack[:, :, -1]
    completed_row = np.any(np.isfinite(final_column) & (final_column != 0), axis=0)
    rows = np.flatnonzero(completed_row)
    if rows.size == 0:
        return stack, Ny
    last_row = int(rows.max())
    new_Ny = last_row + 1
    return stack[:, :new_Ny, :], new_Ny


def percentile_clip(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    """Return (vmin, vmax) from finite values using percentile clipping."""
    from probeflow.processing.display import clip_range_from_array

    return clip_range_from_array(arr, low, high)


def to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map [vmin, vmax] → [0, 255] uint8."""
    from probeflow.processing.display import array_to_uint8

    return array_to_uint8(arr, vmin=vmin, vmax=vmax)


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger for CLI tools."""
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


# Canonical definitions live in core.common; re-exported here for callers
# that still import from probeflow.io.common.
from probeflow.core.common import _f, _i  # noqa: F401


def check_overwrite(input_path: Path, output_path: Path) -> None:
    """Raise ValueError if output_path resolves to the same file as input_path."""
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError(
            f"Output path would overwrite the source: {output_path!r}"
        )


def check_output_available(output_path: Path, *, overwrite: bool = False) -> None:
    """Raise FileExistsError when an output artifact already exists."""
    out = Path(output_path)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {out}. Pass overwrite=True/--force."
        )
