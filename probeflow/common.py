"""Shared utilities for Nanonis file conversion tools."""

import re
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# Hardware constants — unified across both tools.
# Nanonis DAQ uses a ±10 V reference over 2^DAC_BITS counts.
DAC_BITS_DEFAULT = 20
DAC_VOLTAGE_REF = 10.0  # V


def v_per_dac(bits: int = DAC_BITS_DEFAULT) -> float:
    """Return volts per DAC count: V_ref / 2^bits."""
    return DAC_VOLTAGE_REF / (2 ** bits)


def parse_header(hb: bytes) -> dict:
    """Parse key=value lines from a Nanonis .dat header block."""
    hdr: dict = {}
    for line in hb.splitlines():
        if b"=" in line:
            k, v = line.split(b"=", 1)
            key = k.decode("ascii", errors="ignore").split("/")[-1].strip()
            val = v.decode("ascii", errors="ignore").strip()
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
    Prefers the explicit Dacto[A]z header field (units: Å/DAC).
    Falls back to GainZ * ZPiezoconst (Å/V) * V/DAC.
    """
    dz = _f(find_hdr(hdr, "Dacto[A]z", None))
    if dz is not None:
        return dz * 1e-10  # Å/DAC → m/DAC

    gz = _f(find_hdr(hdr, "GainZ", 10.0), 10.0)
    zp = _f(find_hdr(hdr, "ZPiezoconst", 19.2), 19.2)  # Å/V typical
    # V/DAC * dimensionless * Å/V = Å/DAC → × 1e-10 → m/DAC
    return vpd * gz * zp * 1e-10


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
    Remove trailing incomplete scan rows using channel 0 as a reference.
    Returns (trimmed_stack, new_Ny).
    """
    ch0 = stack[0]
    Ny, Nx = ch0.shape
    valid = np.logical_and(~np.isnan(ch0), ch0 != 0)
    rows, cols = np.where(valid)
    if rows.size == 0:
        return stack, Ny
    last_row = int(rows.max())
    last_col = int(cols[rows == last_row].max())
    new_Ny = (last_row + 1) if last_col == (Nx - 1) else last_row
    new_Ny = max(1, new_Ny)
    return stack[:, :new_Ny, :], new_Ny


def percentile_clip(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> Tuple[float, float]:
    """Return (vmin, vmax) from finite values using percentile clipping."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        return 0.0, 1.0
    return vmin, vmax


def to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly map [vmin, vmax] → [0, 255] uint8."""
    x = np.array(arr, dtype=np.float32)
    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(x)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger for CLI tools."""
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def _f(x, default=None):
    """Safe float conversion; replaces commas with dots."""
    try:
        return float(str(x).replace(",", "."))
    except (TypeError, ValueError):
        return default


def _i(x, default=None):
    """Safe int conversion."""
    try:
        return int(float(str(x).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def mark_processed_stem(stem: str) -> str:
    """Return stem with _processed appended, unless already present."""
    if stem.endswith("_processed"):
        return stem
    return f"{stem}_processed"
