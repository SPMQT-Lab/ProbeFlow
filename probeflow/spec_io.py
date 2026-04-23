"""Reader for Createc vertical-spectroscopy (.VERT) files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Union

import numpy as np

from .common import _f, _i, find_hdr, get_dac_bits, i_scale_a_per_dac, v_per_dac, z_scale_m_per_dac

log = logging.getLogger(__name__)

# Voltage range below this threshold (mV) → file is a time trace, not a bias sweep.
# Configurable via read_spec_file(time_trace_threshold_mv=...) for unusual sweeps.
_TIME_TRACE_THRESHOLD_MV = 1.0


@dataclass
class SpecData:
    """All data and metadata from one Createc .VERT spectroscopy file.

    Parameters
    ----------
    header : dict[str, str]
        Raw header key-value pairs from the file.
    channels : dict[str, np.ndarray]
        Named data channels in SI units: 'I' (A), 'Z' (m), 'V' (V).
        For bias sweeps, channels['V'] equals x_array and is redundant;
        for time traces it holds the (near-constant) measurement bias.
    x_array : np.ndarray
        Independent variable in SI units (time in s or bias in V).
    x_label : str
        Human-readable axis label, e.g. 'Bias (V)' or 'Time (s)'.
    x_unit : str
        Unit string, e.g. 'V', 's'.
    y_units : dict[str, str]
        Unit string for each channel, e.g. {'I': 'A', 'Z': 'm'}.
    position : tuple[float, float]
        (x_m, y_m) tip position in physical coordinates (metres).
    metadata : dict[str, Any]
        Scan parameters: sweep_type, bias, frequency, title, etc.
    """

    header: dict[str, str]
    channels: dict[str, np.ndarray]
    x_array: np.ndarray
    x_label: str
    x_unit: str
    y_units: dict[str, str]
    position: tuple[float, float]
    metadata: dict[str, Any]


def parse_spec_header(path: Union[str, Path]) -> dict[str, str]:
    """Read only the header of a .VERT file and return it as a dictionary.

    Reads in 64 KB chunks and stops as soon as the DATA marker is found,
    so large spectroscopy files are not loaded entirely into memory.

    Parameters
    ----------
    path : str or Path
        Path to a Createc .VERT file.

    Returns
    -------
    dict[str, str]
        Key-value pairs from the file header.
    """
    path = Path(path)
    chunks: list[bytes] = []
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            blob = b"".join(chunks)
            pos = blob.find(b"DATA")
            if pos >= 0:
                return _parse_vert_header(blob[:pos])
    raise ValueError(f"{path.name}: missing DATA marker")


def read_spec_file(
    path: Union[str, Path],
    *,
    time_trace_threshold_mv: float = _TIME_TRACE_THRESHOLD_MV,
) -> SpecData:
    """Read a Createc .VERT spectroscopy file and return a SpecData object.

    The data is converted to SI units on read. The sweep type (bias sweep vs
    time trace) is detected from the Vpoint header entries first, falling back
    to checking the voltage range in the data column.

    Parameters
    ----------
    path : str or Path
        Path to a Createc .VERT file.
    time_trace_threshold_mv : float
        Voltage-range threshold (mV) below which a file is classified as a
        time trace rather than a bias sweep. Default 1.0 mV.

    Returns
    -------
    SpecData
        Parsed and unit-converted spectroscopy data.
    """
    path = Path(path)
    raw = path.read_bytes()

    data_pos = raw.find(b"DATA")
    if data_pos < 0:
        raise ValueError(f"{path.name}: missing DATA marker")

    hdr = _parse_vert_header(raw[:data_pos])

    # Locate the params line that follows "DATA\r\n" then the data rows.
    data_section = raw[data_pos:]
    eol = b"\r\n" if b"\r\n" in data_section[:6] else b"\n"
    eol_len = len(eol)
    first_eol = data_section.find(eol)
    params_start = first_eol + eol_len
    second_eol = data_section.find(eol, params_start)
    if second_eol < 0:
        second_eol = len(data_section)

    # Parse data rows with np.loadtxt for speed (5–20× faster than a Python loop).
    data_text = data_section[second_eol + eol_len:].decode("latin-1", errors="replace")
    clean = "\n".join(
        ln.rstrip("\t ") for ln in data_text.splitlines() if ln.strip()
    )
    if not clean:
        raise ValueError(f"{path.name}: no data rows found after DATA marker")
    try:
        arr = np.loadtxt(StringIO(clean), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    except ValueError as exc:
        raise ValueError(f"{path.name}: failed to parse data section — {exc}") from exc

    # Column layout: index | bias_mV | current_raw | z_raw (always 4 columns)
    if arr.shape[1] < 4:
        raise ValueError(
            f"{path.name}: expected ≥4 data columns (idx, V_mV, I, Z), "
            f"got {arr.shape[1]}"
        )

    bias_mv = arr[:, 1]

    # Sweep-type detection: prefer header Vpoint entries over data-range heuristic.
    is_time_trace = _detect_time_trace(hdr, bias_mv, time_trace_threshold_mv)

    # Unit conversion using calibration constants from the header.
    # The DAC spans ±10 V (full range 20 V), so the step size is 20/2^bits V/count,
    # which is 2× the value returned by v_per_dac (which uses a 10 V reference).
    bits = get_dac_bits(hdr)
    vpd = v_per_dac(bits) * 2
    zs = z_scale_m_per_dac(hdr, vpd)
    is_ = i_scale_a_per_dac(hdr, vpd, negative=False)  # raw DAC already carries sign

    # Column layout: 0=index, 1=bias_mV, 2=Z_DAC, 3=I_DAC
    z_m = arr[:, 2] * zs
    current_a = arr[:, 3] * is_

    channels: dict[str, np.ndarray] = {
        "I": current_a,
        "Z": z_m,
        # For time traces: constant bias reference. For bias sweeps: equals x_array.
        "V": bias_mv * 1e-3,
    }
    y_units: dict[str, str] = {"I": "A", "Z": "m", "V": "V"}

    # Factor out SpecFreq — used both for x_array and metadata.
    spec_freq = float(_f(find_hdr(hdr, "SpecFreq", "1000"), 1000.0))

    if is_time_trace:
        x_array = arr[:, 0] / spec_freq  # sample index / Hz → seconds
        x_label = "Time (s)"
        x_unit = "s"
    else:
        x_array = bias_mv * 1e-3  # mV → V
        x_label = "Bias (V)"
        x_unit = "V"

    # Tip position in physical coordinates (metres).
    dac_to_a_xy = _f(find_hdr(hdr, "Dacto[A]xy", "1"), 1.0)
    ox_dac = _f(find_hdr(hdr, "OffsetX", "0"), 0.0)
    oy_dac = _f(find_hdr(hdr, "OffsetY", "0"), 0.0)
    pos_x_m = ox_dac * dac_to_a_xy * 1e-10
    pos_y_m = oy_dac * dac_to_a_xy * 1e-10

    bias_raw = find_hdr(hdr, "BiasVolt.[mV]", None) or find_hdr(hdr, "Biasvolt[mV]", "0")
    metadata: dict[str, Any] = {
        "filename": path.name,
        "bias_mv": float(_f(bias_raw, 0.0)),
        "spec_freq_hz": spec_freq,
        "gain_pre_exp": float(_f(find_hdr(hdr, "GainPre 10^", "9"), 9.0)),
        "fb_log": hdr.get("FBLog", "0").strip() == "1",
        "sweep_type": "time_trace" if is_time_trace else "bias_sweep",
        "n_points": len(arr),
        "title": hdr.get("Titel", ""),
    }

    log.info(
        "%s: %s, %d pts, pos=(%.3g, %.3g) m",
        path.name,
        metadata["sweep_type"],
        metadata["n_points"],
        pos_x_m,
        pos_y_m,
    )

    return SpecData(
        header=hdr,
        channels=channels,
        x_array=x_array,
        x_label=x_label,
        x_unit=x_unit,
        y_units=y_units,
        position=(pos_x_m, pos_y_m),
        metadata=metadata,
    )


def _detect_time_trace(
    hdr: dict[str, str],
    bias_mv: np.ndarray,
    threshold_mv: float,
) -> bool:
    """Return True if this file is a time trace rather than a bias sweep.

    Prefers the Vpoint header entries (which encode the programmed sweep
    explicitly) over the data-range heuristic, to avoid misclassifying very
    short bias sweeps or time traces with DAC noise.
    """
    # Collect voltages from Vpoint entries that have a non-zero time span.
    vpoint_volts: list[float] = []
    for i in range(8):
        t = _f(hdr.get(f"Vpoint{i}.t", "0"), 0.0)
        v = _f(hdr.get(f"Vpoint{i}.V", None), None)
        if t is not None and t > 0 and v is not None:
            vpoint_volts.append(v)

    if len(vpoint_volts) >= 2:
        span = max(vpoint_volts) - min(vpoint_volts)
        return span < threshold_mv

    # Fallback: check the actual data column.
    v_range = float(bias_mv.max() - bias_mv.min())
    return v_range < threshold_mv


def _parse_vert_header(hb: bytes) -> dict[str, str]:
    """Parse key=value lines from a Createc .VERT header block.

    Handles both 'internal / display=value' and plain 'key=value' formats.
    Uses latin-1 encoding to preserve special characters (e.g. Å).
    """
    hdr: dict[str, str] = {}
    for line in hb.splitlines():
        line = line.strip()
        if b"=" not in line:
            continue
        k, _, v = line.partition(b"=")
        # "internal / display=value" → use the display name (after slash) as key.
        if b"/" in k:
            k = k.split(b"/")[-1]
        key = k.decode("latin-1", errors="replace").strip()
        val = v.decode("latin-1", errors="replace").strip()
        if key:
            hdr[key] = val
    return hdr
