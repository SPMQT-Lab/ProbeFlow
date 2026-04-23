"""Reader for Createc vertical-spectroscopy (.VERT) files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

from .common import _f, _i, find_hdr, get_dac_bits, i_scale_a_per_dac, v_per_dac, z_scale_m_per_dac

log = logging.getLogger(__name__)

# Voltage range below this threshold (mV) → file is a time trace, not a bias sweep.
_TIME_TRACE_THRESHOLD_MV = 1.0


@dataclass
class SpecData:
    """All data and metadata from one Createc .VERT spectroscopy file.

    Parameters
    ----------
    header : dict
        Raw header key-value pairs from the file.
    channels : dict[str, np.ndarray]
        Named data channels in SI units: 'I' (A), 'Z' (m), 'V' (V).
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
    metadata : dict
        Scan parameters: sweep_type, bias, frequency, title, etc.
    """

    header: dict
    channels: dict[str, np.ndarray]
    x_array: np.ndarray
    x_label: str
    x_unit: str
    y_units: dict[str, str]
    position: tuple[float, float]
    metadata: dict


def parse_spec_header(path: Union[str, Path]) -> dict:
    """Read only the header of a .VERT file and return it as a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to a Createc .VERT file.

    Returns
    -------
    dict
        Key-value pairs from the file header.
    """
    raw = Path(path).read_bytes()
    data_pos = raw.find(b"DATA")
    if data_pos < 0:
        raise ValueError(f"{Path(path).name}: missing DATA marker")
    return _parse_vert_header(raw[:data_pos])


def read_spec_file(path: Union[str, Path]) -> SpecData:
    """Read a Createc .VERT spectroscopy file and return a SpecData object.

    The data is converted to SI units on read. The sweep type (bias sweep vs
    time trace) is detected automatically from whether the bias column is
    constant or varying.

    Parameters
    ----------
    path : str or Path
        Path to a Createc .VERT file.

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

    # Locate the parameters line that follows "DATA\r\n" and then the data rows.
    data_section = raw[data_pos:]
    eol = b"\r\n" if b"\r\n" in data_section[:6] else b"\n"
    eol_len = len(eol)
    first_eol = data_section.find(eol)
    params_start = first_eol + eol_len
    second_eol = data_section.find(eol, params_start)
    if second_eol < 0:
        second_eol = len(data_section)

    params_line = data_section[params_start:second_eol].decode("latin-1", errors="replace").strip()
    params = params_line.split()
    declared_npts = _i(params[0], 0) if params else 0

    # Read tab-separated ASCII data rows that follow the params line.
    text_start = data_pos + second_eol + eol_len
    data_text = raw[text_start:].decode("latin-1", errors="replace")
    rows: list[list[float]] = []
    for line in data_text.splitlines():
        cols = [c.strip() for c in line.split("\t") if c.strip()]
        if not cols:
            continue
        try:
            rows.append([float(c) for c in cols])
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"{path.name}: no data rows found after DATA marker")

    arr = np.array(rows, dtype=np.float64)
    if arr.shape[1] < 3:
        raise ValueError(
            f"{path.name}: expected ≥3 data columns, got {arr.shape[1]}"
        )

    # Column layout (always): index | bias_mV | current_raw | z_raw
    bias_mv = arr[:, 1]
    v_range = float(bias_mv.max() - bias_mv.min())
    is_time_trace = v_range < _TIME_TRACE_THRESHOLD_MV

    # Unit conversion using calibration constants from the header.
    bits = get_dac_bits(hdr)
    vpd = v_per_dac(bits)
    zs = z_scale_m_per_dac(hdr, vpd)
    is_ = i_scale_a_per_dac(hdr, vpd, negative=True)

    # Current: column 2 may be zero when FBLog=1 stores log-mode current differently;
    # we store as-converted and let the user decide which channel is meaningful.
    current_a = arr[:, 2] * abs(is_)
    z_m = arr[:, 3] * zs if arr.shape[1] > 3 else np.zeros(len(arr))

    channels: dict[str, np.ndarray] = {
        "I": current_a,
        "Z": z_m,
        "V": bias_mv * 1e-3,  # mV → V, stored as reference channel
    }
    y_units: dict[str, str] = {"I": "A", "Z": "m", "V": "V"}

    if is_time_trace:
        spec_freq = _f(find_hdr(hdr, "SpecFreq", "1000"), 1000.0)
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

    metadata: dict = {
        "filename": path.name,
        "bias_mv": float(
            _f(
                find_hdr(hdr, "BiasVolt.[mV]", find_hdr(hdr, "Biasvolt[mV]", "0")),
                0.0,
            )
        ),
        "spec_freq_hz": float(_f(find_hdr(hdr, "SpecFreq", "1000"), 1000.0)),
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


def _parse_vert_header(hb: bytes) -> dict:
    """Parse key=value lines from a Createc .VERT header block.

    Handles both 'internal / display=value' and plain 'key=value' formats.
    Uses latin-1 encoding to preserve special characters (e.g. Å).
    """
    hdr: dict = {}
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
