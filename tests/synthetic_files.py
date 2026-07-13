"""Small generated SPM fixtures used instead of committed experimental spectra."""

from __future__ import annotations

from pathlib import Path
import zlib

import numpy as np


def write_createc_vert(
    path: Path,
    *,
    sweep: str = "bias",
    bias_mv: float = -50.0,
    points: int = 25,
) -> Path:
    """Write a compact, anonymous Createc VERT spectrum."""

    if sweep not in {"bias", "time"}:
        raise ValueError("sweep must be 'bias' or 'time'")
    if sweep == "time":
        biases = np.full(points, bias_mv, dtype=float)
        v0 = v1 = bias_mv
    else:
        biases = np.linspace(-300.0, -50.0, points)
        v0, v1 = float(biases[0]), float(biases[-1])

    header = (
        "[ParVERT30]\r\n"
        "DAC-Type=20bit\r\n"
        "GainPre 10^=9\r\n"
        "Dacto[A]xy=0.00083\r\n"
        "Dacto[A]z=0.00018\r\n"
        "GainZ=1\r\n"
        "OffsetX=100\r\n"
        "OffsetY=200\r\n"
        "SpecFreq=1000\r\n"
        f"Biasvolt[mV]={bias_mv}\r\n"
        "FBOff=1\r\n"
        f"Vpoint0.t=10\r\nVpoint0.V={v0}\r\n"
        f"Vpoint1.t=10\r\nVpoint1.V={v1}\r\n"
        "Vpoint2.t=0\r\nVpoint2.V=0\r\n"
        "DATA\r\n"
        f"    {points}    100    200    1\r\n"
    )
    rows = "".join(
        f"{i}\t{bias:.9g}\t0\t{-1000.0 - i:.9g}\r\n"
        for i, bias in enumerate(biases)
    )
    path.write_bytes((header + rows).encode("latin-1"))
    return path


def write_nanonis_spec(path: Path, *, points: int = 25) -> Path:
    """Write a compact, anonymous Nanonis bias spectrum."""

    header = (
        "Experiment\tBias spectroscopy\n"
        "User\t\n"
        "Saved Date\t01.01.2000 00:00:00\n"
        "X (m)\t1e-9\n"
        "Y (m)\t2e-9\n"
        "Bias>Bias (V)\t0.05\n"
        "[DATA]\n"
        "Bias calc (V)\tCurrent (A)\tCurrent [AVG] (A)\t"
        "LockIn [AVG] (A)\tOC M1 Freq. Shift (Hz)\tInput 6 (V)\n"
    )
    rows = []
    for i, bias in enumerate(np.linspace(-0.3, 0.3, points)):
        current = 1e-10 * (i + 1)
        rows.append(
            f"{bias:.9g}\t{current:.9g}\t{current * 0.95:.9g}\t"
            f"{current * 0.1:.9g}\t{-5.0 + i * 0.1:.9g}\t{bias * bias:.9g}\n"
        )
    path.write_text(header + "".join(rows), encoding="latin-1")
    return path


def write_legacy_two_channel_dat(path: Path) -> Path:
    """Write a canonical two-forward-plane Createc image fixture."""

    nx, ny = 8, 6
    header = (
        "[Paramco32]\n"
        f"Num.X={nx}\n"
        f"Num.Y={ny}\n"
        "Length x[A]=80\n"
        "Length y[A]=60\n"
        "DAC-Type=20bit\n"
        "Dacto[A]z=0.00018\n"
        "GainZ=1\n"
        "GainPre 10^=9\n"
        "Biasvolt[mV]=-50\n"
        "Current[A]=1e-10\n"
    ).encode("latin-1")
    z = np.arange(nx * ny, dtype="<f4")
    current = np.arange(nx * ny, 2 * nx * ny, dtype="<f4")
    payload = np.concatenate([z, current]).astype("<f4").tobytes()
    path.write_bytes(header + b"DATA" + zlib.compress(payload))
    return path
