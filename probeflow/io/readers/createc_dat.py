"""Low-level Createc ``.dat`` image decoding and reporting.

This module intentionally stops short of promising that every Createc channel
is STM topography/current.  It decodes the raw container, records the decisions
made along the way, and exposes small adapters for the legacy STM paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import zlib

import numpy as np

from probeflow.io.common import (
    _f,
    _i,
    find_hdr,
    get_dac_bits,
    i_scale_a_per_dac,
    parse_header,
    trim_stack,
    v_per_dac,
    z_scale_m_per_dac,
)


@dataclass(frozen=True)
class CreatecChannelInfo:
    """Best-known interpretation for one decoded Createc image plane."""

    native_index: int
    name: str
    unit: str
    direction: str | None
    scale_factor: float
    semantic: str


@dataclass(frozen=True)
class CreatecDatDecodeReport:
    """Structured report for a decoded Createc image ``.dat`` file."""

    path: Path
    original_header: dict[str, str]
    header: dict[str, str]
    original_Nx: int
    original_Ny: int
    payload_float_count: int
    detected_channel_count: int
    ignored_tail_float_count: int
    raw_channels_dac: np.ndarray | None
    trimmed_Ny: int
    first_column_removed: bool
    first_column_diagnostics: list[dict[str, float | int | None]]
    scale_factors: dict[str, float | int]
    channel_info: tuple[CreatecChannelInfo, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def decoded_Nx(self) -> int:
        return int(self.header.get("Num.X", self.original_Nx))

    @property
    def decoded_Ny(self) -> int:
        return int(self.header.get("Num.Y", self.trimmed_Ny))


def read_createc_dat_report(
    path,
    *,
    include_raw: bool = True,
    remove_first_column: bool = True,
) -> CreatecDatDecodeReport:
    """Decode a Createc image ``.dat`` into a report.

    ``include_raw=False`` still validates/decompresses the payload and inspects
    channel 0 for row trimming, but it does not keep image arrays in the
    returned report.  This is the metadata fast path: callers get corrected
    dimensions and channel names without constructing a full :class:`Scan`.
    """

    path = Path(path)
    raw = path.read_bytes()
    warnings: list[str] = []

    if b"DATA" not in raw:
        raise ValueError(
            f"{path.name}: missing DATA marker — not a valid Createc .dat file"
        )

    hb, comp = raw.split(b"DATA", 1)
    original_header = parse_header(hb)
    header = dict(original_header)

    Nx = _i(find_hdr(header, "Num.X", 0), 0)
    Ny = _i(find_hdr(header, "Num.Y", 0), 0)
    if Nx <= 0 or Ny <= 0:
        raise ValueError(f"{path.name}: invalid dimensions Nx={Nx}, Ny={Ny}")

    if _i(find_hdr(header, "ScanmodeSine", 0), 0) != 0:
        raise NotImplementedError(f"{path.name}: sine scan mode is not supported")

    try:
        payload = zlib.decompress(comp)
    except zlib.error as exc:
        raise ValueError(f"{path.name}: zlib decompression failed — {exc}") from exc

    if len(payload) % 4:
        warnings.append(
            f"decompressed payload has {len(payload) % 4} trailing byte(s) "
            "outside complete float32 values"
        )

    payload_float_count = len(payload) // 4
    num_chan = _detect_channel_count(payload_float_count, Ny, Nx, header)
    needed = num_chan * Ny * Nx
    ignored_tail = payload_float_count - needed
    if ignored_tail:
        warnings.append(
            f"ignored {ignored_tail} trailing float32 value(s) after "
            f"{num_chan} channel(s)"
        )

    arr = np.frombuffer(payload, dtype="<f4", count=needed).reshape((num_chan, Ny, Nx))
    trimmed, trimmed_Ny = trim_stack(arr)
    if trimmed_Ny != Ny:
        warnings.append(f"trimmed image height from {Ny} to {trimmed_Ny} row(s)")
    header["Num.Y"] = str(trimmed_Ny)

    first_column_diagnostics = _first_column_diagnostics(trimmed)
    if remove_first_column:
        if Nx <= 1:
            raise ValueError(f"{path.name}: cannot remove first column from Nx={Nx}")
        decoded = trimmed[:, :, 1:]
        header["Num.X"] = str(Nx - 1)
    else:
        decoded = trimmed
        header["Num.X"] = str(Nx)

    if include_raw:
        raw_channels_dac = np.array(decoded, dtype=np.float32, copy=True)
    else:
        raw_channels_dac = None

    bits = get_dac_bits(header)
    vpd = v_per_dac(bits)
    zs = z_scale_m_per_dac(header, vpd)
    is_ = i_scale_a_per_dac(header, vpd)
    scale_factors: dict[str, float | int] = {
        "dac_bits": bits,
        "v_per_dac": vpd,
        "z_m_per_dac": zs,
        "current_a_per_dac": is_,
    }

    channel_info = _channel_info(num_chan, zs, is_, header)

    return CreatecDatDecodeReport(
        path=path,
        original_header=original_header,
        header=header,
        original_Nx=Nx,
        original_Ny=Ny,
        payload_float_count=payload_float_count,
        detected_channel_count=num_chan,
        ignored_tail_float_count=ignored_tail,
        raw_channels_dac=raw_channels_dac,
        trimmed_Ny=trimmed_Ny,
        first_column_removed=remove_first_column,
        first_column_diagnostics=first_column_diagnostics,
        scale_factors=scale_factors,
        channel_info=channel_info,
        warnings=tuple(warnings),
    )


def scale_channels_for_scan(report: CreatecDatDecodeReport) -> list[np.ndarray]:
    """Return decoded channels scaled according to ``report.channel_info``."""

    if report.raw_channels_dac is None:
        raise ValueError("report does not include raw channel arrays")

    planes: list[np.ndarray] = []
    for info in report.channel_info:
        arr = report.raw_channels_dac[info.native_index] * info.scale_factor
        planes.append(np.asarray(arr, dtype=np.float64))
    return planes


def scan_range_m_from_header(hdr: dict[str, str]) -> tuple[float, float]:
    """Return Createc lateral scan range in metres."""

    lx_a = _f(hdr.get("Length x[A]", "0"), 0.0)
    ly_a = _f(hdr.get("Length y[A]", "0"), 0.0)
    return (lx_a * 1e-10, ly_a * 1e-10)


def _detect_channel_count(
    payload_float_count: int,
    Ny: int,
    Nx: int,
    hdr: dict[str, str] | None = None,
) -> int:
    """Detect channel count from payload size.

    Createc usually records the stored image-plane count in ``Channels``.
    Trust it when the payload contains that many complete planes; otherwise
    keep the legacy 4/2 fallback for older or inconsistent headers.
    """

    pixels = Ny * Nx
    if pixels <= 0:
        raise ValueError(f"Invalid image dimensions Ny={Ny}, Nx={Nx}")

    header_channels = _i(_header_value(hdr or {}, "Channels"), None)
    if header_channels is not None and header_channels > 0:
        needed = header_channels * pixels
        if payload_float_count >= needed:
            return header_channels

    exact, tail = divmod(payload_float_count, pixels)
    if tail == 0 and exact > 0 and exact not in (2, 4):
        return exact

    for n in (4, 2, 1):
        if payload_float_count >= n * pixels:
            return n
    raise ValueError(
        "Payload too small for one image channel "
        f"(Ny={Ny}, Nx={Nx}, payload floats={payload_float_count})"
    )


def has_legacy_stm_two_channel_layout(report: CreatecDatDecodeReport) -> bool:
    """Return True for legacy [Z forward, Current forward] Createc DAT files."""

    if report.detected_channel_count != 2 or len(report.channel_info) != 2:
        return False
    first, second = report.channel_info
    return (
        first.semantic == "z"
        and first.direction == "forward"
        and second.semantic == "current"
        and second.direction == "forward"
    )


def has_canonical_stm_four_channel_layout(report: CreatecDatDecodeReport) -> bool:
    """Return True for native [Z fwd, I fwd, Z bwd, I bwd] DAT files."""

    if report.detected_channel_count != 4 or len(report.channel_info) != 4:
        return False
    expected = (
        ("z", "forward"),
        ("current", "forward"),
        ("z", "backward"),
        ("current", "backward"),
    )
    return all(
        info.semantic == semantic and info.direction == direction
        for info, (semantic, direction) in zip(report.channel_info, expected)
    )


def _first_column_diagnostics(stack: np.ndarray) -> list[dict[str, float | int | None]]:
    """Summarise the first stored column before it is removed."""

    diagnostics: list[dict[str, float | int | None]] = []
    for k, arr in enumerate(stack):
        first = arr[:, 0]
        if arr.shape[1] > 1:
            delta = first - arr[:, 1]
            finite_delta = delta[np.isfinite(delta)]
            mean_abs_delta = (
                float(np.mean(np.abs(finite_delta))) if finite_delta.size else None
            )
            median_abs_delta = (
                float(np.median(np.abs(finite_delta))) if finite_delta.size else None
            )
        else:
            mean_abs_delta = None
            median_abs_delta = None

        finite_first = first[np.isfinite(first)]
        diagnostics.append(
            {
                "channel_index": k,
                "exact_zero_count": int(np.count_nonzero(first == 0.0)),
                "finite_count": int(finite_first.size),
                "mean": float(np.mean(finite_first)) if finite_first.size else None,
                "mean_abs_delta_to_second_column": mean_abs_delta,
                "median_abs_delta_to_second_column": median_abs_delta,
            }
        )
    return diagnostics


def _channel_info(
    num_chan: int,
    z_scale: float,
    current_scale: float,
    hdr: dict[str, str] | None = None,
) -> tuple[CreatecChannelInfo, ...]:
    """Return best-known Createc channel metadata in native channel order."""

    selected = _selected_scan_channels(hdr or {}, num_chan)
    if selected is not None:
        infos: list[CreatecChannelInfo] = []
        half = len(selected)
        for native_index in range(num_chan):
            signal = selected[native_index % half]
            direction = "forward" if native_index < half else "backward"
            base_name, unit, scale, semantic = _selected_signal_metadata(
                signal,
                z_scale,
                current_scale,
            )
            infos.append(
                CreatecChannelInfo(
                    native_index=native_index,
                    name=f"{base_name} {direction}",
                    unit=unit,
                    direction=direction,
                    scale_factor=float(scale),
                    semantic=semantic,
                )
            )
        return tuple(infos)

    # Future Createc AFM support should replace these positional fallbacks with
    # metadata derived from real AFM .dat headers.  The intended shape is:
    # (1) identify any header keys that describe saved channel names/order,
    # (2) map known AFM signals such as frequency shift, amplitude, drive, and
    #     dissipation to conservative units/scales,
    # (3) preserve unknown planes as raw DAC channels with decode warnings.
    known = [
        ("Z forward", "m", "forward", z_scale, "z"),
        ("Current forward", "A", "forward", current_scale, "current"),
        ("Z backward", "m", "backward", z_scale, "z"),
        ("Current backward", "A", "backward", current_scale, "current"),
        ("Frequency shift", "Hz", None, 1.0, "frequency_shift"),
        ("Amplitude", "unknown", None, 1.0, "amplitude"),
        ("Drive", "V", None, 1.0, "drive"),
        ("Phase", "unknown", None, 1.0, "phase"),
    ]

    infos: list[CreatecChannelInfo] = []
    for k in range(num_chan):
        if k < len(known):
            name, unit, direction, scale, semantic = known[k]
        else:
            name, unit, direction, scale, semantic = (
                f"Raw channel {k}",
                "DAC",
                None,
                1.0,
                "unknown",
            )
        infos.append(
            CreatecChannelInfo(
                native_index=k,
                name=name,
                unit=unit,
                direction=direction,
                scale_factor=float(scale),
                semantic=semantic,
            )
        )
    return tuple(infos)


def _selected_scan_channels(
    hdr: dict[str, str],
    num_chan: int,
) -> list[int] | None:
    """Return selected Createc channel bits when they match stored planes."""

    select = _i(_header_value(hdr, "Channelselectval"), None)
    if select is None or select <= 0 or num_chan % 2:
        return None
    bits = [bit for bit in range(32) if select & (1 << bit)]
    if len(bits) * 2 != num_chan:
        return None
    return bits


def _selected_signal_metadata(
    bit: int,
    z_scale: float,
    current_scale: float,
) -> tuple[str, str, float, str]:
    if bit == 0:
        return "Z", "m", z_scale, "z"
    if bit == 1:
        return "Current", "A", current_scale, "current"
    return f"Aux{max(0, bit - 1)}", "DAC", 1.0, "unknown"


def _header_value(hdr: dict[str, str], key: str):
    """Return an exact Createc header value, ignoring case and whitespace."""

    target = key.lower()
    for k, value in hdr.items():
        if str(k).strip().lower() == target:
            return value
    return None
