"""Reader for Createc vertical-spectroscopy (.VERT) files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

from probeflow.io.common import _f, find_hdr
from probeflow.io.createc_interpretation import (
    createc_vert_measurement_metadata,
    normalize_measurement_mode,
)
from probeflow.io.readers.createc_vert import (
    CreatecVertDecodeReport,
    detect_createc_vert_time_trace,
    parse_createc_vert_header,
    read_createc_vert_report,
)

log = logging.getLogger(__name__)

# Voltage range below this threshold (mV) → file is a time trace, not a bias sweep.
# Configurable via read_spec_file(time_trace_threshold_mv=...) for unusual sweeps.
_TIME_TRACE_THRESHOLD_MV = 1.0


@dataclass(frozen=True)
class SpecChannel:
    """Source-aware metadata for one spectroscopy channel.

    ``key`` is the backwards-compatible lookup key in ``SpecData.channels``.
    ``source_name`` and ``source_label`` preserve the vendor/file identity so
    interpretations can live alongside the decoded data instead of replacing it.
    """

    key: str
    source_name: str
    source_label: str
    unit: str
    roles: tuple[str, ...] = ()
    display_label: str = ""

    def __post_init__(self) -> None:
        if not self.display_label:
            object.__setattr__(self, "display_label", self.key)


@dataclass
class SpecData:
    """All data and metadata from one Createc .VERT spectroscopy file.

    Parameters
    ----------
    header : dict[str, str]
        Raw header key-value pairs from the file.
    channels : dict[str, np.ndarray]
        Named data channels. Known channels are converted to SI units:
        'I' (A), 'Z' (m), 'V' (V). Unknown decoded channels are preserved
        in raw numeric units with conservative unit labels. Interpreted
        dI/dz Createc files expose the moving feedback-height counts as
        'Z feedback' and the static programmed column as 'Z command'.
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
    channel_info : dict[str, SpecChannel]
        Source-aware channel metadata keyed like ``channels``.
    """

    header: dict[str, str]
    channels: dict[str, np.ndarray]
    x_array: np.ndarray
    x_label: str
    x_unit: str
    y_units: dict[str, str]
    position: tuple[float, float]
    metadata: dict[str, Any]
    # Ordered list of all channel names as they should appear in the UI.
    # Defaults to empty for backwards compatibility with old constructors.
    channel_order: list[str] = field(default_factory=list)
    # Subset of ``channel_order`` to preselect when a viewer first opens.
    default_channels: list[str] = field(default_factory=list)
    channel_info: dict[str, SpecChannel] = field(default_factory=dict)

    def __repr__(self) -> str:
        n = self.metadata.get("n_points", "?")
        sw = self.metadata.get("sweep_type", "unknown")
        fname = self.metadata.get("filename", "")
        return f"SpecData({fname!r}, {sw}, {n} pts)"


@dataclass(frozen=True)
class SpecMetadata:
    """Lightweight spectroscopy summary for folder indexing.

    Unlike :class:`SpecData`, this object never contains full numeric channel
    arrays. Readers may stream file rows to count points, but should not build
    array payloads just to populate folder-browser metadata.
    """

    path: Path
    source_format: str
    channels: tuple[str, ...]
    units: tuple[str, ...]
    position: tuple[float, float]
    metadata: dict[str, Any]
    bias: float | None = None
    comment: str | None = None
    acquisition_datetime: str | None = None
    raw_header: dict[str, str] = field(default_factory=dict)
    channel_info: tuple[SpecChannel, ...] = field(default_factory=tuple)


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
    return parse_createc_vert_header(path)


def read_spec_file(
    path: Union[str, Path],
    *,
    time_trace_threshold_mv: float = _TIME_TRACE_THRESHOLD_MV,
    measurement_mode: str | None = None,
) -> SpecData:
    """Read a spectroscopy file (Createc .VERT or Nanonis .dat) into SpecData.

    The file type is identified from its content signature, so callers can
    pass either vendor format without worrying about extensions.
    """
    from probeflow.core.loaders import identify_spectrum_file

    sig = identify_spectrum_file(path)
    if sig.source_format == "nanonis_dat_spectrum":
        from probeflow.io.readers.nanonis_spec import read_nanonis_spec
        spec = read_nanonis_spec(sig.path)
        _apply_measurement_override(spec.metadata, measurement_mode)
        return spec
    return _read_createc_vert(
        sig.path,
        time_trace_threshold_mv=time_trace_threshold_mv,
        measurement_mode=measurement_mode,
    )


def read_spec_metadata(
    path: Union[str, Path],
    *,
    time_trace_threshold_mv: float = _TIME_TRACE_THRESHOLD_MV,
    measurement_mode: str | None = None,
) -> SpecMetadata:
    """Read spectroscopy metadata without loading full numeric arrays."""
    from probeflow.core.loaders import identify_spectrum_file

    sig = identify_spectrum_file(path)
    if sig.source_format == "nanonis_dat_spectrum":
        from probeflow.io.readers.nanonis_spec import read_nanonis_spec_metadata
        meta = read_nanonis_spec_metadata(sig.path)
        _apply_measurement_override(meta.metadata, measurement_mode)
        return meta
    return _read_createc_vert_metadata(
        sig.path,
        time_trace_threshold_mv=time_trace_threshold_mv,
        measurement_mode=measurement_mode,
    )


def _read_createc_vert_metadata(
    path: Path,
    *,
    time_trace_threshold_mv: float = _TIME_TRACE_THRESHOLD_MV,
    measurement_mode: str | None = None,
) -> SpecMetadata:
    """Read Createc .VERT metadata without materialising channel arrays."""
    report = read_createc_vert_report(path, include_arrays=False)
    metadata, bias, comment, position, order, units, channel_info = (
        _metadata_from_vert_report(
            report,
            time_trace_threshold_mv=time_trace_threshold_mv,
            measurement_mode=measurement_mode,
        )
    )
    return SpecMetadata(
        path=path,
        source_format="createc_vert",
        channels=tuple(order),
        units=tuple(units[ch] for ch in order),
        position=position,
        metadata=metadata,
        bias=bias,
        comment=comment,
        acquisition_datetime=None,
        raw_header=report.header,
        channel_info=tuple(channel_info[ch] for ch in order),
    )


def _metadata_from_vert_report(
    report: CreatecVertDecodeReport,
    *,
    time_trace_threshold_mv: float,
    measurement_mode: str | None = None,
) -> tuple[
    dict[str, Any],
    float | None,
    str | None,
    tuple[float, float],
    list[str],
    dict[str, str],
    dict[str, SpecChannel],
]:
    """Return public metadata fields derived from a Createc VERT report."""

    hdr = report.header
    if report.raw_columns and "V" in report.raw_columns:
        bias_for_type = np.asarray(report.raw_columns["V"], dtype=np.float64)
    elif report.bias_min_mv is not None and report.bias_max_mv is not None:
        bias_for_type = np.array(
            [report.bias_min_mv, report.bias_max_mv],
            dtype=np.float64,
        )
    else:
        bias_for_type = np.array([], dtype=np.float64)

    is_time_trace = detect_createc_vert_time_trace(
        hdr,
        bias_for_type,
        time_trace_threshold_mv,
    )

    bias_raw = find_hdr(hdr, "BiasVolt.[mV]", None) or find_hdr(
        hdr,
        "Biasvolt[mV]",
        None,
    )
    bias_mv = _f(bias_raw)
    comment = hdr.get("Titel", "").strip() or None
    measurement = createc_vert_measurement_metadata(
        report,
        measurement_mode=measurement_mode,
    )
    measurement = _with_public_height_aliases(report, measurement)
    order = _public_channel_order(report, measurement)
    units = _public_channel_units(report, measurement, order)
    channel_info = _public_channel_info(report, measurement, order)
    source_channels = [
        info.raw_name
        for info in report.channel_info
    ]

    metadata: dict[str, Any] = {
        "filename": report.path.name,
        "bias_mv": float(bias_mv) if bias_mv is not None else None,
        "spec_freq_hz": _f(find_hdr(hdr, "SpecFreq", "1000"), 1000.0),
        "gain_pre_exp": float(_f(find_hdr(hdr, "GainPre 10^", "9"), 9.0)),
        "fb_log": hdr.get("FBLog", "0").strip() == "1",
        "sweep_type": "time_trace" if is_time_trace else "bias_sweep",
        "n_points": report.raw_table_shape[0],
        "title": comment or "",
        "source": dict(report.source),
        "createc_vert": {
            "file_version": report.file_version,
            "params_line": report.params_line,
            "spec_total_points": report.spec_total_points,
            "spec_position_dac": [report.spec_pos_x, report.spec_pos_y],
            "channel_code": report.channel_code,
            "output_channel_count_marker": report.output_channel_count_marker,
            "column_names": list(report.column_names),
            "warnings": list(report.warnings),
        },
        "measurement_family": measurement["measurement_family"],
        "feedback_mode": measurement["feedback_mode"],
        "derivative_label": measurement["derivative_label"],
        "height_channel": measurement.get("height_channel"),
        "height_source_channel": measurement.get("height_source_channel"),
        "z_command_channel": measurement.get("z_command_channel"),
        "measurement_confidence": measurement["confidence"],
        "measurement_evidence": list(measurement["evidence"]),
    }
    _add_channel_metadata_overlay(metadata, channel_info, order, source_channels)

    return (
        metadata,
        (bias_mv / 1000.0) if bias_mv is not None else None,
        comment,
        _position_from_createc_header(hdr),
        order,
        units,
        channel_info,
    )


def _scaled_channels_from_vert_report(
    report: CreatecVertDecodeReport,
    measurement: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if report.raw_columns is None:
        raise ValueError("VERT report does not include raw numeric arrays")

    if measurement is None:
        measurement = createc_vert_measurement_metadata(report)
    measurement = _with_public_height_aliases(report, measurement)

    channels: dict[str, np.ndarray] = {}
    y_units: dict[str, str] = {}
    for info in report.channel_info:
        raw = report.raw_columns.get(info.raw_name)
        if raw is None:
            continue
        public_name = _public_channel_name(info.canonical_name, measurement)
        channels[public_name] = np.asarray(
            raw * info.scale_factor,
            dtype=np.float64,
        )
        y_units[public_name] = _public_channel_unit(
            info.canonical_name,
            info.unit,
            measurement,
        )
    return channels, y_units


def _public_channel_order(
    report: CreatecVertDecodeReport,
    measurement: dict[str, Any] | None = None,
) -> list[str]:
    if measurement is None:
        measurement = createc_vert_measurement_metadata(report)
    decoded = [
        _public_channel_name(info.canonical_name, measurement)
        for info in report.channel_info
    ]
    if _has_feedback_height_alias(measurement):
        preferred_names = (
            "I",
            "Z feedback",
            "dI/dV",
            "di_q",
            "V",
            "Z command",
        )
    else:
        preferred_names = ("I", "Z", "V")
    preferred = [name for name in preferred_names if name in decoded]
    rest = [name for name in decoded if name not in preferred]
    return preferred + rest


def _public_channel_units(
    report: CreatecVertDecodeReport,
    measurement: dict[str, Any],
    order: list[str],
) -> dict[str, str]:
    units = {
        _public_channel_name(info.canonical_name, measurement): _public_channel_unit(
            info.canonical_name,
            info.unit,
            measurement,
        )
        for info in report.channel_info
    }
    return {name: units[name] for name in order if name in units}


def _public_channel_info(
    report: CreatecVertDecodeReport,
    measurement: dict[str, Any],
    order: list[str],
) -> dict[str, SpecChannel]:
    channels: dict[str, SpecChannel] = {}
    for info in report.channel_info:
        key = _public_channel_name(info.canonical_name, measurement)
        unit = _public_channel_unit(info.canonical_name, info.unit, measurement)
        roles = _createc_channel_roles(info.raw_name, key, measurement)
        channels[key] = SpecChannel(
            key=key,
            source_name=info.raw_name,
            source_label=info.raw_name,
            unit=unit,
            roles=roles,
            display_label=_display_label_for_source(key, info.raw_name, roles),
        )
    return {key: channels[key] for key in order if key in channels}


def _public_channel_name(name: str, measurement: dict[str, Any]) -> str:
    if _has_feedback_height_alias(measurement):
        if name == "Raw column 9":
            return "Z feedback"
        if name == "Z":
            return "Z command"
    return name


def _public_channel_unit(
    name: str,
    unit: str,
    measurement: dict[str, Any],
) -> str:
    if _has_feedback_height_alias(measurement) and name == "Raw column 9":
        return "DAC"
    return unit


def _has_feedback_height_alias(measurement: dict[str, Any]) -> bool:
    return (
        measurement.get("measurement_family") == "iz"
        and measurement.get("height_source_channel") == "Raw column 9"
    )


def _with_public_height_aliases(
    report: CreatecVertDecodeReport,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """Attach public dI/dz height aliases when the raw feedback column exists."""

    if (
        measurement.get("measurement_family") == "iz"
        and "Raw column 9" in report.column_names
    ):
        measurement = dict(measurement)
        measurement["height_channel"] = "Z feedback"
        measurement["height_source_channel"] = "Raw column 9"
        if "Z" in report.column_names:
            measurement["z_command_channel"] = "Z command"
        return measurement
    return measurement


def _createc_channel_roles(
    source_name: str,
    key: str,
    measurement: dict[str, Any],
) -> tuple[str, ...]:
    if _has_feedback_height_alias(measurement) and source_name == "Raw column 9":
        return ("z_feedback", "height_counts")
    if _has_feedback_height_alias(measurement) and source_name == "Z":
        return ("z_command",)
    return infer_spec_channel_roles(source_name)


def infer_spec_channel_roles(name: str) -> tuple[str, ...]:
    """Return conservative spectroscopy roles inferred from a channel name."""

    text = name.strip().lower()
    if text in {"v", "bias", "bias calc"} or text.startswith("bias "):
        return ("bias_axis",)
    if text == "i" or text.startswith("current"):
        return ("current",)
    if text in {"z", "z rel", "z-controller"} or text.startswith("z "):
        return ("z",)
    if any(
        token in text
        for token in ("di/dv", "di/dz", "di_q", "di2_q", "lockin", "li demod")
    ):
        return ("lockin_derivative",)
    if (
        text.startswith("adc")
        or text.startswith("dac")
        or text.startswith("input")
        or text.startswith("oc ")
        or text.startswith("na")
    ):
        return ("auxiliary",)
    return ("unknown",)


def _display_label_for_source(
    key: str,
    source_name: str,
    roles: tuple[str, ...],
) -> str:
    if key == source_name:
        return key
    if "z_feedback" in roles:
        return f"{source_name} - {key}"
    if "z_command" in roles:
        return f"{source_name} - command"
    return f"{source_name} - {key}"


def spec_channel_to_dict(channel: SpecChannel) -> dict[str, Any]:
    """Return a JSON-serialisable representation of ``SpecChannel``."""

    return {
        "key": channel.key,
        "source_name": channel.source_name,
        "source_label": channel.source_label,
        "unit": channel.unit,
        "roles": list(channel.roles),
        "display_label": channel.display_label,
    }


def _add_channel_metadata_overlay(
    metadata: dict[str, Any],
    channel_info: dict[str, SpecChannel],
    order: list[str],
    source_channels: list[str] | None = None,
) -> None:
    ordered = [channel_info[key] for key in order if key in channel_info]
    metadata["channel_roles"] = {
        channel.key: list(channel.roles)
        for channel in ordered
    }
    if source_channels is not None:
        metadata["source_channels"] = list(source_channels)
    else:
        metadata["source_channels"] = [
            channel.source_name
            for channel in ordered
        ]
    metadata["channel_info"] = [
        spec_channel_to_dict(channel)
        for channel in ordered
    ]


def _default_spec_channels(channel_order: list[str]) -> list[str]:
    if "I" in channel_order:
        return ["I"]
    current_like = [ch for ch in channel_order if "current" in ch.lower()]
    if current_like:
        return [current_like[0]]
    return channel_order[:1]


def _position_from_createc_header(hdr: dict[str, str]) -> tuple[float, float]:
    dac_to_a_xy = _f(find_hdr(hdr, "Dacto[A]xy", "1"), 1.0)
    ox_dac = _f(find_hdr(hdr, "OffsetX", "0"), 0.0)
    oy_dac = _f(find_hdr(hdr, "OffsetY", "0"), 0.0)
    return (ox_dac * dac_to_a_xy * 1e-10, oy_dac * dac_to_a_xy * 1e-10)


def _read_createc_vert(
    path: Path,
    *,
    time_trace_threshold_mv: float = _TIME_TRACE_THRESHOLD_MV,
    measurement_mode: str | None = None,
) -> SpecData:
    """Read a Createc .VERT spectroscopy file and return a SpecData object.

    The data is converted to SI units on read. The sweep type (bias sweep vs
    time trace) is detected from the Vpoint header entries first, falling back
    to checking the voltage range in the data column.
    """
    report = read_createc_vert_report(path, include_arrays=True)
    if report.raw_columns is None:
        raise ValueError(f"{path.name}: internal VERT report has no numeric arrays")

    metadata, _bias, _comment, position, channel_order, _units, channel_info = (
        _metadata_from_vert_report(
            report,
            time_trace_threshold_mv=time_trace_threshold_mv,
            measurement_mode=measurement_mode,
        )
    )
    measurement = {
        "measurement_family": metadata.get("measurement_family"),
        "height_source_channel": metadata.get("height_source_channel"),
    }
    channels, y_units = _scaled_channels_from_vert_report(report, measurement)

    idx = report.raw_columns.get("idx")
    if idx is None:
        raise ValueError(f"{path.name}: missing idx column in VERT data")
    spec_freq = metadata["spec_freq_hz"]
    if metadata["sweep_type"] == "time_trace":
        x_array = idx / spec_freq  # sample index / Hz → seconds
        x_label = "Time (s)"
        x_unit = "s"
    else:
        x_array = channels["V"]
        x_label = "Bias (V)"
        x_unit = "V"

    log.info(
        "%s: %s, %d pts, pos=(%.3g, %.3g) m",
        path.name,
        metadata["sweep_type"],
        metadata["n_points"],
        position[0],
        position[1],
    )

    return SpecData(
        header=report.header,
        channels=channels,
        x_array=x_array,
        x_label=x_label,
        x_unit=x_unit,
        y_units=y_units,
        position=position,
        metadata=metadata,
        channel_order=channel_order,
        default_channels=_default_spec_channels(channel_order),
        channel_info=channel_info,
    )


def _apply_measurement_override(
    metadata: dict[str, Any],
    measurement_mode: str | None,
) -> None:
    """Populate generic spectroscopy measurement metadata for non-Createc paths."""

    mode = normalize_measurement_mode(measurement_mode)
    if mode == "iz":
        metadata.update(
            {
                "measurement_family": "iz",
                "feedback_mode": "on",
                "derivative_label": "dI/dz",
                "height_channel": metadata.get("height_channel"),
                "height_source_channel": metadata.get("height_source_channel"),
                "z_command_channel": metadata.get("z_command_channel"),
                "measurement_confidence": "override",
                "measurement_evidence": ["measurement_mode override: iz"],
            }
        )
        return
    if mode == "sts" or "measurement_family" not in metadata:
        metadata.update(
            {
                "measurement_family": "sts",
                "feedback_mode": metadata.get("feedback_mode", "unknown"),
                "derivative_label": metadata.get("derivative_label", "dI/dV"),
                "height_channel": metadata.get("height_channel"),
                "height_source_channel": metadata.get("height_source_channel"),
                "z_command_channel": metadata.get("z_command_channel"),
                "measurement_confidence": (
                    "override" if mode == "sts" else metadata.get(
                        "measurement_confidence",
                        "low",
                    )
                ),
                "measurement_evidence": metadata.get("measurement_evidence", []),
            }
        )
