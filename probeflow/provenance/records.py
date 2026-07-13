"""Lightweight, graph-ready provenance records for ProbeFlow image exports.

The model here is intentionally linear:

    raw file -> opened image -> processing steps -> export/session

Each processing step still records input and output state ids so the same data
can later be lifted into a graph without changing sidecar files.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from probeflow.core.source_identity import privacy_safe_path, sanitize_export_data


SCHEMA_VERSION = 1

PROCESSED_EXPORT_WARNING = (
    "This file was exported by ProbeFlow and is not raw data.\n"
    "Original file: {filename}\n"
    "Processing steps: {steps}\n"
    "For critical quantitative measurements, compare against the original raw file."
)

DAT_TO_SXM_CONVERSION_WARNING = (
    "This file was converted from Createc .dat to Nanonis-compatible .sxm by ProbeFlow.\n"
    "This is a compatibility export, not the original measurement file.\n"
    "Original file: {filename}\n"
    "Converter version: {version}"
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version() -> str | None:
    try:
        from probeflow import __version__

        return str(__version__) if __version__ else None
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _hash_payload(payload: Any, prefix: str) -> str:
    text = json.dumps(_json_safe(payload), sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _file_hash(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class SourceRecord:
    """Structured identity for the raw file and selected opened image channel."""

    source_filename: str | None
    source_path: str | None
    source_file_type: str | None
    channel: str | None
    loader_name: str | None
    loader_version: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    file_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return sanitize_export_data({
            "source_filename": self.source_filename,
            "source_path": self.source_path,
            "source_file_type": self.source_file_type,
            "channel": self.channel,
            "loader_name": self.loader_name,
            "loader_version": self.loader_version,
            "metadata": _json_safe(self.metadata),
            "file_hash": self.file_hash,
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRecord":
        return cls(
            source_filename=data.get("source_filename"),
            source_path=data.get("source_path"),
            source_file_type=data.get("source_file_type"),
            channel=data.get("channel"),
            loader_name=data.get("loader_name"),
            loader_version=data.get("loader_version"),
            metadata=dict(data.get("metadata") or {}),
            file_hash=data.get("file_hash"),
        )


@dataclass(frozen=True)
class ProvenanceStep:
    """One provenance-aware operation step.

    ``input_state_id`` and ``output_state_id`` are linear today but graph-ready:
    future graph conversion can treat them as DataNode identifiers.

    .. note::
       Review arch-backend #12 (2026-05-28) — this class was previously
       named ``ProcessingStep``, which collided with
       :class:`probeflow.processing.state.ProcessingStep` (a different
       dataclass with different fields).  The legacy name is kept as
       an alias at the bottom of this module for backward compatibility,
       but new code should use ``ProvenanceStep``.
    """

    step_id: str
    operation_id: str
    operation_name: str
    operation_version: str | None
    parameters: dict[str, Any]
    input_state_id: str
    output_state_id: str
    timestamp: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return sanitize_export_data({
            "step_id": self.step_id,
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "operation_version": self.operation_version,
            "parameters": _json_safe(self.parameters),
            "input_state_id": self.input_state_id,
            "output_state_id": self.output_state_id,
            "timestamp": self.timestamp,
            "warnings": list(self.warnings),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceStep":
        return cls(
            step_id=str(data["step_id"]),
            operation_id=str(data["operation_id"]),
            operation_name=str(data.get("operation_name") or data["operation_id"]),
            operation_version=data.get("operation_version"),
            parameters=dict(data.get("parameters") or {}),
            input_state_id=str(data["input_state_id"]),
            output_state_id=str(data["output_state_id"]),
            timestamp=str(data.get("timestamp") or _utc_now()),
            warnings=tuple(str(w) for w in data.get("warnings") or ()),
        )


# Backward-compatibility alias.  See ProvenanceStep docstring for the
# motivation (arch-backend #12).
ProcessingStep = ProvenanceStep


@dataclass
class ProcessingHistory:
    """Linear history for one opened image/channel."""

    source_record: SourceRecord
    steps: list[ProvenanceStep] = field(default_factory=list)
    current_state_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.current_state_id is None:
            self.current_state_id = _hash_payload(self.source_record.to_dict(), "source")

    def append_step(
        self,
        *,
        operation_id: str,
        operation_name: str | None = None,
        operation_version: str | None = None,
        parameters: dict[str, Any] | None = None,
        warnings: list[str] | tuple[str, ...] | None = None,
        timestamp: str | None = None,
        output_state_id: str | None = None,
    ) -> ProvenanceStep:
        input_state_id = str(self.current_state_id)
        params = copy.deepcopy(dict(parameters or {}))
        step_index = len(self.steps) + 1
        step_id = f"step-{step_index:04d}"
        output = output_state_id or _hash_payload(
            {
                "input_state_id": input_state_id,
                "operation_id": operation_id,
                "parameters": params,
                "step_index": step_index,
            },
            "state",
        )
        step = ProvenanceStep(
            step_id=step_id,
            operation_id=str(operation_id),
            operation_name=str(operation_name or operation_id),
            operation_version=operation_version,
            parameters=params,
            input_state_id=input_state_id,
            output_state_id=output,
            timestamp=timestamp or _utc_now(),
            warnings=tuple(str(w) for w in warnings or ()),
        )
        self.steps.append(step)
        self.current_state_id = output
        return step

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "source_record": self.source_record.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "current_state_id": self.current_state_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingHistory":
        source_data = data.get("source_record") or data.get("source") or {}
        history = cls(
            source_record=SourceRecord.from_dict(source_data),
            steps=[
                ProvenanceStep.from_dict(step)
                for step in data.get("steps") or []
            ],
            current_state_id=data.get("current_state_id"),
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
        )
        if history.steps and not data.get("current_state_id"):
            history.current_state_id = history.steps[-1].output_state_id
        return history

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "ProcessingHistory":
        return cls.from_dict(json.loads(payload))

    def short_summary(self) -> str:
        if not self.steps:
            return "No processing steps"
        return "; ".join(_step_summary(step) for step in self.steps)


@dataclass(frozen=True)
class ExportRecord:
    """Reliable sidecar record for one exported artifact."""

    export_path: str | None
    export_format: str
    source_info: SourceRecord
    processing_history: ProcessingHistory
    display_settings: dict[str, Any]
    timestamp: str
    warning: str | None = None
    warnings: tuple[str, ...] = ()
    rois: dict[str, Any] | None = None
    masks: dict[str, Any] | None = None
    schema_version: int = SCHEMA_VERSION
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        warnings = list(self.warnings)
        if self.warning and self.warning not in warnings:
            warnings.insert(0, self.warning)
        return sanitize_export_data({
            "record_type": "probeflow_export",
            "schema_version": int(self.schema_version),
            "export_path": self.export_path,
            "export_format": self.export_format,
            "source_info": self.source_info.to_dict(),
            "processing_history": self.processing_history.to_dict(),
            "display_settings": _json_safe(self.display_settings),
            "timestamp": self.timestamp,
            "warning": self.warning,
            "warnings": warnings,
            "rois": _json_safe(self.rois),
            "masks": _json_safe(self.masks),
            "summary": self.summary,
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportRecord":
        return cls(
            export_path=data.get("export_path"),
            export_format=str(data.get("export_format") or ""),
            source_info=SourceRecord.from_dict(data.get("source_info") or {}),
            processing_history=ProcessingHistory.from_dict(
                data.get("processing_history") or {}
            ),
            display_settings=dict(data.get("display_settings") or {}),
            timestamp=str(data.get("timestamp") or _utc_now()),
            warning=data.get("warning"),
            warnings=tuple(str(w) for w in data.get("warnings") or ()),
            rois=data.get("rois") if isinstance(data.get("rois"), dict) else None,
            masks=data.get("masks") if isinstance(data.get("masks"), dict) else None,
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
            summary=data.get("summary"),
        )

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "ExportRecord":
        return cls.from_dict(json.loads(payload))


def source_record_from_scan(
    scan: Any,
    *,
    channel_index: int | None = 0,
    channel_name: str | None = None,
    include_file_hash: bool = False,
) -> SourceRecord:
    """Build a SourceRecord from a loaded Scan-like object."""

    source_path = getattr(scan, "source_path", None)
    path = Path(source_path) if source_path is not None else None
    source_format = getattr(scan, "source_format", None)
    loader_name = {
        "dat": "Createc .dat reader",
        "sxm": "Nanonis .sxm reader",
    }.get(str(source_format), f"{source_format} reader" if source_format else None)
    file_type = {
        "dat": "Createc .dat",
        "sxm": "Nanonis .sxm",
    }.get(str(source_format), source_format)

    if channel_name is None and channel_index is not None:
        try:
            names = list(getattr(scan, "plane_names", []) or [])
            if 0 <= int(channel_index) < len(names):
                channel_name = str(names[int(channel_index)])
        except Exception:
            channel_name = None

    metadata: dict[str, Any] = {}
    try:
        if channel_index is not None and 0 <= int(channel_index) < len(scan.planes):
            metadata["array_shape"] = list(scan.planes[int(channel_index)].shape)
    except Exception:
        pass
    try:
        metadata["scan_range_m"] = list(getattr(scan, "scan_range_m"))
    except Exception:
        pass
    try:
        if channel_index is not None and 0 <= int(channel_index) < len(scan.plane_units):
            metadata["unit"] = str(scan.plane_units[int(channel_index)])
    except Exception:
        pass
    experiment = getattr(scan, "experiment_metadata", None)
    if experiment:
        metadata["experiment_metadata"] = _json_safe(dict(experiment))

    header = getattr(scan, "header", None) or {}
    if isinstance(header, dict):
        interesting = (
            "REC_DATE",
            "REC_TIME",
            "BIAS",
            "SCAN_PIXELS",
            "SCAN_RANGE",
            "SCAN_DIR",
            "Num.X",
            "Num.Y",
            "Biasvolt[mV]",
            "SetPoint",
        )
        header_subset = {
            key: header[key]
            for key in interesting
            if key in header
        }
        if header_subset:
            metadata["header"] = _json_safe(header_subset)

    return SourceRecord(
        source_filename=path.name if path is not None else None,
        source_path=privacy_safe_path(path),
        source_file_type=str(file_type) if file_type is not None else None,
        channel=channel_name,
        loader_name=loader_name,
        loader_version=_version(),
        metadata=metadata,
        file_hash=_file_hash(path) if include_file_hash else None,
    )


def processing_history_from_scan(
    scan: Any,
    *,
    channel_index: int | None = 0,
    channel_name: str | None = None,
    processing_state: Any | None = None,
    include_file_hash: bool = False,
) -> ProcessingHistory:
    """Create a linear opened-image history from a Scan and ProcessingState."""

    source = source_record_from_scan(
        scan,
        channel_index=channel_index,
        channel_name=channel_name,
        include_file_hash=include_file_hash,
    )
    history = ProcessingHistory(source_record=source)
    load_name = _load_operation_name(source.source_file_type)
    history.append_step(
        operation_id="file_load",
        operation_name=load_name,
        operation_version=source.loader_version,
        parameters={
            "source_path": source.source_path,
            "source_file_type": source.source_file_type,
            "channel": source.channel,
            "loader_name": source.loader_name,
        },
    )

    if processing_state is None:
        processing_state = getattr(scan, "processing_state", None)
    append_processing_state(history, processing_state)
    return history


def append_processing_state(
    history: ProcessingHistory,
    processing_state: Any | None,
) -> ProcessingHistory:
    """Append canonical ProcessingState steps to a ProcessingHistory."""

    if processing_state is None:
        return history
    if hasattr(processing_state, "to_dict"):
        state_data = processing_state.to_dict()
    else:
        state_data = dict(processing_state)

    for step in state_data.get("steps") or []:
        op = str(step.get("op") or step.get("operation_id") or "")
        if not op:
            continue
        params = dict(step.get("params") or step.get("parameters") or {})
        op_name = _operation_name(op, params)
        history.append_step(
            operation_id=op,
            operation_name=op_name,
            operation_version=_version(),
            parameters=params,
        )
    return history


def build_export_record(
    history: ProcessingHistory,
    *,
    export_path: str | Path | None,
    export_format: str,
    display_settings: dict[str, Any] | None = None,
    export_parameters: dict[str, Any] | None = None,
    warnings: list[str] | tuple[str, ...] | None = None,
    conversion: str | None = None,
    rois: dict[str, Any] | None = None,
    masks: dict[str, Any] | None = None,
) -> ExportRecord:
    """Build the sidecar ExportRecord and append export/conversion steps."""

    export_history = ProcessingHistory.from_dict(history.to_dict())
    warning_list = [str(w) for w in warnings or ()]

    if conversion == "dat_to_sxm":
        conversion_warning = DAT_TO_SXM_CONVERSION_WARNING.format(
            filename=history.source_record.source_filename or "unknown",
            version=_version() or "unknown",
        )
        if conversion_warning not in warning_list:
            warning_list.append(conversion_warning)
        export_history.append_step(
            operation_id="dat_to_sxm",
            operation_name="Createc .dat to Nanonis-compatible .sxm conversion",
            operation_version=_version(),
            parameters={
                "source_format": history.source_record.source_file_type,
                "export_format": "sxm",
                "converter_version": _version(),
            },
            warnings=(conversion_warning,),
        )

    export_params = dict(export_parameters or {})
    if display_settings:
        export_params.setdefault("display_settings", copy.deepcopy(display_settings))
    op_id = f"export_{str(export_format).lower()}"
    export_history.append_step(
        operation_id=op_id,
        operation_name=f"Export: {str(export_format).upper()}",
        operation_version=_version(),
        parameters=export_params,
    )

    primary_warning = None
    processed_warning = None
    if _has_data_processing(export_history):
        processed_warning = PROCESSED_EXPORT_WARNING.format(
            filename=history.source_record.source_filename or "unknown",
            steps=_processing_steps_summary(export_history),
        )
    if conversion == "dat_to_sxm":
        primary_warning = warning_list[0] if warning_list else None
        if processed_warning and processed_warning not in warning_list:
            warning_list.append(processed_warning)
    elif processed_warning:
        primary_warning = processed_warning
        if processed_warning not in warning_list:
            warning_list.insert(0, processed_warning)

    summary = human_readable_export_summary(
        source=history.source_record,
        processing_history=export_history,
        primary_warning=primary_warning,
    )
    if conversion == "dat_to_sxm" and processed_warning:
        summary = f"{summary}\n\n{processed_warning}"
    return ExportRecord(
        export_path=privacy_safe_path(export_path),
        export_format=str(export_format).lower(),
        source_info=history.source_record,
        processing_history=export_history,
        display_settings=dict(display_settings or {}),
        timestamp=_utc_now(),
        warning=primary_warning,
        warnings=tuple(warning_list),
        rois=copy.deepcopy(rois) if rois is not None else None,
        masks=copy.deepcopy(masks) if masks is not None else None,
        summary=summary,
    )


def human_readable_export_summary(
    *,
    source: SourceRecord,
    processing_history: ProcessingHistory,
    primary_warning: str | None = None,
) -> str:
    if primary_warning:
        return primary_warning
    return (
        "This file was exported by ProbeFlow.\n"
        f"Original file: {source.source_filename or 'unknown'}\n"
        f"Processing steps: {_processing_steps_summary(processing_history)}"
    )


def display_lines(history: ProcessingHistory) -> list[str]:
    """Return compact lines for the GUI History panel."""

    lines = [
        f"Source: {history.source_record.source_filename or 'unknown'}",
        f"Channel: {history.source_record.channel or 'unknown'}",
        "",
    ]
    for index, step in enumerate(history.steps, 1):
        lines.append(f"{index}. {_step_summary(step)}")
    return lines


def _load_operation_name(source_file_type: str | None) -> str:
    if source_file_type == "Createc .dat":
        return "Loaded Createc .dat"
    if source_file_type == "Nanonis .sxm":
        return "Loaded Nanonis .sxm"
    return "Loaded image file"


def _operation_name(op: str, params: dict[str, Any]) -> str:
    if op == "align_rows":
        return "Row alignment"
    if op == "remove_bad_lines":
        return "Bad-line removal"
    if op in {"plane_bg", "stm_line_bg", "stm_background", "facet_level"}:
        return "Background subtraction"
    if op == "smooth":
        return "Gaussian blur/smoothing"
    if op == "gaussian_high_pass":
        return "Gaussian high-pass filter"
    if op == "edge_detect":
        return "Edge detection"
    if op in {"fourier_filter", "fft_soft_border", "periodic_notch_filter"}:
        return "FFT filtering"
    if op == "linear_undistort":
        return "Linear undistort"
    if op == "set_zero_point":
        return "Set zero point"
    if op == "set_zero_plane":
        return "Set zero plane"
    if op in ("roi", "mask"):
        if op == "roi" and isinstance(params, dict) and params.get("scope_kind") == "region":
            scope = "Region"
        else:
            scope = "ROI" if op == "roi" else "Mask"
        nested = params.get("step") if isinstance(params, dict) else None
        nested_op = nested.get("op") if isinstance(nested, dict) else None
        if nested_op:
            return f"{scope}-scoped {_operation_name(str(nested_op), nested.get('params', {}))}"
        return f"{scope}-scoped processing"
    if op.startswith("export_"):
        return f"Export: {op.removeprefix('export_').upper()}"
    return op.replace("_", " ").title()


def _step_summary(step: ProvenanceStep) -> str:
    p = step.parameters or {}
    op = step.operation_id
    if op == "file_load":
        return step.operation_name
    if op == "align_rows":
        return f"Row alignment: {p.get('method', 'median')}"
    if op == "remove_bad_lines":
        bits = [
            f"threshold={p.get('threshold_mad', p.get('threshold', 5.0))}",
            f"method={p.get('method', 'mad')}",
        ]
        if "min_segment_length_px" in p:
            bits.append(f"px={p['min_segment_length_px']}")
        return "Bad lines: " + ", ".join(bits)
    if op == "plane_bg":
        return f"Background: plane subtraction order={p.get('order', 1)}"
    if op == "stm_background":
        return f"Background: {p.get('model', 'linear')} ({p.get('line_statistic', 'median')})"
    if op == "smooth":
        return f"Gaussian blur/smoothing: sigma={p.get('sigma_px', 1.0)} px"
    if op == "edge_detect":
        method = str(p.get("method", "laplacian"))
        if method in ("sobel", "scharr"):
            return f"Edge detection: {method} gradient magnitude"
        return f"Edge detection: {method} (sigma={p.get('sigma', 1.0)} px)"
    if op in ("roi", "mask"):
        is_region = op == "roi" and p.get("scope_kind") == "region"
        scope = "Region" if is_region else ("ROI" if op == "roi" else "Mask")
        nested = p.get("step") if isinstance(p, dict) else None
        if isinstance(nested, dict):
            nested_op = str(nested.get("op") or "")
            nested_params = dict(nested.get("params") or {})
            nested_name = _operation_name(nested_op, nested_params)
            label = p.get("roi") or p.get("mask") or ("region" if is_region else scope)
            summary = f"{scope}-scoped {nested_name} ({label})"
            # Be explicit that the op is computed full-image then pasted inside
            # the scope (so non-local ops are influenced by out-of-scope data).
            if p.get("scope_semantics") == "full_image_compute_masked_paste":
                summary += " — computed full-image, applied inside scope"
            if p.get("frozen_geometry") is not None:
                summary += " [frozen geometry]"
            elif p.get("frozen_mask") is not None:
                summary += " [frozen mask]"
            return summary
        return f"{scope}-scoped processing"
    if op.startswith("export_"):
        fmt = op.removeprefix("export_").upper()
        display = p.get("display_settings") if isinstance(p, dict) else None
        if isinstance(display, dict):
            hist = _histogram_summary(display)
            cmap = display.get("colormap")
            tail = ", ".join(item for item in (hist, f"cmap={cmap}" if cmap else "") if item)
            return f"Export: {fmt}" + (f", {tail}" if tail else "")
        return f"Export: {fmt}"
    if op == "dat_to_sxm":
        return "Converted Createc .dat to Nanonis-compatible .sxm"
    if p:
        params = ", ".join(f"{key}={value}" for key, value in sorted(p.items()))
        return f"{step.operation_name}: {params}"
    return step.operation_name


def _histogram_summary(display: dict[str, Any]) -> str:
    low = display.get("low_pct")
    high = display.get("high_pct")
    if low is not None and high is not None:
        return f"histogram=[{low:g},{high:g}]"
    vmin = display.get("vmin")
    vmax = display.get("vmax")
    if vmin is not None and vmax is not None:
        return f"range=[{vmin},{vmax}]"
    return ""


def _has_data_processing(history: ProcessingHistory) -> bool:
    non_data = {"file_load"}
    for step in history.steps:
        if step.operation_id in non_data:
            continue
        if step.operation_id.startswith("export_"):
            continue
        if step.operation_id == "dat_to_sxm":
            continue
        return True
    return False


def _processing_steps_summary(history: ProcessingHistory) -> str:
    summaries = [
        _step_summary(step)
        for step in history.steps
        if step.operation_id != "file_load" and not step.operation_id.startswith("export_")
    ]
    return "; ".join(summaries) if summaries else "none"
