"""ExportProvenance — records where exported files came from and how.

The provenance chain is:
    source file -> Scan -> channel -> ProcessingState -> DisplayRangeState -> export file

All fields serialise to plain JSON via :meth:`ExportProvenance.to_dict`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from probeflow.processing.display_state import DisplayRangeState
    from probeflow.processing.state import ProcessingState
    from probeflow.core.scan_model import Scan


def _get_version() -> str | None:
    """Return the ProbeFlow package version string, or None if unavailable."""
    try:
        from probeflow import __version__
        if __version__:
            return str(__version__)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("probeflow")
    except Exception:
        return None


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string ending in Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ExportProvenance:
    """Immutable provenance record for one exported file.

    All values must be JSON-serialisable (strings, numbers, lists, dicts, None).
    Nested objects (processing_state, display_state) are stored pre-serialised
    so ``to_dict()`` never needs to know their internal structure.
    """

    source_file:        str | None
    source_format:      str | None
    item_type:          str                         # "scan" | "spectrum" | …
    channel_name:       str | None
    channel_index:      int | None
    array_shape:        tuple[int, int] | None       # (Ny, Nx)
    scan_range_m:       tuple[float, float] | None   # (width_m, height_m)
    units:              str | None
    processing_state:   dict[str, Any]               # ProcessingState.to_dict()
    display_state:      dict[str, Any]               # DisplayRangeState.to_dict()
    probeflow_version:  str | None
    export_timestamp:   str                          # ISO-8601 UTC
    export_kind:        str = "export"                # "png" | "prepared_png" | …
    output_path:        str | None = None
    source_id:          str | None = None
    channel_id:         str | None = None
    processing_state_hash: str | None = None
    artifact_id:        str | None = None
    warnings:           tuple[str, ...] = ()
    rois:               dict[str, Any] | None = None

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-compatible dict of all provenance fields."""
        return {
            "source_file":       self.source_file,
            "source_format":     self.source_format,
            "item_type":         self.item_type,
            "channel_name":      self.channel_name,
            "channel_index":     self.channel_index,
            "array_shape":       list(self.array_shape) if self.array_shape else None,
            "scan_range_m":      list(self.scan_range_m) if self.scan_range_m else None,
            "units":             self.units,
            "processing_state":  self.processing_state,
            "display_state":     self.display_state,
            "probeflow_version": self.probeflow_version,
            "export_timestamp":  self.export_timestamp,
            "export_kind":       self.export_kind,
            "output_path":       self.output_path,
            "source_id":         self.source_id,
            "channel_id":        self.channel_id,
            "processing_state_hash": self.processing_state_hash,
            "artifact_id":       self.artifact_id,
            "warnings":          list(self.warnings),
            "rois":              self.rois,
        }

    # ── Convenience constructor ───────────────────────────────────────────────

    @classmethod
    def from_scan_export(
        cls,
        scan: "Scan",
        *,
        channel_index: int | None = None,
        channel_name:  str | None = None,
        processing_state: "ProcessingState | None" = None,
        display_state:    "DisplayRangeState | None" = None,
        item_type: str = "scan",
    ) -> "ExportProvenance":
        """Construct provenance from a loaded Scan and current settings.

        Defensive: missing or unavailable fields default to None without raising.
        """
        # Source identity
        source_file   = str(scan.source_path) if getattr(scan, "source_path", None) else None
        source_format = getattr(scan, "source_format", None)

        # Array shape for the selected channel
        array_shape: tuple[int, int] | None = None
        try:
            planes = scan.planes
            idx = channel_index if channel_index is not None else 0
            if 0 <= idx < len(planes):
                ny, nx = planes[idx].shape
                array_shape = (int(ny), int(nx))
            elif planes:
                ny, nx = planes[0].shape
                array_shape = (int(ny), int(nx))
        except Exception:
            pass

        # Scan range
        scan_range_m: tuple[float, float] | None = None
        try:
            w, h = scan.scan_range_m
            scan_range_m = (float(w), float(h))
        except Exception:
            pass

        # Units for the selected channel
        units: str | None = None
        try:
            plane_units = scan.plane_units
            idx = channel_index if channel_index is not None else 0
            if 0 <= idx < len(plane_units):
                units = str(plane_units[idx])
        except Exception:
            pass

        # Channel name (prefer explicit argument, fall back to scan plane_names)
        if channel_name is None:
            try:
                plane_names = scan.plane_names
                idx = channel_index if channel_index is not None else 0
                if 0 <= idx < len(plane_names):
                    channel_name = str(plane_names[idx])
            except Exception:
                pass

        # Serialise state objects
        ps_dict = processing_state.to_dict() if processing_state is not None else {"steps": []}
        ds_dict = display_state.to_dict()    if display_state    is not None else {}

        return cls(
            source_file=source_file,
            source_format=source_format,
            item_type=item_type,
            channel_name=channel_name,
            channel_index=channel_index,
            array_shape=array_shape,
            scan_range_m=scan_range_m,
            units=units,
            processing_state=ps_dict,
            display_state=ds_dict,
            probeflow_version=_get_version(),
            export_timestamp=_utc_now(),
            source_id=_source_id(source_file),
            channel_id=_channel_id(source_file, channel_index, channel_name),
            processing_state_hash=processing_state_hash(ps_dict),
        )


def processing_state_hash(processing_state: dict[str, Any]) -> str:
    """Return a stable short hash for a serialised ProcessingState dict."""
    payload = _json.dumps(processing_state or {"steps": []}, sort_keys=True, default=str)
    return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def processing_state_from_history(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Convert ``Scan.processing_history`` into ProcessingState-like JSON.

    This keeps writer-level provenance useful even when the caller has already
    mutated a Scan and only the lightweight history entries remain available.
    Timestamps are intentionally omitted from the canonical processing state;
    they describe when a step was applied, not what numerical operation it is.
    """
    from probeflow.processing.history import processing_state_dict_from_history
    return processing_state_dict_from_history(history)


def background_processing_warnings(processing_state: dict[str, Any]) -> tuple[str, ...]:
    """Warnings for exports intended as downstream analysis inputs."""
    bg_ops = {"plane_bg", "stm_line_bg", "stm_background", "facet_level"}
    steps = processing_state.get("steps", []) if processing_state else []
    if any(step.get("op") in bg_ops for step in steps):
        return ()
    return (
        "No background/line-leveling operation is recorded; downstream tools "
        "that consume PNGs may be sensitive to large tilt or background.",
    )


def build_scan_export_provenance(
    scan: "Scan",
    *,
    channel_index: int = 0,
    channel_name: str | None = None,
    processing_state: "ProcessingState | dict[str, Any] | None" = None,
    display_state: "DisplayRangeState | dict[str, Any] | None" = None,
    export_kind: str = "export",
    output_path=None,
    warnings: tuple[str, ...] | list[str] | None = None,
    roi_set=None,
) -> ExportProvenance:
    """Shared provenance constructor for GUI, CLI, writers, and handoffs."""
    if processing_state is None:
        scan_state = getattr(scan, "processing_state", None)
        if scan_state is not None and hasattr(scan_state, "to_dict"):
            ps_dict = scan_state.to_dict()
        else:
            ps_dict = processing_state_from_history(getattr(scan, "processing_history", []))
    elif hasattr(processing_state, "to_dict"):
        ps_dict = processing_state.to_dict()
    else:
        ps_dict = dict(processing_state)

    if display_state is None:
        ds_dict = {}
    elif hasattr(display_state, "to_dict"):
        ds_dict = display_state.to_dict()
    else:
        ds_dict = dict(display_state)

    class _State:
        def __init__(self, data):
            self._data = data
        def to_dict(self):
            return self._data

    prov = ExportProvenance.from_scan_export(
        scan,
        channel_index=channel_index,
        channel_name=channel_name,
        processing_state=_State(ps_dict),
        display_state=_State(ds_dict),
    )
    out_str = str(output_path) if output_path is not None else None
    artifact_id = _artifact_id(out_str, ps_dict)
    return ExportProvenance(
        source_file=prov.source_file,
        source_format=prov.source_format,
        item_type=prov.item_type,
        channel_name=prov.channel_name,
        channel_index=prov.channel_index,
        array_shape=prov.array_shape,
        scan_range_m=prov.scan_range_m,
        units=prov.units,
        processing_state=ps_dict,
        display_state=ds_dict,
        probeflow_version=prov.probeflow_version,
        export_timestamp=prov.export_timestamp,
        export_kind=str(export_kind),
        output_path=out_str,
        source_id=prov.source_id,
        channel_id=prov.channel_id,
        processing_state_hash=processing_state_hash(ps_dict),
        artifact_id=artifact_id,
        warnings=tuple(warnings or ()),
        rois=roi_set.to_dict() if roi_set is not None else None,
    )


def png_display_state(
    display_state: "DisplayRangeState | dict[str, Any] | None" = None,
    *,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    colormap: str | None = None,
    add_scalebar: bool | None = None,
    scalebar_unit: str | None = None,
    scalebar_pos: str | None = None,
) -> dict[str, Any]:
    """Return display/export state for PNG provenance sidecars.

    ``DisplayRangeState`` deliberately tracks only contrast limits. PNG export
    also has visual/export choices such as colormap and scale-bar placement;
    keeping those keys here makes Viewer, Convert, CLI, and writer-level PNGs
    describe display state in the same shape without treating those choices as
    numerical processing.
    """
    if display_state is None:
        from probeflow.processing.display_state import DisplayRangeState
        data: dict[str, Any] = DisplayRangeState(
            low_pct=float(clip_low),
            high_pct=float(clip_high),
        ).to_dict()
    elif hasattr(display_state, "to_dict"):
        data = display_state.to_dict()
    else:
        data = dict(display_state)

    if colormap is not None:
        data["colormap"] = str(colormap)
    if add_scalebar is not None:
        data["add_scalebar"] = bool(add_scalebar)
    if scalebar_unit is not None:
        data["scalebar_unit"] = str(scalebar_unit)
    if scalebar_pos is not None:
        data["scalebar_pos"] = str(scalebar_pos)
    return data


def _source_id(source_file: str | None) -> str | None:
    if not source_file:
        return None
    p = Path(source_file)
    payload = f"{p.name}:{p.as_posix()}"
    return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _channel_id(
    source_file: str | None,
    channel_index: int | None,
    channel_name: str | None,
) -> str | None:
    if source_file is None and channel_index is None and channel_name is None:
        return None
    payload = (
        f"{source_file or ''}:"
        f"{channel_index if channel_index is not None else ''}:"
        f"{channel_name or ''}"
    )
    return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _artifact_id(output_path: str | None, processing_state: dict[str, Any]) -> str | None:
    if output_path is None:
        return None
    payload = f"{output_path}:{processing_state_hash(processing_state)}"
    return _hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
