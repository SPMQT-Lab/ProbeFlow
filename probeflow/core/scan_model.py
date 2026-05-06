"""Pure data model for loaded STM image scans.

This module deliberately contains no top-level reader, writer, validation, GUI,
or CLI imports. Keep it that way: readers, writers, validation, metadata, CLI,
and GUI can all depend on the :class:`Scan` model without creating import
cycles. The ``save_*`` convenience methods retain lazy writer imports only for
public API compatibility.

The public compatibility import remains ``from probeflow.core.scan_model import Scan``;
new internal code that only needs the dataclass should prefer importing from
``probeflow.core.scan_model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


PLANE_CANON_NAMES: tuple[str, ...] = (
    "Z forward", "Z backward", "Current forward", "Current backward",
)
PLANE_CANON_UNITS: tuple[str, ...] = ("m", "m", "A", "A")


class _ProcessingHistoryView(list):
    """Mutable compatibility view over ``Scan.processing_state``."""

    def __init__(self, scan: "Scan"):
        self._scan = scan
        super().__init__(self._wrap_entries(scan._processing_history_entries()))

    def _wrap(self, value):
        return _wrap_history_value(value, self._sync_from_view)

    def _wrap_entries(self, entries):
        return [self._wrap(entry) for entry in entries]

    def _sync_from_view(self) -> None:
        self._scan.processing_history = [_plain_history_value(entry) for entry in self]

    def _replace(self, entries) -> None:
        self._scan.processing_history = [_plain_history_value(entry) for entry in entries]
        super().clear()
        super().extend(self._wrap_entries(self._scan._processing_history_entries()))

    def append(self, entry) -> None:
        self._replace([*list(self), entry])

    def extend(self, entries) -> None:
        self._replace([*list(self), *list(entries)])

    def insert(self, index: int, entry) -> None:
        entries = list(self)
        entries.insert(index, entry)
        self._replace(entries)

    def clear(self) -> None:
        self._replace([])

    def pop(self, index: int = -1):
        entries = list(self)
        item = entries.pop(index)
        self._replace(entries)
        return _plain_history_value(item)

    def remove(self, entry) -> None:
        entries = list(self)
        entries.remove(entry)
        self._replace(entries)

    def reverse(self) -> None:
        entries = list(self)
        entries.reverse()
        self._replace(entries)

    def sort(self, *args, **kwargs) -> None:
        entries = list(self)
        entries.sort(*args, **kwargs)
        self._replace(entries)

    def __setitem__(self, index, value) -> None:
        entries = list(self)
        entries[index] = value
        self._replace(entries)

    def __delitem__(self, index) -> None:
        entries = list(self)
        del entries[index]
        self._replace(entries)

    def __iadd__(self, entries):
        self.extend(entries)
        return self

    def __imul__(self, n: int):
        self._replace(list(self) * n)
        return self


class _TrackedHistoryDict(dict):
    """Dict proxy that syncs legacy history edits back to the owning Scan."""

    def __init__(self, data, on_change):
        self._on_change = on_change
        dict.__init__(self)
        for key, value in dict(data).items():
            dict.__setitem__(self, key, _wrap_history_value(value, on_change))

    def _changed(self):
        self._on_change()

    def __setitem__(self, key, value) -> None:
        dict.__setitem__(self, key, _wrap_history_value(value, self._on_change))
        self._changed()

    def __delitem__(self, key) -> None:
        dict.__delitem__(self, key)
        self._changed()

    def clear(self) -> None:
        dict.clear(self)
        self._changed()

    _MISSING = object()

    def pop(self, key, default=_MISSING):
        if default is self._MISSING:
            value = dict.pop(self, key)
        else:
            value = dict.pop(self, key, default)
        self._changed()
        return value

    def popitem(self):
        item = dict.popitem(self)
        self._changed()
        return item

    def setdefault(self, key, default=None):
        if key not in self:
            dict.__setitem__(self, key, _wrap_history_value(default, self._on_change))
            self._changed()
        return dict.__getitem__(self, key)

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            dict.__setitem__(self, key, _wrap_history_value(value, self._on_change))
        self._changed()


class _TrackedHistoryList(list):
    """List proxy that syncs nested legacy history edits back to the owning Scan."""

    def __init__(self, data, on_change):
        self._on_change = on_change
        list.__init__(self, [_wrap_history_value(value, on_change) for value in data])

    def _changed(self):
        self._on_change()

    def append(self, value) -> None:
        list.append(self, _wrap_history_value(value, self._on_change))
        self._changed()

    def extend(self, values) -> None:
        list.extend(self, [_wrap_history_value(value, self._on_change) for value in values])
        self._changed()

    def insert(self, index: int, value) -> None:
        list.insert(self, index, _wrap_history_value(value, self._on_change))
        self._changed()

    def clear(self) -> None:
        list.clear(self)
        self._changed()

    def pop(self, index: int = -1):
        value = list.pop(self, index)
        self._changed()
        return value

    def remove(self, value) -> None:
        list.remove(self, value)
        self._changed()

    def reverse(self) -> None:
        list.reverse(self)
        self._changed()

    def sort(self, *args, **kwargs) -> None:
        list.sort(self, *args, **kwargs)
        self._changed()

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            value = [_wrap_history_value(item, self._on_change) for item in value]
        else:
            value = _wrap_history_value(value, self._on_change)
        list.__setitem__(self, index, value)
        self._changed()

    def __delitem__(self, index) -> None:
        list.__delitem__(self, index)
        self._changed()

    def __iadd__(self, values):
        self.extend(values)
        return self

    def __imul__(self, n: int):
        list.__imul__(self, n)
        self._changed()
        return self


def _wrap_history_value(value, on_change):
    if isinstance(value, (_TrackedHistoryDict, _TrackedHistoryList)):
        return value
    if isinstance(value, dict):
        return _TrackedHistoryDict(value, on_change)
    if isinstance(value, list):
        return _TrackedHistoryList(value, on_change)
    return value


def _plain_history_value(value):
    if isinstance(value, dict):
        return {
            _plain_history_value(key): _plain_history_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_plain_history_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_history_value(item) for item in value)
    return value


@dataclass(init=False)
class Scan:
    """A parsed STM topography scan with all planes in display orientation.

    Attributes
    ----------
    planes
        List of 2-D float64 arrays. Canonical STM scans use SI units and the
        public order Z forward, Z backward, Current forward, Current backward.
        Selected-channel or auxiliary layouts preserve native channel order
        with units recorded in ``plane_units``. Each array is oriented for
        display: origin at the top-left and scan direction left-to-right.
    plane_names, plane_units
        Parallel lists describing each plane.
    plane_synthetic
        True when a plane was synthesised, e.g. backward mirrored from forward
        because the original file only had forward channels.
    header
        The raw vendor header dict.
    scan_range_m
        Physical ``(width_m, height_m)``.
    source_path
        Absolute path to the file we loaded from.
    source_format
        ``"sxm"`` | ``"dat"`` identifies the reader that produced this Scan.
    """

    planes: List[np.ndarray]
    plane_names: List[str]
    plane_units: List[str]
    plane_synthetic: List[bool]
    header: dict
    scan_range_m: Tuple[float, float]
    source_path: Path
    source_format: str
    experiment_metadata: dict[str, Any] = field(default_factory=dict)
    _processing_state: Any = field(default=None, init=False, repr=False)
    _processing_history_timestamps: list[str | None] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __init__(
        self,
        planes: List[np.ndarray],
        plane_names: List[str],
        plane_units: List[str],
        plane_synthetic: List[bool],
        header: dict,
        scan_range_m: Tuple[float, float],
        source_path: Path,
        source_format: str,
        processing_state: Any | None = None,
        processing_history: List[dict] | None = None,
        experiment_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.planes = planes
        self.plane_names = plane_names
        self.plane_units = plane_units
        self.plane_synthetic = plane_synthetic
        self.header = header
        self.scan_range_m = scan_range_m
        self.source_path = source_path
        self.source_format = source_format
        self.experiment_metadata = dict(experiment_metadata or {})
        self._processing_state = None
        self._processing_history_timestamps = []
        if processing_state is not None and processing_history is not None:
            raise ValueError(
                "Pass either processing_state or processing_history, not both."
            )
        self.processing_state = processing_state
        if processing_history is not None:
            self.processing_history = processing_history

    @staticmethod
    def _empty_processing_state():
        from probeflow.processing.state import ProcessingState
        return ProcessingState()

    @staticmethod
    def _coerce_processing_state(value):
        from probeflow.processing.state import ProcessingState
        if value is None:
            return ProcessingState()
        if isinstance(value, ProcessingState):
            return ProcessingState.from_dict(value.to_dict())
        if isinstance(value, dict):
            return ProcessingState.from_dict(value)
        if hasattr(value, "to_dict"):
            return ProcessingState.from_dict(value.to_dict())
        raise TypeError(f"Unsupported processing_state: {type(value).__name__}")

    @property
    def processing_state(self):
        """Canonical numerical processing state for this scan."""
        if self._processing_state is None:
            self._processing_state = self._empty_processing_state()
        return self._processing_state

    @processing_state.setter
    def processing_state(self, value) -> None:
        state = self._coerce_processing_state(value)
        self._processing_state = state
        self._processing_history_timestamps = [None] * len(state.steps)

    def record_processing_state(
        self,
        state,
        *,
        timestamp: str | None = None,
    ) -> None:
        """Append canonical processing steps to this scan."""
        from datetime import datetime
        from probeflow.processing.state import ProcessingState

        state = self._coerce_processing_state(state)
        if not state.steps:
            return
        current = self.processing_state
        self._processing_state = ProcessingState(steps=[
            *current.steps,
            *state.steps,
        ])
        ts = timestamp or datetime.now().isoformat()
        self._processing_history_timestamps.extend([ts] * len(state.steps))

    def _processing_history_entries(self) -> list[dict[str, Any]]:
        from probeflow.processing.history import processing_history_entries_from_state
        return processing_history_entries_from_state(
            self.processing_state,
            timestamps=self._processing_history_timestamps,
        )

    @property
    def processing_history(self) -> List[dict]:
        """Legacy history view derived from ``processing_state``."""
        return _ProcessingHistoryView(self)

    @processing_history.setter
    def processing_history(self, entries: List[dict] | None) -> None:
        from probeflow.processing.history import processing_state_from_history
        self._processing_state = processing_state_from_history(entries)
        self._processing_history_timestamps = [
            entry.get("timestamp") if isinstance(entry, dict) else None
            for entry in (entries or [])
            if isinstance(entry, dict) and entry.get("op")
        ]

    @property
    def n_planes(self) -> int:
        return len(self.planes)

    @property
    def dims(self) -> Tuple[int, int]:
        """Scan dimensions as ``(Nx, Ny)`` (matches Nanonis SCAN_PIXELS)."""
        if not self.planes:
            return (0, 0)
        Ny, Nx = self.planes[0].shape
        return (Nx, Ny)

    def save_sxm(self, out_path) -> None:
        """Write this Scan to a Nanonis ``.sxm`` file."""
        from probeflow.io.writers.sxm import write_sxm
        write_sxm(self, out_path)

    def save_png(
        self,
        out_path,
        plane_idx: int = 0,
        *,
        colormap: str = "gray",
        clip_low: float = 1.0,
        clip_high: float = 99.0,
        add_scalebar: bool = True,
        scalebar_unit: str = "nm",
        scalebar_pos: str = "bottom-right",
        provenance=None,
    ) -> None:
        """Render one plane to a colourised PNG with an optional scale bar."""
        from probeflow.io.writers.png import write_png
        write_png(
            self, out_path, plane_idx=plane_idx,
            colormap=colormap, clip_low=clip_low, clip_high=clip_high,
            add_scalebar=add_scalebar,
            scalebar_unit=scalebar_unit, scalebar_pos=scalebar_pos,
            provenance=provenance,
        )

    def save_pdf(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Render one plane to a publication-ready PDF."""
        from probeflow.io.writers.pdf import write_pdf
        write_pdf(self, out_path, plane_idx=plane_idx, **kwargs)

    def save_csv(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Dump one plane as a 2-D CSV grid."""
        from probeflow.io.writers.csv import write_csv
        write_csv(self, out_path, plane_idx=plane_idx, **kwargs)

    def save_gwy(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Write this Scan to a Gwyddion ``.gwy`` file."""
        from probeflow.io.writers.gwy import write_gwy
        write_gwy(self, out_path, plane_idx=plane_idx, **kwargs)

    def save(self, out_path, plane_idx: int = 0, **kwargs) -> None:
        """Suffix-driven save: ``.sxm`` / ``.gwy`` / ``.png`` / ``.pdf`` / ``.csv``."""
        from probeflow.io.writers import save_scan
        save_scan(self, out_path, plane_idx=plane_idx, **kwargs)
