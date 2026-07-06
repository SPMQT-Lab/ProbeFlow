"""Shared helpers for spectroscopy viewer dialogs."""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QAbstractSpinBox, QPushButton, QWidget

from probeflow.spectroscopy.measurement import SpectrumDeltaMeasurement
from probeflow.spectroscopy.models import DisplayedSpectrum

_DISPLAY_PIPELINE_TOOLTIP = (
    "Displayed/exported spectra are derived copies: raw x/y -> smoothing -> "
    "numerical derivative -> normalization -> outlier mask -> vertical offset. "
    "The loaded raw spectroscopy arrays and source files are not modified."
)
_DERIVATIVE_NUMERIC_LABEL = "Numerical dy/dx"


def _derivative_enabled(label: str) -> bool:
    return label in {_DERIVATIVE_NUMERIC_LABEL, "dI/dV"}

_SMOOTHING_TOOLTIP = (
    "Smoothing is applied only to displayed copies. Savitzky-Golay uses an odd "
    "window length greater than the polynomial order; invalid GUI values are "
    "rejected before plotting."
)
_NORMALIZATION_TOOLTIP = (
    "Normalization is applied only to displayed copies. Options divide by the "
    "spectrum metadata setpoint, a user constant, max(abs(y)), or a denominator "
    "channel."
)


def _plain_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setDefault(False)
    button.setAutoDefault(False)
    return button


def _shorten_filename(name: str, max_chars: int = 36) -> str:
    if len(name) <= max_chars:
        return name
    if max_chars < 9:
        return name[:max_chars]
    suffix_len = max(10, max_chars // 2)
    prefix_len = max_chars - suffix_len - 3
    return f"{name[:prefix_len]}...{name[-suffix_len:]}"


def _trace_key(trace: DisplayedSpectrum) -> tuple[str, str, str]:
    return (trace.source_file, trace.spectrum_id, trace.y_channel)


def _shared_scale_values(displayed_list: list[DisplayedSpectrum]) -> dict[str, np.ndarray]:
    """Concatenated finite y values per y_unit, for shared display scaling.

    Traces co-plotted on one axis must share a single SI-prefix scale per
    unit; picking the prefix per trace from its own values silently rescales
    the traces relative to each other (a 50 pA and a 2 nA spectrum would plot
    as 50 vs 2).  Feed the returned per-unit sample into the scale chooser so
    every trace in the group gets the same factor.
    """
    groups: dict[str, list[np.ndarray]] = {}
    for displayed in displayed_list:
        arr = np.asarray(displayed.y_display, dtype=float)
        groups.setdefault(displayed.y_unit, []).append(arr[np.isfinite(arr)])
    return {
        unit: np.concatenate(vals) if vals else np.array([], dtype=float)
        for unit, vals in groups.items()
    }


def _displayed_trace_for_measurement(
    traces: list[DisplayedSpectrum],
    measurement: SpectrumDeltaMeasurement | None,
) -> DisplayedSpectrum | None:
    if measurement is None:
        return None
    key = measurement.point1.trace_key
    for trace in traces:
        if _trace_key(trace) == key:
            return trace
    return None


def _focus_in_parameter_inputs(focus: QWidget | None, inputs: list[QWidget]) -> bool:
    if focus is None:
        return False
    for widget in inputs:
        if focus is widget:
            return True
        line_edit = widget.lineEdit() if hasattr(widget, "lineEdit") else None
        if line_edit is not None and focus is line_edit:
            return True
    return False


class _NoWheelFilter(QObject):
    """Ignores wheel events on spinboxes so panel scrolling doesn't change values."""

    def eventFilter(self, obj, event):
        if isinstance(obj, QAbstractSpinBox) and event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return False


_NO_WHEEL_FILTER: _NoWheelFilter | None = None


def _no_wheel_filter() -> _NoWheelFilter:
    global _NO_WHEEL_FILTER
    if _NO_WHEEL_FILTER is None:
        _NO_WHEEL_FILTER = _NoWheelFilter()
    return _NO_WHEEL_FILTER
