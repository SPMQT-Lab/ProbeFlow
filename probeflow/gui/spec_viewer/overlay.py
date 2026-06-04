"""Multi-spectrum overlay viewer dialog."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from probeflow.gui.models import VertFile
from probeflow.gui.widgets import MeasurementResultsTable
from probeflow.spectroscopy.export import (
    displayed_spectra_to_clipboard_text,
    displayed_spectra_to_csv_text,
    displayed_spectra_to_json_text,
    displayed_spectra_to_txt_text,
)
from probeflow.spectroscopy.measurement import (
    SpectrumDeltaMeasurement,
    SpectrumMeasurementPoint,
)
from probeflow.spectroscopy.models import (
    DisplayedSpectrum,
    SpectrumDisplayOptions,
    SpectrumTrace,
)
from probeflow.spectroscopy.normalization import (
    NORMALIZATION_LABELS,
    normalization_formula_text,
    normalize_mode,
)
from probeflow.spectroscopy.smoothing import savgol_validation_message
from probeflow.spectroscopy.transforms import make_displayed_spectrum

from .shared import (
    _DERIVATIVE_NUMERIC_LABEL,
    _DISPLAY_PIPELINE_TOOLTIP,
    _NORMALIZATION_TOOLTIP,
    _SMOOTHING_TOOLTIP,
    _derivative_enabled,
    _focus_in_parameter_inputs,
    _plain_button,
)
from .single import SpecViewerDialog

class SpecOverlayDialog(QDialog):
    """Overlay or waterfall plot for multiple selected spectroscopy files."""

    _BG = SpecViewerDialog._BG
    _FG = SpecViewerDialog._FG
    _COLORS = SpecViewerDialog._COLORS

    def __init__(self, entries: list[VertFile], t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectra overlay")
        self.setMinimumSize(1050, 620)
        self.resize(1160, 700)
        self._entries = list(entries)
        self._t = t
        self._specs = []
        self._canvas = None
        self._fig = None
        self._displayed_traces: list[DisplayedSpectrum] = []
        self._displayed_trace_axes = []
        self._measure_enabled = False
        self._measure_points: list[SpectrumMeasurementPoint] = []
        self._measurement: SpectrumDeltaMeasurement | None = None
        self._measurement_artists = []
        self._crosshair_artists = []
        self._parameter_inputs: list[QWidget] = []
        self._build()
        self._load()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        title = QLabel("Selected spectra")
        title.setFont(ui_font(12, weight=QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        body = QSplitter(Qt.Horizontal)

        controls = QWidget()
        controls.setMinimumWidth(300)
        controls.setMaximumWidth(400)
        controls.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        controls.setToolTip(_DISPLAY_PIPELINE_TOOLTIP)
        self._controls_panel = controls
        ctrl_lay = QVBoxLayout(controls)
        ctrl_lay.setContentsMargins(6, 6, 6, 6)
        ctrl_lay.setSpacing(5)

        self._count_lbl = QLabel("Loading…")
        self._count_lbl.setFont(ui_font(9))
        self._count_lbl.setWordWrap(True)
        ctrl_lay.addWidget(self._count_lbl)

        self._channel_cb = QComboBox()
        self._channel_cb.setFont(ui_font(9))
        self._channel_cb.setToolTip("Common signal channel to overlay/export as displayed data.")
        self._channel_cb.currentTextChanged.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Signal axis"))
        ctrl_lay.addWidget(self._channel_cb)

        self._smoothing_cb = QComboBox()
        self._smoothing_cb.addItems(["None", "Gaussian", "Savitzky-Golay"])
        self._smoothing_cb.setToolTip(_SMOOTHING_TOOLTIP)
        self._smoothing_cb.currentTextChanged.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Smoothing"))
        ctrl_lay.addWidget(self._smoothing_cb)

        self._smooth_points_spin = QSpinBox()
        self._smooth_points_spin.setRange(3, 9999)
        self._smooth_points_spin.setSingleStep(2)
        self._smooth_points_spin.setValue(7)
        self._smooth_points_spin.setToolTip(
            "Smoothing window in points. Savitzky-Golay requires an odd value "
            "greater than the polynomial order."
        )
        self._smooth_points_spin.valueChanged.connect(self._redraw)
        self._smooth_points_spin.editingFinished.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Window/points"))
        ctrl_lay.addWidget(self._smooth_points_spin)

        self._savgol_order_spin = QSpinBox()
        self._savgol_order_spin.setRange(0, 12)
        self._savgol_order_spin.setValue(2)
        self._savgol_order_spin.setToolTip(
            "Savitzky-Golay polynomial order. Must be smaller than the odd window length."
        )
        self._savgol_order_spin.valueChanged.connect(self._redraw)
        self._savgol_order_spin.editingFinished.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Poly order"))
        ctrl_lay.addWidget(self._savgol_order_spin)

        self._derivative_cb = QComboBox()
        self._derivative_cb.addItems(["Off", _DERIVATIVE_NUMERIC_LABEL])
        self._derivative_cb.setToolTip(
            "Compute a numerical derivative of the displayed y channel with respect to x."
        )
        self._derivative_cb.currentTextChanged.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Derivative"))
        ctrl_lay.addWidget(self._derivative_cb)

        self._normalize_cb = QComboBox()
        self._normalize_cb.addItems(NORMALIZATION_LABELS)
        self._normalize_cb.setToolTip(_NORMALIZATION_TOOLTIP)
        self._normalize_cb.currentTextChanged.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Normalize"))
        ctrl_lay.addWidget(self._normalize_cb)

        self._norm_constant_spin = QDoubleSpinBox()
        self._norm_constant_spin.setRange(-1e12, 1e12)
        self._norm_constant_spin.setDecimals(6)
        self._norm_constant_spin.setSingleStep(1.0)
        self._norm_constant_spin.setValue(1.0)
        self._norm_constant_spin.setToolTip(
            "Constant normalization uses y_display = y_input / constant. "
            "The original data are not changed."
        )
        self._norm_constant_spin.valueChanged.connect(self._redraw)
        self._norm_constant_spin.editingFinished.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Constant"))
        ctrl_lay.addWidget(self._norm_constant_spin)

        self._norm_channel_cb = QComboBox()
        self._norm_channel_cb.setToolTip(
            "Channel normalization uses y_display = y_input / selected_channel."
        )
        self._norm_channel_cb.currentTextChanged.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Norm channel"))
        ctrl_lay.addWidget(self._norm_channel_cb)

        self._formula_lbl = QLabel("Display: y")
        self._formula_lbl.setFont(ui_font(8))
        self._formula_lbl.setWordWrap(True)
        self._formula_lbl.setToolTip(_DISPLAY_PIPELINE_TOOLTIP)
        ctrl_lay.addWidget(QLabel("Formula"))
        ctrl_lay.addWidget(self._formula_lbl)

        self._outlier_cb = QComboBox()
        self._outlier_cb.addItems(["Off", "MAD", "Jump"])
        self._outlier_cb.setToolTip(
            "Mask outliers from displayed/exported arrays; original loaded data stay intact."
        )
        self._outlier_cb.currentTextChanged.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Outliers"))
        ctrl_lay.addWidget(self._outlier_cb)

        self._outlier_threshold_spin = QDoubleSpinBox()
        self._outlier_threshold_spin.setRange(0.1, 1e6)
        self._outlier_threshold_spin.setDecimals(2)
        self._outlier_threshold_spin.setSingleStep(0.5)
        self._outlier_threshold_spin.setValue(6.0)
        self._outlier_threshold_spin.setToolTip("Robust outlier threshold for display masking.")
        self._outlier_threshold_spin.valueChanged.connect(self._redraw)
        self._outlier_threshold_spin.editingFinished.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Threshold"))
        ctrl_lay.addWidget(self._outlier_threshold_spin)

        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-1e12, 1e12)
        self._offset_spin.setDecimals(4)
        self._offset_spin.setSingleStep(1.0)
        self._offset_spin.setValue(0.0)
        self._offset_spin.setToolTip(
            "Vertical offset applied last, in displayed Y units. Raw data are unchanged."
        )
        self._offset_spin.valueChanged.connect(self._redraw)
        self._offset_spin.editingFinished.connect(self._redraw)
        ctrl_lay.addWidget(QLabel("Waterfall offset"))
        ctrl_lay.addWidget(self._offset_spin)

        self._measure_lbl = QLabel("Measurement: off")
        self._measure_lbl.setFont(ui_font(9))
        self._measure_lbl.setWordWrap(True)
        self._measure_lbl.setToolTip(
            "Crosshair measurements use the displayed trace, not the raw spectrum."
        )
        ctrl_lay.addWidget(self._measure_lbl)

        self._measure_btn = _plain_button("Measure Δ")
        self._measure_btn.setCheckable(True)
        self._measure_btn.setToolTip(
            "Enable crosshair measurement on displayed spectrum data."
        )
        self._measure_btn.toggled.connect(self._set_measure_mode)
        ctrl_lay.addWidget(self._measure_btn)

        self._copy_measure_btn = _plain_button("Copy measurement")
        self._copy_measure_btn.clicked.connect(self._copy_measurement)
        ctrl_lay.addWidget(self._copy_measure_btn)

        self._add_measure_btn = _plain_button("Add to measurements")
        self._add_measure_btn.clicked.connect(self._add_measurement_result)
        ctrl_lay.addWidget(self._add_measure_btn)

        self._clear_measure_btn = _plain_button("Clear measurement")
        self._clear_measure_btn.clicked.connect(self._clear_measurement)
        ctrl_lay.addWidget(self._clear_measure_btn)

        self._measurement_table = MeasurementResultsTable()
        self._measurement_table.setMaximumHeight(170)
        ctrl_lay.addWidget(self._measurement_table)

        self._parameter_inputs.extend([
            self._channel_cb,
            self._smoothing_cb,
            self._smooth_points_spin,
            self._savgol_order_spin,
            self._derivative_cb,
            self._normalize_cb,
            self._norm_constant_spin,
            self._norm_channel_cb,
            self._outlier_cb,
            self._outlier_threshold_spin,
            self._offset_spin,
        ])

        copy_btn = _plain_button("Copy data")
        copy_btn.clicked.connect(self._copy_csv)
        ctrl_lay.addWidget(copy_btn)

        export_btn = _plain_button("Export CSV…")
        export_btn.clicked.connect(self._export_csv)
        ctrl_lay.addWidget(export_btn)

        export_json_btn = _plain_button("Export JSON…")
        export_json_btn.clicked.connect(self._export_json)
        ctrl_lay.addWidget(export_json_btn)

        export_txt_btn = _plain_button("Export TXT…")
        export_txt_btn.clicked.connect(self._export_txt)
        ctrl_lay.addWidget(export_txt_btn)

        ctrl_lay.addStretch(1)
        body.addWidget(controls)

        self._canvas_widget = QWidget()
        self._canvas_widget.setMinimumWidth(560)
        self._canvas_lay = QVBoxLayout(self._canvas_widget)
        self._canvas_lay.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self._canvas_widget)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setCollapsible(0, False)
        body.setCollapsible(1, False)
        body.setSizes([320, 840])
        self._splitter = body
        lay.addWidget(body, 1)

        self._status = QLabel("")
        lay.addWidget(self._status)

        self._cursor_lbl = QLabel("Cursor: —")
        lay.addWidget(self._cursor_lbl)

        row = QHBoxLayout()
        row.addStretch(1)
        close_btn = _plain_button("Close")
        close_btn.clicked.connect(self.accept)
        row.addWidget(close_btn)
        lay.addLayout(row)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self._measure_enabled:
            self._clear_measurement()
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and _focus_in_parameter_inputs(
            self.focusWidget(), self._parameter_inputs
        ):
            self._redraw()
            event.accept()
            return
        super().keyPressEvent(event)

    def _savgol_validation_error(self) -> str | None:
        _ref_entry, ref_spec = self._reference_for_channel(self._channel_cb.currentText())
        n_points = int(len(ref_spec.x_array)) if ref_spec is not None else None
        return savgol_validation_message(
            self._smoothing_cb.currentText(),
            int(self._smooth_points_spin.value()),
            int(self._savgol_order_spin.value()),
            n_points,
        )

    def _refresh_formula(self) -> None:
        if hasattr(self, "_formula_lbl"):
            self._formula_lbl.setText(f"Display: {self._normalization_formula()}")

    def _normalization_formula(self) -> str:
        return normalization_formula_text(
            derivative=_derivative_enabled(self._derivative_cb.currentText()),
            mode_label=self._normalize_cb.currentText(),
            constant=float(self._norm_constant_spin.value()),
            channel=self._norm_channel_cb.currentText(),
            offset=float(self._offset_spin.value()),
        )

    def _load(self) -> None:
        from probeflow.io.spectroscopy import read_spec_file

        loaded = []
        errors = []
        for entry in self._entries:
            try:
                loaded.append((entry, read_spec_file(entry.path)))
            except Exception as exc:
                errors.append(f"{entry.stem}: {exc}")
        self._specs = loaded
        self._count_lbl.setText(f"{len(loaded)} spectra loaded")
        if errors:
            self._status.setText("; ".join(errors[:3]))

        common = None
        for _entry, spec in loaded:
            order = spec.channel_order or spec.channels.keys()
            channels = {ch for ch in order if ch in spec.channels}
            common = channels if common is None else common & channels
        ordered = []
        if loaded and common:
            first_order = loaded[0][1].channel_order or list(loaded[0][1].channels.keys())
            ordered = [ch for ch in first_order if ch in common]
        self._channel_cb.blockSignals(True)
        self._channel_cb.clear()
        self._channel_cb.addItems(ordered)
        if "I" in ordered:
            self._channel_cb.setCurrentText("I")
        self._channel_cb.blockSignals(False)
        self._norm_channel_cb.blockSignals(True)
        self._norm_channel_cb.clear()
        self._norm_channel_cb.addItems(ordered)
        self._norm_channel_cb.blockSignals(False)
        self._refresh_formula()
        self._redraw()

    def _display_options(self, vertical_offset: float = 0.0) -> SpectrumDisplayOptions:
        smoothing = {
            "None": "none",
            "Gaussian": "gaussian",
            "Savitzky-Golay": "savgol",
        }.get(self._smoothing_cb.currentText(), "none")
        normalize = normalize_mode(self._normalize_cb.currentText())
        outliers = {
            "Off": "none",
            "MAD": "mad",
            "Jump": "jump",
        }.get(self._outlier_cb.currentText(), "none")
        return SpectrumDisplayOptions(
            smoothing_mode=smoothing,
            smoothing_points=int(self._smooth_points_spin.value()),
            savgol_polyorder=int(self._savgol_order_spin.value()),
            derivative=_derivative_enabled(self._derivative_cb.currentText()),
            normalize_mode=normalize,
            normalize_constant=float(self._norm_constant_spin.value()),
            normalize_channel=self._norm_channel_cb.currentText() or None,
            outlier_mode=outliers,
            outlier_threshold=float(self._outlier_threshold_spin.value()),
            vertical_offset=float(vertical_offset),
        )

    def _raw_trace(self, entry, spec, channel: str) -> SpectrumTrace:
        info = getattr(spec, "channel_info", {}).get(channel)
        label = getattr(info, "display_label", channel) if info is not None else channel
        return SpectrumTrace(
            source_file=str(entry.path),
            spectrum_id=entry.stem,
            x_channel=spec.x_label,
            y_channel=channel,
            x_raw=spec.x_array,
            y_raw=spec.channels[channel],
            x_label=spec.x_label,
            y_label=label,
            x_unit=spec.x_unit,
            y_unit=spec.y_units.get(channel, ""),
            metadata={
                "source_file": str(entry.path),
                "spectrum_label": entry.stem,
                "raw_points": int(len(spec.x_array)),
                "setpoint_a": spec.metadata.get("setpoint_a"),
            },
        )

    def _displayed(self, entry, spec, channel: str, vertical_offset: float) -> DisplayedSpectrum:
        displayed = make_displayed_spectrum(
            self._raw_trace(entry, spec, channel),
            self._display_options(vertical_offset=0.0),
            channel_lookup=spec.channels,
        )
        scale, unit = self._scale(displayed.y_unit, displayed.y_display)
        metadata = dict(displayed.metadata)
        metadata.update({
            "display_scale": scale,
            "display_unit": unit,
            "vertical_offset_units": "displayed_y",
        })
        return replace(
            displayed,
            y_display=displayed.y_display * scale + float(vertical_offset),
            y_unit=unit,
            options=replace(displayed.options, vertical_offset=float(vertical_offset)),
            metadata=metadata,
        )

    def _scale(self, unit: str, values: np.ndarray) -> tuple[float, str]:
        from probeflow.analysis.spec_plot import choose_display_unit

        if "/" in unit:
            numerator, denominator = unit.split("/", 1)
            scale, disp_num = choose_display_unit(numerator, values)
            return scale, f"{disp_num}/{denominator}" if denominator else disp_num
        return choose_display_unit(unit, values)

    def _reference_for_channel(self, channel: str):
        for entry, spec in self._specs:
            if channel in spec.channels:
                return entry, spec
        return None, None

    def _x_compatibility_issue(self, ref_spec, spec, channel: str) -> str | None:
        if ref_spec is None:
            return None
        if spec.x_label != ref_spec.x_label or spec.x_unit != ref_spec.x_unit:
            return (
                f"x-axis {spec.x_label!r}/{spec.x_unit!r} differs from "
                f"{ref_spec.x_label!r}/{ref_spec.x_unit!r}"
            )
        if len(spec.x_array) != len(spec.channels[channel]):
            return "x/y length mismatch"
        if len(spec.x_array) == 0:
            return "empty x-axis"
        if len(spec.x_array) != len(ref_spec.x_array):
            return "x-axis length differs from reference"
        if not np.allclose(spec.x_array, ref_spec.x_array, rtol=1e-7, atol=1e-12, equal_nan=True):
            return "x-axis values differ from reference"
        return None

    def _redraw(self) -> None:
        self._refresh_formula()
        error = self._savgol_validation_error()
        if error is not None:
            self._status.setText(error)
            return
        self._displayed_traces = []
        self._displayed_trace_axes = []
        self._reset_measurement_state()
        if self._canvas is not None:
            self._canvas_lay.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas = None
            self._fig = None

        fig = Figure(figsize=(8.5, 4.5), tight_layout=True)
        fig.patch.set_facecolor(self._BG)
        ax = fig.subplots()
        ax.set_facecolor(self._BG)
        channel = self._channel_cb.currentText()
        offset = float(self._offset_spin.value())
        _ref_entry, ref_spec = self._reference_for_channel(channel)
        units = []
        plotted = 0
        skipped = []
        for i, (entry, spec) in enumerate(self._specs):
            if not channel or channel not in spec.channels:
                continue
            issue = self._x_compatibility_issue(ref_spec, spec, channel)
            if issue is not None:
                skipped.append(f"{entry.stem}: {issue}")
                continue
            try:
                displayed = self._displayed(entry, spec, channel, i * offset)
            except Exception as exc:
                self._status.setText(f"Plot error: {exc}")
                continue
            self._displayed_traces.append(displayed)
            self._displayed_trace_axes.append((displayed, ax))
            units.append(displayed.y_unit)
            ax.plot(
                displayed.x_display,
                displayed.y_display,
                linewidth=1.0,
                color=self._COLORS[i % len(self._COLORS)],
                label=entry.stem,
            )
            plotted += 1
        ax.set_xlabel(self._specs[0][1].x_label if self._specs else "x", color=self._FG)
        ax.set_ylabel(units[0] if len(set(units)) == 1 else "value", color=self._FG)
        ax.tick_params(colors=self._FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(self._FG)
        if plotted:
            ax.legend(fontsize=7)
            message = f"Showing {plotted} spectra"
            if skipped:
                message += f"; skipped {len(skipped)} incompatible"
            self._status.setText(message)
        else:
            if skipped:
                self._status.setText(f"No compatible spectra to plot: {skipped[0]}")
            else:
                self._status.setText("No common signal channel to plot.")

        canvas = FigureCanvasQTAgg(fig)
        self._canvas_lay.addWidget(canvas)
        self._canvas = canvas
        self._fig = fig
        canvas.mpl_connect("motion_notify_event", self._on_cursor_motion)
        canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _current_displayed_spectra(self) -> list[DisplayedSpectrum]:
        channel = self._channel_cb.currentText()
        if not channel:
            return []
        offset = float(self._offset_spin.value())
        _ref_entry, ref_spec = self._reference_for_channel(channel)
        rows = []
        skipped = []
        for i, (entry, spec) in enumerate(self._specs):
            if channel not in spec.channels:
                skipped.append(f"{entry.stem}: missing {channel}")
                continue
            issue = self._x_compatibility_issue(ref_spec, spec, channel)
            if issue is not None:
                skipped.append(f"{entry.stem}: {issue}")
                continue
            try:
                rows.append(self._displayed(entry, spec, channel, i * offset))
            except Exception as exc:
                skipped.append(f"{entry.stem}: {exc}")
        if skipped:
            self._status.setText(f"Skipped {len(skipped)} spectra: {skipped[0]}")
        return rows

    def _current_csv_text(self) -> str:
        return displayed_spectra_to_csv_text(self._current_displayed_spectra())

    def _current_json_text(self) -> str:
        return displayed_spectra_to_json_text(self._current_displayed_spectra())

    def _current_txt_text(self) -> str:
        return displayed_spectra_to_txt_text(self._current_displayed_spectra())

    def _on_cursor_motion(self, event) -> None:
        SpecViewerDialog._on_cursor_motion(self, event)

    def _on_canvas_click(self, event) -> None:
        SpecViewerDialog._on_canvas_click(self, event)

    def _set_measure_mode(self, checked: bool) -> None:
        SpecViewerDialog._set_measure_mode(self, checked)

    def _copy_measurement(self) -> None:
        SpecViewerDialog._copy_measurement(self)

    def _add_measurement_result(self) -> None:
        SpecViewerDialog._add_measurement_result(self)

    def _clear_measurement(self) -> None:
        SpecViewerDialog._clear_measurement(self)

    def _reset_measurement_state(self) -> None:
        SpecViewerDialog._reset_measurement_state(self)

    def _traces_for_axis(self, axis) -> list[DisplayedSpectrum]:
        return SpecViewerDialog._traces_for_axis(self, axis)

    def _axis_for_point(self, point: SpectrumMeasurementPoint):
        return SpecViewerDialog._axis_for_point(self, point)

    def _update_crosshair(self, event) -> None:
        SpecViewerDialog._update_crosshair(self, event)

    def _draw_measurement_artists(self) -> None:
        SpecViewerDialog._draw_measurement_artists(self)

    def _remove_crosshair_artists(self) -> None:
        SpecViewerDialog._remove_crosshair_artists(self)

    def _remove_measurement_artists(self) -> None:
        SpecViewerDialog._remove_measurement_artists(self)

    def _copy_csv(self) -> None:
        text = displayed_spectra_to_clipboard_text(self._current_displayed_spectra())
        if not text:
            self._status.setText("No spectrum data to copy.")
            return
        QApplication.clipboard().setText(text)
        self._status.setText("Copied overlay data as CSV.")

    def _export_csv(self) -> None:
        text = self._current_csv_text()
        if not text:
            self._status.setText("No spectrum data to export.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export spectra overlay CSV",
            str(Path.home() / "spectra_overlay.csv"),
            "CSV files (*.csv)",
        )
        if not out_path:
            return
        try:
            Path(out_path).write_text(text, encoding="utf-8")
            self._status.setText(f"CSV → {out_path}")
        except Exception as exc:
            self._status.setText(f"CSV export error: {exc}")

    def _export_json(self) -> None:
        text = self._current_json_text()
        if not text:
            self._status.setText("No spectrum data to export.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export spectra overlay JSON",
            str(Path.home() / "spectra_overlay.displayed.json"),
            "JSON files (*.json)",
        )
        if not out_path:
            return
        try:
            Path(out_path).write_text(text, encoding="utf-8")
            self._status.setText(f"JSON → {out_path}")
        except Exception as exc:
            self._status.setText(f"JSON export error: {exc}")

    def _export_txt(self) -> None:
        text = self._current_txt_text()
        if not text:
            self._status.setText("No spectrum data to export.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export spectra overlay TXT",
            str(Path.home() / "spectra_overlay.txt"),
            "Text files (*.txt)",
        )
        if not out_path:
            return
        try:
            Path(out_path).write_text(text, encoding="utf-8")
            self._status.setText(f"TXT → {out_path}")
        except Exception as exc:
            self._status.setText(f"TXT export error: {exc}")
