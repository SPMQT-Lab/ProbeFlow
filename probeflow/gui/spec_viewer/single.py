"""Single-spectrum viewer dialog."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from probeflow.gui.models import VertFile
from probeflow.gui.widgets import MeasurementResultsTable
from probeflow.measurements.spectrum import spectrum_delta_to_result
from probeflow.spectroscopy.export import (
    displayed_spectra_to_clipboard_text,
    displayed_spectra_to_csv_text,
    displayed_spectra_to_json_text,
    displayed_spectra_to_txt_text,
)
from probeflow.spectroscopy.measurement import (
    SpectrumDeltaMeasurement,
    SpectrumMeasurementPoint,
    format_measurement_summary,
    measure_delta,
    measurement_to_tsv,
    nearest_point_across_traces,
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
    _displayed_trace_for_measurement,
    _focus_in_parameter_inputs,
    _no_wheel_filter,
    _plain_button,
    _shorten_filename,
    _trace_key,
)

class SpecViewerDialog(QDialog):
    """Full-size viewer for a spectroscopy file (Createc .VERT or Nanonis .dat).

    The viewer is channel-agnostic: it builds a toggleable list from
    ``spec.channel_order`` and stacks one subplot per selected channel.
    """

    # Dark-theme colours for plot elements.
    _BG = "#1e1e2e"
    _FG = "#cdd6f4"
    # Plot curve colours, cycled across selected channels.
    _COLORS = ("#89b4fa", "#a6e3a1", "#fab387", "#f5c2e7",
               "#94e2d5", "#f9e2af", "#cba6f7", "#f38ba8")

    def __init__(self, entry: VertFile, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowTitle(entry.stem)
        self.setMinimumSize(900, 560)
        self.resize(1100, 640)
        self._entry = entry
        self._t = t
        self._spec = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._channel_check_widgets: list[QCheckBox] = []
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
        # Unit-override choice per base SI unit. "Auto" means use
        # choose_display_unit; otherwise lookup_unit_scale picks a fixed scale.
        self._unit_choice: dict[str, str] = {"m": "Auto", "A": "Auto", "V": "Auto"}
        self._build()
        self._load()

    # ── UI construction ─────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        self._title = QLabel(self._entry.stem)
        self._title.setFont(QFont("Helvetica", 12, QFont.Bold))
        self._title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._title)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: scrollable channel list.
        self._channels_panel = QWidget()
        self._channels_lay = QVBoxLayout(self._channels_panel)
        self._channels_lay.setContentsMargins(6, 6, 6, 6)
        self._channels_lay.setSpacing(4)
        ch_header = QLabel("Channels")
        ch_header.setFont(QFont("Helvetica", 10, QFont.Bold))
        self._channels_lay.addWidget(ch_header)

        # Unit-override selectors for height (Z) and current channels.
        unit_box = QGroupBox("Display units")
        unit_box.setFont(QFont("Helvetica", 9, QFont.Bold))
        unit_lay = QGridLayout(unit_box)
        unit_lay.setContentsMargins(6, 4, 6, 4)
        unit_lay.setSpacing(2)

        z_lbl = QLabel("Z:")
        z_lbl.setFont(QFont("Helvetica", 9))
        self._z_unit_cb = QComboBox()
        self._z_unit_cb.addItems(["Auto", "pm", "Å", "nm", "µm", "m"])
        self._z_unit_cb.setFont(QFont("Helvetica", 9))
        self._z_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("m", v))

        i_lbl = QLabel("I:")
        i_lbl.setFont(QFont("Helvetica", 9))
        self._i_unit_cb = QComboBox()
        self._i_unit_cb.addItems(["Auto", "fA", "pA", "nA", "µA", "A"])
        self._i_unit_cb.setFont(QFont("Helvetica", 9))
        self._i_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("A", v))

        v_lbl = QLabel("V:")
        v_lbl.setFont(QFont("Helvetica", 9))
        self._v_unit_cb = QComboBox()
        self._v_unit_cb.addItems(["Auto", "µV", "mV", "V"])
        self._v_unit_cb.setFont(QFont("Helvetica", 9))
        self._v_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("V", v))

        unit_lay.addWidget(z_lbl, 0, 0)
        unit_lay.addWidget(self._z_unit_cb, 0, 1)
        unit_lay.addWidget(i_lbl, 1, 0)
        unit_lay.addWidget(self._i_unit_cb, 1, 1)
        unit_lay.addWidget(v_lbl, 2, 0)
        unit_lay.addWidget(self._v_unit_cb, 2, 1)

        self._channels_lay.addWidget(unit_box)

        analysis_box = QGroupBox("Spectrum display")
        analysis_box.setFont(QFont("Helvetica", 9, QFont.Bold))
        analysis_box.setToolTip(_DISPLAY_PIPELINE_TOOLTIP)
        analysis_lay = QGridLayout(analysis_box)
        analysis_lay.setContentsMargins(6, 4, 6, 4)
        analysis_lay.setSpacing(3)

        self._file_lbl = QLabel(_shorten_filename(self._entry.path.name))
        self._file_lbl.setFont(QFont("Helvetica", 8))
        self._file_lbl.setWordWrap(False)
        self._file_lbl.setToolTip(str(self._entry.path))
        self._x_axis_lbl = QLabel("x: —")
        self._x_axis_lbl.setFont(QFont("Helvetica", 8))

        self._signal_cb = QComboBox()
        self._signal_cb.setFont(QFont("Helvetica", 9))
        self._signal_cb.setToolTip("Choose the channel to plot/export as displayed data.")
        self._signal_cb.currentTextChanged.connect(self._on_signal_changed)

        self._smoothing_cb = QComboBox()
        self._smoothing_cb.addItems(["None", "Gaussian", "Savitzky-Golay"])
        self._smoothing_cb.setFont(QFont("Helvetica", 9))
        self._smoothing_cb.setToolTip(_SMOOTHING_TOOLTIP)
        self._smoothing_cb.currentTextChanged.connect(self._on_processing_changed)

        self._smooth_points_spin = QSpinBox()
        self._smooth_points_spin.setRange(3, 9999)
        self._smooth_points_spin.setSingleStep(2)
        self._smooth_points_spin.setValue(7)
        self._smooth_points_spin.setFont(QFont("Helvetica", 9))
        self._smooth_points_spin.setToolTip(
            "Smoothing window in points. Savitzky-Golay requires an odd value "
            "greater than the polynomial order."
        )
        self._smooth_points_spin.valueChanged.connect(self._on_processing_changed)
        self._smooth_points_spin.editingFinished.connect(self._on_processing_changed)

        self._savgol_order_spin = QSpinBox()
        self._savgol_order_spin.setRange(0, 12)
        self._savgol_order_spin.setValue(2)
        self._savgol_order_spin.setFont(QFont("Helvetica", 9))
        self._savgol_order_spin.setToolTip(
            "Savitzky-Golay polynomial order. Must be smaller than the odd window length."
        )
        self._savgol_order_spin.valueChanged.connect(self._on_processing_changed)
        self._savgol_order_spin.editingFinished.connect(self._on_processing_changed)

        self._derivative_cb = QComboBox()
        self._derivative_cb.addItems(["Off", _DERIVATIVE_NUMERIC_LABEL])
        self._derivative_cb.setFont(QFont("Helvetica", 9))
        self._derivative_cb.setToolTip(
            "Compute a numerical derivative of the displayed y channel with respect to x."
        )
        self._derivative_cb.currentTextChanged.connect(self._on_processing_changed)

        self._normalize_cb = QComboBox()
        self._normalize_cb.addItems(NORMALIZATION_LABELS)
        self._normalize_cb.setFont(QFont("Helvetica", 9))
        self._normalize_cb.setToolTip(_NORMALIZATION_TOOLTIP)
        self._normalize_cb.currentTextChanged.connect(self._on_processing_changed)

        self._norm_constant_spin = QDoubleSpinBox()
        self._norm_constant_spin.setRange(-1e12, 1e12)
        self._norm_constant_spin.setDecimals(6)
        self._norm_constant_spin.setSingleStep(1.0)
        self._norm_constant_spin.setValue(1.0)
        self._norm_constant_spin.setFont(QFont("Helvetica", 9))
        self._norm_constant_spin.setToolTip(
            "Constant normalization uses y_display = y_input / constant. "
            "The original data are not changed."
        )
        self._norm_constant_spin.valueChanged.connect(self._on_processing_changed)
        self._norm_constant_spin.editingFinished.connect(self._on_processing_changed)

        self._norm_channel_cb = QComboBox()
        self._norm_channel_cb.setFont(QFont("Helvetica", 9))
        self._norm_channel_cb.setToolTip(
            "Channel normalization uses y_display = y_input / selected_channel."
        )
        self._norm_channel_cb.currentTextChanged.connect(self._on_processing_changed)

        self._formula_lbl = QLabel("Display: y")
        self._formula_lbl.setFont(QFont("Helvetica", 8))
        self._formula_lbl.setWordWrap(True)
        self._formula_lbl.setToolTip(_DISPLAY_PIPELINE_TOOLTIP)

        self._outlier_cb = QComboBox()
        self._outlier_cb.addItems(["Off", "MAD", "Jump"])
        self._outlier_cb.setFont(QFont("Helvetica", 9))
        self._outlier_cb.setToolTip(
            "Mask outliers from displayed/exported arrays; original loaded data stay intact."
        )
        self._outlier_cb.currentTextChanged.connect(self._on_processing_changed)

        self._outlier_threshold_spin = QDoubleSpinBox()
        self._outlier_threshold_spin.setRange(0.1, 1e6)
        self._outlier_threshold_spin.setDecimals(2)
        self._outlier_threshold_spin.setSingleStep(0.5)
        self._outlier_threshold_spin.setValue(6.0)
        self._outlier_threshold_spin.setFont(QFont("Helvetica", 9))
        self._outlier_threshold_spin.setToolTip("Robust outlier threshold for display masking.")
        self._outlier_threshold_spin.valueChanged.connect(self._on_processing_changed)
        self._outlier_threshold_spin.editingFinished.connect(self._on_processing_changed)

        self._plot_mode_cb = QComboBox()
        self._plot_mode_cb.addItems(["Separate", "Overlay", "Waterfall"])
        self._plot_mode_cb.setFont(QFont("Helvetica", 9))
        self._plot_mode_cb.setToolTip("Choose separate axes, overlaid traces, or waterfall offset.")
        self._plot_mode_cb.currentTextChanged.connect(self._redraw)

        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-1e12, 1e12)
        self._offset_spin.setDecimals(4)
        self._offset_spin.setSingleStep(1.0)
        self._offset_spin.setValue(0.0)
        self._offset_spin.setFont(QFont("Helvetica", 9))
        self._offset_spin.setToolTip(
            "Vertical offset applied last, in displayed Y units. Raw data are unchanged."
        )
        self._offset_spin.valueChanged.connect(self._redraw)
        self._offset_spin.editingFinished.connect(self._redraw)

        self._parameter_inputs.extend([
            self._signal_cb,
            self._smoothing_cb,
            self._smooth_points_spin,
            self._savgol_order_spin,
            self._derivative_cb,
            self._normalize_cb,
            self._norm_constant_spin,
            self._norm_channel_cb,
            self._outlier_cb,
            self._outlier_threshold_spin,
            self._plot_mode_cb,
            self._offset_spin,
        ])
        _filt = _no_wheel_filter()
        for _w in self._parameter_inputs:
            if isinstance(_w, QAbstractSpinBox):
                _w.installEventFilter(_filt)
                _w.setFocusPolicy(Qt.StrongFocus)

        analysis_lay.addWidget(QLabel("File:"), 0, 0)
        analysis_lay.addWidget(self._file_lbl, 0, 1)
        analysis_lay.addWidget(QLabel("Bias/x:"), 1, 0)
        analysis_lay.addWidget(self._x_axis_lbl, 1, 1)
        analysis_lay.addWidget(QLabel("Signal:"), 2, 0)
        analysis_lay.addWidget(self._signal_cb, 2, 1)
        analysis_lay.addWidget(QLabel("Smoothing:"), 3, 0)
        analysis_lay.addWidget(self._smoothing_cb, 3, 1)
        analysis_lay.addWidget(QLabel("Window:"), 4, 0)
        analysis_lay.addWidget(self._smooth_points_spin, 4, 1)
        analysis_lay.addWidget(QLabel("Poly order:"), 5, 0)
        analysis_lay.addWidget(self._savgol_order_spin, 5, 1)
        analysis_lay.addWidget(QLabel("Derivative:"), 6, 0)
        analysis_lay.addWidget(self._derivative_cb, 6, 1)
        analysis_lay.addWidget(QLabel("Normalize:"), 7, 0)
        analysis_lay.addWidget(self._normalize_cb, 7, 1)
        analysis_lay.addWidget(QLabel("Constant:"), 8, 0)
        analysis_lay.addWidget(self._norm_constant_spin, 8, 1)
        analysis_lay.addWidget(QLabel("Norm ch:"), 9, 0)
        analysis_lay.addWidget(self._norm_channel_cb, 9, 1)
        analysis_lay.addWidget(QLabel("Formula:"), 10, 0)
        analysis_lay.addWidget(self._formula_lbl, 10, 1)
        analysis_lay.addWidget(QLabel("Outliers:"), 11, 0)
        analysis_lay.addWidget(self._outlier_cb, 11, 1)
        analysis_lay.addWidget(QLabel("Threshold:"), 12, 0)
        analysis_lay.addWidget(self._outlier_threshold_spin, 12, 1)
        analysis_lay.addWidget(QLabel("Plot:"), 13, 0)
        analysis_lay.addWidget(self._plot_mode_cb, 13, 1)
        analysis_lay.addWidget(QLabel("Offset:"), 14, 0)
        analysis_lay.addWidget(self._offset_spin, 14, 1)
        self._channels_lay.addWidget(analysis_box)
        self._channels_lay.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self._channels_panel)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)
        left_scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._left_scroll = left_scroll
        splitter.addWidget(left_scroll)

        # Right panel: plot canvas.
        self._canvas_widget = QWidget()
        self._canvas_widget.setMinimumWidth(520)
        self._canvas_lay = QVBoxLayout(self._canvas_widget)
        self._canvas_lay.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._canvas_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setSizes([340, 760])
        self._splitter = splitter
        lay.addWidget(splitter, 1)

        self._status = QLabel("Loading…")
        self._status.setFont(QFont("Helvetica", 9))
        lay.addWidget(self._status)

        self._cursor_lbl = QLabel("Cursor: —")
        self._cursor_lbl.setFont(QFont("Helvetica", 9))
        lay.addWidget(self._cursor_lbl)

        self._measure_lbl = QLabel("Measurement: off")
        self._measure_lbl.setFont(QFont("Helvetica", 9))
        self._measure_lbl.setWordWrap(True)
        self._measure_lbl.setToolTip(
            "Crosshair measurements use the displayed trace, not the raw spectrum."
        )
        lay.addWidget(self._measure_lbl)

        measure_row = QHBoxLayout()
        self._measure_btn = _plain_button("Measure Δ")
        self._measure_btn.setCheckable(True)
        self._measure_btn.setFixedWidth(110)
        self._measure_btn.setToolTip(
            "Enable crosshair measurement on displayed spectrum data."
        )
        self._measure_btn.toggled.connect(self._set_measure_mode)
        measure_row.addWidget(self._measure_btn)

        self._copy_measure_btn = _plain_button("Copy measurement")
        self._copy_measure_btn.setFixedWidth(150)
        self._copy_measure_btn.clicked.connect(self._copy_measurement)
        measure_row.addWidget(self._copy_measure_btn)

        self._add_measure_btn = _plain_button("Add to measurements")
        self._add_measure_btn.setFixedWidth(170)
        self._add_measure_btn.clicked.connect(self._add_measurement_result)
        measure_row.addWidget(self._add_measure_btn)

        self._clear_measure_btn = _plain_button("Clear measurement")
        self._clear_measure_btn.setFixedWidth(150)
        self._clear_measure_btn.clicked.connect(self._clear_measurement)
        measure_row.addWidget(self._clear_measure_btn)
        measure_row.addStretch(1)
        lay.addLayout(measure_row)

        self._measurement_table = MeasurementResultsTable()
        self._measurement_table.setMaximumHeight(170)
        lay.addWidget(self._measurement_table)

        btn_row = QHBoxLayout()
        self._raw_btn = _plain_button("Show raw data")
        self._raw_btn.setFixedWidth(140)
        self._raw_btn.clicked.connect(self._show_raw_data)
        btn_row.addWidget(self._raw_btn)

        self._export_csv_btn = _plain_button("Export CSV…")
        self._export_csv_btn.setFixedWidth(120)
        self._export_csv_btn.setToolTip(
            "Save the spectrum as a CSV file with one column per selected channel.")
        self._export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self._export_csv_btn)

        self._copy_csv_btn = _plain_button("Copy data")
        self._copy_csv_btn.setFixedWidth(100)
        self._copy_csv_btn.setToolTip(
            "Copy the currently displayed spectrum data as CSV text.")
        self._copy_csv_btn.clicked.connect(self._copy_csv)
        btn_row.addWidget(self._copy_csv_btn)

        self._export_json_btn = _plain_button("Export JSON…")
        self._export_json_btn.setFixedWidth(120)
        self._export_json_btn.clicked.connect(self._export_json)
        btn_row.addWidget(self._export_json_btn)

        self._export_txt_btn = _plain_button("Export TXT…")
        self._export_txt_btn.setFixedWidth(110)
        self._export_txt_btn.clicked.connect(self._export_txt)
        btn_row.addWidget(self._export_txt_btn)

        self._export_grace_btn = _plain_button("Export xmgrace…")
        self._export_grace_btn.setFixedWidth(160)
        self._export_grace_btn.setToolTip(
            "Render via xmgrace (Helvetica default). "
            "Produces three files in the chosen folder: "
            ".agr (re-editable Grace project), .png, and .pdf.")
        self._export_grace_btn.clicked.connect(self._export_xmgrace)
        btn_row.addWidget(self._export_grace_btn)

        btn_row.addStretch()
        close_btn = _plain_button("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self._measure_enabled:
            self._clear_measurement()
            event.accept()
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and _focus_in_parameter_inputs(
            self.focusWidget(), self._parameter_inputs
        ):
            self._on_processing_changed()
            event.accept()
            return
        super().keyPressEvent(event)

    # ── Data load + channel list population ─────────────────────────────

    def _load(self) -> None:
        from probeflow.io.spectroscopy import read_spec_file

        try:
            spec = read_spec_file(self._entry.path)
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
            return
        self._spec = spec
        self._x_axis_lbl.setText(f"{spec.x_label}")

        # Pull channel_order off the spec; fall back to whatever keys are
        # present for old SpecData objects that don't carry it.
        order = list(spec.channel_order) if spec.channel_order else list(spec.channels.keys())
        defaults = set(spec.default_channels)

        # Remove only channel checkboxes from prior loads. Static controls such
        # as the unit QGroupBox own combo boxes and must survive dialog lifetime.
        for w in self._channel_check_widgets:
            self._channels_lay.removeWidget(w)
            w.deleteLater()
        self._channel_check_widgets.clear()
        self._checkboxes.clear()
        for ch in order:
            if ch not in spec.channels:
                continue
            cb = QCheckBox(self._channel_display_label(ch))
            cb.setChecked(ch in defaults)
            cb.toggled.connect(self._redraw)
            self._channels_lay.insertWidget(self._channels_lay.count() - 1, cb)
            self._checkboxes[ch] = cb
            self._channel_check_widgets.append(cb)

        self._signal_cb.blockSignals(True)
        self._signal_cb.clear()
        self._signal_cb.addItems([ch for ch in order if ch in spec.channels])
        first_signal = next((ch for ch in spec.default_channels if ch in spec.channels), None)
        if first_signal is not None:
            self._signal_cb.setCurrentText(first_signal)
        self._signal_cb.blockSignals(False)

        self._norm_channel_cb.blockSignals(True)
        self._norm_channel_cb.clear()
        self._norm_channel_cb.addItems([ch for ch in order if ch in spec.channels])
        self._norm_channel_cb.blockSignals(False)

        sweep = spec.metadata.get("sweep_type", "").replace("_", " ")
        n_pts = spec.metadata.get("n_points", 0)
        try:
            pos = spec.position
            pos_str = f"pos ({pos[0]*1e9:.2f}, {pos[1]*1e9:.2f}) nm"
        except (TypeError, IndexError, KeyError):
            pos_str = "pos unknown"
        self._status.setText(f"{sweep}  |  {n_pts} points  |  {pos_str}")

        self._refresh_formula()
        self._redraw()

    # ── Plotting ────────────────────────────────────────────────────────

    def _on_unit_changed(self, base: str, label: str) -> None:
        self._unit_choice[base] = label
        self._refresh_channel_labels()
        self._redraw()

    def _on_signal_changed(self, ch: str) -> None:
        if not ch or ch not in self._checkboxes:
            return
        for key, cb in self._checkboxes.items():
            cb.blockSignals(True)
            cb.setChecked(key == ch)
            cb.blockSignals(False)
        self._redraw()

    def _on_processing_changed(self, _value: str = "") -> None:
        self._refresh_formula()
        error = self._savgol_validation_error()
        if error is not None:
            self._status.setText(error)
            return
        self._refresh_channel_labels()
        self._redraw()

    def _savgol_validation_error(self, point_count: int | None = None) -> str | None:
        n_points = point_count
        if n_points is None and self._spec is not None:
            n_points = int(len(self._spec.x_array))
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

        points = int(self._smooth_points_spin.value())
        polyorder = int(self._savgol_order_spin.value())

        return SpectrumDisplayOptions(
            smoothing_mode=smoothing,
            smoothing_points=points,
            savgol_polyorder=polyorder,
            derivative=_derivative_enabled(self._derivative_cb.currentText()),
            normalize_mode=normalize,
            normalize_constant=float(self._norm_constant_spin.value()),
            normalize_channel=self._norm_channel_cb.currentText() or None,
            outlier_mode=outliers,
            outlier_threshold=float(self._outlier_threshold_spin.value()),
            vertical_offset=float(vertical_offset),
        )

    def _raw_trace_for_channel(self, ch: str) -> SpectrumTrace | None:
        if self._spec is None or ch not in self._spec.channels:
            return None
        return SpectrumTrace(
            source_file=str(self._entry.path),
            spectrum_id=self._entry.stem,
            x_channel=self._spec.x_label,
            y_channel=ch,
            x_raw=self._spec.x_array,
            y_raw=self._spec.channels[ch],
            x_label=self._spec.x_label,
            y_label=self._channel_label(ch),
            x_unit=self._spec.x_unit,
            y_unit=self._spec.y_units.get(ch, ""),
            metadata={
                "source_file": str(self._entry.path),
                "spectrum_label": self._entry.stem,
                "raw_points": int(len(self._spec.x_array)),
                "setpoint_a": self._spec.metadata.get("setpoint_a"),
            },
        )

    def _displayed_for_channel(self, ch: str, vertical_offset: float = 0.0):
        if self._spec is None:
            return None
        raw = self._raw_trace_for_channel(ch)
        if raw is None:
            return None
        displayed = make_displayed_spectrum(
            raw,
            self._display_options(vertical_offset=0.0),
            channel_lookup=self._spec.channels,
        )
        scale, disp_unit = self._display_scale(displayed.y_unit, displayed.y_display)
        metadata = dict(displayed.metadata)
        metadata.update({
            "display_scale": scale,
            "display_unit": disp_unit,
            "vertical_offset_units": "displayed_y",
        })
        return replace(
            displayed,
            y_display=displayed.y_display * scale + float(vertical_offset),
            y_unit=disp_unit,
            options=replace(displayed.options, vertical_offset=float(vertical_offset)),
            metadata=metadata,
        )

    def _display_scale(self, unit: str, values: np.ndarray) -> tuple[float, str]:
        from probeflow.analysis.spec_plot import choose_display_unit, lookup_unit_scale

        if "/" in unit:
            numerator, denominator = unit.split("/", 1)
            choice = self._unit_choice.get(numerator, "Auto")
            override = lookup_unit_scale(numerator, choice) if choice != "Auto" else None
            scale, disp_num = override or choose_display_unit(numerator, values)
            return scale, f"{disp_num}/{denominator}" if denominator else disp_num
        choice = self._unit_choice.get(unit, "Auto")
        override = lookup_unit_scale(unit, choice) if choice != "Auto" else None
        if override is not None:
            return override
        return choose_display_unit(unit, values)

    def _display_values_for_channel(self, ch: str) -> tuple[np.ndarray, str]:
        displayed = self._displayed_for_channel(ch)
        if displayed is None:
            return np.array([], dtype=float), ""
        return displayed.y_display, displayed.y_unit

    def _channel_display_label(self, ch: str) -> str:
        try:
            displayed = self._displayed_for_channel(ch)
            if displayed is None:
                return ch
            label = displayed.y_label
            disp_unit = displayed.y_unit
        except Exception as exc:
            self._status.setText(f"Display option unavailable for {ch}: {exc}")
            label = self._channel_label(ch)
            disp_unit = self._spec.y_units.get(ch, "") if self._spec is not None else ""
        return f"{label}  ({disp_unit})" if disp_unit else label

    def _channel_label(self, ch: str) -> str:
        if self._spec is None:
            return ch
        info = getattr(self._spec, "channel_info", {}).get(ch)
        return getattr(info, "display_label", ch) if info is not None else ch

    def _refresh_channel_labels(self) -> None:
        for ch, cb in self._checkboxes.items():
            cb.setText(self._channel_display_label(ch))

    def _redraw(self) -> None:
        if self._spec is None:
            return
        self._refresh_formula()
        error = self._savgol_validation_error()
        if error is not None:
            self._status.setText(error)
            return
        self._displayed_traces = []
        self._displayed_trace_axes = []
        self._reset_measurement_state()
        # Remove existing canvas if present.
        if self._canvas is not None:
            self._canvas_lay.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas = None
            self._fig = None

        selected = [ch for ch, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            # Empty figure keeps the area from collapsing.
            fig = Figure(figsize=(8.5, 4.5), tight_layout=True)
            fig.patch.set_facecolor(self._BG)
            canvas = FigureCanvasQTAgg(fig)
            self._canvas_lay.addWidget(canvas)
            self._canvas = canvas
            self._fig = fig
            canvas.mpl_connect("motion_notify_event", self._on_cursor_motion)
            canvas.mpl_connect("button_press_event", self._on_canvas_click)
            return

        fig = Figure(figsize=(8.5, 4.5), tight_layout=True)
        fig.patch.set_facecolor(self._BG)
        axes = fig.subplots(nrows=len(selected), ncols=1, sharex=True)
        if len(selected) == 1:
            axes = [axes]

        plot_mode = self._plot_mode_cb.currentText()
        base_offset = float(self._offset_spin.value())
        if plot_mode in {"Overlay", "Waterfall"}:
            axes = [axes[0]]
            for extra_ax in fig.axes[1:]:
                extra_ax.remove()
            ax = axes[0]
            ax.set_facecolor(self._BG)
            units = []
            for i, ch in enumerate(selected):
                offset = i * base_offset if plot_mode == "Waterfall" else base_offset
                try:
                    displayed = self._displayed_for_channel(ch, offset)
                    if displayed is None:
                        continue
                except Exception as exc:
                    self._status.setText(f"Plot error for {ch}: {exc}")
                    continue
                self._displayed_traces.append(displayed)
                self._displayed_trace_axes.append((displayed, ax))
                y_disp, disp_unit = displayed.y_display, displayed.y_unit
                units.append(disp_unit)
                label = displayed.y_label
                if disp_unit:
                    label = f"{label} ({disp_unit})"
                ax.plot(displayed.x_display, y_disp, linewidth=1.0,
                        color=self._COLORS[i % len(self._COLORS)], label=label)
            ax.set_ylabel(units[0] if len(set(units)) == 1 else "value",
                          color=self._FG, fontsize=8)
            ax.tick_params(colors=self._FG, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(self._FG)
                spine.set_linewidth(0.6)
            ax.legend(fontsize=7)
        else:
            for i, (ch, ax) in enumerate(zip(selected, axes)):
                try:
                    displayed = self._displayed_for_channel(ch, base_offset)
                    if displayed is None:
                        continue
                except Exception as exc:
                    self._status.setText(f"Plot error for {ch}: {exc}")
                    continue
                self._displayed_traces.append(displayed)
                self._displayed_trace_axes.append((displayed, ax))
                y_disp, disp_unit = displayed.y_display, displayed.y_unit

                ax.set_facecolor(self._BG)
                ax.plot(displayed.x_display, y_disp, linewidth=1.0,
                        color=self._COLORS[i % len(self._COLORS)])
                label = displayed.y_label
                ax.set_ylabel(f"{label} ({disp_unit})" if disp_unit else label,
                              color=self._FG, fontsize=8)
                ax.tick_params(colors=self._FG, labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor(self._FG)
                    spine.set_linewidth(0.6)
                if i < len(selected) - 1:
                    ax.tick_params(axis="x", labelbottom=False)

        axes[-1].set_xlabel(self._spec.x_label, color=self._FG, fontsize=8)

        canvas = FigureCanvasQTAgg(fig)
        self._canvas_lay.addWidget(canvas)
        self._canvas = canvas
        self._fig = fig
        canvas.mpl_connect("motion_notify_event", self._on_cursor_motion)
        canvas.mpl_connect("button_press_event", self._on_canvas_click)

    # ── Export ──────────────────────────────────────────────────────────

    def _current_displayed_spectra(self) -> list[DisplayedSpectrum]:
        spec = self._spec
        if spec is None:
            return []
        selected = [ch for ch, cb in self._checkboxes.items() if cb.isChecked()]
        plot_mode = self._plot_mode_cb.currentText()
        base_offset = float(self._offset_spin.value())
        rows = []
        for i, ch in enumerate(selected):
            if ch not in spec.channels:
                continue
            offset = i * base_offset if plot_mode == "Waterfall" else base_offset
            try:
                displayed = self._displayed_for_channel(ch, offset)
            except Exception as exc:
                self._status.setText(f"Display option unavailable for {ch}: {exc}")
                continue
            if displayed is not None:
                rows.append(displayed)
        return rows

    def _on_cursor_motion(self, event) -> None:
        self._update_crosshair(event)
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self._cursor_lbl.setText("Cursor: —")
            return
        if not self._displayed_traces:
            self._cursor_lbl.setText("Cursor: —")
            return

        point = nearest_point_across_traces(
            self._traces_for_axis(event.inaxes),
            event.xdata,
            event.ydata,
            max_normalized_distance=None,
        )
        if point is None:
            self._cursor_lbl.setText("Cursor: —")
            return

        x_unit = f" {point.x_unit}" if point.x_unit else ""
        y_unit = f" {point.y_unit}" if point.y_unit else ""
        self._cursor_lbl.setText(
            f"Cursor (displayed): {point.x_label} = {point.x:.6g}{x_unit}, "
            f"{point.y_label} = {point.y:.6g}{y_unit}  |  "
            f"Trace: {Path(point.source_file).name} | channel: {point.y_channel}"
        )

    def _set_measure_mode(self, checked: bool) -> None:
        self._measure_enabled = checked
        self._remove_crosshair_artists()
        if checked:
            self._measure_lbl.setText("Measurement: click point 1 on a displayed trace.")
            self._status.setText("Crosshair measurements use displayed data, not raw data.")
        else:
            self._measure_lbl.setText(
                format_measurement_summary(self._measurement)
                if self._measurement is not None else "Measurement: off"
            )
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _on_canvas_click(self, event) -> None:
        if not self._measure_enabled:
            return
        if getattr(event, "button", 1) == 3:
            self._clear_measurement()
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self._status.setText("Click inside a spectrum axis to set a measurement point.")
            return

        traces = self._traces_for_axis(event.inaxes)
        if len(self._measure_points) == 1:
            key = self._measure_points[0].trace_key
            traces = [trace for trace in traces if _trace_key(trace) == key]
        point = nearest_point_across_traces(
            traces,
            event.xdata,
            event.ydata,
            max_normalized_distance=0.08,
        )
        if point is None:
            self._status.setText("No finite displayed point close enough to measure.")
            return

        if len(self._measure_points) != 1:
            self._measure_points = [point]
            self._measurement = None
            self._measure_lbl.setText(
                f"Measurement point 1: {point.trace_name}, "
                f"{point.x_label}={point.x:.6g} {point.x_unit}, "
                f"{point.y_label}={point.y:.6g} {point.y_unit}"
            )
            self._status.setText("Measurement point 1 set; click point 2 on the same trace.")
        else:
            try:
                self._measurement = measure_delta(self._measure_points[0], point)
            except ValueError as exc:
                self._status.setText(str(exc))
                return
            self._measure_points.append(point)
            self._measure_lbl.setText(format_measurement_summary(self._measurement))
            self._status.setText("Displayed trace measurement complete.")
        self._draw_measurement_artists()

    def _copy_measurement(self) -> None:
        if self._measurement is None:
            self._status.setText("No completed crosshair measurement to copy.")
            return
        QApplication.clipboard().setText(measurement_to_tsv(self._measurement))
        self._status.setText("Copied displayed-trace measurement.")

    def _add_measurement_result(self) -> None:
        if self._measurement is None:
            self._status.setText("No completed crosshair measurement to add.")
            return
        trace = _displayed_trace_for_measurement(self._displayed_traces, self._measurement)
        result = spectrum_delta_to_result(
            self._measurement,
            measurement_id=self._measurement_table.next_measurement_id(),
            trace=trace,
        )
        self._measurement_table.add_result(result)
        self._status.setText(f"Added measurement {result.measurement_id}.")

    def _clear_measurement(self) -> None:
        self._measure_points.clear()
        self._measurement = None
        self._remove_measurement_artists()
        self._measure_lbl.setText(
            "Measurement: click point 1 on a displayed trace."
            if self._measure_enabled else "Measurement: off"
        )
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _reset_measurement_state(self) -> None:
        self._measure_points.clear()
        self._measurement = None
        self._remove_measurement_artists()
        self._remove_crosshair_artists()
        if hasattr(self, "_measure_lbl"):
            self._measure_lbl.setText(
                "Measurement: click point 1 on a displayed trace."
                if self._measure_enabled else "Measurement: off"
            )

    def _traces_for_axis(self, axis) -> list[DisplayedSpectrum]:
        traces = [
            trace
            for trace, trace_axis in getattr(self, "_displayed_trace_axes", [])
            if trace_axis is axis
        ]
        return traces or list(self._displayed_traces)

    def _axis_for_point(self, point: SpectrumMeasurementPoint):
        for trace, axis in getattr(self, "_displayed_trace_axes", []):
            if _trace_key(trace) == point.trace_key:
                return axis
        return None

    def _update_crosshair(self, event) -> None:
        if not self._measure_enabled:
            return
        self._remove_crosshair_artists()
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        axis = event.inaxes
        self._crosshair_artists = [
            axis.axvline(event.xdata, color="#f9e2af", linewidth=0.8, alpha=0.85),
            axis.axhline(event.ydata, color="#f9e2af", linewidth=0.8, alpha=0.85),
        ]
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _draw_measurement_artists(self) -> None:
        self._remove_measurement_artists()
        axes_and_points = [
            (self._axis_for_point(point), point)
            for point in self._measure_points
        ]
        axes_and_points = [(axis, point) for axis, point in axes_and_points if axis is not None]
        for axis, point in axes_and_points:
            marker = axis.plot(
                [point.x],
                [point.y],
                marker="o",
                markersize=5,
                color="#f38ba8",
                markeredgecolor="#11111b",
                linestyle="None",
                zorder=6,
            )[0]
            self._measurement_artists.append(marker)
        if len(axes_and_points) == 2 and axes_and_points[0][0] is axes_and_points[1][0]:
            axis = axes_and_points[0][0]
            p1 = axes_and_points[0][1]
            p2 = axes_and_points[1][1]
            line = axis.plot(
                [p1.x, p2.x],
                [p1.y, p2.y],
                color="#f38ba8",
                linewidth=1.0,
                alpha=0.9,
                zorder=5,
            )[0]
            self._measurement_artists.append(line)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _remove_crosshair_artists(self) -> None:
        for artist in getattr(self, "_crosshair_artists", []):
            try:
                artist.remove()
            except ValueError:
                pass
        self._crosshair_artists = []

    def _remove_measurement_artists(self) -> None:
        for artist in getattr(self, "_measurement_artists", []):
            try:
                artist.remove()
            except ValueError:
                pass
        self._measurement_artists = []

    def _selected_channels_in_display_units(self):
        for displayed in self._current_displayed_spectra():
            yield displayed.y_channel, displayed.y_label, displayed.y_display, displayed.y_unit

    def _current_csv_text(self) -> str:
        return displayed_spectra_to_csv_text(self._current_displayed_spectra())

    def _current_json_text(self) -> str:
        return displayed_spectra_to_json_text(self._current_displayed_spectra())

    def _current_txt_text(self) -> str:
        return displayed_spectra_to_txt_text(self._current_displayed_spectra())

    def _export_csv(self) -> None:
        if self._spec is None:
            self._status.setText("Nothing to export — spectrum failed to load.")
            return
        csv_text = self._current_csv_text()
        if not csv_text:
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export spectrum CSV",
            str(Path.home() / f"{self._entry.stem}.csv"),
            "CSV files (*.csv)")
        if not out_path:
            return
        try:
            Path(out_path).write_text(csv_text, encoding="utf-8")
            self._status.setText(f"CSV → {out_path}")
        except Exception as exc:
            self._status.setText(f"CSV export error: {exc}")

    def _copy_csv(self) -> None:
        csv_text = displayed_spectra_to_clipboard_text(self._current_displayed_spectra())
        if not csv_text:
            self._status.setText("Tick at least one channel before copying.")
            return
        QApplication.clipboard().setText(csv_text)
        self._status.setText("Copied displayed spectrum data as CSV.")

    def _export_json(self) -> None:
        text = self._current_json_text()
        if not text:
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export displayed spectrum JSON",
            str(Path.home() / f"{self._entry.stem}.displayed.json"),
            "JSON files (*.json)")
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
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export displayed spectrum TXT",
            str(Path.home() / f"{self._entry.stem}.txt"),
            "Text files (*.txt)")
        if not out_path:
            return
        try:
            Path(out_path).write_text(text, encoding="utf-8")
            self._status.setText(f"TXT → {out_path}")
        except Exception as exc:
            self._status.setText(f"TXT export error: {exc}")

    def _export_xmgrace(self) -> None:
        if self._spec is None:
            self._status.setText("Nothing to export — spectrum failed to load.")
            return
        displayed = self._current_displayed_spectra()
        if not displayed:
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output folder for xmgrace export",
            str(Path.home()))
        if not out_dir:
            return
        try:
            from probeflow.analysis.xmgrace_export import Curve, export_bundle
        except ImportError as exc:
            self._status.setText(f"xmgrace export unavailable: {exc}")
            return
        first_x = displayed[0].x_display
        if any(d.x_display.shape != first_x.shape or not np.allclose(d.x_display, first_x)
               for d in displayed[1:]):
            self._status.setText(
                "xmgrace export needs matching displayed x values; use CSV/JSON/TXT.")
            return

        units = {d.y_unit for d in displayed}
        if len(units) == 1:
            y_label = f"value ({displayed[0].y_unit})" if displayed[0].y_unit else "value"
        else:
            y_label = "value"
        curves = [
            Curve(
                name=d.y_channel,
                y=d.y_display,
                legend=f"{Path(d.source_file).stem}: {d.y_label} ({d.y_unit})"
                       if d.y_unit else f"{Path(d.source_file).stem}: {d.y_label}",
            )
            for d in displayed
        ]
        try:
            paths = export_bundle(
                Path(out_dir),
                self._entry.stem,
                first_x,
                curves,
                x_label=displayed[0].x_label,
                y_label=y_label,
                title=self._entry.stem,
                subtitle="ProbeFlow xmgrace export",
                font="Helvetica",
            )
        except FileNotFoundError as exc:
            self._status.setText(f"Export error: {exc}")
            return
        except Exception as exc:
            self._status.setText(f"xmgrace failed: {exc}")
            return
        names = ", ".join(p.name for p in paths.values())
        self._status.setText(f"Exported to {out_dir}: {names}")

    # ── Raw-data table ──────────────────────────────────────────────────

    def _show_raw_data(self) -> None:
        if self._spec is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Raw data — {self._entry.stem}")
        dlg.resize(640, 400)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(8, 8, 8, 8)

        table = self._raw_data_table()
        v.addWidget(table)

        close_btn = _plain_button("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)

        dlg.exec()

    def _raw_data_table(self) -> QTableWidget:
        spec = self._spec
        if spec is None:
            return QTableWidget(0, 0)
        order = [ch for ch in spec.channel_order if ch in spec.channels]
        if not order:
            order = list(spec.channels.keys())
        n_rows = len(spec.x_array)
        n_cols = 1 + len(order)

        table = QTableWidget(n_rows, n_cols)
        display_rows = []
        for ch in order:
            values = np.asarray(spec.channels[ch])
            scale, disp_unit = self._display_scale(spec.y_units.get(ch, ""), values)
            display_rows.append((ch, self._channel_label(ch), values * scale, disp_unit))
        headers = [spec.x_label] + [
            f"{label} ({unit})" if unit else label
            for _ch, label, _values, unit in display_rows
        ]
        table.setHorizontalHeaderLabels(headers)

        for r in range(n_rows):
            table.setItem(r, 0, QTableWidgetItem(f"{spec.x_array[r]:.6g}"))
            for c, (_ch, _label, values, _unit) in enumerate(display_rows, start=1):
                table.setItem(r, c, QTableWidgetItem(f"{values[r]:.6g}"))
        table.horizontalHeader().setStretchLastSection(True)
        return table
