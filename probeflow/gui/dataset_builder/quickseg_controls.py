"""QuickSeg controls for the Dataset Builder right panel."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from probeflow.dataset_builder.quickseg import EDGE_SCALE_PRESETS, QuickSegParams
from probeflow.gui.typography import ui_font


class _Section(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._toggle = QPushButton(f"> {title}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setFont(ui_font(9, weight=QFont.Bold))
        self._toggle.setFixedHeight(28)
        self._toggle.setCursor(QCursor(Qt.PointingHandCursor))

        self._body = QFrame()
        self._body.setVisible(False)
        self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(0, 6, 0, 0)
        self._body_lay.setSpacing(8)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self._toggle)
        lay.addWidget(self._body)
        self._toggle.toggled.connect(self._on_toggled)

    @property
    def body_layout(self) -> QVBoxLayout:
        return self._body_lay

    def _on_toggled(self, checked: bool) -> None:
        self._body.setVisible(bool(checked))
        self._toggle.setText(("v " if checked else "> ") + self._toggle.text()[2:])


class QuickSegControlsWidget(QWidget):
    """Task-specific controls for terrace segmentation."""

    parameters_changed = Signal()
    overlay_changed = Signal()
    preview_changed = Signal()
    apply_requested = Signal()
    reset_requested = Signal()
    set_default_requested = Signal()
    new_label_requested = Signal()
    undo_seed_requested = Signal()
    clear_seeds_requested = Signal()
    refresh_requested = Signal()
    clear_result_requested = Signal()
    save_requested = Signal()
    save_next_requested = Signal()
    accept_requested = Signal()
    uncertain_requested = Signal()
    reject_requested = Signal()

    def __init__(self, theme: dict, parent=None):
        super().__init__(parent)
        self._theme = dict(theme)
        self._default_params = QuickSegParams()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        header = QLabel("<b>QuickSeg terrace segmentation</b>")
        root.addWidget(header)

        status_form = QFormLayout()
        status_form.setContentsMargins(0, 0, 0, 0)
        status_form.setSpacing(4)

        self._current_label = QSpinBox()
        self._current_label.setRange(1, 9999)
        self._current_label.setValue(1)
        self._current_label.setReadOnly(True)
        self._current_label.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self._seed_mode_lbl = QLabel("Add seed mode")
        self._seed_mode_lbl.setWordWrap(True)

        self._result_lbl = QLabel("No watershed result")
        self._result_lbl.setWordWrap(True)

        status_form.addRow("Current terrace label", self._current_label)
        status_form.addRow("Seed mode", self._seed_mode_lbl)
        status_form.addRow("Result", self._result_lbl)
        root.addLayout(status_form)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._new_label_btn = QPushButton("New terrace / next label")
        self._undo_btn = QPushButton("Undo seed")
        btn_row.addWidget(self._new_label_btn)
        btn_row.addWidget(self._undo_btn)
        root.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(6)
        self._clear_seeds_btn = QPushButton("Clear seeds")
        self._refresh_btn = QPushButton("R / Refresh watershed")
        self._clear_result_btn = QPushButton("Clear result")
        btn_row2.addWidget(self._clear_seeds_btn)
        btn_row2.addWidget(self._refresh_btn)
        btn_row2.addWidget(self._clear_result_btn)
        root.addLayout(btn_row2)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("accentBtn")
        self._reset_btn = QPushButton("Reset")
        self._set_default_btn = QPushButton("Set current parameters as default")
        self._auto_refresh_chk = QCheckBox("Auto-refresh after Apply")
        self._show_seeds_chk = QCheckBox("Show seeds")
        self._show_seeds_chk.setChecked(True)
        self._show_boundaries_chk = QCheckBox("Show boundaries")
        self._show_boundaries_chk.setChecked(True)
        self._show_filled_chk = QCheckBox("Show filled regions")
        self._show_filled_chk.setChecked(True)

        root.addWidget(self._apply_btn)
        root.addWidget(self._reset_btn)
        root.addWidget(self._set_default_btn)
        root.addWidget(self._auto_refresh_chk)
        root.addWidget(self._show_seeds_chk)
        root.addWidget(self._show_boundaries_chk)
        root.addWidget(self._show_filled_chk)

        opacity_form = QFormLayout()
        opacity_form.setContentsMargins(0, 0, 0, 0)
        opacity_form.setSpacing(4)
        self._opacity = QDoubleSpinBox()
        self._opacity.setRange(0.0, 1.0)
        self._opacity.setDecimals(2)
        self._opacity.setSingleStep(0.05)
        self._opacity.setValue(0.55)
        opacity_form.addRow("Overlay opacity", self._opacity)
        root.addLayout(opacity_form)

        preview_form = QFormLayout()
        preview_form.setContentsMargins(0, 0, 0, 0)
        preview_form.setSpacing(4)
        self._show_preprocessing_preview = QCheckBox("Show preprocessing preview")
        self._preview_stage = QComboBox()
        self._preview_stage.addItem("Watershed elevation", "watershed_elevation")
        self._preview_stage.addItem("Gradient contrast", "gradient_contrast")
        self._preview_stage.addItem("Connected edge mask", "connected_edge_mask")
        self._preview_stage.addItem("Anisotropic blur", "anisotropic_blur")
        self._preview_stage.addItem("Denoised", "denoised")
        self._preview_stage.addItem("Flattened display", "flat_display")
        preview_form.addRow("", self._show_preprocessing_preview)
        preview_form.addRow("Preview stage", self._preview_stage)
        root.addLayout(preview_form)

        self._advanced = _Section("Tuning")
        root.addWidget(self._advanced)
        adv_form = QFormLayout()
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_form.setSpacing(6)
        adv_form.setHorizontalSpacing(10)
        adv_form.setVerticalSpacing(6)
        adv_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._denoise_strength = self._make_float_spinbox(0.04, 0.02, 0.16, 0.005, decimals=3)
        self._smooth_along_scan = self._make_float_spinbox(1.2, 0.8, 3.5, 0.1, decimals=2)
        self._smooth_across_scan = self._make_float_spinbox(0.7, 0.4, 1.2, 0.05, decimals=2)
        self._edge_scale = QComboBox()
        for name in EDGE_SCALE_PRESETS:
            self._edge_scale.addItem(name, name)
        self._edge_sensitivity = self._make_float_spinbox(84.0, 80.0, 92.0, 0.5, decimals=1)
        self._min_edge_size = QSpinBox()
        self._min_edge_size.setRange(10, 120)
        self._min_edge_size.setValue(40)
        self._min_edge_size.setMinimumWidth(120)
        self._edge_connect_strength = self._make_float_spinbox(0.45, 0.25, 0.75, 0.05, decimals=2)
        self._barrier_strength = self._make_float_spinbox(0.18, 0.10, 0.45, 0.02, decimals=2)
        adv_form.addRow("Denoise strength", self._denoise_strength)
        adv_form.addRow("Smooth along scan", self._smooth_along_scan)
        adv_form.addRow("Smooth across scan", self._smooth_across_scan)
        adv_form.addRow("Edge scale", self._edge_scale)
        adv_form.addRow("Edge sensitivity", self._edge_sensitivity)
        adv_form.addRow("Min edge size", self._min_edge_size)
        adv_form.addRow("Edge connect strength", self._edge_connect_strength)
        adv_form.addRow("Barrier strength", self._barrier_strength)
        self._advanced.body_layout.addLayout(adv_form)

        review_row = QHBoxLayout()
        review_row.setSpacing(6)
        self._save_btn = QPushButton("Save")
        self._accept_btn = QPushButton("Accept")
        self._uncertain_btn = QPushButton("Uncertain")
        self._reject_btn = QPushButton("Reject")
        self._save_next_btn = QPushButton("Save + Next")
        review_row.addWidget(self._save_btn)
        review_row.addWidget(self._accept_btn)
        review_row.addWidget(self._uncertain_btn)
        review_row.addWidget(self._reject_btn)
        root.addLayout(review_row)
        root.addWidget(self._save_next_btn)
        root.addStretch(1)

        self._connect_changes()

    def _make_float_spinbox(
        self,
        value: float,
        lo: float,
        hi: float,
        step: float,
        *,
        decimals: int = 6,
    ) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(decimals)
        sb.setSingleStep(step)
        sb.setValue(value)
        sb.setMinimumWidth(120)
        return sb

    def _make_percent_spinbox(self, value: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(0.0, 100.0)
        sb.setDecimals(1)
        sb.setSingleStep(1.0)
        sb.setSuffix(" %")
        sb.setValue(value)
        sb.setMinimumWidth(120)
        return sb

    def _connect_changes(self) -> None:
        for widget in (
            self._denoise_strength,
            self._smooth_along_scan,
            self._smooth_across_scan,
            self._edge_scale,
            self._edge_sensitivity,
            self._min_edge_size,
            self._edge_connect_strength,
            self._barrier_strength,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.parameters_changed.emit)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.parameters_changed.emit)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self.parameters_changed.emit)

        for widget in (
            self._show_seeds_chk,
            self._show_boundaries_chk,
            self._show_filled_chk,
            self._opacity,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(lambda _value: self.overlay_changed.emit())
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(lambda _checked: self.overlay_changed.emit())

        self._apply_btn.clicked.connect(self.apply_requested.emit)
        self._reset_btn.clicked.connect(self.reset_requested.emit)
        self._set_default_btn.clicked.connect(self.set_default_requested.emit)
        self._new_label_btn.clicked.connect(self.new_label_requested.emit)
        self._undo_btn.clicked.connect(self.undo_seed_requested.emit)
        self._clear_seeds_btn.clicked.connect(self.clear_seeds_requested.emit)
        self._refresh_btn.clicked.connect(self.refresh_requested.emit)
        self._clear_result_btn.clicked.connect(self.clear_result_requested.emit)
        self._save_btn.clicked.connect(self.save_requested.emit)
        self._accept_btn.clicked.connect(self.accept_requested.emit)
        self._uncertain_btn.clicked.connect(self.uncertain_requested.emit)
        self._reject_btn.clicked.connect(self.reject_requested.emit)
        self._save_next_btn.clicked.connect(self.save_next_requested.emit)
        self._show_preprocessing_preview.toggled.connect(lambda _checked: self.preview_changed.emit())
        self._preview_stage.currentIndexChanged.connect(lambda _idx: self.preview_changed.emit())

    def parameters(self) -> QuickSegParams:
        return QuickSegParams(
            denoise_strength=float(self._denoise_strength.value()),
            smooth_along_scan=float(self._smooth_along_scan.value()),
            smooth_across_scan=float(self._smooth_across_scan.value()),
            edge_scale=str(self._edge_scale.currentData() or "balanced"),
            edge_sensitivity=float(self._edge_sensitivity.value()),
            min_edge_size=int(self._min_edge_size.value()),
            edge_connect_strength=float(self._edge_connect_strength.value()),
            barrier_strength=float(self._barrier_strength.value()),
            overlay_opacity=float(self._opacity.value()),
        )

    def set_parameters(self, params: QuickSegParams, *, emit_changed: bool = False) -> None:
        widgets = (
            self._denoise_strength,
            self._smooth_along_scan,
            self._smooth_across_scan,
            self._edge_scale,
            self._edge_sensitivity,
            self._min_edge_size,
            self._edge_connect_strength,
            self._barrier_strength,
            self._show_seeds_chk,
            self._show_boundaries_chk,
            self._show_filled_chk,
            self._opacity,
            self._auto_refresh_chk,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self._denoise_strength.setValue(float(params.denoise_strength))
        self._smooth_along_scan.setValue(float(params.smooth_along_scan))
        self._smooth_across_scan.setValue(float(params.smooth_across_scan))
        idx = self._edge_scale.findData(str(params.edge_scale))
        if idx >= 0:
            self._edge_scale.setCurrentIndex(idx)
        self._edge_sensitivity.setValue(float(params.edge_sensitivity))
        self._min_edge_size.setValue(int(params.min_edge_size))
        self._edge_connect_strength.setValue(float(params.edge_connect_strength))
        self._barrier_strength.setValue(float(params.barrier_strength))
        self._opacity.setValue(float(params.overlay_opacity))
        for widget in widgets:
            widget.blockSignals(False)
        if emit_changed:
            self.parameters_changed.emit()

    def reset_parameters(self) -> None:
        self.set_parameters(self._default_params, emit_changed=False)

    def auto_refresh_after_apply(self) -> bool:
        return bool(self._auto_refresh_chk.isChecked())

    def show_seeds(self) -> bool:
        return bool(self._show_seeds_chk.isChecked())

    def show_boundaries(self) -> bool:
        return bool(self._show_boundaries_chk.isChecked())

    def show_filled_regions(self) -> bool:
        return bool(self._show_filled_chk.isChecked())

    def overlay_opacity(self) -> float:
        return float(self._opacity.value())

    def show_preprocessing_preview(self) -> bool:
        return bool(self._show_preprocessing_preview.isChecked())

    def preview_stage(self) -> str:
        return str(self._preview_stage.currentData() or "watershed_elevation")

    def set_current_label(self, label: int) -> None:
        self._current_label.setValue(int(label))

    def current_label(self) -> int:
        return int(self._current_label.value())

    def set_seed_mode_status(self, text: str) -> None:
        self._seed_mode_lbl.setText(text)

    def set_result_status(self, text: str) -> None:
        self._result_lbl.setText(text)

    def set_review_enabled(self, enabled: bool) -> None:
        for widget in (
            self._save_btn,
            self._accept_btn,
            self._uncertain_btn,
            self._reject_btn,
            self._save_next_btn,
            self._apply_btn,
            self._reset_btn,
            self._set_default_btn,
            self._new_label_btn,
            self._undo_btn,
            self._clear_seeds_btn,
            self._refresh_btn,
            self._clear_result_btn,
        ):
            widget.setEnabled(bool(enabled))

    def apply_theme(self, theme: dict) -> None:
        self._theme = dict(theme)
        border = theme.get("border", "#3a414c")
        bg = theme.get("main_bg", "#16181d")
        fg = theme.get("fg", "#e6e8eb")
        sub_fg = theme.get("sub_fg", "#9aa1ab")
        accent = theme.get("accent_bg", "#4d8dff")
        accent_fg = theme.get("accent_fg", "#0c0e12")
        self.setStyleSheet(f"background: transparent; color: {fg};")
        for btn in (
            self._new_label_btn,
            self._undo_btn,
            self._clear_seeds_btn,
            self._refresh_btn,
            self._clear_result_btn,
            self._save_btn,
            self._accept_btn,
            self._uncertain_btn,
            self._reject_btn,
            self._save_next_btn,
            self._apply_btn,
            self._reset_btn,
            self._set_default_btn,
        ):
            if btn is self._apply_btn:
                btn.setStyleSheet(
                    "QPushButton {"
                    f" background-color: {accent}; color: {accent_fg};"
                    f" border: 1px solid {accent}; border-radius: 6px; padding: 4px 10px; font-weight: 700;"
                    "}"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton {"
                    f" background-color: {bg}; color: {fg};"
                    f" border: 1px solid {border}; border-radius: 6px; padding: 4px 10px;"
                    "}"
                )
        self._seed_mode_lbl.setStyleSheet(f"color: {sub_fg};")
        self._result_lbl.setStyleSheet(f"color: {sub_fg};")
