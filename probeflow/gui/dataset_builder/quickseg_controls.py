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

from probeflow.dataset_builder.quickseg import QuickSegParams
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
        self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
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
    apply_requested = Signal()
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
        self._auto_refresh_chk = QCheckBox("Auto-refresh after Apply")
        self._show_seeds_chk = QCheckBox("Show seeds")
        self._show_seeds_chk.setChecked(True)
        self._show_boundaries_chk = QCheckBox("Show boundaries")
        self._show_boundaries_chk.setChecked(True)
        self._show_filled_chk = QCheckBox("Show filled regions")
        self._show_filled_chk.setChecked(True)

        root.addWidget(self._apply_btn)
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

        self._advanced = _Section("Advanced")
        root.addWidget(self._advanced)
        adv_form = QFormLayout()
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_form.setSpacing(6)
        adv_form.setHorizontalSpacing(10)
        adv_form.setVerticalSpacing(6)
        adv_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._background_mode = QComboBox()
        self._background_mode.addItem("None", "none")
        self._background_mode.addItem("subtract_background(order=1)", "subtract_background_order1")
        self._background_mode.addItem("Old plane fit", "plane_fit")
        self._background_mode.addItem("Facet level", "facet_level")
        self._background_mode.setMinimumWidth(220)
        self._plane_lo = self._make_percent_spinbox(1.0)
        self._plane_hi = self._make_percent_spinbox(99.0)
        self._tv_weight = self._make_float_spinbox(0.25, 0.0, 10.0, 0.05)
        self._tv_iters = QSpinBox()
        self._tv_iters.setRange(1, 5000)
        self._tv_iters.setValue(500)
        self._tv_eps = self._make_float_spinbox(2.0e-5, 1.0e-8, 1.0, 1.0e-5)
        self._gaussian_sigma = self._make_float_spinbox(4.0, 0.0, 100.0, 0.5)
        self._gaussian_order = QSpinBox()
        self._gaussian_order.setRange(0, 5)
        self._gaussian_order.setValue(0)
        self._gaussian_mode = QComboBox()
        self._gaussian_mode.addItems(["reflect", "nearest", "mirror", "wrap", "constant"])
        self._gaussian_mode.setMinimumWidth(120)
        self._watershed_connectivity = QSpinBox()
        self._watershed_connectivity.setRange(1, 5)
        self._watershed_connectivity.setValue(1)
        self._compactness = self._make_float_spinbox(0.0, 0.0, 1000.0, 0.05)
        self._watershed_line = QCheckBox("Use watershed line")
        adv_form.addRow("Plane/background mode", self._background_mode)
        adv_form.addRow("Plane percentile min", self._plane_lo)
        adv_form.addRow("Plane percentile max", self._plane_hi)
        adv_form.addRow("TV denoise weight", self._tv_weight)
        adv_form.addRow("TV iterations", self._tv_iters)
        adv_form.addRow("TV eps", self._tv_eps)
        adv_form.addRow("Gaussian sigma", self._gaussian_sigma)
        adv_form.addRow("Gaussian order", self._gaussian_order)
        adv_form.addRow("Gaussian mode", self._gaussian_mode)
        adv_form.addRow("Watershed connectivity", self._watershed_connectivity)
        adv_form.addRow("Watershed compactness", self._compactness)
        adv_form.addRow("", self._watershed_line)
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

    def _make_float_spinbox(self, value: float, lo: float, hi: float, step: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(6)
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
            self._background_mode,
            self._plane_lo,
            self._plane_hi,
            self._tv_weight,
            self._tv_iters,
            self._tv_eps,
            self._gaussian_sigma,
            self._gaussian_order,
            self._gaussian_mode,
            self._watershed_connectivity,
            self._compactness,
            self._watershed_line,
            self._show_seeds_chk,
            self._show_boundaries_chk,
            self._show_filled_chk,
            self._opacity,
            self._auto_refresh_chk,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.parameters_changed.emit)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.parameters_changed.emit)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self.parameters_changed.emit)

        self._apply_btn.clicked.connect(self.apply_requested.emit)
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

    def parameters(self) -> QuickSegParams:
        return QuickSegParams(
            background_mode=str(self._background_mode.currentData() or "none"),
            plane_percentile_low=float(self._plane_lo.value()),
            plane_percentile_high=float(self._plane_hi.value()),
            tv_weight=float(self._tv_weight.value()),
            tv_iterations=int(self._tv_iters.value()),
            tv_eps=float(self._tv_eps.value()),
            gaussian_sigma=float(self._gaussian_sigma.value()),
            gaussian_order=int(self._gaussian_order.value()),
            gaussian_mode=str(self._gaussian_mode.currentText()),
            gaussian_axes=(1,),
            watershed_connectivity=int(self._watershed_connectivity.value()),
            watershed_compactness=float(self._compactness.value()),
            watershed_line=bool(self._watershed_line.isChecked()),
            overlay_opacity=float(self._opacity.value()),
        )

    def set_parameters(self, params: QuickSegParams) -> None:
        self._background_mode.setCurrentIndex(max(0, self._background_mode.findData(params.background_mode)))
        self._plane_lo.setValue(float(params.plane_percentile_low))
        self._plane_hi.setValue(float(params.plane_percentile_high))
        self._tv_weight.setValue(float(params.tv_weight))
        self._tv_iters.setValue(int(params.tv_iterations))
        self._tv_eps.setValue(float(params.tv_eps))
        self._gaussian_sigma.setValue(float(params.gaussian_sigma))
        self._gaussian_order.setValue(int(params.gaussian_order))
        idx = self._gaussian_mode.findText(str(params.gaussian_mode))
        if idx >= 0:
            self._gaussian_mode.setCurrentIndex(idx)
        self._watershed_connectivity.setValue(int(params.watershed_connectivity))
        self._compactness.setValue(float(params.watershed_compactness))
        self._watershed_line.setChecked(bool(params.watershed_line))
        self._opacity.setValue(float(params.overlay_opacity))

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
