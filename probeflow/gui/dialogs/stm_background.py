"""STM scan-line background subtraction dialog."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from probeflow.processing import STMBackgroundParams, preview_stm_background
from probeflow.processing.display import array_to_uint8


_MODEL_LABELS = {
    "Linear": "linear",
    "2nd order polynomial": "poly2",
    "3rd order polynomial": "poly3",
    "Low-pass": "low_pass",
    "Line by line": "line_by_line",
}


class STMBackgroundDialog(QDialog):
    """Closeable utility dialog for previewing scan-line background subtraction."""

    applied = Signal(dict)

    def __init__(
        self,
        image: np.ndarray,
        *,
        theme: dict | None = None,
        active_roi_mask: np.ndarray | None = None,
        active_roi_id: str | None = None,
        active_roi_name: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("STM Background")
        self.resize(840, 640)
        self.setModal(False)

        self._image = np.asarray(image, dtype=np.float64).copy()
        self._theme = theme or {}
        self._active_roi_mask = (
            None if active_roi_mask is None else np.asarray(active_roi_mask, dtype=bool)
        )
        self._active_roi_id = active_roi_id
        self._active_roi_name = active_roi_name or active_roi_id or "active ROI"
        self._last_result = None
        self._last_mode = "corrected"

        root = QVBoxLayout(self)
        root.setSpacing(8)

        intro = QLabel(
            "Estimate a scan-line background from the fit region, then subtract "
            "that fitted background from the full image."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        controls = QFormLayout()
        controls.setLabelAlignment(Qt.AlignRight)

        self._fit_region_combo = QComboBox()
        self._fit_region_combo.addItems(["Whole image", "Active ROI"])
        if self._active_roi_mask is None:
            self._fit_region_combo.model().item(1).setEnabled(False)
            self._fit_region_combo.setToolTip("Draw/select an area ROI to enable ROI fitting.")
        else:
            self._fit_region_combo.setToolTip(
                f"Active ROI fits use {self._active_roi_name}; subtraction still applies to the full image."
            )
        controls.addRow("Fit region:", self._fit_region_combo)

        self._stat_combo = QComboBox()
        self._stat_combo.addItems(["Median", "Mean"])
        self._stat_combo.setToolTip(
            "Median is robust to adsorbates, pits, and spikes; mean follows all pixels."
        )
        controls.addRow("Line statistic:", self._stat_combo)

        self._model_combo = QComboBox()
        self._model_combo.addItems(list(_MODEL_LABELS))
        controls.addRow("Background model:", self._model_combo)

        self._linear_x_cb = QCheckBox("Linear fit in x first")
        self._linear_x_cb.setToolTip(
            "Optionally remove a per-line x slope before fitting the y background profile."
        )
        controls.addRow("", self._linear_x_cb)

        self._blur_spin = QDoubleSpinBox()
        self._blur_spin.setRange(0.5, 200.0)
        self._blur_spin.setDecimals(1)
        self._blur_spin.setSingleStep(0.5)
        self._blur_spin.setValue(5.0)
        self._blur_spin.setSuffix(" px")
        self._blur_spin.setToolTip("Only used by the low-pass background model.")
        controls.addRow("Blur length:", self._blur_spin)

        jump_row = QHBoxLayout()
        jump_row.setContentsMargins(0, 0, 0, 0)
        self._jump_cb = QCheckBox()
        self._jump_spin = QDoubleSpinBox()
        self._jump_spin.setRange(1e-15, 1e15)
        self._jump_spin.setDecimals(3)
        self._jump_spin.setSingleStep(1.0)
        self._jump_spin.setValue(0.0)
        self._jump_spin.setEnabled(False)
        self._jump_spin.setToolTip(
            "Optional profile-jump suppression before fitting; units match the image data."
        )
        self._jump_cb.toggled.connect(self._jump_spin.setEnabled)
        jump_row.addWidget(self._jump_cb)
        jump_row.addWidget(self._jump_spin, 1)
        controls.addRow("Eliminate jumps above:", jump_row)

        root.addLayout(controls)

        self._status_lbl = QLabel("Preview not run.")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        body = QHBoxLayout()
        self._preview_lbl = QLabel("Preview")
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setMinimumSize(260, 260)
        self._preview_lbl.setStyleSheet(
            f"background: {self._theme.get('sidebar_bg', '#181825')};"
        )
        body.addWidget(self._preview_lbl, 1)

        self._fig = Figure(figsize=(4.2, 3.0), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        body.addWidget(self._canvas, 1)
        root.addLayout(body, 1)

        buttons = QHBoxLayout()
        self._background_btn = QPushButton("Preview background")
        self._corrected_btn = QPushButton("Preview corrected image")
        self._apply_btn = QPushButton("Apply")
        close_btn = QPushButton("Close")
        buttons.addWidget(self._background_btn)
        buttons.addWidget(self._corrected_btn)
        buttons.addStretch()
        buttons.addWidget(self._apply_btn)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        self._background_btn.clicked.connect(lambda: self._preview("background"))
        self._corrected_btn.clicked.connect(lambda: self._preview("corrected"))
        self._apply_btn.clicked.connect(self._apply)
        close_btn.clicked.connect(self.close)
        self._model_combo.currentTextChanged.connect(self._sync_controls)
        self._sync_controls()

    def processing_params(self) -> dict:
        model = _MODEL_LABELS[self._model_combo.currentText()]
        fit_region = "active_roi" if self._fit_region_combo.currentIndex() == 1 else "whole_image"
        params = {
            "fit_region": fit_region,
            "line_statistic": self._stat_combo.currentText().lower(),
            "model": model,
            "linear_x_first": self._linear_x_cb.isChecked(),
            "preserve_level": "median",
        }
        if model == "low_pass":
            params["blur_length"] = float(self._blur_spin.value())
        else:
            params["blur_length"] = None
        params["jump_threshold"] = (
            float(self._jump_spin.value()) if self._jump_cb.isChecked() else None
        )
        if fit_region == "active_roi" and self._active_roi_id:
            params["fit_roi_id"] = self._active_roi_id
            params["applied_to"] = "whole_image"
        return params

    def _params_obj(self) -> STMBackgroundParams:
        params = self.processing_params()
        return STMBackgroundParams(
            fit_region=str(params["fit_region"]),
            line_statistic=str(params["line_statistic"]),
            model=str(params["model"]),
            linear_x_first=bool(params["linear_x_first"]),
            blur_length=params.get("blur_length"),
            jump_threshold=params.get("jump_threshold"),
            preserve_level=str(params["preserve_level"]),
        )

    def _fit_mask(self) -> np.ndarray | None:
        if self._fit_region_combo.currentIndex() == 1:
            return self._active_roi_mask
        return None

    def _preview(self, mode: str) -> None:
        self._last_mode = mode
        try:
            result = preview_stm_background(
                self._image,
                self._params_obj(),
                mask=self._fit_mask(),
            )
        except Exception as exc:
            self._last_result = None
            self._status_lbl.setText(f"Fit failed: {exc}")
            QMessageBox.warning(
                self,
                "STM Background",
                f"Fit failed: {exc}\n\nTry linear, polynomial, or low-pass background.",
            )
            return
        self._last_result = result
        image = result.background_image if mode == "background" else result.corrected
        self._show_image(image)
        self._plot_profile(result.line_profile, result.fitted_profile)
        label = "background" if mode == "background" else "corrected image"
        fit_region = self.processing_params()["fit_region"].replace("_", " ")
        self._status_lbl.setText(
            f"Previewed {label}; fit region: {fit_region}; model: {result.params.model}."
        )

    def _apply(self) -> None:
        if self._last_result is None:
            self._preview("corrected")
            if self._last_result is None:
                return
        self.applied.emit(self.processing_params())
        self.close()

    def _show_image(self, arr: np.ndarray) -> None:
        try:
            gray = array_to_uint8(arr)
        except Exception:
            gray = np.zeros(np.asarray(arr).shape[:2], dtype=np.uint8)
        qimg = QImage(
            gray.data,
            gray.shape[1],
            gray.shape[0],
            gray.strides[0],
            QImage.Format_Grayscale8,
        ).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self._preview_lbl.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._preview_lbl.setPixmap(pix)

    def _plot_profile(self, profile: np.ndarray, fitted: np.ndarray) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        y = np.arange(profile.size)
        ax.plot(y, profile, ".", markersize=3, label="line statistic")
        ax.plot(y, fitted, "-", linewidth=1.5, label="fitted background")
        ax.set_xlabel("scan line")
        ax.set_ylabel("height")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _sync_controls(self) -> None:
        is_low_pass = _MODEL_LABELS[self._model_combo.currentText()] == "low_pass"
        self._blur_spin.setEnabled(is_low_pass)
