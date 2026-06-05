"""Advanced Edge Detection dialog (Canny + Sobel/Scharr).

A modeless utility dialog built like :class:`STMBackgroundDialog`: a live,
non-destructive overlay preview plus explicit output actions that turn the
result into reusable analysis objects — an overlay, a new image, an active
mask, or ROI(s).  The dialog never mutates the source image.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from probeflow.processing.display import array_to_uint8
from probeflow.processing.edge_detection import (
    CANNY_PRESETS,
    EdgeDetectionResult,
    canny_edges,
    gradient_filter,
)

_PRESET_CUSTOM = "Custom"
_OVERLAY_RGBA = (255, 59, 48, 170)  # red, matching the bad-segment overlay hue


class EdgeDetectionDialog(QDialog):
    """Preview Canny / Sobel-Scharr edge detection and emit reusable outputs."""

    # Output signals — the viewer connects to these.
    overlay_requested = Signal(object)        # EdgeDetectionResult
    overlay_cleared = Signal()
    mask_created = Signal(object)             # ImageMask
    rois_created = Signal(list)               # list[ROI]
    image_created = Signal(object, dict)      # (display_image ndarray, provenance dict)

    def __init__(
        self,
        image: np.ndarray,
        *,
        theme: dict | None = None,
        pixel_size_nm: float | None = None,
        pixel_size_x_nm: float | None = None,
        pixel_size_y_nm: float | None = None,
        active_roi_mask: np.ndarray | None = None,
        source_channel: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Advanced Edge Detection")
        self.resize(880, 680)
        self.setModal(False)

        self._image = np.asarray(image, dtype=np.float64).copy()
        self._theme = theme or {}
        self._pixel_size_nm = pixel_size_nm
        self._pixel_size_x_nm = pixel_size_x_nm or pixel_size_nm
        self._pixel_size_y_nm = pixel_size_y_nm or pixel_size_nm
        self._active_roi_mask = (
            None if active_roi_mask is None else np.asarray(active_roi_mask, dtype=bool)
        )
        self._source_channel = source_channel
        self._result: EdgeDetectionResult | None = None

        root = QVBoxLayout(self)
        root.setSpacing(8)

        intro = QLabel(
            "Detect edges without altering the image. Preview as an overlay, then "
            "create a mask, ROI(s), or a new image from the result."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Debounced live recompute — created before the panels because building
        # them fires control signals that call ``_schedule``.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._recompute)

        # ── Method selector ────────────────────────────────────────────────
        method_row = QFormLayout()
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Canny", "Sobel / Scharr"])
        method_row.addRow("Method:", self._method_combo)
        root.addLayout(method_row)

        # ── Parameter panels (swapped by the method selector) ────────────────
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_canny_panel())
        self._stack.addWidget(self._build_gradient_panel())
        root.addWidget(self._stack)

        # ── Preview ──────────────────────────────────────────────────────────
        self._preview_lbl = QLabel("Preview")
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setMinimumSize(360, 320)
        self._preview_lbl.setStyleSheet(
            f"background: {self._theme.get('sidebar_bg', '#181825')};"
        )
        root.addWidget(self._preview_lbl, 1)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        # ── Output buttons ─────────────────────────────────────────────────
        buttons = QHBoxLayout()
        self._overlay_btn = QPushButton("Overlay on image")
        self._image_btn = QPushButton("Open as new image")
        self._mask_btn = QPushButton("Create mask")
        self._roi_btn = QPushButton("Convert to ROI(s)")
        close_btn = QPushButton("Close")
        for b in (self._overlay_btn, self._image_btn, self._mask_btn, self._roi_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        self._overlay_btn.clicked.connect(self._emit_overlay)
        self._image_btn.clicked.connect(self._emit_image)
        self._mask_btn.clicked.connect(self._emit_mask)
        self._roi_btn.clicked.connect(self._emit_rois)
        close_btn.clicked.connect(self._on_close)

        self._method_combo.currentIndexChanged.connect(self._stack.setCurrentIndex)
        self._method_combo.currentIndexChanged.connect(lambda _: self._schedule())

        self._apply_control_sizing()
        self._recompute()

    def _apply_control_sizing(self) -> None:
        """Stop combo/label truncation and enlarge the tiny spin-box arrows."""
        for combo in self.findChildren(QComboBox):
            combo.setMinimumWidth(180)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            # Widen the popup so item text (e.g. "Sobel"/"Scharr") is not clipped.
            combo.view().setMinimumWidth(180)
        for spin in self.findChildren(QAbstractSpinBox):
            spin.setMinimumHeight(28)
            spin.setMinimumWidth(120)
            spin.setButtonSymbols(QAbstractSpinBox.UpDownArrows)

    # ── Panel construction ──────────────────────────────────────────────────

    def _build_canny_panel(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._canny_preset = QComboBox()
        self._canny_preset.addItem(_PRESET_CUSTOM)
        self._canny_preset.addItems(list(CANNY_PRESETS))
        self._canny_preset.setToolTip("Presets fill sigma and thresholds for common STM cases.")
        form.addRow("Preset:", self._canny_preset)

        self._canny_sigma = QDoubleSpinBox()
        self._canny_sigma.setRange(0.0, 20.0)
        self._canny_sigma.setSingleStep(0.1)
        self._canny_sigma.setDecimals(2)
        self._canny_sigma.setValue(1.0)
        self._canny_sigma.setSuffix(" px")
        self._canny_sigma_lbl = QLabel("")
        srow = QHBoxLayout()
        srow.setContentsMargins(0, 0, 0, 0)
        srow.addWidget(self._canny_sigma, 1)
        srow.addWidget(self._canny_sigma_lbl)
        form.addRow("Gaussian sigma:", _wrap(srow))

        self._canny_mode = QComboBox()
        self._canny_mode.addItems(["Percentile", "Absolute"])
        self._canny_mode.setToolTip(
            "Percentile thresholds are robust across STM channels whose absolute "
            "scale varies; absolute uses raw gradient values."
        )
        form.addRow("Threshold mode:", self._canny_mode)

        self._canny_low = QDoubleSpinBox()
        self._canny_low.setRange(0.0, 100.0)
        self._canny_low.setValue(70.0)
        self._canny_high = QDoubleSpinBox()
        self._canny_high.setRange(0.0, 100.0)
        self._canny_high.setValue(90.0)
        form.addRow("Low threshold:", self._canny_low)
        form.addRow("High threshold:", self._canny_high)

        self._canny_roi = QCheckBox("Apply within active ROI only")
        self._canny_roi.setEnabled(self._active_roi_mask is not None)
        if self._active_roi_mask is None:
            self._canny_roi.setToolTip("Select an area ROI to enable.")
        form.addRow("", self._canny_roi)

        self._canny_preset.currentTextChanged.connect(self._apply_canny_preset)
        for sb in (self._canny_sigma, self._canny_low, self._canny_high):
            sb.valueChanged.connect(lambda _: self._on_canny_edited())
        self._canny_mode.currentIndexChanged.connect(lambda _: self._on_mode_changed())
        self._canny_roi.toggled.connect(lambda _: self._schedule())
        self._on_mode_changed()
        self._update_sigma_label()
        return w

    def _build_gradient_panel(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._grad_operator = QComboBox()
        self._grad_operator.addItems(["Sobel", "Scharr"])
        form.addRow("Operator:", self._grad_operator)

        self._grad_output = QComboBox()
        self._grad_output.addItems(["Gradient magnitude", "X gradient", "Y gradient",
                                     "Gradient orientation"])
        form.addRow("Output:", self._grad_output)

        self._grad_normalize = QCheckBox("Normalize output")
        self._grad_normalize.setChecked(True)
        form.addRow("", self._grad_normalize)

        self._grad_threshold_cb = QCheckBox("Threshold magnitude to mask")
        self._grad_threshold = QDoubleSpinBox()
        self._grad_threshold.setRange(0.0, 100.0)
        self._grad_threshold.setValue(90.0)
        self._grad_threshold.setSuffix(" %")
        self._grad_threshold.setEnabled(False)
        trow = QHBoxLayout()
        trow.setContentsMargins(0, 0, 0, 0)
        trow.addWidget(self._grad_threshold_cb)
        trow.addWidget(self._grad_threshold, 1)
        form.addRow("Threshold:", _wrap(trow))

        self._grad_roi = QCheckBox("Apply within active ROI only")
        self._grad_roi.setEnabled(self._active_roi_mask is not None)
        if self._active_roi_mask is None:
            self._grad_roi.setToolTip("Select an area ROI to enable.")
        form.addRow("", self._grad_roi)

        self._grad_threshold_cb.toggled.connect(self._grad_threshold.setEnabled)
        self._grad_operator.currentIndexChanged.connect(lambda _: self._schedule())
        self._grad_output.currentIndexChanged.connect(lambda _: self._schedule())
        self._grad_normalize.toggled.connect(lambda _: self._schedule())
        self._grad_threshold_cb.toggled.connect(lambda _: self._schedule())
        self._grad_threshold.valueChanged.connect(lambda _: self._schedule())
        self._grad_roi.toggled.connect(lambda _: self._schedule())

        # ── ROI-conversion options (shared by the Convert button) ─────────────
        roi_box = QGroupBox("ROI conversion")
        roi_form = QFormLayout(roi_box)
        self._roi_min_size = QSpinBox()
        self._roi_min_size.setRange(0, 100000)
        self._roi_min_size.setValue(20)
        self._roi_min_size.setSuffix(" px")
        roi_form.addRow("Min component size:", self._roi_min_size)
        self._roi_fill = QCheckBox("Fill enclosed regions first")
        self._roi_fill.setChecked(True)
        self._roi_fill.setToolTip(
            "Edge masks are thin boundaries; filling enclosed regions turns them "
            "into solid area ROIs. Disable to convert the raw boundary pixels."
        )
        roi_form.addRow("", self._roi_fill)
        self._roi_simplify = QCheckBox("Simplify polylines")
        roi_form.addRow("", self._roi_simplify)
        self._roi_one_per = QCheckBox("One ROI per component")
        self._roi_one_per.setChecked(True)
        roi_form.addRow("", self._roi_one_per)
        form.addRow(roi_box)
        return w

    # ── Preset / mode handling ────────────────────────────────────────────────

    def _apply_canny_preset(self, name: str) -> None:
        if name == _PRESET_CUSTOM or name not in CANNY_PRESETS:
            return
        p = CANNY_PRESETS[name]
        # Avoid re-entrancy resetting the combo to Custom while we fill the spins.
        with _blocked(self._canny_sigma, self._canny_low, self._canny_high):
            self._canny_mode.setCurrentText("Percentile")
            self._canny_sigma.setValue(p["sigma"])
            self._canny_low.setValue(p["low"])
            self._canny_high.setValue(p["high"])
        self._update_sigma_label()
        self._schedule()

    def _on_canny_edited(self) -> None:
        # Manual edits drop the preset back to Custom.
        if self._canny_preset.currentText() != _PRESET_CUSTOM:
            with _blocked(self._canny_preset):
                self._canny_preset.setCurrentText(_PRESET_CUSTOM)
        self._update_sigma_label()
        self._schedule()

    def _on_mode_changed(self) -> None:
        percentile = self._canny_mode.currentText() == "Percentile"
        suffix = " %" if percentile else ""
        for sb in (self._canny_low, self._canny_high):
            sb.setSuffix(suffix)
            sb.setRange(0.0, 100.0 if percentile else 1e9)
        self._schedule()

    def _update_sigma_label(self) -> None:
        if self._pixel_size_nm:
            nm = self._canny_sigma.value() * self._pixel_size_nm
            self._canny_sigma_lbl.setText(f"≈ {nm:.3g} nm")
        else:
            self._canny_sigma_lbl.setText("")

    # ── Recompute / preview ────────────────────────────────────────────────────

    def _schedule(self) -> None:
        self._debounce.start()

    def _recompute(self) -> None:
        try:
            self._result = self._compute()
        except Exception as exc:  # noqa: BLE001 — surface any backend error to the user
            self._result = None
            self._status_lbl.setText(f"Edge detection failed: {exc}")
            return
        self._render_preview(self._result)
        self._status_lbl.setText(self._summarize(self._result))

    def _compute(self) -> EdgeDetectionResult:
        if self._method_combo.currentIndex() == 0:
            roi = self._active_roi_mask if self._canny_roi.isChecked() else None
            return canny_edges(
                self._image,
                sigma=float(self._canny_sigma.value()),
                threshold_mode="percentile" if self._canny_mode.currentText() == "Percentile"
                else "absolute",
                low=float(self._canny_low.value()),
                high=float(self._canny_high.value()),
                roi_mask=roi,
                pixel_size_nm=self._pixel_size_nm,
                source_channel=self._source_channel,
            )
        roi = self._active_roi_mask if self._grad_roi.isChecked() else None
        output_map = {
            "Gradient magnitude": "magnitude",
            "X gradient": "x",
            "Y gradient": "y",
            "Gradient orientation": "orientation",
        }
        return gradient_filter(
            self._image,
            operator=self._grad_operator.currentText().lower(),
            output=output_map[self._grad_output.currentText()],
            normalize=self._grad_normalize.isChecked(),
            threshold_to_mask=self._grad_threshold_cb.isChecked(),
            threshold=float(self._grad_threshold.value()),
            roi_mask=roi,
            pixel_size_nm=self._pixel_size_nm,
            pixel_size_x_nm=self._pixel_size_x_nm,
            pixel_size_y_nm=self._pixel_size_y_nm,
            source_channel=self._source_channel,
        )

    def _render_preview(self, result: EdgeDetectionResult | None) -> None:
        if result is None:
            return
        base = array_to_uint8(self._image)  # grayscale uint8 (Ny, Nx)
        h, w = base.shape[:2]
        rgb = np.repeat(base[:, :, None], 3, axis=2).copy()
        if result.edge_mask is not None and result.edge_mask.any():
            r, g, b, a = _OVERLAY_RGBA
            mask = result.edge_mask
            alpha = a / 255.0
            for ch, val in zip(range(3), (r, g, b)):
                rgb[..., ch][mask] = (
                    (1 - alpha) * rgb[..., ch][mask] + alpha * val
                ).astype(np.uint8)
        elif result.display_image is not None:
            # No mask (e.g. gradient/orientation) — show the response itself.
            rgb = np.repeat(array_to_uint8(result.display_image)[:, :, None], 3, axis=2).copy()

        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self._preview_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._preview_lbl.setPixmap(pix)

    def _summarize(self, result: EdgeDetectionResult | None) -> str:
        if result is None:
            return ""
        if result.edge_mask is not None:
            return f"{result.method}: {int(result.edge_mask.sum())} edge pixels."
        return f"{result.method}: {result.parameters.get('output', 'response')} preview."

    # ── Output actions ──────────────────────────────────────────────────────────

    def _ensure_result(self) -> EdgeDetectionResult | None:
        if self._result is None:
            self._recompute()
        return self._result

    def _ensure_mask(self) -> np.ndarray | None:
        result = self._ensure_result()
        if result is None:
            return None
        if result.edge_mask is None or not result.edge_mask.any():
            QMessageBox.information(
                self, "Advanced Edge Detection",
                "No binary mask available. For Sobel/Scharr, enable "
                "“Threshold magnitude to mask”.",
            )
            return None
        return result.edge_mask

    def _emit_overlay(self) -> None:
        result = self._ensure_result()
        if result is not None:
            self.overlay_requested.emit(result)

    def _emit_image(self) -> None:
        result = self._ensure_result()
        if result is None or result.display_image is None:
            return
        provenance = {"op": "advanced_edge_detection", **result.parameters}
        self.image_created.emit(result.display_image, provenance)

    def _emit_mask(self) -> None:
        from probeflow.core.mask import ImageMask
        mask = self._ensure_mask()
        if mask is None:
            return
        result = self._result
        image_mask = ImageMask.new(mask, method=result.method, parameters=dict(result.parameters))
        self.mask_created.emit(image_mask)
        self._status_lbl.setText(f"Created mask “{image_mask.name}”.")

    def _emit_rois(self) -> None:
        from probeflow.core.roi import roi_from_mask
        mask = self._ensure_mask()
        if mask is None:
            return
        # ROI-conversion options only exist on the gradient panel; fall back to
        # sensible defaults when on the Canny panel.
        min_size = self._roi_min_size.value() if hasattr(self, "_roi_min_size") else 20
        simplify = self._roi_simplify.isChecked() if hasattr(self, "_roi_simplify") else False
        one_per = self._roi_one_per.isChecked() if hasattr(self, "_roi_one_per") else True
        fill = self._roi_fill.isChecked() if hasattr(self, "_roi_fill") else True
        if fill:
            # Edge masks are thin boundaries; fill enclosed regions so they
            # become solid area ROIs rather than skinny/broken outlines.
            from probeflow.processing.mask_ops import fill_holes
            mask = fill_holes(mask)
        rois = roi_from_mask(
            mask, min_size_px=int(min_size), simplify=bool(simplify),
            one_per_component=bool(one_per),
            name_prefix=self._result.method if self._result else "mask",
        )
        if not rois:
            QMessageBox.information(
                self, "Advanced Edge Detection",
                "No closed regions found to convert. Try a higher threshold or "
                "morphological cleanup (fill holes) first.",
            )
            return
        self.rois_created.emit(rois)
        self._status_lbl.setText(f"Created {len(rois)} ROI(s) from the mask.")

    def _on_close(self) -> None:
        self.overlay_cleared.emit()
        self.close()


# ── small helpers ──────────────────────────────────────────────────────────────

def _wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


class _blocked:
    """Context manager: block Qt signals on the given widgets for its body."""

    def __init__(self, *widgets):
        self._widgets = widgets

    def __enter__(self):
        self._prev = [w.blockSignals(True) for w in self._widgets]
        return self

    def __exit__(self, *exc):
        for w, prev in zip(self._widgets, self._prev):
            w.blockSignals(prev)
        return False
