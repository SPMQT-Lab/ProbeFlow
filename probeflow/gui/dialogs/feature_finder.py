"""Feature finder dialog — local maxima/minima with selective FFT export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from probeflow.analysis.feature_finder import (
    FeatureDetectionResult,
    feature_points_to_csv,
    feature_points_to_image,
    find_image_features,
)
from probeflow.processing.display import clip_range_from_array as _clip_range


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(1)
    return line


class FeatureFinderDialog(QDialog):
    """Find local maxima/minima, preview as point overlays, and export.

    Accepts the current display image and optional ROI mask. All numerical
    work is delegated to probeflow.analysis.feature_finder.
    """

    def __init__(
        self,
        arr: np.ndarray,
        *,
        pixel_size_x_m: float = 1e-10,
        pixel_size_y_m: float = 1e-10,
        roi_mask: np.ndarray | None = None,
        theme: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Feature finder")
        self.resize(1000, 680)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._arr = np.asarray(arr, dtype=np.float64)
        self._px_x_m = float(pixel_size_x_m)
        self._px_y_m = float(pixel_size_y_m)
        self._roi_mask = roi_mask
        self._t = theme or {}
        self._result: FeatureDetectionResult | None = None

        self._build()
        self._sync_threshold_controls()

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(6)

        # Left: matplotlib canvas
        self._fig = Figure(figsize=(6, 6), dpi=90)
        self._fig.patch.set_alpha(0)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._canvas = FigureCanvasQTAgg(self._fig)
        main.addWidget(self._canvas, 1)

        # Right: scrollable controls sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setMinimumWidth(280)
        sidebar_scroll.setMaximumWidth(320)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # ── Detection mode ──────────────────────────────────────────────────
        mode_lbl = QLabel("Detection mode")
        mode_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        lay.addWidget(mode_lbl)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._maxima_btn = QPushButton("Maxima")
        self._maxima_btn.setCheckable(True)
        self._maxima_btn.setChecked(True)
        self._minima_btn = QPushButton("Minima")
        self._minima_btn.setCheckable(True)
        for btn in (self._maxima_btn, self._minima_btn):
            btn.setFixedHeight(26)
            self._mode_group.addButton(btn)
            mode_row.addWidget(btn)
        lay.addLayout(mode_row)

        # ── Threshold mode ──────────────────────────────────────────────────
        thr_lbl = QLabel("Threshold mode")
        thr_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        lay.addWidget(thr_lbl)

        thr_row = QHBoxLayout()
        thr_row.setSpacing(4)
        self._thr_group = QButtonGroup(self)
        self._thr_group.setExclusive(True)
        self._above_btn = QPushButton("Above")
        self._above_btn.setCheckable(True)
        self._above_btn.setChecked(True)
        self._below_btn = QPushButton("Below")
        self._below_btn.setCheckable(True)
        self._between_btn = QPushButton("Between")
        self._between_btn.setCheckable(True)
        for btn in (self._above_btn, self._below_btn, self._between_btn):
            btn.setFixedHeight(26)
            self._thr_group.addButton(btn)
            thr_row.addWidget(btn)
        lay.addLayout(thr_row)
        self._above_btn.toggled.connect(self._sync_threshold_controls)
        self._below_btn.toggled.connect(self._sync_threshold_controls)
        self._between_btn.toggled.connect(self._sync_threshold_controls)

        arr_finite = self._arr[np.isfinite(self._arr)]
        v_min = float(arr_finite.min()) if arr_finite.size else 0.0
        v_max = float(arr_finite.max()) if arr_finite.size else 1.0
        v_range = v_max - v_min or 1.0
        step = v_range / 100.0
        decimals = max(3, -int(np.floor(np.log10(abs(step) + 1e-30))))

        # Low threshold
        self._low_row = QHBoxLayout()
        self._low_lbl = QLabel("Low threshold:")
        self._low_spin = QDoubleSpinBox()
        self._low_spin.setRange(-1e15, 1e15)
        self._low_spin.setDecimals(decimals)
        self._low_spin.setSingleStep(step)
        self._low_spin.setValue(v_min + 0.5 * v_range)
        self._low_row.addWidget(self._low_lbl)
        self._low_row.addWidget(self._low_spin)
        lay.addLayout(self._low_row)

        # High threshold
        self._high_row = QHBoxLayout()
        self._high_lbl = QLabel("High threshold:")
        self._high_spin = QDoubleSpinBox()
        self._high_spin.setRange(-1e15, 1e15)
        self._high_spin.setDecimals(decimals)
        self._high_spin.setSingleStep(step)
        self._high_spin.setValue(v_max)
        self._high_row.addWidget(self._high_lbl)
        self._high_row.addWidget(self._high_spin)
        lay.addLayout(self._high_row)

        # ── Detection settings ──────────────────────────────────────────────
        lay.addWidget(_sep())
        det_lbl = QLabel("Detection settings")
        det_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        lay.addWidget(det_lbl)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("Min distance:"))
        self._dist_spin = QDoubleSpinBox()
        self._dist_spin.setRange(1.0, 10000.0)
        self._dist_spin.setDecimals(1)
        self._dist_spin.setSingleStep(1.0)
        self._dist_spin.setValue(3.0)
        self._dist_spin.setSuffix(" px")
        dist_row.addWidget(self._dist_spin)
        lay.addLayout(dist_row)

        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Pre-smooth σ:"))
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.0, 100.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.25)
        self._smooth_spin.setValue(0.0)
        self._smooth_spin.setSuffix(" px")
        smooth_row.addWidget(self._smooth_spin)
        lay.addLayout(smooth_row)

        if self._roi_mask is not None:
            self._roi_cb = QCheckBox("Restrict to active ROI")
            self._roi_cb.setChecked(True)
            lay.addWidget(self._roi_cb)
        else:
            self._roi_cb = None

        preview_btn = QPushButton("Update preview")
        preview_btn.setFixedHeight(30)
        preview_btn.setFont(QFont("Helvetica", 10, QFont.Bold))
        preview_btn.clicked.connect(self._run_detection)
        lay.addWidget(preview_btn)

        self._count_lbl = QLabel("Detected features: —")
        self._count_lbl.setFont(QFont("Helvetica", 9))
        lay.addWidget(self._count_lbl)

        # ── Export coordinates ──────────────────────────────────────────────
        lay.addWidget(_sep())
        export_lbl = QLabel("Export")
        export_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        lay.addWidget(export_lbl)

        export_csv_btn = QPushButton("Export coordinates CSV…")
        export_csv_btn.setFixedHeight(26)
        export_csv_btn.clicked.connect(self._export_csv)
        lay.addWidget(export_csv_btn)

        # ── Feature image ───────────────────────────────────────────────────
        lay.addWidget(_sep())
        feat_lbl = QLabel("Feature image")
        feat_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        lay.addWidget(feat_lbl)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Disk radius:"))
        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.0, 200.0)
        self._radius_spin.setDecimals(1)
        self._radius_spin.setSingleStep(0.5)
        self._radius_spin.setValue(2.0)
        self._radius_spin.setSuffix(" px")
        radius_row.addWidget(self._radius_spin)
        lay.addLayout(radius_row)

        feat_smooth_row = QHBoxLayout()
        feat_smooth_row.addWidget(QLabel("Smoothing σ:"))
        self._feat_smooth_spin = QDoubleSpinBox()
        self._feat_smooth_spin.setRange(0.0, 100.0)
        self._feat_smooth_spin.setDecimals(2)
        self._feat_smooth_spin.setSingleStep(0.25)
        self._feat_smooth_spin.setValue(0.5)
        self._feat_smooth_spin.setSuffix(" px")
        feat_smooth_row.addWidget(self._feat_smooth_spin)
        lay.addLayout(feat_smooth_row)

        export_feat_btn = QPushButton("Export feature image PNG…")
        export_feat_btn.setFixedHeight(26)
        export_feat_btn.clicked.connect(self._export_feature_image)
        lay.addWidget(export_feat_btn)

        fft_btn = QPushButton("FFT feature image…")
        fft_btn.setFixedHeight(26)
        fft_btn.clicked.connect(self._open_fft)
        lay.addWidget(fft_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)

        lay.addStretch(1)

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(28)
        close_btn.clicked.connect(self.close)
        lay.addWidget(close_btn)

        sidebar_scroll.setWidget(inner)
        main.addWidget(sidebar_scroll)

        self._redraw()

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detection_mode(self) -> str:
        return "maxima" if self._maxima_btn.isChecked() else "minima"

    def _threshold_mode(self) -> str:
        if self._above_btn.isChecked():
            return "above"
        if self._below_btn.isChecked():
            return "below"
        return "between"

    def _sync_threshold_controls(self) -> None:
        thr = self._threshold_mode()
        show_low = thr in ("above", "between")
        show_high = thr in ("below", "between")
        self._low_lbl.setVisible(show_low)
        self._low_spin.setVisible(show_low)
        self._high_lbl.setVisible(show_high)
        self._high_spin.setVisible(show_high)

    def _run_detection(self) -> None:
        roi = (
            self._roi_mask
            if self._roi_cb is not None and self._roi_cb.isChecked()
            else None
        )
        thr = self._threshold_mode()
        tlo = self._low_spin.value() if thr in ("above", "between") else None
        thi = self._high_spin.value() if thr in ("below", "between") else None
        try:
            self._result = find_image_features(
                self._arr,
                mode=self._detection_mode(),
                threshold_mode=thr,
                threshold_low=tlo,
                threshold_high=thi,
                min_distance_px=self._dist_spin.value(),
                smoothing_sigma_px=self._smooth_spin.value(),
                roi_mask=roi,
            )
            n = len(self._result.points)
            self._count_lbl.setText(f"Detected features: {n}")
            self._status_lbl.setText(self._result.message)
        except Exception as exc:
            self._result = None
            self._count_lbl.setText("Detected features: —")
            self._status_lbl.setText(f"Error: {exc}")
        self._redraw()

    # ── Render ────────────────────────────────────────────────────────────────

    def _redraw(self) -> None:
        self._ax.clear()
        self._ax.set_axis_off()
        bg = self._t.get("bg", "#1e1e2e")
        self._fig.patch.set_facecolor(bg)
        self._ax.set_facecolor(bg)
        try:
            vmin, vmax = _clip_range(self._arr, 1.0, 99.0)
        except Exception:
            vmin, vmax = float(np.nanmin(self._arr)), float(np.nanmax(self._arr))
        self._ax.imshow(self._arr, cmap="gray", vmin=vmin, vmax=vmax,
                        interpolation="nearest", origin="upper")
        if self._result is not None and self._result.points:
            xs = [pt.x_px for pt in self._result.points]
            ys = [pt.y_px for pt in self._result.points]
            self._ax.scatter(xs, ys, marker="+", c="#f38ba8", s=60, linewidths=1.2,
                             zorder=5)
        self._canvas.draw_idle()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_csv(self) -> None:
        if self._result is None or not self._result.points:
            self._status_lbl.setText("Run detection first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export coordinates CSV",
            str(Path.home() / "probeflow_features.csv"),
            "CSV files (*.csv)",
        )
        if not path:
            return
        px_x_nm = self._px_x_m * 1e9
        px_y_nm = self._px_y_m * 1e9
        text = feature_points_to_csv(
            self._result.points,
            pixel_size_x_nm=px_x_nm,
            pixel_size_y_nm=px_y_nm,
        )
        Path(path).write_text(text, encoding="utf-8")
        self._status_lbl.setText(f"Saved {len(self._result.points)} points → {path}")

    def _build_feature_image(self) -> np.ndarray:
        if self._result is None or not self._result.points:
            raise ValueError("No detected features. Run detection first.")
        return feature_points_to_image(
            self._result.points,
            self._arr.shape,
            radius_px=self._radius_spin.value(),
            smoothing_sigma_px=self._feat_smooth_spin.value(),
        )

    def _export_feature_image(self) -> None:
        try:
            feat_img = self._build_feature_image()
        except ValueError as exc:
            self._status_lbl.setText(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export feature image PNG",
            str(Path.home() / "probeflow_feature_image.png"),
            "PNG files (*.png)",
        )
        if not path:
            return
        try:
            from PIL import Image as PILImage
            arr_u8 = (feat_img / max(float(feat_img.max()), 1e-12) * 255).astype(np.uint8)
            PILImage.fromarray(arr_u8).save(path)
        except ImportError:
            import matplotlib.pyplot as _plt
            _plt.imsave(path, feat_img, cmap="gray")
        self._status_lbl.setText(f"Feature image saved → {path}")

    def _open_fft(self) -> None:
        try:
            feat_img = self._build_feature_image()
        except ValueError as exc:
            self._status_lbl.setText(str(exc))
            return
        from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
        ny, nx = feat_img.shape
        scan_range_m = (nx * self._px_x_m, ny * self._px_y_m)
        dlg = FFTViewerDialog(
            feat_img,
            scan_range_m,
            colormap="gray",
            theme=self._t,
            parent=self,
        )
        dlg.setWindowTitle("FFT — feature image")
        dlg.show()
