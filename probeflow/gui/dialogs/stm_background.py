"""STM scan-line background subtraction dialog."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QImage, QPixmap, QStandardItem, QStandardItemModel
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
from probeflow.processing.background import compute_scanline_profile
from probeflow.processing.display import array_to_uint8


_MODEL_LABELS = {
    "Linear": "linear",
    "2nd order polynomial": "poly2",
    "3rd order polynomial": "poly3",
    "Low-pass": "low_pass",
    "Line by line": "line_by_line",
    "Piezo creep": "piezo_creep",
    "Piezo creep + y²": "piezo_creep_x2",
    "Piezo creep + y³": "piezo_creep_x3",
    "Sqrt creep": "sqrt_creep",
}

_MODEL_TOOLTIPS = {
    "Linear":
        "B(y) = a + b·y\n"
        "Least-squares line fitted to the row profile. Removes a constant "
        "tilt along the slow-scan direction.",
    "2nd order polynomial":
        "B(y) = a + b·y + c·y²\n"
        "Least-squares quadratic. Removes tilt and gentle bowl-shaped "
        "background curvature.",
    "3rd order polynomial":
        "B(y) = a + b·y + c·y² + d·y³\n"
        "Least-squares cubic. Handles more complex slow-scan drift.",
    "Low-pass":
        "B(y) = Gaussian-smoothed row profile (width = blur length).\n"
        "Non-parametric; captures any slowly-varying background without "
        "assuming a functional form.",
    "Line by line":
        "B(y) = raw per-row statistic (median or mean).\n"
        "Each scan line is zeroed independently — strongest correction, "
        "but removes genuine large-scale topography.",
    "Piezo creep":
        "B(y) = a + b·y + c·log(|y − d|)\n"
        "Logarithmic creep model. d is the fitted singularity anchor "
        "(typically before scan start). Best for images showing a rapid "
        "height drift that decays logarithmically from the first line.",
    "Piezo creep + y²":
        "B(y) = a + b·y + c·log(|y − d|) + e·y²\n"
        "Logarithmic creep plus a quadratic term. Handles residual "
        "parabolic background on top of the creep drift.",
    "Piezo creep + y³":
        "B(y) = a + b·y + c·log(|y − d|) + e·y³\n"
        "Logarithmic creep plus a cubic term. Use when the residual "
        "background is asymmetric across the scan.",
    "Sqrt creep":
        "B(y) = a + b·y + c·√|y − d|\n"
        "Square-root creep variant. Useful when the drift grows more "
        "slowly than a logarithm (intermediate between linear and log).",
}


def _auto_unit(values_m: np.ndarray) -> tuple[float, str]:
    """Pick a human-readable unit for height values assumed to be in metres."""
    finite = values_m[np.isfinite(values_m)]
    if finite.size == 0:
        return 1e12, "pm"
    nonzero = finite[finite != 0.0]
    if nonzero.size == 0:
        return 1e12, "pm"
    peak = float(np.max(np.abs(nonzero)))
    if peak < 5e-10:
        return 1e12, "pm"
    elif peak < 5e-6:
        return 1e9, "nm"
    elif peak < 5e-3:
        return 1e6, "μm"
    else:
        return 1e3, "mm"


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
        prior_row_alignment: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("STM Background")
        self.resize(900, 700)
        self.setModal(False)

        self._image = np.asarray(image, dtype=np.float64).copy()
        self._theme = theme or {}
        self._active_roi_mask = (
            None if active_roi_mask is None else np.asarray(active_roi_mask, dtype=bool)
        )
        self._active_roi_id = active_roi_id
        self._active_roi_name = active_roi_name or active_roi_id or "active ROI"
        self._prior_row_alignment = (
            prior_row_alignment.lower().strip() if prior_row_alignment else None
        )
        self._last_result = None
        self._last_mode = "corrected"

        # Determine display unit from image profile so the jump threshold
        # spin box uses the same height units as the plot axes.
        _init_prof = compute_scanline_profile(self._image, mask=None, statistic="median")
        _fp = _init_prof[np.isfinite(_init_prof)]
        self._unit_scale, self._unit_label = _auto_unit(_fp) if _fp.size > 0 else (1e9, "nm")

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
        _model_item_model = QStandardItemModel(self._model_combo)
        for label in _MODEL_LABELS:
            item = QStandardItem(label)
            item.setToolTip(_MODEL_TOOLTIPS.get(label, ""))
            _model_item_model.appendRow(item)
        self._model_combo.setModel(_model_item_model)
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
        self._jump_spin.setRange(1e-6, 1e6)   # in display units (nm / pm / μm)
        self._jump_spin.setDecimals(4)
        _default_display = round(1e-10 * self._unit_scale, 6)  # 100 pm in display units
        self._jump_spin.setSingleStep(max(1e-4, _default_display / 10))
        self._jump_spin.setValue(_default_display)
        self._jump_spin.setSuffix(f" {self._unit_label}")
        self._jump_spin.setEnabled(False)
        self._jump_spin.setToolTip(
            "Detect abrupt row-to-row changes in the median/mean height profile "
            "— tip changes between scan lines, not lateral steps within a row "
            "(a terrace edge crossing the image is invisible to the row profile; "
            "use the step-tolerant simple background or facet level for those). "
            f"Threshold in {self._unit_label} — the same unit shown on the plot. "
            "Detected jumps are removed before fitting the smooth background, "
            "then added back to the fitted background before subtraction."
        )
        self._jump_cb.toggled.connect(self._jump_spin.setEnabled)
        jump_row.addWidget(self._jump_cb)
        jump_row.addWidget(self._jump_spin, 1)
        controls.addRow("Handle profile jumps above:", jump_row)

        root.addLayout(controls)

        self._warning_lbl = QLabel("")
        self._warning_lbl.setWordWrap(True)
        self._warning_lbl.setStyleSheet(
            "color: #b8860b; background: #fffbe6; border: 1px solid #e6d26e; "
            "border-radius: 3px; padding: 4px 6px;"
        )
        self._warning_lbl.hide()
        root.addWidget(self._warning_lbl)

        self._status_lbl = QLabel("Preview not run.")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        # Action buttons sit directly under the controls (above the preview
        # panes) so Preview/Apply stay reachable without scrolling past the
        # large image + plot area.
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

        body = QHBoxLayout()
        self._preview_lbl = QLabel("Preview")
        self._preview_lbl.setAlignment(Qt.AlignCenter)
        self._preview_lbl.setMinimumSize(260, 260)
        self._preview_lbl.setStyleSheet(
            f"background: {self._theme.get('sidebar_bg', '#181825')};"
        )
        body.addWidget(self._preview_lbl, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(4)

        self._fig = Figure(figsize=(4.4, 4.8), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setToolTip(
            "Top: each dot is the per-row median (or mean) height — one point per scan line.\n"
            "The orange line is the background model fitted to those row statistics.\n"
            "Bottom: residuals = row statistics − fitted background.\n"
            "Applying subtracts that 1-D curve from every column of the 2-D image."
        )
        right_col.addWidget(self._canvas, 1)

        self._stats_lbl = QLabel("")
        stats_font = QFont()
        stats_font.setFamily("monospace")
        stats_font.setStyleHint(QFont.StyleHint.Monospace)
        stats_font.setPointSize(8)
        self._stats_lbl.setFont(stats_font)
        self._stats_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._stats_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        right_col.addWidget(self._stats_lbl)

        body.addLayout(right_col, 1)
        root.addLayout(body, 1)

        self._background_btn.clicked.connect(lambda: self._preview("background"))
        self._corrected_btn.clicked.connect(lambda: self._preview("corrected"))
        self._apply_btn.clicked.connect(self._apply)
        close_btn.clicked.connect(self.close)
        self._model_combo.currentTextChanged.connect(lambda _: self._on_model_changed())
        self._fit_region_combo.currentIndexChanged.connect(lambda _: self._invalidate_preview())
        self._stat_combo.currentIndexChanged.connect(lambda _: self._invalidate_preview())
        self._stat_combo.currentIndexChanged.connect(lambda _: self._update_alignment_warning())
        self._linear_x_cb.toggled.connect(lambda _: self._invalidate_preview())
        self._blur_spin.valueChanged.connect(lambda _: self._invalidate_preview())
        self._jump_cb.toggled.connect(lambda _: self._invalidate_preview())
        self._jump_spin.valueChanged.connect(lambda _: self._invalidate_preview())
        self._sync_controls()
        self._update_alignment_warning()

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
            float(self._jump_spin.value()) / self._unit_scale
            if self._jump_cb.isChecked() else None
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

        # Compute grey reference profile for excluded rows (ROI mode only)
        grey_profile = None
        mask = self._fit_mask()
        if mask is not None:
            excluded = ~np.isfinite(result.line_profile)
            if excluded.any():
                full_prof = compute_scanline_profile(
                    self._image, mask=None, statistic=result.params.line_statistic
                )
                grey_profile = np.where(excluded, full_prof, np.nan)

        self._plot_profile(
            result.line_profile, result.fitted_profile,
            grey_profile=grey_profile,
            jump_positions=result.jump_positions,
        )
        self._update_stats(result)

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

    def _plot_profile(
        self,
        profile: np.ndarray,
        fitted: np.ndarray,
        *,
        grey_profile: np.ndarray | None = None,
        jump_positions: tuple[int, ...] = (),
    ) -> None:
        self._fig.clear()
        gs = GridSpec(2, 1, figure=self._fig, height_ratios=[3, 2], hspace=0.5)
        ax_top = self._fig.add_subplot(gs[0])
        ax_bot = self._fig.add_subplot(gs[1])

        y = np.arange(profile.size)
        finite = np.isfinite(profile)

        # Choose display units from the combined range of profile and fitted values
        vals_for_unit = profile[finite] if finite.any() else fitted
        scale, unit = _auto_unit(vals_for_unit)

        # ── Top panel: row statistics + fitted background ──────────────────
        if grey_profile is not None:
            grey_finite = np.isfinite(grey_profile)
            if grey_finite.any():
                ax_top.scatter(
                    y[grey_finite], grey_profile[grey_finite] * scale,
                    s=2, color="gray", alpha=0.5, label="excluded rows", zorder=2,
                )
        if finite.any():
            ax_top.scatter(
                y[finite], profile[finite] * scale,
                s=2, color="tab:blue", label="line statistic", zorder=3,
            )
        ax_top.plot(
            y, fitted * scale, "-", color="tab:orange", linewidth=1.2,
            label="fitted background", zorder=4,
        )
        for pos in jump_positions:
            ax_top.axvline(pos, color="tab:red", linewidth=0.8, linestyle="--", alpha=0.7)
        ax_top.set_ylabel(f"height ({unit})", fontsize=8)
        ax_top.legend(loc="best", fontsize=7, markerscale=2.5, handlelength=1.5)
        ax_top.grid(True, alpha=0.25)
        ax_top.tick_params(labelsize=7)

        # ── Bottom panel: residuals ────────────────────────────────────────
        residual = profile - fitted
        res_finite = np.isfinite(residual)
        res_scale, res_unit = _auto_unit(residual[res_finite]) if res_finite.any() else (1e12, "pm")

        if res_finite.any():
            ax_bot.scatter(
                y[res_finite], residual[res_finite] * res_scale,
                s=2, color="tab:purple", zorder=3,
            )

            # Flag near-numerical-precision residuals
            rms = float(np.sqrt(np.mean(residual[res_finite] ** 2)))
            prof_range = float(np.ptp(profile[finite])) if finite.any() else 0.0
            if prof_range > 0 and rms / prof_range < 1e-8:
                ax_bot.text(
                    0.5, 0.5,
                    "Residuals within numerical precision",
                    transform=ax_bot.transAxes,
                    ha="center", va="center", fontsize=7, color="gray", style="italic",
                )

        ax_bot.axhline(0, color="gray", linewidth=0.8, alpha=0.5, linestyle="--")
        ax_bot.set_xlabel("scan line", fontsize=8)
        ax_bot.set_ylabel(f"residual ({res_unit})", fontsize=8)
        ax_bot.grid(True, alpha=0.25)
        ax_bot.tick_params(labelsize=7)

        # GridSpec already sets the spacing; tight_layout only tidies margins and
        # warns about the height-ratio axes — silence that cosmetic warning.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._fig.tight_layout()
        self._canvas.draw_idle()

    def _update_stats(self, result) -> None:
        profile = result.line_profile
        fitted = result.fitted_profile
        residual = profile - fitted
        finite = np.isfinite(profile)
        rows_total = profile.size
        rows_used = int(finite.sum())

        prof_range = float(np.ptp(profile[finite])) if finite.any() else 0.0
        bg_range = float(np.ptp(fitted))

        res_finite = np.isfinite(residual)
        if res_finite.any():
            res_rms = float(np.sqrt(np.mean(residual[res_finite] ** 2)))
            res_max = float(np.max(np.abs(residual[res_finite])))
        else:
            res_rms = res_max = 0.0

        all_vals = profile[finite] if finite.any() else np.array([0.0])
        scale, unit = _auto_unit(all_vals)
        res_arr = residual[res_finite] if res_finite.any() else np.array([0.0])
        res_scale, res_unit = _auto_unit(res_arr)

        stat = result.params.line_statistic
        model = result.params.model.replace("_", " ")
        fit_region = self.processing_params()["fit_region"].replace("_", " ")

        lines = [
            f"Statistic:        {stat}",
            f"Model:            {model}",
            f"Fit region:       {fit_region}",
            f"Rows used:        {rows_used} / {rows_total}",
            f"Row-stat range:   {prof_range * scale:.3g} {unit}",
            f"Background range: {bg_range * scale:.3g} {unit}",
            f"Residual RMS:     {res_rms * res_scale:.3g} {res_unit}",
            f"Max |residual|:   {res_max * res_scale:.3g} {res_unit}",
        ]

        # Largest adjacent row-to-row difference (regardless of threshold)
        p_arr = result.line_profile.copy()
        p_arr[~np.isfinite(p_arr)] = np.nan
        consec_diffs = np.abs(np.diff(p_arr))
        fin_diffs = consec_diffs[np.isfinite(consec_diffs)]
        if fin_diffs.size > 0:
            largest_adj = float(np.max(fin_diffs))
            adj_scale, adj_unit = _auto_unit(fin_diffs)
        else:
            largest_adj, adj_scale, adj_unit = 0.0, scale, unit

        jump_threshold = result.params.jump_threshold
        if jump_threshold is not None:
            thr_scale, thr_unit = _auto_unit(np.array([jump_threshold]))
            lines += [
                "Jump handling:    on",
                f"Jump threshold:   {jump_threshold * thr_scale:.3g} {thr_unit}",
                f"Largest adj jump: {largest_adj * adj_scale:.3g} {adj_unit}",
                f"Detected jumps:   {len(result.jump_positions)}",
            ]
            if result.jump_sizes:
                largest_det = float(np.max(np.abs(result.jump_sizes)))
                det_scale, det_unit = _auto_unit(np.abs(np.array(list(result.jump_sizes))))
                lines.append(f"Largest detected: {largest_det * det_scale:.3g} {det_unit}")
        else:
            lines.append("Jump handling:    off")

        self._stats_lbl.setText("\n".join(lines))

    def _update_alignment_warning(self) -> None:
        if self._prior_row_alignment is None:
            self._warning_lbl.hide()
            return
        current_stat = self._stat_combo.currentText().lower()
        if self._prior_row_alignment == current_stat:
            self._warning_lbl.setText(
                f"Previous operation: row alignment using {self._prior_row_alignment}. "
                f"The current row-{self._prior_row_alignment} background is expected to be near zero."
            )
            self._warning_lbl.show()
        else:
            self._warning_lbl.hide()

    def _on_model_changed(self) -> None:
        # Switching the background model is a bigger change than tweaking one of
        # its parameters, so refresh an existing preview automatically rather
        # than just marking it stale.  (In-model tweaks still use the manual
        # Preview buttons via ``_invalidate_preview``.)
        self._sync_controls()
        if self._last_result is not None:
            self._preview(self._last_mode)

    def _invalidate_preview(self) -> None:
        if self._last_result is not None:
            self._last_result = None
            self._status_lbl.setText("Parameters changed — run a preview before applying.")
            self._stats_lbl.setText("")

    def _sync_controls(self) -> None:
        label = self._model_combo.currentText()
        is_low_pass = _MODEL_LABELS[label] == "low_pass"
        self._blur_spin.setEnabled(is_low_pass)
        self._model_combo.setToolTip(_MODEL_TOOLTIPS.get(label, ""))
