"""Symmetrize tab for the FFT viewer — n-fold rotational averaging.

Mixin split out of ``fft_viewer.py`` (same pattern as the Inverse FFT tab).
It relies on attributes owned by ``FFTViewerDialog`` (``self._arr``,
``self._scan_range_m``, ``self._fft_source`` / ``self._roi_id``, the Bragg
reference controls ``self._bragg_sym_combo`` / ``self._bragg_a_spin`` /
``self._bragg_unit_combo``, the ``self._apply_correction_fn`` /
``self._new_image_fn`` callbacks, ``self._fft_pixel_sizes_m`` and the shared
FFT-preview helpers ``_show_fft_preview`` / ``_hide_fft_preview``).

Symmetrization fabricates data — it paints the assumed symmetry over defects
and domain boundaries — so the residual (original − result) is presented as a
first-class preview, mirroring the Inverse FFT tab's honesty-first design.
"""

from __future__ import annotations

import math

import numpy as np
from probeflow.gui._tooltips import tip as _tip
from probeflow.gui.typography import ui_font
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
    QWidget,
)


class FFTViewerSymmetrizeMixin:
    """Enforce an n-fold symmetry by rotate-and-average; preview and apply."""

    def _build_symmetrize_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        intro = QLabel(
            "Enforce an n-fold symmetry by averaging the image with its rotated "
            "copies (equivalent to rotating FFT segments onto each other). "
            "Always inspect the residual: it is everything symmetrization "
            "removed — noise, but also real defects and domain boundaries.")
        intro.setWordWrap(True)
        intro.setFont(ui_font(9))
        intro.setToolTip(_tip(
            "Averaging the image over its symmetry rotations keeps everything "
            "n-fold symmetric (the lattice) and suppresses everything that is "
            "not (noise, defects) by ~1/n. Each rotated copy is registered "
            "back onto the original first, so the symmetry axis does not have "
            "to sit at the image centre. The residual = original − result "
            "shows exactly what was removed — if a defect matters, it is in "
            "there."))
        lay.addWidget(intro)

        # ── symmetry ────────────────────────────────────────────────────────
        sgrp = QGroupBox("Symmetry")
        sg = QVBoxLayout(sgrp)
        sg.setSpacing(4)
        fold_row = QHBoxLayout()
        fold_row.addWidget(QLabel("Fold order:"))
        self._symm_fold_spin = QSpinBox()
        self._symm_fold_spin.setRange(2, 12)
        self._symm_fold_spin.setValue(6)
        self._symm_fold_spin.setToolTip(_tip(
            "Rotation order n: copies rotated by 360°/n are averaged. Use 6 "
            "for hexagonal lattices, 4 for square, 3 for threefold surfaces, "
            "2 for twofold."))
        self._symm_fold_spin.setMaximumWidth(72)
        self._symm_fold_spin.valueChanged.connect(
            lambda _v: self._update_symmetrize_status())
        fold_row.addWidget(self._symm_fold_spin)
        fold_row.addStretch(1)
        sg.addLayout(fold_row)

        self._symm_mirror_cb = QCheckBox("Also mirror (dihedral symmetry)")
        self._symm_mirror_cb.setToolTip(_tip(
            "Additionally average the mirrored copies (point group D_n instead "
            "of C_n). Unlike pure rotations, mirrors need the axis angle below "
            "to be aligned with the lattice."))
        self._symm_mirror_cb.toggled.connect(self._on_symm_mirror_toggled)
        sg.addWidget(self._symm_mirror_cb)

        axis_row = QHBoxLayout()
        self._symm_axis_lbl = QLabel("Mirror axis:")
        axis_row.addWidget(self._symm_axis_lbl)
        self._symm_axis_spin = QDoubleSpinBox()
        self._symm_axis_spin.setRange(0.0, 180.0)
        self._symm_axis_spin.setDecimals(1)
        self._symm_axis_spin.setSingleStep(0.5)
        self._symm_axis_spin.setSuffix(" °")
        self._symm_axis_spin.setToolTip(_tip(
            "Angle of the mirror line through the image centre. 0° is "
            "horizontal (+x). Only used when mirroring is on — pure rotations "
            "need no axis."))
        self._symm_axis_spin.setMaximumWidth(90)
        axis_row.addWidget(self._symm_axis_spin)
        self._symm_align_btn = QPushButton("Align to Bragg peaks")
        self._symm_align_btn.setToolTip(_tip(
            "Detect the Bragg peaks (using the reference lattice constant and "
            "symmetry from the Inspect tab) and set the mirror axis to the "
            "measured lattice direction."))
        self._symm_align_btn.clicked.connect(self._on_symm_align_axis)
        axis_row.addWidget(self._symm_align_btn)
        axis_row.addStretch(1)
        sg.addLayout(axis_row)

        # ── options ─────────────────────────────────────────────────────────
        ogrp = QGroupBox("Options")
        og = QVBoxLayout(ogrp)
        og.setSpacing(4)
        self._symm_register_cb = QCheckBox("Auto-centre (register rotated copies)")
        self._symm_register_cb.setChecked(True)
        self._symm_register_cb.setToolTip(_tip(
            "Shift each rotated copy back onto the original by cross-"
            "correlation before averaging. Keep this on: without it the "
            "symmetry axis must pass exactly through the image centre, or the "
            "average destroys the lattice contrast instead of cleaning it."))
        og.addWidget(self._symm_register_cb)
        interp_row = QHBoxLayout()
        interp_row.addWidget(QLabel("Interpolation:"))
        self._symm_interp_combo = QComboBox()
        self._symm_interp_combo.addItems(["Linear", "Cubic"])
        self._symm_interp_combo.setToolTip(_tip(
            "Resampling used for the rotations. Linear is the default; Cubic "
            "keeps sharp lattices slightly crisper at the cost of mild "
            "ringing."))
        self._symm_interp_combo.setMaximumWidth(100)
        interp_row.addWidget(self._symm_interp_combo)
        interp_row.addStretch(1)
        og.addLayout(interp_row)
        self._symm_strict_cb = QCheckBox("Blank pixels not covered by every copy")
        self._symm_strict_cb.setToolTip(_tip(
            "Rotated copies leave the frame near the corners, so those pixels "
            "average fewer copies. Off (default): renormalize by the available "
            "copies. On: set them to NaN so the output only contains fully "
            "symmetrized pixels."))
        og.addWidget(self._symm_strict_cb)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        top_row.addWidget(sgrp, 1)
        top_row.addWidget(ogrp, 1)
        lay.addLayout(top_row)

        # ── preview / apply ─────────────────────────────────────────────────
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Preview:"))
        self._symm_view_combo = QComboBox()
        self._symm_view_combo.addItems(["Result", "Residual", "Original"])
        self._symm_view_combo.setToolTip(_tip(
            "Which image the left preview shows: the symmetrized Result, the "
            "Residual (everything symmetrization removed), or the Original."))
        self._symm_view_combo.currentIndexChanged.connect(
            lambda _=0: self._on_symmetrize_preview()
            if self._symmetrize_preview_active else None)
        self._symm_view_combo.setMaximumWidth(140)
        view_row.addWidget(self._symm_view_combo)
        view_row.addStretch(1)
        lay.addLayout(view_row)

        pv_row = QHBoxLayout()
        self._symm_preview_btn = QPushButton("Preview symmetrized image")
        self._symm_preview_btn.setToolTip(_tip(
            "Symmetrize with the current settings and show the outcome in the "
            "left preview, without changing your data. Check the residual "
            "before applying."))
        self._symm_preview_btn.clicked.connect(self._on_symmetrize_preview)
        self._symm_clear_btn = QPushButton("Clear preview")
        self._symm_clear_btn.setEnabled(False)
        self._symm_clear_btn.clicked.connect(self._on_symmetrize_clear)
        pv_row.addWidget(self._symm_preview_btn)
        pv_row.addWidget(self._symm_clear_btn)
        lay.addLayout(pv_row)

        ap_row = QHBoxLayout()
        self._symm_apply_btn = QPushButton("Apply to image")
        self._symm_apply_btn.setObjectName("accentBtn")
        self._symm_apply_btn.setToolTip(_tip(
            "Apply the symmetrization. On a whole-image FFT this is recorded "
            "as a reproducible step in the processing history. On an ROI it "
            "creates a new corrected cropped image (never edits the original "
            "in place)."))
        self._symm_apply_btn.clicked.connect(self._on_symmetrize_apply)
        ap_row.addWidget(self._symm_apply_btn)
        lay.addLayout(ap_row)

        ex_row = QHBoxLayout()
        exp_res = QPushButton("Export result")
        exp_res.setToolTip(_tip("Save the symmetrized image to a file."))
        exp_res.clicked.connect(lambda: self._on_symmetrize_export("result"))
        exp_resid = QPushButton("Export residual")
        exp_resid.setToolTip(_tip("Save the residual (removed component) to a file."))
        exp_resid.clicked.connect(lambda: self._on_symmetrize_export("residual"))
        ex_row.addWidget(exp_res)
        ex_row.addWidget(exp_resid)
        lay.addLayout(ex_row)

        self._symm_status_lbl = QLabel("")
        self._symm_status_lbl.setWordWrap(True)
        self._symm_status_lbl.setFont(ui_font(8))
        lay.addWidget(self._symm_status_lbl)

        lay.addStretch(1)
        scroll.setWidget(page)
        self._on_symm_mirror_toggled(False)
        self._update_symmetrize_status()
        return scroll

    # ── control state ────────────────────────────────────────────────────────

    def _on_symm_mirror_toggled(self, checked: bool) -> None:
        for w in (self._symm_axis_lbl, self._symm_axis_spin, self._symm_align_btn):
            w.setEnabled(bool(checked))
        self._update_symmetrize_status()

    def _symmetrize_kwargs(self) -> dict:
        return {
            "mirror": bool(self._symm_mirror_cb.isChecked()),
            "mirror_axis_deg": float(self._symm_axis_spin.value()),
            "register": bool(self._symm_register_cb.isChecked()),
            "interpolation": ("linear"
                              if self._symm_interp_combo.currentIndex() == 0
                              else "cubic"),
            "strict_coverage": bool(self._symm_strict_cb.isChecked()),
        }

    def _anisotropy_warning(self) -> str | None:
        px = self._fft_pixel_sizes_m()
        if px is None:
            return None
        px_x, px_y = px
        if px_x <= 0 or px_y <= 0:
            return None
        ratio = max(px_x, px_y) / min(px_x, px_y)
        if ratio <= 1.01:
            return None
        return (f"⚠ Pixels are anisotropic (x/y ratio {ratio:.2f}): rotating "
                "in pixel space is not a physical rotation, so the enforced "
                "symmetry will be distorted. Resample to square pixels first.")

    # ── align to Bragg ───────────────────────────────────────────────────────

    def _on_symm_align_axis(self) -> None:
        """Set the mirror axis from detected Bragg peaks.

        Uses the reference lattice (constant + symmetry) from the Inspect tab
        to predict the first-order Bragg radius, finds the peaks in that
        annulus, and takes their common n-fold axis.
        """
        arr = np.asarray(self._arr, dtype=np.float64)
        from probeflow.processing.bragg import (
            find_bragg_peaks_in_annulus, predicted_bragg_radius)
        from probeflow.processing.symmetrize import fold_axis_from_peaks

        symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hex"
        a_val = float(self._bragg_a_spin.value())
        unit = self._bragg_unit_combo.currentText()
        a_m = a_val * 1e-10 if unit == "Å" else a_val * 1e-9
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
            scan_m = math.sqrt(w_m * h_m)   # geometric mean for non-square scans
            r_px = predicted_bragg_radius(
                a_m, symmetry, scan_m, min(arr.shape[:2]))
        except (ValueError, TypeError) as exc:
            self._symm_status_lbl.setText(f"Cannot predict Bragg radius: {exc}")
            return

        n = int(self._symm_fold_spin.value())
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, float(arr[finite].mean()) if finite.any() else 0.0)
        mag = np.abs(np.fft.fftshift(np.fft.fft2(filled - filled.mean())))
        peaks = find_bragg_peaks_in_annulus(mag, r_px, expected_count=n)

        # The detector returns local maxima even when the annulus holds only
        # the noise floor (e.g. a wrong lattice constant), so gate the picks on
        # how far they stand above the annulus statistics before trusting them
        # (real Bragg peaks measure z ≈ 7–10, leakage maxima z ≈ 2–4).
        if peaks.shape[0]:
            Ny, Nx = mag.shape
            cy, cx = Ny / 2.0, Nx / 2.0
            gy, gx = np.ogrid[:Ny, :Nx]
            annulus = mag[(np.hypot(gx - cx, gy - cy) >= r_px * 0.8)
                          & (np.hypot(gx - cx, gy - cy) <= r_px * 1.2)]
            mean_a = float(annulus.mean())
            std_a = float(annulus.std()) + 1e-12
            strong = [
                (dx, dy) for dx, dy in peaks
                if (mag[int(round(cy + dy)), int(round(cx + dx))] - mean_a)
                / std_a >= 5.0]
            peaks = np.array(strong, dtype=np.float64).reshape(-1, 2)
        if peaks.shape[0] < 2:
            self._symm_status_lbl.setText(
                "Could not detect clear Bragg peaks near the predicted "
                "radius — check the reference lattice constant and symmetry "
                "on the Inspect tab, then try again.")
            return
        try:
            axis = fold_axis_from_peaks(peaks, n)
        except ValueError:
            self._symm_status_lbl.setText(
                f"The {peaks.shape[0]} detected peaks do not share a "
                f"{n}-fold axis — is the fold order right for this lattice?")
            return
        self._symm_axis_spin.setValue(axis)
        self._symm_status_lbl.setText(
            f"Mirror axis set to {axis:.1f}° from {peaks.shape[0]} Bragg "
            f"peak{'s' if peaks.shape[0] != 1 else ''} "
            f"(axis repeats every {360.0 / n:.0f}°).")

    # ── preview / apply / export ─────────────────────────────────────────────

    def _compute_symmetrization(self, array):
        from probeflow.processing.symmetrize import symmetrize_image
        return symmetrize_image(
            np.asarray(array, dtype=np.float64),
            int(self._symm_fold_spin.value()),
            **self._symmetrize_kwargs())

    def _on_symmetrize_preview(self) -> None:
        arr = np.asarray(self._arr, dtype=np.float64)
        try:
            res = self._compute_symmetrization(arr)
        except ValueError as exc:
            self._symm_status_lbl.setText(f"Symmetrize failed: {exc}")
            return
        view = self._symm_view_combo.currentText()
        disp = {"Result": res.result, "Residual": res.residual, "Original": arr}[view]
        self._show_fft_preview(disp)
        self._symmetrize_preview_active = True
        self._symm_clear_btn.setEnabled(True)
        self._update_symmetrize_status(res)

    def _on_symmetrize_clear(self) -> None:
        self._hide_fft_preview()
        self._symmetrize_preview_active = False
        self._symm_clear_btn.setEnabled(False)
        self._update_symmetrize_status()

    def _symmetrize_op_params(self) -> dict:
        params = {"n_fold": int(self._symm_fold_spin.value()),
                  **self._symmetrize_kwargs(),
                  "fft_source": self._fft_source}
        if self._fft_source == "active_roi" and self._roi_id is not None:
            params["fft_roi_id"] = self._roi_id
        return params

    def _on_symmetrize_apply(self) -> None:
        params = self._symmetrize_op_params()
        if self._fft_source == "active_roi":
            self._apply_symmetrize_to_new_image(params)
            return
        if self._apply_correction_fn is None:
            self._symm_status_lbl.setText("Apply is unavailable in this context.")
            return
        if self._symmetrize_preview_active:
            self._hide_fft_preview()
            self._symmetrize_preview_active = False
            self._symm_clear_btn.setEnabled(False)
        self._apply_correction_fn("symmetrize_fft", params)
        self._symm_status_lbl.setText(
            f"Applied {params['n_fold']}-fold symmetrization.")

    def _apply_symmetrize_to_new_image(self, params: dict) -> None:
        arr = np.asarray(self._arr, dtype=np.float64)
        try:
            res = self._compute_symmetrization(arr)
        except ValueError as exc:
            self._symm_status_lbl.setText(f"Symmetrize failed: {exc}")
            return
        fn = getattr(self, "_new_image_fn", None)
        if fn is None:
            self._symm_status_lbl.setText(
                "ROI apply needs a host that can open a new image — export the "
                "result instead.")
            return
        fn(res.result, tuple(self._scan_range_m),
           {"op": "symmetrize_fft", "params": params})
        self._symm_status_lbl.setText("Created a new symmetrized image from the ROI.")

    def _on_symmetrize_export(self, which: str) -> None:
        arr = np.asarray(self._arr, dtype=np.float64)
        try:
            res = self._compute_symmetrization(arr)
        except ValueError as exc:
            self._symm_status_lbl.setText(f"Symmetrize failed: {exc}")
            return
        data = res.result if which == "result" else res.residual
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {which}", f"symmetrized_{which}.csv",
            "CSV (*.csv);;NumPy (*.npy)")
        if not path:
            return
        try:
            if path.lower().endswith(".npy"):
                np.save(path, data)
            else:
                np.savetxt(path, np.nan_to_num(data), delimiter=",")
            self._symm_status_lbl.setText(f"Exported {which} → {path}")
        except Exception as exc:
            self._symm_status_lbl.setText(f"Export failed: {exc}")

    def _update_symmetrize_status(self, res=None) -> None:
        if not getattr(self, "_symm_status_lbl", None):
            return
        n = int(self._symm_fold_spin.value())
        group = f"D{n}" if self._symm_mirror_cb.isChecked() else f"C{n}"
        src = "ROI" if self._fft_source == "active_roi" else "whole image"
        bits = [f"Symmetry: {group} ({n}-fold"
                f"{' + mirror' if self._symm_mirror_cb.isChecked() else ''}) · "
                f"FFT source: {src}"]
        if res is not None:
            pct = res.symmetry_residual_norm * 100.0
            bits.append(
                f"Removed asymmetry: {pct:.1f}% of image contrast "
                "(inspect the Residual preview — defects live there)")
        warn = self._anisotropy_warning()
        if warn:
            bits.append(warn)
        self._symm_status_lbl.setText("\n".join(bits))
