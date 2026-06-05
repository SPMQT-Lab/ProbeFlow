"""Inverse-FFT / Fourier-reconstruction tab for the FFT viewer.

Mixin split out of ``fft_viewer.py``. It relies on attributes owned by
``FFTViewerDialog`` (``self._arr``, ``self._qx`` / ``self._qy``, ``self._ax_fft``,
``self._canvas_fft``, ``self._tab_widget`` / ``self._reconstruct_tab_index``,
``self._fft_selection_overlay``, ``self._fft_source`` / ``self._roi_id``,
``self._scan_range_m``, the ``self._apply_correction_fn`` / ``self._new_image_fn``
callbacks, and the shared FFT-preview helpers ``_show_fft_preview`` /
``_hide_fft_preview``).
"""

from __future__ import annotations

import numpy as np
from probeflow.core.resources import asset_path
from probeflow.gui._tooltips import tip as _tip
from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

# Shared paint palette (mirrors the feature-finder MASK_COLORS).
_PAINT_COLORS: dict[str, tuple[int, int, int]] = {
    "Cyan":    (137, 220, 235),
    "Red":     (243, 139, 168),
    "Green":   (166, 227, 161),
    "Yellow":  (249, 226, 175),
    "Blue":    (137, 180, 250),
    "Magenta": (203, 166, 247),
}


class FFTViewerReconstructMixin:
    """Select Fourier features, preview the inverse FFT, and apply/export."""

    def _build_reconstruct_tab(self) -> QWidget:
        """Draw Fourier selections (ellipse / rectangle / paint), preview the
        inverse FFT result + residual, and apply/export."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        intro = QLabel(
            "Select Fourier features, then preview the inverse FFT to see exactly "
            "what is removed or isolated in real space. The residual is "
            "original − result.")
        intro.setWordWrap(True)
        intro.setFont(ui_font(9))
        intro.setToolTip(_tip(
            "Make FFT filtering auditable: draw ellipse/rectangle/paint "
            "selections on Fourier features, choose Remove or Keep, and preview "
            "the reconstructed image and the residual before applying. Use it to "
            "confirm a periodic artefact is really gone (or to isolate one "
            "periodic component)."))
        lay.addWidget(intro)

        # ── selections ─────────────────────────────────────────────────────────
        sgrp = QGroupBox("Fourier selections")
        sg = QVBoxLayout(sgrp)
        sg.setSpacing(4)
        # Checkable draw tools: pick one, then drag on the FFT to draw. Holding
        # Shift draws a regular shape (circle / square). Click an active tool
        # again to return to edit mode (move/resize existing selections).
        tool_row = QHBoxLayout()
        self._recon_tool_btns: dict[str, QToolButton] = {}
        tool_specs = [
            ("ellipse", "Ellipse", "ellipse",
             "Draw an ellipse: drag a box on the FFT. Hold Shift for a circle. "
             "Its conjugate partner (dashed) is added automatically."),
            ("rect", "Rectangle", "rectangle",
             "Draw a rectangle: drag a box on the FFT. Hold Shift for a square."),
            ("paint", "Paint", "freehand",
             "Freehand brush: drag to paint an irregular Fourier region. The "
             "mirrored (conjugate) region is grabbed too."),
        ]
        for kind, label, icon, tip in tool_specs:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            ipath = asset_path(f"toolbar/{icon}.png")
            if ipath.exists():
                btn.setIcon(QIcon(str(ipath)))
                btn.setIconSize(QSize(16, 16))
            btn.setToolTip(_tip(tip))
            btn.clicked.connect(lambda _checked=False, k=kind: self._on_tool_clicked(k))
            self._recon_tool_btns[kind] = btn
            tool_row.addWidget(btn)
        tool_row.addStretch(1)
        sg.addLayout(tool_row)

        # Paint brush controls — only relevant while the Paint tool is active.
        self._recon_paint_row = QWidget()
        paint_lay = QHBoxLayout(self._recon_paint_row)
        paint_lay.setContentsMargins(0, 0, 0, 0)
        paint_lay.setSpacing(4)
        paint_lay.addWidget(QLabel("Brush:"))
        self._recon_brush_spin = QSpinBox()
        self._recon_brush_spin.setRange(1, 100)
        self._recon_brush_spin.setValue(8)
        self._recon_brush_spin.setSuffix(" px")
        self._recon_brush_spin.setToolTip(_tip("Paint brush radius, in FFT pixels."))
        self._recon_brush_spin.setMaximumWidth(80)
        self._recon_brush_spin.valueChanged.connect(self._on_brush_size_changed)
        paint_lay.addWidget(self._recon_brush_spin)
        paint_lay.addWidget(QLabel("Color:"))
        self._recon_color_combo = QComboBox()
        self._recon_color_combo.addItems(list(_PAINT_COLORS.keys()))
        self._recon_color_combo.setToolTip(_tip("Paint overlay color."))
        self._recon_color_combo.setMaximumWidth(110)
        self._recon_color_combo.currentIndexChanged.connect(self._on_paint_color_changed)
        paint_lay.addWidget(self._recon_color_combo)
        paint_lay.addStretch(1)
        self._recon_paint_row.setVisible(False)
        sg.addWidget(self._recon_paint_row)

        del_btn = QPushButton("Delete selected")
        del_btn.setToolTip(_tip("Remove the currently-selected Fourier region."))
        del_btn.clicked.connect(self._on_delete_selection)
        clr_btn = QPushButton("Clear selections")
        clr_btn.setToolTip(_tip("Remove all Fourier selections."))
        clr_btn.clicked.connect(self._on_clear_selections)
        for b in (del_btn, clr_btn):
            b.setMaximumWidth(150)
        del_row = QHBoxLayout()
        del_row.addWidget(del_btn)
        del_row.addWidget(clr_btn)
        del_row.addStretch(1)
        sg.addLayout(del_row)

        # ── mask options ───────────────────────────────────────────────────────
        ogrp = QGroupBox("Mask")
        og = QVBoxLayout(ogrp)
        og.setSpacing(4)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._recon_mode_combo = QComboBox()
        self._recon_mode_combo.addItems(["Remove selected", "Keep selected"])
        self._recon_mode_combo.setToolTip(_tip(
            "Remove selected: suppress the selected Fourier features and rebuild "
            "the corrected image. Keep selected: rebuild only the selected "
            "periodic component (everything else removed)."))
        # Drop the combo's int index — _update_reconstruct_status treats its
        # first positional arg as a reconstruction result, so passing the index
        # straight through made it call res.imag_residual_norm on an int.
        self._recon_mode_combo.currentIndexChanged.connect(
            lambda _idx: self._update_reconstruct_status())
        self._recon_mode_combo.setMaximumWidth(140)
        mode_row.addWidget(self._recon_mode_combo)
        mode_row.addStretch(1)
        og.addLayout(mode_row)
        self._recon_conj_cb = QCheckBox("Conjugate symmetry (keep result real)")
        self._recon_conj_cb.setChecked(True)
        self._recon_conj_cb.setToolTip(_tip(
            "Include each selection's mirror feature at the opposite side of DC. "
            "Keep this on: it is what makes the reconstructed image real-valued. "
            "Turn off only for expert phase experiments."))
        og.addWidget(self._recon_conj_cb)
        soft_row = QHBoxLayout()
        soft_row.addWidget(QLabel("Soft edge:"))
        self._recon_soft_spin = QSpinBox()
        self._recon_soft_spin.setRange(0, 20)
        self._recon_soft_spin.setValue(0)
        self._recon_soft_spin.setSuffix(" px")
        self._recon_soft_spin.setToolTip(_tip(
            "Feather the selection boundary by this many pixels. A few pixels "
            "reduces ringing in the reconstructed image; 0 is a hard edge."))
        self._recon_soft_spin.setMaximumWidth(96)
        soft_row.addWidget(self._recon_soft_spin)
        soft_row.addStretch(1)
        og.addLayout(soft_row)

        # Selections + Mask share one row — the FFT dialog is wide enough.
        top_row = QHBoxLayout()
        top_row.setSpacing(6)
        top_row.addWidget(sgrp, 1)
        top_row.addWidget(ogrp, 1)
        lay.addLayout(top_row)

        # ── preview / apply ─────────────────────────────────────────────────────
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Preview:"))
        self._recon_view_combo = QComboBox()
        self._recon_view_combo.addItems(["Result", "Residual", "Original"])
        self._recon_view_combo.setToolTip(_tip(
            "Which image the left preview shows: the reconstructed Result, the "
            "Residual (what was removed), or the Original for comparison."))
        self._recon_view_combo.currentIndexChanged.connect(
            lambda _=0: self._on_reconstruct_preview() if self._reconstruct_preview_active else None)
        self._recon_view_combo.setMaximumWidth(140)
        view_row.addWidget(self._recon_view_combo)
        view_row.addStretch(1)
        lay.addLayout(view_row)

        pv_row = QHBoxLayout()
        self._recon_preview_btn = QPushButton("Preview inverse FFT")
        self._recon_preview_btn.setToolTip(_tip(
            "Reconstruct the image with the current mask and show it in the left "
            "preview, without changing your data. Inspect the residual before "
            "applying."))
        self._recon_preview_btn.clicked.connect(self._on_reconstruct_preview)
        self._recon_clear_btn = QPushButton("Clear preview")
        self._recon_clear_btn.setEnabled(False)
        self._recon_clear_btn.clicked.connect(self._on_reconstruct_clear)
        pv_row.addWidget(self._recon_preview_btn)
        pv_row.addWidget(self._recon_clear_btn)
        lay.addLayout(pv_row)

        ap_row = QHBoxLayout()
        self._recon_apply_btn = QPushButton("Apply to image")
        self._recon_apply_btn.setObjectName("accentBtn")
        self._recon_apply_btn.setToolTip(_tip(
            "Apply the inverse-FFT reconstruction. On a whole-image FFT this is "
            "recorded as a reproducible step in the processing history. On an "
            "ROI it creates a new corrected cropped image (never edits the "
            "original in place)."))
        self._recon_apply_btn.clicked.connect(self._on_reconstruct_apply)
        ap_row.addWidget(self._recon_apply_btn)
        lay.addLayout(ap_row)

        ex_row = QHBoxLayout()
        exp_res = QPushButton("Export result")
        exp_res.setToolTip(_tip("Save the reconstructed image to a file."))
        exp_res.clicked.connect(lambda: self._on_reconstruct_export("result"))
        exp_resid = QPushButton("Export residual")
        exp_resid.setToolTip(_tip("Save the residual (removed component) to a file."))
        exp_resid.clicked.connect(lambda: self._on_reconstruct_export("residual"))
        ex_row.addWidget(exp_res)
        ex_row.addWidget(exp_resid)
        lay.addLayout(ex_row)

        self._recon_status_lbl = QLabel("")
        self._recon_status_lbl.setWordWrap(True)
        self._recon_status_lbl.setFont(ui_font(8))
        lay.addWidget(self._recon_status_lbl)

        lay.addStretch(1)
        scroll.setWidget(page)
        self._update_reconstruct_status()
        return scroll

    def _reconstruct_active(self) -> bool:
        return (self._reconstruct_tab_index >= 0
                and self._tab_widget.currentIndex() == self._reconstruct_tab_index)

    def _ensure_selection_overlay(self):
        if self._qx is None or self._arr is None:
            return None
        from probeflow.gui.dialogs.fft_selection import FFTSelectionOverlay
        if self._fft_selection_overlay is None:
            self._fft_selection_overlay = FFTSelectionOverlay(
                self._ax_fft, self._qx, self._qy, self._arr.shape,
                on_change=self._on_selection_changed)
        else:
            self._fft_selection_overlay.set_qaxes(self._qx, self._qy, self._arr.shape)
        return self._fft_selection_overlay

    def _draw_selection_overlay(self) -> None:
        ov = self._ensure_selection_overlay()
        if ov is not None and ov.count() > 0:
            ov.draw()

    def _on_selection_changed(self) -> None:
        ov = self._fft_selection_overlay
        if ov is None:
            return
        for art in ov._artists:
            try:
                art.remove()
            except Exception:
                pass
        ov.draw()
        self._canvas_fft.draw_idle()
        self._update_reconstruct_status()

    def _on_tool_clicked(self, kind: str) -> None:
        ov = self._ensure_selection_overlay()
        if ov is None:
            self._recon_tool_btns[kind].setChecked(False)
            self._recon_status_lbl.setText("Load a scan first.")
            return
        # Toggle: a second click on the active tool returns to edit mode.
        active = None if ov.tool() == kind else kind
        ov.set_tool(active)
        for k, btn in self._recon_tool_btns.items():
            btn.setChecked(k == active)
        self._recon_paint_row.setVisible(active == "paint")
        if active == "paint":
            ov.set_brush_radius_px(self._recon_brush_spin.value())
            ov.set_paint_color(_PAINT_COLORS[self._recon_color_combo.currentText()])
        self._update_reconstruct_status()

    def _on_brush_size_changed(self, value: int) -> None:
        ov = self._fft_selection_overlay
        if ov is not None:
            ov.set_brush_radius_px(int(value))

    def _on_paint_color_changed(self, _idx: int) -> None:
        ov = self._fft_selection_overlay
        if ov is not None:
            ov.set_paint_color(_PAINT_COLORS[self._recon_color_combo.currentText()])
            self._on_selection_changed()

    def _on_delete_selection(self) -> None:
        if self._fft_selection_overlay is not None:
            self._fft_selection_overlay.delete_selected()
            self._on_selection_changed()

    def _on_clear_selections(self) -> None:
        if self._fft_selection_overlay is not None:
            self._fft_selection_overlay.clear()
            self._on_selection_changed()

    def _reconstruct_mode(self) -> str:
        return "remove_selected" if self._recon_mode_combo.currentIndex() == 0 else "keep_selected"

    def _compute_reconstruction(self, array):
        ov = self._ensure_selection_overlay()
        if ov is None or ov.count() == 0:
            return None
        from probeflow.processing.inverse_fft import (
            fourier_region_from_dict, fourier_region_mask, inverse_fft_from_mask)
        regions = [fourier_region_from_dict(d) for d in ov.to_regions()]
        mask = fourier_region_mask(
            array.shape, regions,
            conjugate=self._recon_conj_cb.isChecked(),
            soft_px=float(self._recon_soft_spin.value()))
        return inverse_fft_from_mask(array, mask, mode=self._reconstruct_mode())

    def _on_reconstruct_preview(self) -> None:
        arr = np.asarray(self._arr, dtype=np.float64)
        res = self._compute_reconstruction(arr)
        if res is None:
            self._recon_status_lbl.setText("Add a Fourier selection first.")
            return
        view = self._recon_view_combo.currentText()
        disp = {"Result": res.result, "Residual": res.residual, "Original": arr}[view]
        self._show_fft_preview(disp)
        self._reconstruct_preview_active = True
        self._recon_clear_btn.setEnabled(True)
        self._update_reconstruct_status(res)

    def _on_reconstruct_clear(self) -> None:
        self._hide_fft_preview()
        self._reconstruct_preview_active = False
        self._recon_clear_btn.setEnabled(False)
        self._update_reconstruct_status()

    def _reconstruct_op_params(self) -> dict:
        ov = self._fft_selection_overlay
        # to_regions() already returns JSON-safe dicts carrying the per-kind
        # geometry (ellipse/rect in FFT px, paint as a pixel stamp list) plus
        # q-space provenance.
        params = {
            "selections": ov.to_regions() if ov is not None else [],
            "mode": self._reconstruct_mode(),
            "conjugate_symmetric": bool(self._recon_conj_cb.isChecked()),
            "soft_px": float(self._recon_soft_spin.value()),
            "fft_source": self._fft_source,
        }
        if self._fft_source == "active_roi" and self._roi_id is not None:
            params["fft_roi_id"] = self._roi_id
        return params

    def _on_reconstruct_apply(self) -> None:
        ov = self._ensure_selection_overlay()
        if ov is None or ov.count() == 0:
            self._recon_status_lbl.setText("Add a Fourier selection first.")
            return
        params = self._reconstruct_op_params()
        if self._fft_source == "active_roi":
            self._apply_reconstruct_to_new_image(params)
            return
        if self._apply_correction_fn is None:
            self._recon_status_lbl.setText("Apply is unavailable in this context.")
            return
        if self._reconstruct_preview_active:
            self._hide_fft_preview()
            self._reconstruct_preview_active = False
            self._recon_clear_btn.setEnabled(False)
        self._apply_correction_fn("inverse_fft_filter", params)
        self._recon_status_lbl.setText("Applied inverse FFT reconstruction.")

    def _apply_reconstruct_to_new_image(self, params: dict) -> None:
        arr = np.asarray(self._arr, dtype=np.float64)
        res = self._compute_reconstruction(arr)
        if res is None:
            return
        fn = getattr(self, "_new_image_fn", None)
        if fn is None:
            self._recon_status_lbl.setText(
                "ROI apply needs a host that can open a new image — export the "
                "result instead.")
            return
        fn(res.result, tuple(self._scan_range_m),
           {"op": "inverse_fft_filter", "params": params})
        self._recon_status_lbl.setText("Created a new corrected image from the ROI.")

    def _on_reconstruct_export(self, which: str) -> None:
        arr = np.asarray(self._arr, dtype=np.float64)
        res = self._compute_reconstruction(arr)
        if res is None:
            self._recon_status_lbl.setText("Add a Fourier selection first.")
            return
        data = res.result if which == "result" else res.residual
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export {which}", f"inverse_fft_{which}.csv",
            "CSV (*.csv);;NumPy (*.npy)")
        if not path:
            return
        try:
            if path.lower().endswith(".npy"):
                np.save(path, data)
            else:
                np.savetxt(path, np.nan_to_num(data), delimiter=",")
            self._recon_status_lbl.setText(f"Exported {which} → {path}")
        except Exception as exc:
            self._recon_status_lbl.setText(f"Export failed: {exc}")

    def _update_reconstruct_status(self, res=None) -> None:
        if not getattr(self, "_recon_status_lbl", None):
            return
        ov = self._fft_selection_overlay
        n = ov.count() if ov is not None else 0
        mode = "remove selected" if self._reconstruct_mode() == "remove_selected" else "keep selected"
        src = "ROI" if self._fft_source == "active_roi" else "whole image"
        if n == 0:
            tool = ov.tool() if ov is not None else None
            hint = ("Drag on the FFT to draw (Shift = regular shape)."
                    if tool else "Pick Ellipse, Rectangle or Paint to begin.")
            self._recon_status_lbl.setText(f"FFT source: {src}. {hint}")
            return
        conj = " + conjugates" if self._recon_conj_cb.isChecked() else ""
        txt = (f"FFT mask: {n} region{'s' if n != 1 else ''}{conj} · "
               f"Mode: {mode} · Residual = original − result")
        if res is not None and res.imag_residual_norm > 1e-6:
            txt += f" · imag {res.imag_residual_norm:.1e}"
        self._recon_status_lbl.setText(txt)
