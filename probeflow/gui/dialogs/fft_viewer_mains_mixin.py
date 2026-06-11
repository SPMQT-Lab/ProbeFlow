"""Mains-pickup (50/60 Hz) diagnostic + removal tab for the FFT viewer.

Mixin split out of ``fft_viewer.py``. It relies on attributes owned by
``FFTViewerDialog`` (``self._arr``, ``self._qx`` / ``self._qy``, ``self._ax_fft``,
``self._canvas_fft``, ``self._scan_range_m`` / ``self._full_scan_range_m``,
``self._fft_source`` / ``self._roi_id``, the ``self._get_image_fn`` /
``self._apply_correction_fn`` callbacks, and the shared FFT-preview helpers
``_show_fft_preview`` / ``_hide_fft_preview``).
"""

from __future__ import annotations

import math

import numpy as np
from probeflow.gui._tooltips import tip as _tip
from probeflow.gui.typography import ui_font
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)


# Compact width caps so value fields don't stretch across the wide sidebar.
_FIELD_W = 96
_WIDE_FIELD_W = 124


class FFTViewerMainsMixin:
    """Predict / overlay / inspect / preview / apply mains-pickup suppression."""

    def _build_mains_tab(self) -> QWidget:
        """Predict / overlay / inspect / preview / apply 50–60 Hz mains pickup."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        intro = QLabel(
            "Mains pickup (50/60 Hz) appears at a predictable spot in the FFT "
            "set by the fast-scan speed. Mark it, check the cursor frequency, "
            "then preview and apply a symmetric notch.")
        intro.setWordWrap(True)
        intro.setFont(ui_font(9))
        intro.setToolTip(_tip(
            "Find and remove mains pickup — the 50/60 Hz electrical hum that "
            "appears as faint, regularly-spaced stripes. Use it when a scan "
            "shows fine periodic banding that may line up with the power-line "
            "frequency."))
        lay.addWidget(intro)

        # ── overlay + prediction controls (two-column grid) ─────────────────────
        grp = QGroupBox("Predict & overlay")
        gl = QGridLayout(grp)
        gl.setHorizontalSpacing(8)
        gl.setVerticalSpacing(4)
        gl.setContentsMargins(8, 7, 8, 4)

        self._mains_overlay_cb = QCheckBox("Show mains-pickup overlay")
        self._mains_overlay_cb.setToolTip(_tip(
            "Mark where 50/60 Hz pickup and its harmonics should appear in this "
            "FFT, from the fast-scan speed. Nothing is changed — it's a "
            "diagnostic guide. Turn it on to check whether a suspicious FFT "
            "spot is really mains before removing anything."))
        self._mains_overlay_cb.toggled.connect(self._on_mains_changed)

        self._mains_freq_combo = QComboBox()
        self._mains_freq_combo.addItems(["50 Hz", "60 Hz"])
        self._mains_freq_combo.setToolTip(_tip(
            "Your local mains frequency — 50 Hz (most of the world) or 60 Hz "
            "(North America). Choose the one for where the data was taken."))
        self._mains_freq_combo.currentIndexChanged.connect(self._on_mains_changed)
        self._mains_freq_combo.setMaximumWidth(_FIELD_W)

        self._mains_fast_combo = QComboBox()
        self._mains_fast_combo.addItems(["Horizontal", "Vertical"])
        self._mains_fast_combo.setToolTip(_tip(
            "Which image direction was scanned fast. Mains stripes run across "
            "this direction. Leave on Horizontal unless the scan was rotated "
            "or transposed."))
        self._mains_fast_combo.currentIndexChanged.connect(self._on_mains_changed)
        self._mains_fast_combo.setMaximumWidth(_FIELD_W)

        self._mains_speed_spin = QDoubleSpinBox()
        self._mains_speed_spin.setRange(0.0, 1e6)
        self._mains_speed_spin.setDecimals(3)
        self._mains_speed_spin.setSuffix(" nm/s")
        if self._scan_speed_m_per_s and self._scan_speed_m_per_s > 0:
            self._mains_speed_spin.setValue(self._scan_speed_m_per_s * 1e9)
        self._mains_speed_spin.setToolTip(_tip(
            "Fast-scan tip speed, read from the file when available — it sets "
            "where mains lands in the FFT. If blank or wrong, type the value "
            "from your scan parameters; the overlay needs it."))
        self._mains_speed_spin.valueChanged.connect(self._on_mains_changed)
        self._mains_speed_spin.setMaximumWidth(_WIDE_FIELD_W)

        self._mains_harm_spin = QSpinBox()
        self._mains_harm_spin.setRange(1, 20)
        self._mains_harm_spin.setValue(3)
        self._mains_harm_spin.setEnabled(False)
        self._mains_harm_spin.setToolTip(_tip(
            "Manual number of mains harmonics to mark when Auto is off."))
        self._mains_harm_spin.valueChanged.connect(self._on_mains_changed)
        self._mains_harm_spin.setMaximumWidth(_FIELD_W)

        self._mains_auto_cb = QCheckBox("Auto harmonics to Nyquist")
        self._mains_auto_cb.setChecked(True)
        self._mains_auto_cb.setToolTip(_tip(
            "Mark every mains harmonic that fits before the FFT Nyquist limit."))
        self._mains_auto_cb.toggled.connect(self._on_mains_changed)

        gl.addWidget(self._mains_overlay_cb, 0, 0, 1, 4)
        gl.addWidget(QLabel("Frequency:"), 1, 0)
        gl.addWidget(self._mains_freq_combo, 1, 1)
        gl.addWidget(QLabel("Fast axis:"), 1, 2)
        gl.addWidget(self._mains_fast_combo, 1, 3)
        gl.addWidget(QLabel("Scan speed:"), 2, 0)
        gl.addWidget(self._mains_speed_spin, 2, 1)
        gl.addWidget(QLabel("Harmonics:"), 2, 2)
        gl.addWidget(self._mains_harm_spin, 2, 3)
        gl.addWidget(self._mains_auto_cb, 3, 0, 1, 4)
        gl.setColumnStretch(4, 1)

        self._mains_status_lbl = QLabel("")
        self._mains_status_lbl.setWordWrap(True)
        self._mains_status_lbl.setFont(ui_font(8))
        gl.addWidget(self._mains_status_lbl, 4, 0, 1, 5)
        lay.addWidget(grp)

        # ── suppression controls (two-column grid) ──────────────────────────────
        sgrp = QGroupBox("Suppress")
        sl = QGridLayout(sgrp)
        sl.setHorizontalSpacing(8)
        sl.setVerticalSpacing(4)
        sl.setContentsMargins(8, 7, 8, 4)

        self._mains_radius_spin = QSpinBox()
        self._mains_radius_spin.setRange(1, 20)
        self._mains_radius_spin.setValue(3)
        self._mains_radius_spin.setSuffix(" px")
        self._mains_radius_spin.setToolTip(_tip(
            "Width of each mains streak notch in FFT pixels. The shaded band "
            "around each overlay line shows this width on the FFT."))
        # The overlay band must track the radius live, or the control reads
        # as doing nothing.
        self._mains_radius_spin.valueChanged.connect(self._on_mains_changed)
        self._mains_radius_spin.setMaximumWidth(_FIELD_W)

        self._mains_min_q_spin = QDoubleSpinBox()
        self._mains_min_q_spin.setRange(0.0, 1e6)
        self._mains_min_q_spin.setDecimals(3)
        self._mains_min_q_spin.setSingleStep(0.1)
        self._mains_min_q_spin.setSuffix(" nm⁻¹")
        self._mains_min_q_spin.setValue(0.0)
        self._mains_min_q_spin.setToolTip(_tip(
            "Protect low-q signal by leaving a central circle unchanged."))
        self._mains_min_q_spin.valueChanged.connect(self._on_mains_changed)
        self._mains_min_q_spin.setMaximumWidth(_WIDE_FIELD_W)

        sl.addWidget(QLabel("Notch radius:"), 0, 0)
        sl.addWidget(self._mains_radius_spin, 0, 1)
        sl.addWidget(QLabel("Minimum |q|:"), 0, 2)
        sl.addWidget(self._mains_min_q_spin, 0, 3)
        sl.setColumnStretch(4, 1)

        self._mains_fill_cb = QCheckBox("Fill notches to background")
        self._mains_fill_cb.setToolTip(_tip(
            "Instead of zeroing the notched FFT bins (which leaves black "
            "streak-shaped gaps and removes the genuine noise floor too), "
            "bring them down to the local background magnitude — only the "
            "excess pickup energy is removed and the FFT shows background "
            "where the streak was."))
        sl.addWidget(self._mains_fill_cb, 1, 0, 1, 5)

        # ── custom streak pairs (pickup at non-mains frequencies) ───────────────
        custom_row = QHBoxLayout()
        self._mains_add_streak_btn = QPushButton("Add streak pair")
        self._mains_add_streak_btn.setToolTip(_tip(
            "Add a symmetric ±q pair of notch lines for a vertical streak "
            "that is not at a mains harmonic. Drag either line on the FFT "
            "to put it on the streak; the partner mirrors automatically."))
        self._mains_add_streak_btn.clicked.connect(self._on_mains_add_streak)
        self._mains_remove_streak_btn = QPushButton("Remove pair")
        self._mains_remove_streak_btn.setEnabled(False)
        self._mains_remove_streak_btn.setToolTip(_tip(
            "Remove the most recently added/dragged custom streak pair."))
        self._mains_remove_streak_btn.clicked.connect(self._on_mains_remove_streak)
        for b in (self._mains_add_streak_btn, self._mains_remove_streak_btn):
            b.setMaximumWidth(130)
        custom_row.addWidget(self._mains_add_streak_btn)
        custom_row.addWidget(self._mains_remove_streak_btn)
        custom_row.addStretch(1)
        sl.addLayout(custom_row, 2, 0, 1, 5)

        self._mains_residual_cb = QCheckBox("Show residual (removed signal)")
        self._mains_residual_cb.setToolTip(_tip(
            "Preview the residual (original − filtered) instead of the filtered "
            "image. Check it looks like noise/stripes, not real features, "
            "before applying."))
        sl.addWidget(self._mains_residual_cb, 3, 0, 1, 5)

        btn_row = QHBoxLayout()
        self._mains_preview_btn = QPushButton("Preview")
        self._mains_preview_btn.setToolTip(_tip(
            "Show the filtered result (or the residual) without changing your "
            "image. Check the residual looks like noise, not real features, "
            "before applying."))
        self._mains_preview_btn.clicked.connect(self._on_mains_preview)
        self._mains_clear_btn = QPushButton("Clear preview")
        self._mains_clear_btn.setEnabled(False)
        self._mains_clear_btn.clicked.connect(self._on_mains_clear)
        self._mains_apply_btn = QPushButton("Apply")
        self._mains_apply_btn.setObjectName("accentBtn")
        self._mains_apply_btn.setToolTip(_tip(
            "Apply the previewed mains suppression and record the frequency, "
            "harmonics, scan speed and notch settings in the processing "
            "history, so the removal is reproducible."))
        self._mains_apply_btn.clicked.connect(self._on_mains_apply)
        for b in (self._mains_preview_btn, self._mains_clear_btn, self._mains_apply_btn):
            b.setMaximumWidth(110)
        btn_row.addWidget(self._mains_preview_btn)
        btn_row.addWidget(self._mains_clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._mains_apply_btn)
        sl.addLayout(btn_row, 4, 0, 1, 5)
        lay.addWidget(sgrp)

        lay.addStretch(1)
        scroll.setWidget(page)
        self._update_mains_status()
        return scroll

    def _mains_speed_m_per_s(self) -> float | None:
        v = float(self._mains_speed_spin.value()) * 1e-9
        return v if v > 0 else None

    def _mains_harmonics(self) -> int | None:
        auto = getattr(self, "_mains_auto_cb", None)
        if auto is not None and auto.isChecked():
            return None
        return int(self._mains_harm_spin.value())

    def _mains_min_q_nm_inv(self) -> float:
        spin = getattr(self, "_mains_min_q_spin", None)
        return max(0.0, float(spin.value())) if spin is not None else 0.0

    def _mains_predictions(self) -> list:
        v = self._mains_speed_m_per_s()
        if not v or self._arr is None:
            return []
        from probeflow.processing.mains_pickup import predict_mains_fft_positions
        fast = self._mains_fast_axis
        n_fast = self._arr.shape[1] if fast == "x" else self._arr.shape[0]
        width = self._scan_range_m[0] if fast == "x" else self._scan_range_m[1]
        f = 50.0 if self._mains_freq_combo.currentIndex() == 0 else 60.0
        return predict_mains_fft_positions(
            n_fast, width, v, mains_frequency_hz=f,
            harmonics=self._mains_harmonics(), fast_axis=fast)

    @staticmethod
    def _axis_segments_with_radial_floor(
        fixed_q: float,
        axis_values: np.ndarray,
        min_q: float,
    ) -> list[tuple[float, float]]:
        if axis_values is None or len(axis_values) == 0:
            return []
        lo = float(np.nanmin(axis_values))
        hi = float(np.nanmax(axis_values))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return []
        if min_q <= abs(float(fixed_q)):
            return [(lo, hi)]
        gap = math.sqrt(max(0.0, min_q * min_q - float(fixed_q) * float(fixed_q)))
        segments: list[tuple[float, float]] = []
        if lo < -gap:
            segments.append((lo, min(hi, -gap)))
        if hi > gap:
            segments.append((max(lo, gap), hi))
        return [(a, b) for a, b in segments if b > a]

    def _mains_q_per_px(self) -> float:
        """FFT-pixel spacing of the fast axis in nm⁻¹ (0.0 when unknown)."""
        axis = self._qx if self._mains_fast_axis == "x" else self._qy
        if axis is None or len(axis) < 2:
            return 0.0
        return float(abs(axis[1] - axis[0]))

    def _mains_custom_streaks(self) -> list[float]:
        """Positive-|q| positions (nm⁻¹) of the user-placed streak pairs."""
        if not hasattr(self, "_mains_custom_q"):
            self._mains_custom_q = []
        return self._mains_custom_q

    def _draw_mains_overlay(self) -> None:
        """Lines at the predicted mains q positions plus user streak pairs.

        Each line carries a translucent band of half-width = notch radius so
        the removal width is visible and tracks the radius control.
        Rebuilt on every FFT redraw (the axes are cleared by ``ax.cla()``).
        """
        self._mains_artists = []
        if not getattr(self, "_mains_overlay_cb", None):
            return
        if not self._mains_overlay_cb.isChecked() or self._qx is None or self._qy is None:
            return
        min_q = self._mains_min_q_nm_inv()
        half_w = float(self._mains_radius_spin.value()) * self._mains_q_per_px()

        def _draw_pair(q: float, color: str, style: str) -> None:
            for qq in (q, -q):
                if self._mains_fast_axis == "x":
                    for y0, y1 in self._axis_segments_with_radial_floor(qq, self._qy, min_q):
                        art, = self._ax_fft.plot(
                            [qq, qq], [y0, y1], color=color, lw=0.9,
                            ls=style, alpha=0.85, zorder=7,
                        )
                        self._mains_artists.append(art)
                        if half_w > 0:
                            band = self._ax_fft.fill_betweenx(
                                [y0, y1], qq - half_w, qq + half_w,
                                color=color, alpha=0.16, lw=0, zorder=6,
                            )
                            self._mains_artists.append(band)
                else:
                    for x0, x1 in self._axis_segments_with_radial_floor(qq, self._qx, min_q):
                        art, = self._ax_fft.plot(
                            [x0, x1], [qq, qq], color=color, lw=0.9,
                            ls=style, alpha=0.85, zorder=7,
                        )
                        self._mains_artists.append(art)
                        if half_w > 0:
                            band = self._ax_fft.fill_between(
                                [x0, x1], qq - half_w, qq + half_w,
                                color=color, alpha=0.16, lw=0, zorder=6,
                            )
                            self._mains_artists.append(band)

        for p in self._mains_predictions():
            _draw_pair(p["q_nm_inv"], "#f9e2af", "--")
        # User streak pairs: solid cyan so they read as hand-placed, not
        # predicted; draggable on the FFT while the Mains tab is active.
        for q in self._mains_custom_streaks():
            _draw_pair(q, "#89dceb", "-")

    def _on_mains_changed(self) -> None:
        """Fast path: refresh the overlay + status when a control changes."""
        self._mains_fast_axis = "x" if self._mains_fast_combo.currentIndex() == 0 else "y"
        auto = getattr(self, "_mains_auto_cb", None)
        if auto is not None and hasattr(self, "_mains_harm_spin"):
            self._mains_harm_spin.setEnabled(not auto.isChecked())
        for art in self._mains_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._mains_artists = []
        self._draw_mains_overlay()
        self._canvas_fft.draw_idle()
        self._update_mains_status()

    def _update_mains_status(self) -> None:
        if not getattr(self, "_mains_status_lbl", None):
            return
        n_custom = len(self._mains_custom_streaks())
        custom_note = (
            f"  +{n_custom} custom streak pair(s)." if n_custom else ""
        )
        if self._mains_speed_m_per_s() is None:
            self._mains_status_lbl.setText(
                "Scan speed unavailable; enter nm/s to show the mains overlay."
                + custom_note)
            return
        preds = self._mains_predictions()
        if not preds:
            self._mains_status_lbl.setText(
                "No mains harmonics fall within this FFT (check speed/frequency)."
                + custom_note)
            return
        src = "ROI" if self._fft_source == "active_roi" else "whole image"
        if self._mains_harmonics() is None and len(preds) > 4:
            parts = [
                f"{len(preds)} harmonics",
                f"{preds[0]['freq_hz']:.0f}-{preds[-1]['freq_hz']:.0f} Hz",
                f"q={preds[0]['q_nm_inv']:.2f}-{preds[-1]['q_nm_inv']:.2f} nm⁻¹",
            ]
        else:
            parts = [f"{p['freq_hz']:.0f} Hz → q={p['q_nm_inv']:.2f} nm⁻¹" for p in preds]
        min_q = self._mains_min_q_nm_inv()
        floor = f"  |q|≥{min_q:.2f} nm⁻¹." if min_q > 0 else ""
        self._mains_status_lbl.setText(f"FFT source: {src}.  " + " · ".join(parts) + floor)

    # ── Custom streak pairs ───────────────────────────────────────────────────

    def _mains_extra_streaks_px(self) -> list[int]:
        dq = self._mains_q_per_px()
        if dq <= 0:
            return []
        seen: set[int] = set()
        out: list[int] = []
        for q in self._mains_custom_streaks():
            px = int(round(abs(float(q)) / dq))
            if px > 0 and px not in seen:
                seen.add(px)
                out.append(px)
        return out

    def _on_mains_add_streak(self) -> None:
        axis = self._qx if self._mains_fast_axis == "x" else self._qy
        if axis is None or len(axis) < 2:
            self._mains_status_lbl.setText("Open an FFT first.")
            return
        q_max = float(np.nanmax(np.abs(axis)))
        # Stagger new pairs so consecutive adds don't stack invisibly.
        n = len(self._mains_custom_streaks())
        q_new = q_max * (0.35 + 0.12 * (n % 5))
        self._mains_custom_streaks().append(q_new)
        self._mains_remove_streak_btn.setEnabled(True)
        self._on_mains_changed()
        self._mains_status_lbl.setText(
            f"Streak pair added at ±{q_new:.2f} nm⁻¹ — drag either line onto "
            "the streak."
        )

    def _on_mains_remove_streak(self) -> None:
        streaks = self._mains_custom_streaks()
        if streaks:
            streaks.pop()
        self._mains_remove_streak_btn.setEnabled(bool(streaks))
        self._on_mains_changed()

    def _mains_tab_active(self) -> bool:
        idx = getattr(self, "_mains_tab_index", None)
        tabs = getattr(self, "_tab_widget", None)
        return idx is not None and tabs is not None and tabs.currentIndex() == idx

    def _mains_handle_press(self, event) -> bool:
        """Begin dragging the custom streak pair nearest the click, if close."""
        self._mains_drag_idx = getattr(self, "_mains_drag_idx", None)
        if (not self._mains_tab_active() or event.inaxes is not self._ax_fft
                or event.button != 1 or not self._mains_custom_streaks()):
            return False
        coord = event.xdata if self._mains_fast_axis == "x" else event.ydata
        if coord is None:
            return False
        axis = self._qx if self._mains_fast_axis == "x" else self._qy
        tol = max(3.0 * self._mains_q_per_px(),
                  0.02 * float(np.nanmax(np.abs(axis))))
        best_idx, best_dist = None, tol
        for i, q in enumerate(self._mains_custom_streaks()):
            dist = abs(abs(float(coord)) - q)
            if dist <= best_dist:
                best_idx, best_dist = i, dist
        if best_idx is None:
            return False
        self._mains_drag_idx = best_idx
        return True

    def _mains_handle_motion(self, event) -> bool:
        idx = getattr(self, "_mains_drag_idx", None)
        if idx is None:
            return False
        if event.inaxes is not self._ax_fft:
            return True  # keep the drag captured while skimming the edge
        coord = event.xdata if self._mains_fast_axis == "x" else event.ydata
        if coord is None:
            return True
        dq = self._mains_q_per_px()
        streaks = self._mains_custom_streaks()
        if 0 <= idx < len(streaks):
            streaks[idx] = max(abs(float(coord)), dq)
            self._on_mains_changed()
        return True

    def _mains_handle_release(self, _event) -> bool:
        if getattr(self, "_mains_drag_idx", None) is None:
            return False
        self._mains_drag_idx = None
        self._update_mains_status()
        return True

    def _mains_op_params(self) -> dict:
        params = {
            "scan_speed_m_per_s": self._mains_speed_m_per_s(),
            "scan_range_m": [float(self._full_scan_range_m[0]),
                             float(self._full_scan_range_m[1])],
            "mains_frequency_hz": 50.0 if self._mains_freq_combo.currentIndex() == 0 else 60.0,
            "harmonics": self._mains_harmonics(),
            "notch_radius_px": float(self._mains_radius_spin.value()),
            "fast_axis": self._mains_fast_axis,
            "snap_window_px": 2,
            "notch_shape": "streak",
            "min_q_nm_inv": self._mains_min_q_nm_inv(),
            "extra_streaks_px": self._mains_extra_streaks_px(),
            "notch_fill": (
                "background"
                if getattr(self, "_mains_fill_cb", None) is not None
                and self._mains_fill_cb.isChecked() else "zero"
            ),
            "fft_source": self._fft_source,
        }
        if self._fft_source == "active_roi" and self._roi_id is not None:
            params["fft_roi_id"] = self._roi_id
        return params

    def _on_mains_preview(self) -> None:
        v = self._mains_speed_m_per_s()
        if not v and not self._mains_extra_streaks_px():
            self._mains_status_lbl.setText(
                "Enter a scan speed (nm/s) or add a custom streak pair first.")
            return
        arr = self._get_image_fn() if self._get_image_fn is not None else self._full_arr
        if arr is None:
            return
        from probeflow.processing.mains_pickup import mains_pickup_suppression
        p = self._mains_op_params()
        try:
            filtered = mains_pickup_suppression(
                np.asarray(arr, dtype=np.float64),
                scan_speed_m_per_s=v, scan_range_m=tuple(self._full_scan_range_m),
                mains_frequency_hz=p["mains_frequency_hz"], harmonics=p["harmonics"],
                notch_radius_px=p["notch_radius_px"], fast_axis=p["fast_axis"],
                snap_window_px=p["snap_window_px"], notch_shape=p["notch_shape"],
                min_q_nm_inv=p["min_q_nm_inv"],
                extra_streaks_px=p["extra_streaks_px"],
                notch_fill=p["notch_fill"])
        except Exception as exc:
            self._mains_status_lbl.setText(f"Preview failed: {exc}")
            return
        if self._mains_residual_cb.isChecked():
            self._show_fft_preview(np.asarray(arr, dtype=np.float64) - filtered)
            self._mains_status_lbl.setText("Residual preview (what would be removed).")
        else:
            self._show_fft_preview(filtered)
            self._mains_status_lbl.setText("Filtered preview shown.")
        self._mains_preview_active = True
        self._mains_clear_btn.setEnabled(True)

    def _on_mains_clear(self) -> None:
        self._hide_fft_preview()
        self._mains_preview_active = False
        self._mains_clear_btn.setEnabled(False)
        self._update_mains_status()

    def _on_mains_apply(self) -> None:
        if self._apply_correction_fn is None:
            self._mains_status_lbl.setText("Apply is unavailable in this context.")
            return
        if self._mains_speed_m_per_s() is None and not self._mains_extra_streaks_px():
            self._mains_status_lbl.setText(
                "Enter a scan speed (nm/s) or add a custom streak pair first.")
            return
        if self._mains_preview_active:
            self._hide_fft_preview()
            self._mains_preview_active = False
            self._mains_clear_btn.setEnabled(False)
        self._apply_correction_fn("mains_pickup_suppression", self._mains_op_params())
        self._mains_status_lbl.setText("Applied mains-pickup suppression.")
