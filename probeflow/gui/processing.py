"""GUI processing controls and processing-state adapter compatibility."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from probeflow.processing.gui_adapter import *


class ProcessingControlPanel(QWidget):
    """Internal processing controls shared by Browse and Viewer."""

    bad_line_preview_requested = Signal()
    bad_line_preview_settings_changed = Signal()
    stm_background_requested = Signal()

    QUICK_KEYS = ("align_rows", "remove_bad_lines")

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        if mode not in ("browse_quick", "viewer_full"):
            raise ValueError(f"Unknown processing panel mode: {mode}")
        self._mode = mode
        self._build()

    _TWO_COL_THRESHOLD = 360  # px — switch to 1-col below this panel width

    def resizeEvent(self, event: "QResizeEvent"):
        super().resizeEvent(event)
        if not hasattr(self, "_two_col"):
            return
        use_two_col = event.size().width() >= self._TWO_COL_THRESHOLD
        if use_two_col == self._using_two_col:
            return
        self._using_two_col = use_two_col
        hbox = self._two_col.layout()
        if use_two_col:
            # Move bg_section back into the right half of two_col
            hbox.addWidget(self._bg_section, 1)
        else:
            # Pull bg_section out of two_col, place it below in the main column
            hbox.removeWidget(self._bg_section)
            idx = self._lay.indexOf(self._two_col)
            self._lay.insertWidget(idx + 1, self._bg_section)
        self._bg_section.show()

    def _build(self):
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 2, 0, 2)
        self._lay.setSpacing(4)
        lay = self._lay  # local alias for closures

        def _combo_row(label: str, items: list[str],
                       target=None, lbl_width: int = 90) -> QComboBox:
            if target is None:
                target = lay
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            if lbl_width >= 90:
                lbl.setFixedWidth(lbl_width)
            else:
                lbl.setMaximumWidth(lbl_width)
                lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            cb = QComboBox()
            cb.addItems(items)
            cb.setFont(QFont("Helvetica", 8))
            cb.setMinimumWidth(0)
            cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(lbl)
            row.addWidget(cb, 1)
            target.addLayout(row)
            return cb

        def _sub_slider(label: str, mn: int, mx: int, init: int,
                        fmt="{v}") -> tuple[QWidget, QSlider, QLabel]:
            w = QWidget()
            rl = QHBoxLayout(w)
            rl.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setFont(QFont("Helvetica", 8))
            lbl.setMaximumWidth(44)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(mn, mx)
            sl.setValue(init)
            val_lbl = QLabel(fmt.format(v=init))
            val_lbl.setFont(QFont("Helvetica", 8))
            val_lbl.setFixedWidth(28)
            sl.valueChanged.connect(
                lambda v, vl=val_lbl, f=fmt: vl.setText(f.format(v=v)))
            rl.addWidget(lbl)
            rl.addWidget(sl, 1)
            rl.addWidget(val_lbl)
            return w, sl, val_lbl

        def _col_lbl(text: str, target):
            lbl = QLabel(text)
            lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            target.addWidget(lbl)

        # ── Line corrections (full-width in both modes) ───────────────────────
        line_lbl = QLabel("Line corrections")
        line_lbl.setFont(QFont("Helvetica", 7, QFont.Bold))
        line_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(line_lbl)

        self._align_combo = _combo_row("Align rows:", ["None", "Median", "Mean"])

        self._bad_lines_combo = _combo_row(
            "Bad lines:", ["None", "Step segments", "MAD/outlier segments"]
        )
        self._bad_lines_combo.setToolTip(
            "Detects bad fast-scan line segments and repairs only those segments "
            "from neighbouring scan lines."
        )

        if self._mode == "browse_quick":
            lay.addStretch()
            return

        self._bad_line_options = QWidget()
        blo = QVBoxLayout(self._bad_line_options)
        blo.setContentsMargins(0, 0, 0, 0)
        blo.setSpacing(3)
        polarity_row = QHBoxLayout()
        polarity_row.setContentsMargins(0, 0, 0, 0)
        polarity_label = QLabel("Polarity:")
        polarity_label.setFont(QFont("Helvetica", 8))
        self._bad_line_polarity_combo = QComboBox()
        self._bad_line_polarity_combo.addItems(["Bright", "Dark"])
        self._bad_line_polarity_combo.setFont(QFont("Helvetica", 8))
        self._bad_line_polarity_combo.setToolTip(
            "Bright detects segments higher than nearby scan lines. "
            "Dark detects segments lower than nearby scan lines."
        )
        polarity_row.addWidget(polarity_label)
        polarity_row.addWidget(self._bad_line_polarity_combo, 1)
        blo.addLayout(polarity_row)
        threshold_row = QHBoxLayout()
        threshold_row.setContentsMargins(0, 0, 0, 0)
        threshold_label = QLabel("Threshold:")
        threshold_label.setFont(QFont("Helvetica", 8))
        self._bad_line_threshold_spin = QDoubleSpinBox()
        self._bad_line_threshold_spin.setRange(0.5, 50.0)
        self._bad_line_threshold_spin.setSingleStep(0.5)
        self._bad_line_threshold_spin.setDecimals(1)
        self._bad_line_threshold_spin.setValue(5.0)
        self._bad_line_threshold_spin.setToolTip(
            "Robust z-like multiplier. Higher values detect fewer segments; "
            "lower values are more aggressive."
        )
        threshold_row.addWidget(threshold_label)
        threshold_row.addWidget(self._bad_line_threshold_spin, 1)
        blo.addLayout(threshold_row)
        min_len_row = QHBoxLayout()
        min_len_row.setContentsMargins(0, 0, 0, 0)
        min_len_label = QLabel("Min length:")
        min_len_label.setFont(QFont("Helvetica", 8))
        self._bad_line_min_len_spin = QDoubleSpinBox()
        self._bad_line_min_len_spin.setRange(1, 512)
        self._bad_line_min_len_spin.setSingleStep(1)
        self._bad_line_min_len_spin.setDecimals(0)
        self._bad_line_min_len_spin.setValue(2)
        self._bad_line_min_len_spin.setSuffix(" px")
        self._bad_line_min_len_spin.setToolTip(
            "Shortest run of neighbouring pixels along the fast-scan direction "
            "that can be treated as a bad segment."
        )
        min_len_row.addWidget(min_len_label)
        min_len_row.addWidget(self._bad_line_min_len_spin, 1)
        blo.addLayout(min_len_row)
        adj_row = QHBoxLayout()
        adj_row.setContentsMargins(0, 0, 0, 0)
        adj_label = QLabel("Max adjacent:")
        adj_label.setFont(QFont("Helvetica", 8))
        self._bad_line_adjacent_spin = QDoubleSpinBox()
        self._bad_line_adjacent_spin.setRange(1, 32)
        self._bad_line_adjacent_spin.setSingleStep(1)
        self._bad_line_adjacent_spin.setDecimals(0)
        self._bad_line_adjacent_spin.setValue(1)
        self._bad_line_adjacent_spin.setSuffix(" lines")
        self._bad_line_adjacent_spin.setToolTip(
            "Maximum neighbouring scan lines to repair as a local artifact. "
            "Broader adjacent groups are previewed but skipped."
        )
        adj_row.addWidget(adj_label)
        adj_row.addWidget(self._bad_line_adjacent_spin, 1)
        blo.addLayout(adj_row)
        preview_row = QHBoxLayout()
        preview_row.setContentsMargins(0, 0, 0, 0)
        self._bad_line_preview_btn = QPushButton("Preview detection")
        self._bad_line_preview_btn.setFont(QFont("Helvetica", 8))
        self._bad_line_preview_btn.setToolTip(
            "Highlight candidate bad scan-line segments without modifying data."
        )
        preview_row.addWidget(self._bad_line_preview_btn)
        blo.addLayout(preview_row)
        self._bad_line_preview_lbl = QLabel("Preview: not run")
        self._bad_line_preview_lbl.setFont(QFont("Helvetica", 8))
        self._bad_line_preview_lbl.setWordWrap(True)
        blo.addWidget(self._bad_line_preview_lbl)
        lay.addWidget(self._bad_line_options)
        self._bad_lines_combo.currentIndexChanged.connect(
            self._sync_bad_line_controls)
        self._bad_line_threshold_spin.valueChanged.connect(
            lambda _value: self.bad_line_preview_settings_changed.emit())
        self._bad_line_polarity_combo.currentIndexChanged.connect(
            lambda _index: self.bad_line_preview_settings_changed.emit())
        self._bad_line_min_len_spin.valueChanged.connect(
            lambda _value: self.bad_line_preview_settings_changed.emit())
        self._bad_line_adjacent_spin.valueChanged.connect(
            lambda _value: self.bad_line_preview_settings_changed.emit())
        self._bad_lines_combo.currentIndexChanged.connect(
            lambda _index: self.bad_line_preview_settings_changed.emit())
        self._bad_line_preview_btn.clicked.connect(
            self.bad_line_preview_requested.emit)
        self._sync_bad_line_controls()

        # ── Filter section (left column / full-width in 1-col) ────────────────
        self._filter_section = QWidget()
        L = QVBoxLayout(self._filter_section)
        L.setContentsMargins(0, 0, 0, 0)
        L.setSpacing(3)

        _col_lbl("Filters", L)

        self._smooth_combo = _combo_row("Smooth:", ["None", "Gaussian"], L, 54)
        self._smooth_sigma_w, self._smooth_sigma_sl, _ = _sub_slider(
            "sigma:", 1, 20, 1, "{v}px")
        L.addWidget(self._smooth_sigma_w)
        self._smooth_sigma_w.setVisible(False)
        self._smooth_combo.currentIndexChanged.connect(
            lambda i: self._smooth_sigma_w.setVisible(i != 0))

        self._highpass_combo = _combo_row("Hi-pass:", ["None", "Gaussian"], L, 54)
        self._highpass_sigma_w, self._highpass_sigma_sl, _ = _sub_slider(
            "sigma:", 1, 80, 8, "{v}px")
        L.addWidget(self._highpass_sigma_w)
        self._highpass_sigma_w.setVisible(False)
        self._highpass_combo.setToolTip(
            "ImageJ-like high-pass: subtracts a broad Gaussian-blurred background."
        )
        self._highpass_combo.currentIndexChanged.connect(
            lambda i: self._highpass_sigma_w.setVisible(i != 0))

        self._edge_combo = _combo_row("Edge:", ["None", "Laplacian", "LoG", "DoG"], L, 54)
        self._edge_sigma_w, self._edge_sigma_sl, _ = _sub_slider(
            "sigma:", 1, 20, 1, "{v}px")
        L.addWidget(self._edge_sigma_w)
        self._edge_sigma_w.setVisible(False)
        self._edge_combo.currentIndexChanged.connect(
            lambda i: self._edge_sigma_w.setVisible(i != 0))

        _col_lbl("Radial FFT", L)

        self._fft_combo = _combo_row("Mode:", ["None", "Low-pass", "High-pass"], L, 54)
        self._fft_combo.setToolTip(
            "Simple global radial low/high-pass filter. "
            "This is not the ImageJ Periodic Filter workflow."
        )
        self._fft_cutoff_widget, self._fft_sl, _ = _sub_slider(
            "cutoff:", 1, 50, 10, "{v}%")
        L.addWidget(self._fft_cutoff_widget)
        self._fft_cutoff_widget.setVisible(False)
        self._fft_combo.currentIndexChanged.connect(
            lambda i: self._fft_cutoff_widget.setVisible(i != 0))

        self._fft_soft_cb = QCheckBox("Soft border")
        self._fft_soft_cb.setFont(QFont("Helvetica", 8))
        self._fft_soft_cb.setToolTip(
            "Cosine-taper the image edges before FFT to suppress ringing artefacts "
            "(ImageJ FFT_Soft_Border approach)."
        )
        L.addWidget(self._fft_soft_cb)
        L.addStretch()

        # ── Background section (right column / below filters in 1-col) ────────
        self._bg_section = QWidget()
        R = QVBoxLayout(self._bg_section)
        R.setContentsMargins(0, 0, 0, 0)
        R.setSpacing(3)

        _col_lbl("Background", R)
        self._stm_background_btn = QPushButton("STM Background...")
        self._stm_background_btn.setFont(QFont("Helvetica", 8))
        self._stm_background_btn.setToolTip(
            "Open the ImageJ-style scan-line background tool with profile and image previews."
        )
        self._stm_background_btn.clicked.connect(self.stm_background_requested.emit)
        R.addWidget(self._stm_background_btn)
        R.addStretch()

        # ── Two-column container (2-col mode: filter | bg side by side) ───────
        self._two_col = QWidget()
        hbox = QHBoxLayout(self._two_col)
        hbox.setContentsMargins(0, 2, 0, 0)
        hbox.setSpacing(4)
        hbox.addWidget(self._filter_section, 1)
        hbox.addWidget(self._bg_section, 1)
        lay.addWidget(self._two_col)
        self._using_two_col = True

    def state(self) -> dict:
        align_map = {0: None, 1: "median", 2: "mean"}
        bad_map = {0: None, 1: "step", 2: "mad"}
        cfg = {
            "align_rows": align_map[self._align_combo.currentIndex()],
            "remove_bad_lines": bad_map[self._bad_lines_combo.currentIndex()],
        }
        if self._mode == "browse_quick":
            return {k: cfg[k] for k in self.QUICK_KEYS}

        fft_map = {0: None, 1: "low_pass", 2: "high_pass"}
        edge_map = {0: None, 1: "laplacian", 2: "log", 3: "dog"}
        smooth_i = self._smooth_combo.currentIndex()
        highpass_i = self._highpass_combo.currentIndex()
        edge_i = self._edge_combo.currentIndex()
        fft_idx = self._fft_combo.currentIndex()
        cfg.update({
            "remove_bad_lines_threshold": (
                self._bad_line_threshold_spin.value()
                if cfg["remove_bad_lines"] is not None else None
            ),
            "remove_bad_lines_polarity": (
                "bright" if self._bad_line_polarity_combo.currentIndex() == 0 else "dark"
            ),
            "remove_bad_lines_min_segment_length_px": int(
                self._bad_line_min_len_spin.value()
            ),
            "remove_bad_lines_max_adjacent_bad_lines": int(
                self._bad_line_adjacent_spin.value()
            ),
            "smooth_sigma": self._smooth_sigma_sl.value() if smooth_i != 0 else None,
            "highpass_sigma": self._highpass_sigma_sl.value() if highpass_i != 0 else None,
            "edge_method": edge_map[edge_i],
            "edge_sigma": self._edge_sigma_sl.value(),
            "edge_sigma2": self._edge_sigma_sl.value() * 2,
            "fft_mode": fft_map[fft_idx],
            "fft_cutoff": self._fft_sl.value() / 100.0,
            "fft_window": "hanning",
            "fft_soft_border": self._fft_soft_cb.isChecked(),
            "fft_soft_mode": fft_map.get(fft_idx) or "low_pass",
            "fft_soft_cutoff": self._fft_sl.value() / 100.0,
            "fft_soft_border_frac": 0.12,
        })
        return cfg

    def set_state(self, state: dict | None) -> None:
        state = state or {}
        self._align_combo.setCurrentIndex(
            {None: 0, "median": 1, "mean": 2}.get(state.get("align_rows"), 0))
        self._bad_lines_combo.setCurrentIndex(
            {None: 0, "step": 1, "step_segments": 1,
             "mad": 2, "mad_segments": 2}.get(state.get("remove_bad_lines"), 0))
        if self._mode == "browse_quick":
            return
        threshold = state.get("remove_bad_lines_threshold")
        if threshold is None:
            threshold = state.get("threshold_mad", 5.0)
        self._bad_line_threshold_spin.setValue(float(threshold))
        self._bad_line_polarity_combo.setCurrentIndex(
            {"bright": 0, "dark": 1}.get(
                state.get("remove_bad_lines_polarity", "bright"), 0))
        self._bad_line_min_len_spin.setValue(float(
            state.get("remove_bad_lines_min_segment_length_px", 2)))
        self._bad_line_adjacent_spin.setValue(float(
            state.get("remove_bad_lines_max_adjacent_bad_lines", 1)))

        sigma = state.get("smooth_sigma")
        if sigma:
            self._smooth_combo.setCurrentIndex(1)
            self._smooth_sigma_sl.setValue(int(sigma))
        else:
            self._smooth_combo.setCurrentIndex(0)

        highpass = state.get("highpass_sigma")
        if highpass:
            self._highpass_combo.setCurrentIndex(1)
            self._highpass_sigma_sl.setValue(int(highpass))
        else:
            self._highpass_combo.setCurrentIndex(0)

        edge = state.get("edge_method")
        self._edge_combo.setCurrentIndex(
            {None: 0, "laplacian": 1, "log": 2, "dog": 3}.get(edge, 0))
        self._edge_sigma_sl.setValue(int(state.get("edge_sigma", 1)))

        fft_mode = state.get("fft_mode")
        self._fft_combo.setCurrentIndex(
            {None: 0, "low_pass": 1, "high_pass": 2}.get(fft_mode, 0))
        self._fft_sl.setValue(int(round(float(state.get("fft_cutoff", 0.10)) * 100)))
        self._fft_soft_cb.setChecked(bool(state.get("fft_soft_border", False)))

    def bad_line_method(self) -> str | None:
        return self.state().get("remove_bad_lines")

    def bad_line_threshold(self) -> float:
        if not hasattr(self, "_bad_line_threshold_spin"):
            return 5.0
        return float(self._bad_line_threshold_spin.value())

    def bad_line_polarity(self) -> str:
        if not hasattr(self, "_bad_line_polarity_combo"):
            return "bright"
        return "bright" if self._bad_line_polarity_combo.currentIndex() == 0 else "dark"

    def bad_line_min_segment_length_px(self) -> int:
        if not hasattr(self, "_bad_line_min_len_spin"):
            return 2
        return int(self._bad_line_min_len_spin.value())

    def bad_line_max_adjacent_bad_lines(self) -> int:
        if not hasattr(self, "_bad_line_adjacent_spin"):
            return 1
        return int(self._bad_line_adjacent_spin.value())

    def set_bad_line_preview_summary(self, text: str) -> None:
        if hasattr(self, "_bad_line_preview_lbl"):
            self._bad_line_preview_lbl.setText(str(text))

    def _sync_bad_line_controls(self) -> None:
        if not hasattr(self, "_bad_line_options"):
            return
        enabled = self._bad_lines_combo.currentIndex() != 0
        self._bad_line_options.setVisible(enabled)
        self._bad_line_threshold_spin.setEnabled(enabled)
        self._bad_line_polarity_combo.setEnabled(enabled)
        self._bad_line_min_len_spin.setEnabled(enabled)
        self._bad_line_adjacent_spin.setEnabled(enabled)
        self._bad_line_preview_btn.setEnabled(enabled)
        if not enabled:
            self.set_bad_line_preview_summary("Preview: select a method")
