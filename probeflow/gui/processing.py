"""GUI processing controls and processing-state adapter compatibility."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QSizePolicy, QSlider,
    QVBoxLayout, QWidget,
)

from probeflow.processing.gui_adapter import *


class ProcessingControlPanel(QWidget):
    """Internal processing controls shared by Browse and Viewer."""

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
            "Bad lines:", ["None", "MAD (rows)", "Step (cols)"]
        )
        self._bad_lines_combo.setToolTip(
            "MAD: flags rows whose median deviates by >5× MAD from the image median.\n"
            "Step: per-column scan for large vertical jumps; detects partial bad lines."
        )

        if self._mode == "browse_quick":
            lay.addStretch()
            return

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

        self._bg_combo = _combo_row(
            "Order:",
            ["None", "Plane", "Quad.", "Cubic", "Quart."],
            R, 46,
        )
        self._bg_step_cb = QCheckBox("Step-tolerant")
        self._bg_step_cb.setFont(QFont("Helvetica", 8))
        self._bg_step_cb.setToolTip(
            "Ignores steep pixels during polynomial surface fitting. "
            "This is not the STM line-background algorithm."
        )
        R.addWidget(self._bg_step_cb)

        self._stm_line_bg_combo = _combo_row(
            "STM line:",
            ["None", "Step-tol."],
            R, 54,
        )

        self._facet_cb = QCheckBox("Facet level")
        self._facet_cb.setFont(QFont("Helvetica", 8))
        self._facet_cb.setToolTip(
            "Level each atomically flat terrace to a common height reference."
        )
        R.addWidget(self._facet_cb)
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
        bad_map = {0: None, 1: "mad", 2: "step"}
        cfg = {
            "align_rows": align_map[self._align_combo.currentIndex()],
            "remove_bad_lines": bad_map[self._bad_lines_combo.currentIndex()],
        }
        if self._mode == "browse_quick":
            return {k: cfg[k] for k in self.QUICK_KEYS}

        bg_map = {0: None, 1: 1, 2: 2, 3: 3, 4: 4}
        fft_map = {0: None, 1: "low_pass", 2: "high_pass"}
        edge_map = {0: None, 1: "laplacian", 2: "log", 3: "dog"}
        smooth_i = self._smooth_combo.currentIndex()
        highpass_i = self._highpass_combo.currentIndex()
        edge_i = self._edge_combo.currentIndex()
        fft_idx = self._fft_combo.currentIndex()
        cfg.update({
            "bg_order": bg_map[self._bg_combo.currentIndex()],
            "bg_step_tolerance": self._bg_step_cb.isChecked(),
            "stm_line_bg": (
                "step_tolerant"
                if self._stm_line_bg_combo.currentIndex() == 1
                else None
            ),
            "facet_level": self._facet_cb.isChecked(),
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
            {None: 0, "mad": 1, "step": 2}.get(state.get("remove_bad_lines"), 0))
        if self._mode == "browse_quick":
            return

        self._bg_combo.setCurrentIndex(
            {None: 0, 1: 1, 2: 2, 3: 3, 4: 4}.get(state.get("bg_order"), 0))
        self._bg_step_cb.setChecked(bool(state.get("bg_step_tolerance", False)))
        self._stm_line_bg_combo.setCurrentIndex(
            {"step_tolerant": 1}.get(state.get("stm_line_bg"), 0))
        self._facet_cb.setChecked(bool(state.get("facet_level", False)))

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
