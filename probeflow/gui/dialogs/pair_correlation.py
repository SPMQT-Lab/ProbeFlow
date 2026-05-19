"""Pair-correlation dialog for ProbeFlow."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from probeflow.analysis.pair_correlation import PairCorrelationResult, compute_pair_correlation
from probeflow.measurements.models import MeasurementResult


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(1)
    return line


class PairCorrelationDialog(QDialog):
    """Compute g(r) from detected feature points or point ROIs.

    Parameters
    ----------
    sources
        ``{name: (N,2) array of (x,y) in metres}`` — one entry per available
        point source (e.g. "Feature result", "All point ROIs").
    roi_area_m2
        Area of the measurement region; used for g(r) normalisation.
    on_add_result
        Optional callback called with a ``MeasurementResult`` when the user
        clicks "Add to measurement table".
    """

    def __init__(
        self,
        sources: dict[str, np.ndarray],
        *,
        roi_area_m2: float | None = None,
        pixel_size_x_m: float = 1e-10,
        pixel_size_y_m: float = 1e-10,
        source_label: str = "",
        source_path: str | None = None,
        channel: str = "",
        on_add_result: Callable[[MeasurementResult], None] | None = None,
        theme: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pair correlation")
        self.resize(900, 600)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._sources = sources
        self._roi_area_m2 = roi_area_m2
        self._px_x_m = float(pixel_size_x_m)
        self._px_y_m = float(pixel_size_y_m)
        self._source_label = source_label
        self._source_path = source_path
        self._channel = channel
        self._on_add_result = on_add_result
        self._t = theme or {}
        self._result: PairCorrelationResult | None = None

        self._build()

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)

        # Left: matplotlib canvas.
        self._fig = Figure(figsize=(5, 4), tight_layout=True)
        bg = self._t.get("figure.facecolor", "#ffffff")
        self._fig.patch.set_facecolor(bg)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(bg)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumWidth(400)
        main.addWidget(self._canvas, 1)

        # Right: controls in a scroll area.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedWidth(260)
        sidebar = QWidget()
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)
        scroll.setWidget(sidebar)
        main.addWidget(scroll)

        # Source selector.
        src_lbl = QLabel("Source")
        src_lbl.setFont(QFont("Helvetica", 8))
        lay.addWidget(src_lbl)
        self._src_cb = QComboBox()
        self._src_cb.addItems(list(self._sources.keys()) or ["(no points available)"])
        lay.addWidget(self._src_cb)

        lay.addWidget(_sep())

        # Parameters.
        param_form = QFormLayout()
        param_form.setContentsMargins(0, 0, 0, 0)
        param_form.setSpacing(4)

        self._rmax_sb = QDoubleSpinBox()
        self._rmax_sb.setRange(0.0, 1e6)
        self._rmax_sb.setValue(0.0)
        self._rmax_sb.setSuffix(" nm")
        self._rmax_sb.setDecimals(2)
        self._rmax_sb.setSpecialValueText("Auto")
        self._rmax_sb.setToolTip("Max radius. 0 = auto.")
        param_form.addRow("Max r:", self._rmax_sb)

        self._bw_sb = QDoubleSpinBox()
        self._bw_sb.setRange(0.0, 1e6)
        self._bw_sb.setValue(0.0)
        self._bw_sb.setSuffix(" nm")
        self._bw_sb.setDecimals(3)
        self._bw_sb.setSpecialValueText("Auto")
        self._bw_sb.setToolTip("Bin width. 0 = auto.")
        param_form.addRow("Bin width:", self._bw_sb)

        lay.addLayout(param_form)

        compute_btn = QPushButton("Compute")
        compute_btn.setDefault(False)
        compute_btn.setAutoDefault(False)
        compute_btn.clicked.connect(self._run)
        lay.addWidget(compute_btn)

        lay.addWidget(_sep())

        # Results block.
        self._result_lbl = QLabel("—")
        self._result_lbl.setFont(QFont("Helvetica", 8))
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self._result_lbl)

        self._warning_lbl = QLabel("")
        self._warning_lbl.setFont(QFont("Helvetica", 8))
        self._warning_lbl.setWordWrap(True)
        self._warning_lbl.setStyleSheet("color: #c07000;")
        lay.addWidget(self._warning_lbl)

        lay.addWidget(_sep())

        self._add_btn = QPushButton("Add to measurement table")
        self._add_btn.setDefault(False)
        self._add_btn.setAutoDefault(False)
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._add_to_table)
        lay.addWidget(self._add_btn)

        lay.addStretch(1)

        if not self._sources:
            self._result_lbl.setText(
                "Run Feature finder or select point ROIs first."
            )

    # ── Compute ────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        src_name = self._src_cb.currentText()
        pts = self._sources.get(src_name)
        if pts is None or len(pts) == 0:
            self._result_lbl.setText("No points in selected source.")
            return

        r_max = self._rmax_sb.value()
        bw = self._bw_sb.value()

        result = compute_pair_correlation(
            pts,
            roi_area_m2=self._roi_area_m2,
            r_max_m=(r_max * 1e-9) if r_max > 0 else None,
            bin_width_m=(bw * 1e-9) if bw > 0 else None,
        )
        self._result = result
        self._update_plot(result)
        self._update_result_text(result)
        self._add_btn.setEnabled(result.quality != "failed")

    def _update_plot(self, result: PairCorrelationResult) -> None:
        ax = self._ax
        ax.cla()
        fg = self._t.get("text.color", "#000000")
        bg = self._t.get("figure.facecolor", "#ffffff")
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        ax.tick_params(colors=fg, labelsize=7)

        if len(result.r_m) == 0:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, color=fg, fontsize=9)
        else:
            r_nm = result.r_m * 1e9
            area_known = self._roi_area_m2 is not None
            ax.bar(r_nm, result.g_r, width=(r_nm[1] - r_nm[0]) if len(r_nm) > 1 else 0.1,
                   align="center", color="#4a90d9", alpha=0.8)
            ax.set_xlabel("r  (nm)", color=fg, fontsize=8)
            ylabel = "g(r)" if area_known else "pair count"
            ax.set_ylabel(ylabel, color=fg, fontsize=8)
            ax.set_title("Pair correlation", color=fg, fontsize=9)
            if area_known:
                ax.axhline(1.0, color=fg, lw=0.8, linestyle="--", alpha=0.5)
            if result.first_peak_m is not None:
                ax.axvline(result.first_peak_m * 1e9, color="#d94a4a",
                           lw=1.0, linestyle=":", alpha=0.8,
                           label=f"Peak: {result.first_peak_m * 1e9:.3g} nm")
                ax.legend(fontsize=7, framealpha=0.6,
                          labelcolor=fg, facecolor=bg, edgecolor=fg)

        self._canvas.draw()

    def _update_result_text(self, result: PairCorrelationResult) -> None:
        lines = [f"Points: {result.n_points}"]
        if result.density_m2 is not None:
            d_nm2 = result.density_m2 * 1e-18
            lines.append(f"Density: {d_nm2:.4g} nm⁻²")
        if result.nearest_neighbour_median_m is not None:
            nn_nm = result.nearest_neighbour_median_m * 1e9
            lines.append(f"NN median: {nn_nm:.4g} nm")
        if result.first_peak_m is not None:
            pk_nm = result.first_peak_m * 1e9
            lines.append(f"First peak: {pk_nm:.4g} nm")
        lines.append(f"Quality: {result.quality}")
        self._result_lbl.setText("\n".join(lines))
        self._warning_lbl.setText(result.message if result.message else "")

    # ── Add to table ───────────────────────────────────────────────────────────

    def _add_to_table(self) -> None:
        if self._result is None or self._on_add_result is None:
            return
        r = self._result
        src_name = self._src_cb.currentText()
        lines = [f"Points: {r.n_points}"]
        if r.nearest_neighbour_median_m is not None:
            lines.append(f"NN: {r.nearest_neighbour_median_m * 1e9:.4g} nm")
        if r.first_peak_m is not None:
            lines.append(f"Peak: {r.first_peak_m * 1e9:.4g} nm")
        lines.append(f"Quality: {r.quality}")
        summary = "  ".join(lines)

        values: dict = {"n_points": r.n_points}
        if r.density_m2 is not None:
            values["density_nm2"] = r.density_m2 * 1e-18
        if r.nearest_neighbour_median_m is not None:
            values["nn_median_nm"] = r.nearest_neighbour_median_m * 1e9
        if r.first_peak_m is not None:
            values["first_peak_nm"] = r.first_peak_m * 1e9
        values["quality"] = r.quality
        bin_width_m = None
        if len(r.r_m) > 1:
            bin_width_m = float(r.r_m[1] - r.r_m[0])
        elif self._bw_sb.value() > 0:
            bin_width_m = float(self._bw_sb.value() * 1e-9)
        r_max_m = None
        if len(r.r_m) > 0:
            r_max_m = float(r.r_m[-1] + 0.5 * (bin_width_m or 0.0))
        elif self._rmax_sb.value() > 0:
            r_max_m = float(self._rmax_sb.value() * 1e-9)
        context = {
            "point_source": src_name,
            "source_path": self._source_path,
            "roi_area_m2": self._roi_area_m2,
            "r_max_m": r_max_m,
            "r_max_mode": "manual" if self._rmax_sb.value() > 0 else "auto",
            "bin_width_m": bin_width_m,
            "bin_width_mode": "manual" if self._bw_sb.value() > 0 else "auto",
            "pixel_size_x_m": self._px_x_m,
            "pixel_size_y_m": self._px_y_m,
            "edge_correction": "not_applied",
            "message": r.message,
            "data_basis": "feature_points_physical",
            "summary": summary,
        }

        result = MeasurementResult(
            measurement_id="M?",
            kind="pair_corr",
            source_label=self._source_label,
            source_path=self._source_path,
            channel=self._channel,
            x_unit="nm",
            y_unit=None,
            z_unit=None,
            values=values,
            context=context,
            notes=src_name,
        )
        self._on_add_result(result)
