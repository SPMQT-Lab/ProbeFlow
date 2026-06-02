"""Diagnostic plot dialog for the line-profile periodicity tool."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QPushButton, QVBoxLayout

from probeflow.analysis.line_periodicity import PeriodicityDiagnostic, PeriodicityResult, format_period


class PeriodicityPlotDialog(QDialog):
    """Two-panel diagnostic plot: profile + method-specific analysis."""

    def __init__(
        self,
        result: PeriodicityResult,
        diag: PeriodicityDiagnostic,
        *,
        theme: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Line periodicity — diagnostic plot")
        self.resize(900, 480)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._result = result
        self._diag = diag
        self._theme = theme or {}
        self._build()
        self._draw()

    def update_plot(self, result: PeriodicityResult, diag: PeriodicityDiagnostic) -> None:
        self._result = result
        self._diag = diag
        self._draw()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self._fig = Figure(figsize=(9.0, 4.2), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        lay.addWidget(self._canvas, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        export_btn = QPushButton("Export PNG")
        export_btn.setDefault(False)
        export_btn.setAutoDefault(False)
        export_btn.clicked.connect(self._export_png)
        close_btn = QPushButton("Close")
        close_btn.setDefault(False)
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        row.addWidget(export_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)

    def _draw(self) -> None:
        self._fig.clear()
        bg = self._theme.get("bg", "#ffffff")
        fg = self._theme.get("fg", "#111111")
        accent = self._theme.get("accent", "#1f77b4")
        self._fig.patch.set_facecolor(bg)

        ax_prof, ax_diag = self._fig.subplots(1, 2)
        for ax in (ax_prof, ax_diag):
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)

        self._draw_profile(ax_prof, fg, accent)
        method = self._result.method
        if method == "autocorrelation":
            self._draw_autocorr(ax_diag, fg, accent)
        elif method == "peak_spacing":
            self._draw_peak_spacing(ax_diag, fg, accent)
        else:
            self._draw_fft(ax_diag, fg, accent)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

    def _draw_profile(self, ax, fg, accent) -> None:
        diag = self._diag
        s_nm = diag.s_m * 1e9
        z_raw = np.asarray(diag.z_raw, dtype=float)
        z_proc = np.asarray(diag.z_processed, dtype=float)

        # The raw trace keeps the full absolute-height offset and tilt, while the
        # processed trace is background-removed and typically orders of magnitude
        # smaller. Sharing one y-axis lets the raw offset dominate the autoscale
        # and flattens the corrugation we actually want to see, so give the raw
        # trace its own twin axis whenever it differs from the processed one.
        show_raw_twin = (
            z_raw.shape == z_proc.shape
            and z_raw.size > 0
            and not np.allclose(z_raw, z_proc, equal_nan=True)
        )

        handles: list = []
        proc_line, = ax.plot(s_nm, z_proc, color=accent, lw=1.2, label="processed")
        handles.append(proc_line)

        # Mark detected peaks on the processed (analysed) trace.
        if diag.peak_positions_m is not None and len(diag.peak_positions_m) > 0:
            interp_z = np.interp(diag.peak_positions_m, diag.s_m, z_proc)
            peak_line, = ax.plot(
                diag.peak_positions_m * 1e9, interp_z, "v", color="tab:orange",
                ms=5, zorder=5, label="peaks",
            )
            handles.append(peak_line)

        if show_raw_twin:
            ax2 = ax.twinx()
            ax2.set_facecolor(self._theme.get("bg", "#ffffff"))
            raw_line, = ax2.plot(s_nm, z_raw, color=fg, alpha=0.4, lw=0.8, label="raw")
            handles.append(raw_line)
            ax2.set_ylabel("z raw (data units)", fontsize=9, color=fg)
            ax2.tick_params(axis="y", colors=fg, labelsize=8)
            for spine in ax2.spines.values():
                spine.set_color(fg)
            ax.set_ylabel("z processed (data units)", fontsize=9)
        else:
            # Raw and processed coincide (no background removal): one trace, one
            # axis — no need for a second scale.
            if z_raw.size > 0:
                raw_line, = ax.plot(
                    s_nm, z_raw, color=fg, alpha=0.35, lw=0.8, label="raw")
                handles.append(raw_line)
            ax.set_ylabel("z (data units)", fontsize=9)

        # Period marker
        r = self._result
        if not math.isnan(r.period_m) and r.period_m > 0:
            val, unit = format_period(r.period_m)
            unc_str = ""
            if r.uncertainty_m is not None and not math.isnan(r.uncertainty_m):
                scale = 1e10 if unit == "Å" else 1e9
                unc_str = f" ± {r.uncertainty_m * scale:.2g}"
            ax.set_title(
                f"Profile  |  T = {val}{unc_str} {unit}  ({r.quality})",
                fontsize=9, color=fg,
            )
        else:
            ax.set_title(f"Profile  |  {r.quality}: {r.message[:60]}", fontsize=9, color=fg)

        ax.set_xlabel("Distance (nm)", fontsize=9)
        ax.legend(handles, [h.get_label() for h in handles], fontsize=7, framealpha=0.3)

    def _draw_autocorr(self, ax, fg, accent) -> None:
        diag = self._diag
        if diag.autocorr_lag_m is None or diag.autocorr is None:
            ax.set_title("Autocorrelation (no data)", fontsize=9)
            return

        lag_nm = diag.autocorr_lag_m * 1e9
        ax.plot(lag_nm, diag.autocorr, color=accent, lw=1.2)
        ax.axhline(0, color=fg, lw=0.5, alpha=0.4)

        r = self._result
        if not math.isnan(r.period_m):
            ax.axvline(r.period_m * 1e9, color="tab:orange", lw=1.2,
                       linestyle="--", label=f"T = {r.period_m * 1e9:.3g} nm")
            ax.legend(fontsize=7, framealpha=0.3)

        ax.set_title("Autocorrelation", fontsize=9)
        ax.set_xlabel("Lag (nm)", fontsize=9)
        ax.set_ylabel("Normalised autocorrelation", fontsize=9)

    def _draw_peak_spacing(self, ax, fg, accent) -> None:
        diag = self._diag
        s_nm = diag.s_m * 1e9
        ax.plot(s_nm, diag.z_processed, color=accent, lw=1.2)

        if diag.peak_positions_m is not None and len(diag.peak_positions_m) > 0:
            interp_z = np.interp(diag.peak_positions_m, diag.s_m, diag.z_processed)
            ax.plot(diag.peak_positions_m * 1e9, interp_z, "v", color="tab:orange",
                    ms=6, zorder=5, label=f"{len(diag.peak_positions_m)} peaks")
            if len(diag.peak_positions_m) >= 2:
                spacings = np.diff(diag.peak_positions_m) * 1e9
                med = float(np.median(spacings))
                ax.set_title(f"Peak spacing  |  median = {med:.3g} nm", fontsize=9)
            ax.legend(fontsize=7, framealpha=0.3)
        else:
            ax.set_title("Peak spacing (no peaks found)", fontsize=9)

        ax.set_xlabel("Distance (nm)", fontsize=9)
        ax.set_ylabel("z processed", fontsize=9)

    def _draw_fft(self, ax, fg, accent) -> None:
        diag = self._diag
        if diag.fft_freq_m_inv is None or diag.fft_power is None:
            ax.set_title("FFT power (no data)", fontsize=9)
            return

        freqs = diag.fft_freq_m_inv
        power = diag.fft_power
        # Plot as period (nm) on x-axis: only positive, finite periods
        valid = freqs > 0
        periods_nm = np.where(valid, 1.0 / freqs * 1e9, np.nan)

        ax.plot(periods_nm[valid], power[valid], color=accent, lw=1.0)

        r = self._result
        if not math.isnan(r.period_m):
            ax.axvline(r.period_m * 1e9, color="tab:orange", lw=1.2,
                       linestyle="--", label=f"T = {r.period_m * 1e9:.3g} nm")
            ax.legend(fontsize=7, framealpha=0.3)

        ax.set_title("FFT power spectrum", fontsize=9)
        ax.set_xlabel("Period (nm)", fontsize=9)
        ax.set_ylabel("Power", fontsize=9)

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export periodicity plot PNG",
            str(Path.home() / "probeflow_periodicity_plot.png"),
            "PNG files (*.png)",
        )
        if path:
            self._fig.savefig(path, dpi=160, bbox_inches="tight")
