"""Feature-to-lattice comparison dialog for ProbeFlow."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
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

from probeflow.analysis.feature_lattice import (
    FeatureLatticeComparison,
    compare_features_to_lattice,
    default_match_radius,
)
from probeflow.analysis.simple_measurements import _fmt_m
from probeflow.measurements.models import MeasurementResult


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(1)
    return line


class FeatureLatticeDialog(QDialog):
    """Compare detected feature points to an active lattice grid.

    Parameters
    ----------
    sources
        ``{name: (N,2) array of (x,y) in pixel coordinates}`` — available
        feature-point sources.
    lattice_origin_px, a_px, b_px
        Lattice parameters from the active ``LatticeGrid``.
    pixel_size_x_m, pixel_size_y_m
        Physical pixel dimensions (metres) — used to format displacements.
    image_shape
        (H, W) for occupancy estimation.
    on_add_result
        Callback called with a ``MeasurementResult`` when user clicks "Add".
    """

    def __init__(
        self,
        sources: dict[str, np.ndarray],
        *,
        lattice_origin_px: tuple[float, float],
        a_px: tuple[float, float],
        b_px: tuple[float, float],
        pixel_size_x_m: float = 1e-10,
        pixel_size_y_m: float = 1e-10,
        image_shape: tuple[int, int] | None = None,
        source_label: str = "",
        source_path: str | None = None,
        channel: str = "",
        source_metadata: dict[str, dict[str, object]] | None = None,
        on_add_result: Callable[[MeasurementResult], None] | None = None,
        theme: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Feature-to-lattice comparison")
        self.resize(960, 620)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._sources = sources
        self._origin = lattice_origin_px
        self._a = a_px
        self._b = b_px
        self._px_x_m = float(pixel_size_x_m)
        self._px_y_m = float(pixel_size_y_m)
        self._image_shape = image_shape
        self._source_label = source_label
        self._source_path = source_path
        self._channel = channel
        self._source_metadata = source_metadata or {}
        self._on_add_result = on_add_result
        self._t = theme or {}
        self._comparison: FeatureLatticeComparison | None = None
        self._default_radius = default_match_radius(a_px, b_px)

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
        self._canvas.setMinimumWidth(420)
        main.addWidget(self._canvas, 1)

        # Right: sidebar.
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
        src_lbl = QLabel("Feature source")
        src_lbl.setFont(QFont("Helvetica", 8))
        lay.addWidget(src_lbl)
        self._src_cb = QComboBox()
        self._src_cb.addItems(list(self._sources.keys()) or ["(no points)"])
        lay.addWidget(self._src_cb)

        # Lattice status.
        ox, oy = self._origin
        ax, ay = self._a
        bx, by = self._b
        a_nm = math.hypot(ax * self._px_x_m, ay * self._px_y_m) * 1e9
        b_nm = math.hypot(bx * self._px_x_m, by * self._px_y_m) * 1e9
        lat_info = QLabel(
            f"Lattice: a={a_nm:.3g} nm  b={b_nm:.3g} nm"
        )
        lat_info.setFont(QFont("Helvetica", 8))
        lat_info.setWordWrap(True)
        lay.addWidget(lat_info)

        lay.addWidget(_sep())

        param_form = QFormLayout()
        param_form.setContentsMargins(0, 0, 0, 0)
        param_form.setSpacing(4)

        self._radius_sb = QDoubleSpinBox()
        self._radius_sb.setRange(0.0, 1e6)
        self._radius_sb.setValue(0.0)
        self._radius_sb.setSuffix(" px")
        self._radius_sb.setDecimals(2)
        self._radius_sb.setSpecialValueText("Auto")
        self._radius_sb.setToolTip(
            f"Match radius in pixels. Auto = {self._default_radius:.2f} px "
            f"(0.35 × min(|a|, |b|))."
        )
        param_form.addRow("Match radius:", self._radius_sb)
        lay.addLayout(param_form)

        compare_btn = QPushButton("Compare")
        compare_btn.setDefault(False)
        compare_btn.setAutoDefault(False)
        compare_btn.clicked.connect(self._run)
        lay.addWidget(compare_btn)

        lay.addWidget(_sep())

        self._result_lbl = QLabel("—")
        self._result_lbl.setFont(QFont("Helvetica", 8))
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self._result_lbl)

        self._overlay_cb = QCheckBox("Show displacement overlay")
        self._overlay_cb.setFont(QFont("Helvetica", 8))
        self._overlay_cb.setChecked(True)
        self._overlay_cb.toggled.connect(self._refresh_plot)
        lay.addWidget(self._overlay_cb)

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

        r_px = self._radius_sb.value()
        match_r = r_px if r_px > 0 else self._default_radius

        try:
            comp = compare_features_to_lattice(
                pts, self._origin, self._a, self._b,
                match_radius_px=match_r,
                image_shape=self._image_shape,
                pixel_size_x_m=self._px_x_m,
                pixel_size_y_m=self._px_y_m,
            )
        except ValueError as exc:
            self._result_lbl.setText(f"Error: {exc}")
            return

        self._comparison = comp
        self._update_result_text(comp)
        self._refresh_plot()
        self._add_btn.setEnabled(True)

    def _refresh_plot(self) -> None:
        if self._comparison is None:
            return
        self._update_plot(
            self._comparison,
            show_overlay=self._overlay_cb.isChecked(),
        )

    def _update_plot(
        self, comp: FeatureLatticeComparison, *, show_overlay: bool
    ) -> None:
        ax = self._ax
        ax.cla()
        fg = self._t.get("text.color", "#000000")
        bg = self._t.get("figure.facecolor", "#ffffff")
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        ax.tick_params(colors=fg, labelsize=7)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x  (px)", color=fg, fontsize=8)
        ax.set_ylabel("y  (px)", color=fg, fontsize=8)
        ax.set_title("Feature-to-lattice", color=fg, fontsize=9)

        matched = [a for a in comp.assignments if a.matched]
        off = [a for a in comp.assignments if not a.matched]

        if matched:
            xs = [a.feature_xy_px[0] for a in matched]
            ys = [a.feature_xy_px[1] for a in matched]
            ax.scatter(xs, ys, s=18, c="#4a90d9", zorder=3, label="Matched")

        if off:
            xs = [a.feature_xy_px[0] for a in off]
            ys = [a.feature_xy_px[1] for a in off]
            ax.scatter(xs, ys, s=18, c="#d94a4a", marker="x", zorder=3,
                       label="Off-lattice")

        if show_overlay and matched:
            for a in matched:
                fx, fy = a.feature_xy_px
                sx, sy = a.site_xy_px
                ax.plot([fx, sx], [fy, sy], color="#c0c000", lw=0.7,
                        alpha=0.6, zorder=2)
            sxs = [a.site_xy_px[0] for a in matched]
            sys_ = [a.site_xy_px[1] for a in matched]
            ax.scatter(sxs, sys_, s=8, c="#c0c000", marker="+", zorder=2,
                       alpha=0.7, label="Sites")

        if comp.assignments:
            ax.legend(fontsize=7, framealpha=0.6,
                      labelcolor=fg, facecolor=bg, edgecolor=fg)

        self._canvas.draw()

    def _update_result_text(self, comp: FeatureLatticeComparison) -> None:
        lines = [
            f"Features: {comp.n_features}",
            f"Matched: {comp.n_matched}",
            f"Off-lattice: {comp.n_off_lattice}",
            f"Duplicate sites: {comp.n_duplicate_sites}",
        ]
        if comp.rms_displacement_m is not None:
            v, u = _fmt_m(comp.rms_displacement_m)
            lines.append(f"RMS disp.: {v:.4g} {u}")
        if comp.occupancy is not None:
            lines.append(f"Occupancy: {comp.occupancy * 100:.1f} %")
        self._result_lbl.setText("\n".join(lines))

    # ── Add to table ───────────────────────────────────────────────────────────

    def _add_to_table(self) -> None:
        if self._comparison is None or self._on_add_result is None:
            return
        comp = self._comparison
        src_name = self._src_cb.currentText()

        summary_parts = [
            f"Features: {comp.n_features}",
            f"Matched: {comp.n_matched}",
            f"Off-lattice: {comp.n_off_lattice}",
        ]
        if comp.rms_displacement_m is not None:
            v, u = _fmt_m(comp.rms_displacement_m)
            summary_parts.append(f"RMS: {v:.4g} {u}")
        if comp.occupancy is not None:
            summary_parts.append(f"Occ: {comp.occupancy * 100:.1f}%")
        summary = "  ".join(summary_parts)

        values: dict = {
            "n_features": comp.n_features,
            "n_matched": comp.n_matched,
            "n_off_lattice": comp.n_off_lattice,
            "n_duplicate_sites": comp.n_duplicate_sites,
        }
        if comp.rms_displacement_m is not None:
            values["rms_displacement_m"] = comp.rms_displacement_m
        if comp.mean_displacement_m is not None:
            values["mean_displacement_m"] = comp.mean_displacement_m
        if comp.occupancy is not None:
            values["occupancy"] = comp.occupancy
        match_radius_px = (
            float(self._radius_sb.value())
            if self._radius_sb.value() > 0
            else float(self._default_radius)
        )
        context = {
            "point_source": src_name,
            "source_path": self._source_path,
            "match_radius_px": match_radius_px,
            "match_radius_mode": "manual" if self._radius_sb.value() > 0 else "auto",
            "lattice_origin_x_px": float(self._origin[0]),
            "lattice_origin_y_px": float(self._origin[1]),
            "lattice_a_x_px": float(self._a[0]),
            "lattice_a_y_px": float(self._a[1]),
            "lattice_b_x_px": float(self._b[0]),
            "lattice_b_y_px": float(self._b[1]),
            "pixel_size_x_m": self._px_x_m,
            "pixel_size_y_m": self._px_y_m,
            "image_shape_y": int(self._image_shape[0]) if self._image_shape else None,
            "image_shape_x": int(self._image_shape[1]) if self._image_shape else None,
            "occupancy_region": "image_bounds" if self._image_shape else "not_computed",
            "data_basis": "feature_points_pixel_lattice",
            "summary": summary,
        }
        for key, value in self._source_metadata.get(src_name, {}).items():
            if value is None:
                continue
            context_key = key if key.startswith("point_source_") else f"point_source_{key}"
            context[context_key] = value

        mr = MeasurementResult(
            measurement_id="M?",
            kind="feat_lattice",
            source_label=self._source_label,
            source_path=self._source_path,
            channel=self._channel,
            x_unit="px",
            y_unit="px",
            z_unit=None,
            values=values,
            context=context,
            notes=src_name,
        )
        self._on_add_result(mr)
