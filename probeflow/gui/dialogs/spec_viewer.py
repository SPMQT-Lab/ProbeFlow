from __future__ import annotations

from pathlib import Path

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from probeflow.gui.models import VertFile


class SpecViewerDialog(QDialog):
    """Full-size viewer for a spectroscopy file (Createc .VERT or Nanonis .dat).

    The viewer is channel-agnostic: it builds a toggleable list from
    ``spec.channel_order`` and stacks one subplot per selected channel.
    """

    # Dark-theme colours for plot elements.
    _BG = "#1e1e2e"
    _FG = "#cdd6f4"
    # Plot curve colours, cycled across selected channels.
    _COLORS = ("#89b4fa", "#a6e3a1", "#fab387", "#f5c2e7",
               "#94e2d5", "#f9e2af", "#cba6f7", "#f38ba8")

    def __init__(self, entry: VertFile, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(entry.stem)
        self.setMinimumSize(900, 560)
        self.resize(1100, 640)
        self._entry = entry
        self._t = t
        self._spec = None
        self._checkboxes: dict[str, QCheckBox] = {}
        self._channel_check_widgets: list[QCheckBox] = []
        self._canvas = None
        self._fig = None
        # Unit-override choice per base SI unit. "Auto" means use
        # choose_display_unit; otherwise lookup_unit_scale picks a fixed scale.
        self._unit_choice: dict[str, str] = {"m": "Auto", "A": "Auto", "V": "Auto"}
        self._build()
        self._load()

    # ── UI construction ─────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        self._title = QLabel(self._entry.stem)
        self._title.setFont(QFont("Helvetica", 12, QFont.Bold))
        self._title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._title)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: scrollable channel list.
        self._channels_panel = QWidget()
        self._channels_lay = QVBoxLayout(self._channels_panel)
        self._channels_lay.setContentsMargins(6, 6, 6, 6)
        self._channels_lay.setSpacing(4)
        ch_header = QLabel("Channels")
        ch_header.setFont(QFont("Helvetica", 10, QFont.Bold))
        self._channels_lay.addWidget(ch_header)

        # Unit-override selectors for height (Z) and current channels.
        unit_box = QGroupBox("Display units")
        unit_box.setFont(QFont("Helvetica", 9, QFont.Bold))
        unit_lay = QGridLayout(unit_box)
        unit_lay.setContentsMargins(6, 4, 6, 4)
        unit_lay.setSpacing(2)

        z_lbl = QLabel("Z:")
        z_lbl.setFont(QFont("Helvetica", 9))
        self._z_unit_cb = QComboBox()
        self._z_unit_cb.addItems(["Auto", "pm", "Å", "nm", "µm", "m"])
        self._z_unit_cb.setFont(QFont("Helvetica", 9))
        self._z_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("m", v))

        i_lbl = QLabel("I:")
        i_lbl.setFont(QFont("Helvetica", 9))
        self._i_unit_cb = QComboBox()
        self._i_unit_cb.addItems(["Auto", "fA", "pA", "nA", "µA", "A"])
        self._i_unit_cb.setFont(QFont("Helvetica", 9))
        self._i_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("A", v))

        v_lbl = QLabel("V:")
        v_lbl.setFont(QFont("Helvetica", 9))
        self._v_unit_cb = QComboBox()
        self._v_unit_cb.addItems(["Auto", "µV", "mV", "V"])
        self._v_unit_cb.setFont(QFont("Helvetica", 9))
        self._v_unit_cb.currentTextChanged.connect(
            lambda v: self._on_unit_changed("V", v))

        unit_lay.addWidget(z_lbl, 0, 0)
        unit_lay.addWidget(self._z_unit_cb, 0, 1)
        unit_lay.addWidget(i_lbl, 1, 0)
        unit_lay.addWidget(self._i_unit_cb, 1, 1)
        unit_lay.addWidget(v_lbl, 2, 0)
        unit_lay.addWidget(self._v_unit_cb, 2, 1)

        self._channels_lay.addWidget(unit_box)
        self._channels_lay.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self._channels_panel)
        left_scroll.setMinimumWidth(200)
        left_scroll.setMaximumWidth(300)
        splitter.addWidget(left_scroll)

        # Right panel: plot canvas.
        self._canvas_widget = QWidget()
        self._canvas_lay = QVBoxLayout(self._canvas_widget)
        self._canvas_lay.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._canvas_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        lay.addWidget(splitter, 1)

        self._status = QLabel("Loading…")
        self._status.setFont(QFont("Helvetica", 9))
        lay.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._raw_btn = QPushButton("Show raw data")
        self._raw_btn.setFixedWidth(140)
        self._raw_btn.clicked.connect(self._show_raw_data)
        btn_row.addWidget(self._raw_btn)

        self._export_csv_btn = QPushButton("Export CSV…")
        self._export_csv_btn.setFixedWidth(120)
        self._export_csv_btn.setToolTip(
            "Save the spectrum as a CSV file with one column per selected channel.")
        self._export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self._export_csv_btn)

        self._export_grace_btn = QPushButton("Export xmgrace…")
        self._export_grace_btn.setFixedWidth(160)
        self._export_grace_btn.setToolTip(
            "Render via xmgrace (Helvetica default). "
            "Produces three files in the chosen folder: "
            ".agr (re-editable Grace project), .png, and .pdf.")
        self._export_grace_btn.clicked.connect(self._export_xmgrace)
        btn_row.addWidget(self._export_grace_btn)

        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    # ── Data load + channel list population ─────────────────────────────

    def _load(self) -> None:
        from probeflow.io.spectroscopy import read_spec_file

        try:
            spec = read_spec_file(self._entry.path)
        except Exception as exc:
            self._status.setText(f"Error: {exc}")
            return
        self._spec = spec

        # Pull channel_order off the spec; fall back to whatever keys are
        # present for old SpecData objects that don't carry it.
        order = list(spec.channel_order) if spec.channel_order else list(spec.channels.keys())
        defaults = set(spec.default_channels)

        # Remove only channel checkboxes from prior loads. Static controls such
        # as the unit QGroupBox own combo boxes and must survive dialog lifetime.
        for w in self._channel_check_widgets:
            self._channels_lay.removeWidget(w)
            w.deleteLater()
        self._channel_check_widgets.clear()
        self._checkboxes.clear()
        for ch in order:
            if ch not in spec.channels:
                continue
            cb = QCheckBox(self._channel_display_label(ch))
            cb.setChecked(ch in defaults)
            cb.toggled.connect(self._redraw)
            self._channels_lay.insertWidget(self._channels_lay.count() - 1, cb)
            self._checkboxes[ch] = cb
            self._channel_check_widgets.append(cb)

        sweep = spec.metadata.get("sweep_type", "").replace("_", " ")
        n_pts = spec.metadata.get("n_points", 0)
        try:
            pos = spec.position
            pos_str = f"pos ({pos[0]*1e9:.2f}, {pos[1]*1e9:.2f}) nm"
        except (TypeError, IndexError, KeyError):
            pos_str = "pos unknown"
        self._status.setText(f"{sweep}  |  {n_pts} points  |  {pos_str}")

        self._redraw()

    # ── Plotting ────────────────────────────────────────────────────────

    def _on_unit_changed(self, base: str, label: str) -> None:
        self._unit_choice[base] = label
        self._refresh_channel_labels()
        self._redraw()

    def _display_values_for_channel(self, ch: str) -> tuple[np.ndarray, str]:
        if self._spec is None or ch not in self._spec.channels:
            return np.array([], dtype=float), ""
        from probeflow.analysis.spec_plot import choose_display_unit, lookup_unit_scale

        y = np.asarray(self._spec.channels[ch], dtype=float)
        unit = self._spec.y_units.get(ch, "")
        choice = self._unit_choice.get(unit, "Auto")
        override = lookup_unit_scale(unit, choice) if choice != "Auto" else None
        if override is not None:
            scale, disp_unit = override
        else:
            scale, disp_unit = choose_display_unit(unit, y)
        return y * scale, disp_unit

    def _channel_display_label(self, ch: str) -> str:
        _values, disp_unit = self._display_values_for_channel(ch)
        label = self._channel_label(ch)
        return f"{label}  ({disp_unit})" if disp_unit else label

    def _channel_label(self, ch: str) -> str:
        if self._spec is None:
            return ch
        info = getattr(self._spec, "channel_info", {}).get(ch)
        return getattr(info, "display_label", ch) if info is not None else ch

    def _refresh_channel_labels(self) -> None:
        for ch, cb in self._checkboxes.items():
            cb.setText(self._channel_display_label(ch))

    def _redraw(self) -> None:
        if self._spec is None:
            return
        # Remove existing canvas if present.
        if self._canvas is not None:
            self._canvas_lay.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas = None
            self._fig = None

        selected = [ch for ch, cb in self._checkboxes.items() if cb.isChecked()]
        if not selected:
            # Empty figure keeps the area from collapsing.
            fig = Figure(figsize=(8.5, 4.5), tight_layout=True)
            fig.patch.set_facecolor(self._BG)
            canvas = FigureCanvasQTAgg(fig)
            self._canvas_lay.addWidget(canvas)
            self._canvas = canvas
            self._fig = fig
            return

        fig = Figure(figsize=(8.5, 4.5), tight_layout=True)
        fig.patch.set_facecolor(self._BG)
        axes = fig.subplots(nrows=len(selected), ncols=1, sharex=True)
        if len(selected) == 1:
            axes = [axes]

        spec = self._spec
        for i, (ch, ax) in enumerate(zip(selected, axes)):
            y_disp, disp_unit = self._display_values_for_channel(ch)

            ax.set_facecolor(self._BG)
            ax.plot(spec.x_array, y_disp, linewidth=1.0,
                    color=self._COLORS[i % len(self._COLORS)])
            label = self._channel_label(ch)
            ax.set_ylabel(f"{label} ({disp_unit})" if disp_unit else label,
                          color=self._FG, fontsize=8)
            ax.tick_params(colors=self._FG, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(self._FG)
                spine.set_linewidth(0.6)
            if i < len(selected) - 1:
                ax.tick_params(axis="x", labelbottom=False)

        axes[-1].set_xlabel(spec.x_label, color=self._FG, fontsize=8)

        canvas = FigureCanvasQTAgg(fig)
        self._canvas_lay.addWidget(canvas)
        self._canvas = canvas
        self._fig = fig

    # ── Export ──────────────────────────────────────────────────────────

    def _selected_channels_in_display_units(self):
        """Yield (ch, y_disp, disp_unit) for each ticked channel.

        Applies the user's unit override (or auto-pick) so the exported
        values match what's currently shown in the plot.
        """
        spec = self._spec
        if spec is None:
            return
        for ch, cb in self._checkboxes.items():
            if not cb.isChecked() or ch not in spec.channels:
                continue
            y_disp, disp_unit = self._display_values_for_channel(ch)
            yield ch, self._channel_label(ch), y_disp, disp_unit

    def _export_csv(self) -> None:
        if self._spec is None:
            self._status.setText("Nothing to export — spectrum failed to load.")
            return
        rows = list(self._selected_channels_in_display_units())
        if not rows:
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export spectrum CSV",
            str(Path.home() / f"{self._entry.stem}.csv"),
            "CSV files (*.csv)")
        if not out_path:
            return
        try:
            import csv
            x = self._spec.x_array
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow([self._spec.x_label]
                           + [f"{label} ({u})" if u else label
                              for _ch, label, _y, u in rows])
                for i in range(len(x)):
                    w.writerow([f"{x[i]:.10g}"]
                               + [f"{y[i]:.10g}" for _ch, _label, y, _u in rows])
            self._status.setText(f"CSV → {out_path}")
        except Exception as exc:
            self._status.setText(f"CSV export error: {exc}")

    def _export_xmgrace(self) -> None:
        if self._spec is None:
            self._status.setText("Nothing to export — spectrum failed to load.")
            return
        rows = list(self._selected_channels_in_display_units())
        if not rows:
            self._status.setText("Tick at least one channel before exporting.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output folder for xmgrace export",
            str(Path.home()))
        if not out_dir:
            return
        try:
            from probeflow.analysis.xmgrace_export import Curve, export_bundle
        except ImportError as exc:
            self._status.setText(f"xmgrace export unavailable: {exc}")
            return
        # Group all channels under one Y axis label if they share a unit;
        # otherwise we keep the legend per-channel and a generic "value" label.
        units = {u for _, _label, _y, u in rows}
        if len(units) == 1:
            y_label = f"value ({rows[0][3]})" if rows[0][3] else "value"
        else:
            y_label = "value"
        curves = [
            Curve(name=ch, y=y, legend=f"{label} ({u})" if u else label)
            for ch, label, y, u in rows
        ]
        try:
            paths = export_bundle(
                Path(out_dir),
                self._entry.stem,
                self._spec.x_array,
                curves,
                x_label=self._spec.x_label,
                y_label=y_label,
                title=self._entry.stem,
                subtitle="ProbeFlow xmgrace export",
                font="Helvetica",
            )
        except FileNotFoundError as exc:
            self._status.setText(f"Export error: {exc}")
            return
        except Exception as exc:
            self._status.setText(f"xmgrace failed: {exc}")
            return
        names = ", ".join(p.name for p in paths.values())
        self._status.setText(f"Exported to {out_dir}: {names}")

    # ── Raw-data table ──────────────────────────────────────────────────

    def _show_raw_data(self) -> None:
        if self._spec is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Raw data — {self._entry.stem}")
        dlg.resize(640, 400)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(8, 8, 8, 8)

        table = self._raw_data_table()
        v.addWidget(table)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        v.addLayout(btn_row)

        dlg.exec()

    def _raw_data_table(self) -> QTableWidget:
        spec = self._spec
        if spec is None:
            return QTableWidget(0, 0)
        order = [ch for ch in spec.channel_order if ch in spec.channels]
        if not order:
            order = list(spec.channels.keys())
        n_rows = len(spec.x_array)
        n_cols = 1 + len(order)

        table = QTableWidget(n_rows, n_cols)
        display_rows = [
            (ch, self._channel_label(ch), *self._display_values_for_channel(ch))
            for ch in order
        ]
        headers = [spec.x_label] + [
            f"{label} ({unit})" if unit else label
            for _ch, label, _values, unit in display_rows
        ]
        table.setHorizontalHeaderLabels(headers)

        for r in range(n_rows):
            table.setItem(r, 0, QTableWidgetItem(f"{spec.x_array[r]:.6g}"))
            for c, (_ch, _label, values, _unit) in enumerate(display_rows, start=1):
                table.setItem(r, c, QTableWidgetItem(f"{values[r]:.6g}"))
        table.horizontalHeader().setStretchLastSection(True)
        return table
