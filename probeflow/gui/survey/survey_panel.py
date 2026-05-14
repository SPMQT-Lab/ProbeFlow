"""SurveyPanel — list features in a ScanFlow campaign, polish each, export PPTX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QSplitter, QGroupBox, QGridLayout, QFileDialog,
    QMessageBox, QScrollArea, QFrame,
)

log = logging.getLogger(__name__)


class SurveyPanel(QWidget):
    """Main Survey-mode widget. Loads a SurveyManifest and drives editing + export.

    Layout:
        ┌──────────────────────────────────────────────────────────┐
        │ Title bar: campaign name · timestamp · [Export PPTX…]    │
        ├──────────────────────────────────────────────────────────┤
        │ ┌────────────────┐ ┌────────────────────────────────┐    │
        │ │ Wide overview  │ │ Metadata for selected feature  │    │
        │ │ (annotated)    │ │ Vbias / Setpoint / Size / ...  │    │
        │ ├────────────────┤ │ Iteration residuals (Å)        │    │
        │ │ Feature list   │ │ [Process this feature…]        │    │
        │ │ #01 1.4 nm     │ │ [Save polished PNG]            │    │
        │ │ #02 2.8 nm     │ │ Status: raw / polished         │    │
        │ │ ...            │ │                                │    │
        │ └────────────────┘ └────────────────────────────────┘    │
        └──────────────────────────────────────────────────────────┘
    """

    log_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._manifest = None
        self._manifest_dir: Optional[Path] = None
        self._processed_dir: Optional[Path] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QHBoxLayout()
        self._title_label = QLabel("<i>No survey loaded — open a survey.json file</i>")
        self._title_label.setStyleSheet("font-size: 14pt;")
        header.addWidget(self._title_label, 1)

        self._open_btn = QPushButton("Open survey.json…")
        self._open_btn.clicked.connect(self._open_manifest_dialog)
        header.addWidget(self._open_btn)

        self._export_btn = QPushButton("Export PPTX…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_pptx)
        header.addWidget(self._export_btn)
        root.addLayout(header)

        # Body
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        # Left: overview image + feature list
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self._overview_label = QLabel()
        self._overview_label.setAlignment(Qt.AlignCenter)
        self._overview_label.setMinimumHeight(220)
        self._overview_label.setFrameShape(QFrame.StyledPanel)
        self._overview_label.setText("<i>Overview will appear here</i>")
        left_lay.addWidget(self._overview_label, 0)

        self._feature_list = QListWidget()
        self._feature_list.itemSelectionChanged.connect(self._on_select_feature)
        self._feature_list.itemDoubleClicked.connect(
            lambda _it: self._process_feature())
        left_lay.addWidget(self._feature_list, 1)
        splitter.addWidget(left)

        # Right: metadata + actions
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)

        meta_group = QGroupBox("Feature details")
        meta_grid = QGridLayout(meta_group)
        self._meta_labels: dict[str, QLabel] = {}
        rows = [
            ("Index", "index"),
            ("Position offset", "position"),
            ("Feature size", "feature_size"),
            ("Zoom frame", "zoom_size"),
            ("Bias", "bias"),
            ("Setpoint", "setpoint"),
            ("Iterations", "iterations"),
            ("Final residual", "residual"),
            ("Status", "status"),
        ]
        for r, (label, key) in enumerate(rows):
            meta_grid.addWidget(QLabel(f"<b>{label}</b>"), r, 0,
                                alignment=Qt.AlignTop)
            v = QLabel("—")
            v.setWordWrap(True)
            self._meta_labels[key] = v
            meta_grid.addWidget(v, r, 1)
        right_lay.addWidget(meta_group)

        drift_group = QGroupBox("Drift per iteration (Å)")
        drift_lay = QVBoxLayout(drift_group)
        self._drift_label = QLabel("—")
        self._drift_label.setWordWrap(True)
        drift_lay.addWidget(self._drift_label)
        right_lay.addWidget(drift_group)

        actions = QHBoxLayout()
        self._process_btn = QPushButton("Process this feature…")
        self._process_btn.setEnabled(False)
        self._process_btn.clicked.connect(self._process_feature)
        actions.addWidget(self._process_btn)

        self._save_polished_btn = QPushButton("Save polished PNG")
        self._save_polished_btn.setEnabled(False)
        self._save_polished_btn.setToolTip(
            "Render the .dat with the current ProbeFlow processing "
            "settings and store it as the slide image for this feature."
        )
        self._save_polished_btn.clicked.connect(self._save_polished_png)
        actions.addWidget(self._save_polished_btn)
        right_lay.addLayout(actions)

        right_lay.addStretch(1)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def load_manifest(self, manifest_path: Path) -> bool:
        """Load a ScanFlow survey.json and populate the panel."""
        try:
            from scanflow.automation import SurveyManifest
        except ImportError as e:
            self.log_message.emit(f"ScanFlow not importable: {e}")
            QMessageBox.critical(
                self, "ScanFlow missing",
                "ProbeFlow needs the scanflow package to open surveys.\n\n"
                f"pip install scanflow\n\nError: {e}"
            )
            return False

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            QMessageBox.warning(self, "Not found",
                                f"Manifest does not exist:\n{manifest_path}")
            return False

        try:
            manifest = SurveyManifest.load(manifest_path)
        except Exception as e:
            QMessageBox.critical(self, "Load error",
                                 f"Could not load manifest:\n{e}")
            return False

        self._manifest = manifest
        self._manifest_dir = manifest_path.parent
        self._processed_dir = self._manifest_dir / "processed"
        self._processed_dir.mkdir(exist_ok=True)

        self._title_label.setText(
            f"<b>{manifest.name}</b>  ·  {manifest.timestamp}  ·  "
            f"{len(manifest.features)} feature(s)"
        )

        # Overview
        overview_candidates = [
            self._manifest_dir / "wide_annotated.png",
            Path(manifest.wide_preview_path) if manifest.wide_preview_path else None,
        ]
        for c in overview_candidates:
            if c and Path(c).exists():
                pix = QPixmap(str(c))
                self._overview_label.setPixmap(
                    pix.scaledToHeight(280, Qt.SmoothTransformation))
                break
        else:
            self._overview_label.setText("<i>No overview image found</i>")

        # Features
        self._feature_list.clear()
        for f in manifest.features:
            polished = self._processed_dir / f"feature_{f.index:02d}.png"
            status = "✓ polished" if polished.exists() else "raw"
            item = QListWidgetItem(
                f"#{f.index:02d}   size {f.char_dim_nm:.2f} nm   "
                f"zoom {f.zoom_size_nm[0]:.1f} nm   [{status}]"
            )
            item.setData(Qt.UserRole, f.index)
            self._feature_list.addItem(item)

        self._export_btn.setEnabled(True)
        if manifest.features:
            self._feature_list.setCurrentRow(0)

        self.log_message.emit(
            f"Loaded survey: {manifest.name} ({len(manifest.features)} feature(s))"
        )
        return True

    def _open_manifest_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open survey manifest", "",
            "ScanFlow survey (survey.json *.json)"
        )
        if path:
            self.load_manifest(Path(path))

    # ------------------------------------------------------------------
    # Per-feature actions
    # ------------------------------------------------------------------

    def _current_feature(self):
        if not self._manifest:
            return None
        row = self._feature_list.currentRow()
        if row < 0 or row >= len(self._manifest.features):
            return None
        return self._manifest.features[row]

    def _on_select_feature(self) -> None:
        rec = self._current_feature()
        if rec is None:
            return
        m = self._meta_labels
        m["index"].setText(str(rec.index))
        m["position"].setText(
            f"ΔX = {rec.centroid_nm_offset[0]:+.2f} nm,  "
            f"ΔY = {rec.centroid_nm_offset[1]:+.2f} nm"
        )
        m["feature_size"].setText(f"{rec.char_dim_nm:.2f} nm")
        m["zoom_size"].setText(
            f"{rec.zoom_size_nm[0]:.2f} × {rec.zoom_size_nm[1]:.2f} nm"
        )
        m["bias"].setText(f"{rec.bias_V:.4f} V")
        m["setpoint"].setText(f"{rec.setpoint_A * 1e12:.2f} pA")
        m["iterations"].setText(str(len(rec.scan_paths)))
        m["residual"].setText(
            f"dx = {rec.final_residual_angstrom[0]:+.2f} Å,  "
            f"dy = {rec.final_residual_angstrom[1]:+.2f} Å"
        )

        polished = self._processed_dir / f"feature_{rec.index:02d}.png" \
            if self._processed_dir else None
        if polished and polished.exists():
            m["status"].setText(
                f"<span style='color:#2a8a2a'><b>Polished</b></span>  "
                f"<br><small>{polished.name}</small>"
            )
        else:
            m["status"].setText(
                "<span style='color:#888'>Raw (using preview PNG)</span>"
            )

        if rec.drift_log_angstrom:
            lines = [
                f"iter {i+1}:  dx = {dx:+.2f},  dy = {dy:+.2f}"
                for i, (dx, dy) in enumerate(rec.drift_log_angstrom)
            ]
            self._drift_label.setText("<br>".join(lines))
        else:
            self._drift_label.setText("—")

        self._process_btn.setEnabled(bool(rec.scan_paths))
        self._save_polished_btn.setEnabled(bool(rec.scan_paths))

    def _process_feature(self) -> None:
        """Open ProbeFlow's image viewer on the latest iteration of the selected feature.

        After the dialog closes, the user can hit "Save polished PNG" to capture
        the processed view as the slide image for this feature.
        """
        rec = self._current_feature()
        if rec is None or not rec.scan_paths:
            QMessageBox.warning(self, "No scan", "Selected feature has no scan files.")
            return

        dat_path = Path(rec.scan_paths[-1])
        if not dat_path.exists():
            # Try the same filename next to the manifest (in case the survey
            # folder was moved after acquisition)
            local = self._manifest_dir / dat_path.name if self._manifest_dir else None
            if local and local.exists():
                dat_path = local
            else:
                QMessageBox.warning(
                    self, "File missing",
                    f"Could not find scan file:\n{dat_path}"
                )
                return

        try:
            self._open_in_viewer(dat_path)
        except Exception as e:
            log.exception("Viewer launch failed")
            QMessageBox.critical(self, "Viewer error",
                                 f"Could not open {dat_path.name}:\n{e}")

    def _open_in_viewer(self, dat_path: Path) -> None:
        """Launch ProbeFlow's existing ImageViewerDialog on a single .dat file."""
        from probeflow.gui import (
            ImageViewerDialog, SxmFile, scan_image_folder, THEMES, load_config
        )
        # Use the survey folder as the "browser" context so left/right arrows
        # walk through the survey's other .dat files.
        folder = dat_path.parent
        entries = scan_image_folder(folder)
        # Find the matching SxmFile for our path
        target = next((e for e in entries if Path(e.path) == dat_path), None)
        if target is None:
            # Fall back: scan_image_folder might filter; do a direct one-shot
            target = SxmFile.from_path(dat_path) if hasattr(SxmFile, "from_path") else None
        if target is None:
            QMessageBox.warning(self, "Unsupported",
                                f"ProbeFlow could not interpret {dat_path.name}")
            return
        cfg = load_config()
        dark = cfg.get("dark_mode", True)
        t = THEMES["dark" if dark else "light"]
        dlg = ImageViewerDialog(target, entries, "gray", t, parent=self)
        dlg.exec()

    def _save_polished_png(self) -> None:
        """Render the latest .dat for the selected feature using current default
        processing and save it to ``processed/feature_NN.png``.

        For v1, this uses ProbeFlow's default rendering pipeline. A future
        revision can capture the exact state of the most recently opened viewer.
        """
        rec = self._current_feature()
        if rec is None or not rec.scan_paths:
            return
        if self._processed_dir is None:
            return

        dat_path = Path(rec.scan_paths[-1])
        if not dat_path.exists():
            local = self._manifest_dir / dat_path.name if self._manifest_dir else None
            if local and local.exists():
                dat_path = local
            else:
                QMessageBox.warning(self, "File missing", str(dat_path))
                return

        try:
            from probeflow.gui import (
                SxmFile, scan_image_folder, render_with_processing
            )
            entries = scan_image_folder(dat_path.parent)
            target = next((e for e in entries if Path(e.path) == dat_path), None)
            if target is None:
                QMessageBox.warning(self, "Unsupported", dat_path.name)
                return
            # Use default plane, decent contrast, ProbeFlow's standard colormap
            img = render_with_processing(target, colormap="afmhot",
                                         clip_low=1.0, clip_high=99.0)
        except Exception as e:
            log.exception("Polished render failed")
            QMessageBox.critical(self, "Render error", str(e))
            return

        out_path = self._processed_dir / f"feature_{rec.index:02d}.png"
        try:
            if hasattr(img, "save"):
                img.save(str(out_path))
            else:
                import matplotlib.pyplot as plt
                import numpy as np
                arr = np.asarray(img)
                plt.imsave(str(out_path), arr, cmap="afmhot")
        except Exception as e:
            log.exception("Save failed")
            QMessageBox.critical(self, "Save error", str(e))
            return

        # Refresh the list row's status
        row = self._feature_list.currentRow()
        item = self._feature_list.item(row)
        if item is not None:
            item.setText(
                f"#{rec.index:02d}   size {rec.char_dim_nm:.2f} nm   "
                f"zoom {rec.zoom_size_nm[0]:.1f} nm   [✓ polished]"
            )
        self._on_select_feature()
        self.log_message.emit(f"Polished PNG saved: {out_path.name}")

    # ------------------------------------------------------------------
    # PPTX export
    # ------------------------------------------------------------------

    def _export_pptx(self) -> None:
        if self._manifest is None or self._manifest_dir is None:
            return
        default = str(self._manifest_dir / f"{self._manifest.name}.pptx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PPTX", default, "PowerPoint (*.pptx)"
        )
        if not path:
            return

        try:
            from scanflow.io import export_pptx
        except ImportError as e:
            QMessageBox.critical(self, "scanflow missing",
                                 f"Could not import the exporter:\n{e}")
            return

        processed_dir = self._processed_dir

        def _resolver(rec):
            if processed_dir is not None:
                polished = processed_dir / f"feature_{rec.index:02d}.png"
                if polished.exists():
                    return polished
            if rec.preview_paths:
                return Path(rec.preview_paths[-1])
            return None

        try:
            export_pptx(self._manifest, Path(path), image_resolver=_resolver)
        except Exception as e:
            log.exception("PPTX export failed")
            QMessageBox.critical(self, "Export failed", str(e))
            return

        polished_count = sum(
            1 for rec in self._manifest.features
            if processed_dir and (processed_dir / f"feature_{rec.index:02d}.png").exists()
        )
        self.log_message.emit(
            f"PPTX exported: {path}  ({polished_count}/{len(self._manifest.features)} polished)"
        )
        QMessageBox.information(
            self, "PPTX saved",
            f"Wrote {Path(path).name}\n\n"
            f"{polished_count} of {len(self._manifest.features)} features "
            f"used polished images.\n"
            f"Run 'Save polished PNG' on individual features to upgrade those slides."
        )
