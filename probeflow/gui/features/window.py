"""Standalone floating Feature Counting window.

Lets users run particle/template/classify analyses while keeping the Browse
thumbnail grid open in the main window at the same time.

Usage (from ProbeFlowWindow)
----------------------------
    win = FeatureCountingWindow()
    win.load_from_browse_needed.connect(self._on_fc_load_from_browse)
    win.show()
    # When Browse selection changes or "Load" is clicked:
    win.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
"""

from __future__ import annotations

import numpy as np

import os as _os
_os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QSplitter,
    QStatusBar,
)

from probeflow.gui.features import (
    FeaturesPanel,
    FeaturesSidebar,
    _FeaturesWorker,
)


class FeatureCountingWindow(QMainWindow):
    """Floating Feature Counting window — runs alongside the Browse thumbnail grid.

    The main :class:`ProbeFlowWindow` owns this object and bridges the
    ``load_from_browse_needed`` signal so that the selected Browse scan is
    loaded here without switching tabs.

    Two-step workflow (mirrors UniMR)
    ----------------------------------
    Step 1 — Segmentation:
        Set threshold, paint exclusion zones (mask), then click "① Segment"
        in the sidebar.  This populates ``_panel._particles`` and shows
        contour overlays on the image.

    Step 2 — Analysis:
        Choose a mode (Particles / Template / Lattice / Classify) and click
        "② Run".  For Classify, first label a few particles by clicking them
        after Step 1, then Run classifies all remaining particles.
    """

    # Emitted when the user clicks "Load primary scan from Browse".
    # ProbeFlowWindow listens and calls load_entry() with the data.
    load_from_browse_needed = Signal()

    def __init__(self, parent=None):
        # Qt.Window ensures this is an independent top-level window with its own
        # taskbar entry on Windows, not a child that hides behind the main window.
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("ProbeFlow — Feature Counting")
        self.resize(1200, 760)

        self._pool = QThreadPool.globalInstance()
        # Each spawned _FeaturesWorker owns its own signals so two concurrent
        # (or rapidly back-to-back) workers cannot cross-talk.

        # ── Widgets ──────────────────────────────────────────────────────────
        t: dict = {}   # theme dict — window owns its own styling
        self._panel   = FeaturesPanel(t)
        self._sidebar = FeaturesSidebar(t)

        # ── Internal signal wiring ────────────────────────────────────────────
        self._sidebar.load_from_browse_requested.connect(
            self.load_from_browse_needed.emit)
        self._sidebar.segment_requested.connect(self._on_segment_requested)
        self._sidebar.run_requested.connect(self._on_run)
        self._sidebar.export_requested.connect(self._on_export)
        self._sidebar.crop_template_requested.connect(
            self._panel.begin_template_crop)
        self._sidebar.classify_params_changed.connect(
            self._on_classify_params_changed)
        self._sidebar.undo_label_requested.connect(
            self._panel.undo_last_label)
        self._sidebar.mode_changed.connect(self._on_mode_changed)
        self._sidebar.mask_paint_toggled.connect(self._on_mask_paint_toggled)
        self._sidebar.mask_clear_requested.connect(self._on_mask_clear)
        self._sidebar.mask_color_changed.connect(self._panel.set_mask_color)
        # "← Browse" button hides this window (Browse is always in main window)
        self._panel.go_to_browse_requested.connect(self.hide)

        # ── Layout ───────────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._panel)
        splitter.addWidget(self._sidebar)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([860, 340])
        self.setCentralWidget(splitter)

        self._status_bar = QStatusBar()
        self._status_bar.setFont(QFont("Helvetica", 10))
        self.setStatusBar(self._status_bar)

    # ── Public API (called by ProbeFlowWindow) ────────────────────────────────

    def load_entry(self, entry, plane_idx: int, arr: np.ndarray,
                   px_m: float, px_x_m: float, px_y_m: float) -> None:
        """Load a scan plane from Browse into this window."""
        self._panel.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
        self._sidebar.set_status(
            f"Loaded {entry.stem}  (plane {plane_idx},  "
            f"px = {px_m * 1e12:.1f} pm)")
        self._status_bar.showMessage(f"Loaded {entry.stem}")

    # ── Mode / params handlers ────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str) -> None:
        self._panel.set_mode(mode)
        if mode == "classify" and self._panel.get_particles():
            self._panel.set_sample_selection_armed(True)
            self._sidebar.set_status(
                "Click any particle on the image to label it, then press ② Run.")
        elif mode != "classify":
            self._panel.set_sample_selection_armed(False)

    def _on_classify_params_changed(self) -> None:
        self._panel.clear_sample_labels()
        self._sidebar.set_status(
            "Segmentation parameters changed — sample labels cleared.")

    # ── Mask handlers ─────────────────────────────────────────────────────────

    def _on_mask_paint_toggled(self, painting: bool) -> None:
        self._panel.set_mask_painting(painting, self._sidebar.brush_size())
        if painting:
            self._sidebar.set_status(
                "Mask mode — click or drag on the image to paint exclusion zones.")
        else:
            status = ("Mask active — excluded regions shown in colour."
                      if self._panel.has_exclusion_mask()
                      else "Mask drawing stopped.")
            self._sidebar.set_status(status)

    def _on_mask_clear(self) -> None:
        self._panel.clear_exclusion_mask()
        self._sidebar.set_status("Exclusion mask cleared.")

    # ── Step 1: Segment ───────────────────────────────────────────────────────

    def _on_segment_requested(self) -> None:
        """Step 1 — run segmentation with current threshold + exclusion mask.

        After particles are found the image shows contour overlays.  If the
        current analysis mode is 'classify', sample-selection clicking is
        automatically armed so the user can immediately start labeling.
        """
        arr = self._panel.get_analysis_array()   # applies exclusion mask
        if arr is None:
            self._sidebar.set_status("Load a scan first.")
            return
        px_m = self._panel.current_pixel_size()
        px_x_m, px_y_m = self._panel.current_pixel_sizes()
        if px_m <= 0:
            self._sidebar.set_status("Scan has no physical pixel size.")
            return
        params = self._sidebar.particles_params()
        self._sidebar.set_status("Segmenting…")
        worker = _FeaturesWorker(
            "particles", arr, px_m, px_x_m, px_y_m, params,
        )
        worker.signals.finished.connect(self._on_finished)
        self._pool.start(worker)

    # ── Step 2: Run ───────────────────────────────────────────────────────────

    def _on_run(self, mode: str) -> None:
        """Step 2 — run the selected analysis mode."""
        arr = self._panel.get_analysis_array()   # applies exclusion mask if present
        if arr is None:
            self._sidebar.set_status("Load a scan first.")
            return
        px_m = self._panel.current_pixel_size()
        px_x_m, px_y_m = self._panel.current_pixel_sizes()
        if px_m <= 0:
            self._sidebar.set_status("Scan has no physical pixel size.")
            return

        if mode == "particles":
            params = self._sidebar.particles_params()
        elif mode == "template":
            tmpl = self._panel.get_template()
            if tmpl is None:
                self._sidebar.set_status(
                    "Crop a template first (Template → 'Crop template…').")
                return
            params = self._sidebar.template_params()
            params["template"] = tmpl
        elif mode == "lattice":
            params = {}
        elif mode == "classify":
            particles = self._panel.get_particles()
            if not particles:
                self._sidebar.set_status(
                    "Press '① Segment' first to find particles.")
                return
            if not self._panel.has_sample_labels():
                self._sidebar.set_status(
                    "Click particles on the image to label at least one example.")
                return
            idx_to_p = {p.index: p for p in particles}
            samples = [
                (v["name"], idx_to_p[k])
                for k, v in self._panel._sample_labels.items()
                if k in idx_to_p
            ]
            run_p = self._sidebar.classify_run_params()
            params = {"particles": particles, "samples": samples,
                      "use_sharpness": run_p.get("use_sharpness", False)}
        else:
            self._sidebar.set_status(f"Unknown mode {mode!r}")
            return

        self._sidebar.set_status(f"Running {mode}…")
        worker = _FeaturesWorker(
            mode, arr, px_m, px_x_m, px_y_m, params,
        )
        worker.signals.finished.connect(self._on_finished)
        self._pool.start(worker)

    # ── Results ───────────────────────────────────────────────────────────────

    def _on_finished(self, mode: str, result, error: str) -> None:
        if error:
            self._sidebar.set_status(f"{mode} failed: {error}")
            self._status_bar.showMessage(f"{mode} failed: {error}")
            return

        if mode == "particles":
            self._panel.set_particles(result)
            # If the user is in classify mode, auto-arm sample-label clicking
            # so they can immediately click particles to label them.
            current_mode = self._sidebar.current_mode()
            if current_mode == "classify":
                self._panel.set_mode("classify")
                self._panel.set_sample_selection_armed(True)
                self._sidebar.set_status(
                    f"Found {len(result)} particle(s). "
                    "Click any particle to label it, then press ② Run.")
            else:
                self._sidebar.set_status(f"Found {len(result)} particle(s).")
            self._status_bar.showMessage(f"Segmentation: {len(result)} particle(s)")

        elif mode == "template":
            self._panel.set_detections(result)
            self._sidebar.set_status(f"Found {len(result)} match(es).")

        elif mode == "lattice":
            self._panel.set_lattice(result)
            self._sidebar.set_status(
                f"|a|={result.a_length_m * 1e9:.3f} nm  "
                f"|b|={result.b_length_m * 1e9:.3f} nm  "
                f"γ={result.gamma_deg:.1f}°")

        elif mode == "classify":
            # Build a compact "T(30°):3  T(60°):2  other:1" summary
            _BIN = 15
            class_angles: dict = {}
            for c in result:
                class_angles.setdefault(c.class_name, []).append(
                    getattr(c, "particle_orientation_deg", 0.0))
            total = len(result)
            parts = []
            for cls_name in sorted(class_angles):
                angles = class_angles[cls_name]
                if cls_name == "other":
                    pct = 100.0 * len(angles) / total if total > 0 else 0.0
                    parts.append(f"other: {len(angles)} ({pct:.0f}%)")
                    continue
                valid = np.array([a for a in angles if a == a], dtype=float)
                if valid.size == 0:
                    parts.append(f"{cls_name}: {len(angles)}")
                    continue
                bins = np.floor(valid / _BIN).astype(int)
                for b in sorted(set(bins.tolist())):
                    n_b = int((bins == b).sum())
                    mean_a = float(valid[bins == b].mean())
                    pct = 100.0 * n_b / total if total > 0 else 0.0
                    parts.append(f"{cls_name}({mean_a:.0f}°): {n_b} ({pct:.0f}%)")
            summary = "  |  ".join(parts)
            self._panel.set_classifications(result)
            self._sidebar.set_status(
                f"Classified {total} particle(s) — {summary}")
            self._status_bar.showMessage(f"Classify: {summary}")

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export(self, mode: str) -> None:
        from pathlib import Path
        if mode == "particles":
            items = self._panel.get_particles()
            kind, extra = "particles", {"source": None}
        elif mode == "template":
            items = self._panel.get_detections()
            kind, extra = "detections", {"source": None}
        elif mode == "lattice":
            lat = self._panel.get_lattice()
            items = [lat] if lat is not None else []
            kind, extra = "lattice", {"source": None}
        elif mode == "classify":
            items = self._panel.get_classifications()
            kind  = "classifications"
            extra = {
                "samples": self._panel.sample_label_rows(),
                "classification": self._panel._classification_meta,
            }
        else:
            return

        entry = self._panel.current_entry()
        if mode != "classify" and entry:
            extra["source"] = str(entry.path)

        if not items:
            self._sidebar.set_status("Nothing to export — run an analysis first.")
            return

        suggested = Path.home() / f"{entry.stem if entry else 'features'}_{kind}.json"
        out_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {kind} JSON", str(suggested), "JSON (*.json)")
        if not out_path:
            return
        try:
            from probeflow.io.writers.json import write_json
            write_json(out_path, items, kind=kind, extra_meta=extra)
            self._sidebar.set_status(f"Exported → {out_path}")
            self._status_bar.showMessage(f"Exported {kind} → {out_path}")
        except Exception as exc:
            self._sidebar.set_status(f"Export failed: {exc}")
