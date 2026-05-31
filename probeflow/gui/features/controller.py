"""FeatureCounting orchestration controller (review gui-arch #1).

Plain QObject shared by ProbeFlowWindow (in-tab) and FeatureCountingWindow
(floating).  Owns the worker-dispatch loop, live-preview generation counter,
and result/export handlers.

The loading path (Browse → FeaturesPanel) is intentionally kept in each host
because the two hosts load data differently:

  - ProbeFlowWindow: reads Browse selection + applies saved viewer processing.
  - FeatureCountingWindow: receives a pre-loaded array via ``load_entry()``.

All other FeaturesSidebar signals (segment, clear_segmentation, advance_phase2,
clear_classification, run, export, preview, mode change, mask, classify, crop,
undo, colour) are connected in ``__init__`` so callers do not scatter signal
wiring across the host class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

import os as _os
_os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QObject, QThreadPool

from probeflow.gui.features import FeaturesPanel, FeaturesSidebar, _FeaturesWorker

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


def _pct_to_nm2(pct: float, arr: np.ndarray, px_x_m: float, px_y_m: float) -> float:
    """Convert area as % of image (e.g. 0.001 = 0.001%) to nm²."""
    Ny, Nx = arr.shape
    return (pct / 100.0) * float(Nx) * float(Ny) * px_x_m * px_y_m * 1e18


def _classify_summary(result) -> str:
    """Build compact '  |  '-delimited classify summary string."""
    _BIN = 15
    class_angles: dict = {}
    for c in result:
        class_angles.setdefault(c.class_name, []).append(
            getattr(c, "particle_orientation_deg", 0.0))
    total = len(result)
    parts: list[str] = []
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
    return "  |  ".join(parts)


class FeatureCountingController(QObject):
    """Orchestrates FeatureCounting worker dispatch and result handling.

    Parameters
    ----------
    panel, sidebar
        The widget pair that this controller drives.
    pool
        Thread pool for segment/run workers.
    status_cb
        Callable for window-level status bar messages (e.g.
        ``self._status_bar.showMessage``).
    preview_pool
        Dedicated pool for live-preview workers.  If omitted the global
        instance is used (not recommended for preview — use a 1-thread pool).
    parent_widget
        Parent for file dialogs (``None`` centres dialogs on screen).
    """

    def __init__(
        self,
        panel: FeaturesPanel,
        sidebar: FeaturesSidebar,
        pool: QThreadPool,
        status_cb: Callable[[str], None],
        *,
        preview_pool: QThreadPool | None = None,
        parent_widget: QWidget | None = None,
    ) -> None:
        super().__init__()
        self._panel = panel
        self._sidebar = sidebar
        self._pool = pool
        self._status_cb = status_cb
        self._parent_widget = parent_widget

        self._preview_pool = preview_pool or QThreadPool.globalInstance()
        self._preview_generation = 0
        self._pending_preview_worker: _FeaturesWorker | None = None

        # Direct pass-throughs — same in every host.
        sidebar.crop_template_requested.connect(panel.begin_template_crop)
        sidebar.undo_label_requested.connect(panel.undo_last_label)
        sidebar.mask_color_changed.connect(panel.set_mask_color)

        # Orchestration slots.
        sidebar.segment_requested.connect(self._on_segment)
        sidebar.clear_segmentation_requested.connect(self._on_clear_segmentation)
        sidebar.advance_phase2_requested.connect(self._on_advance_phase2)
        sidebar.clear_classification_requested.connect(self._on_clear_classification)
        sidebar.preview_requested.connect(self._on_preview)
        sidebar.run_requested.connect(self._on_run)
        sidebar.export_requested.connect(self._on_export)
        sidebar.classify_params_changed.connect(self._on_classify_params_changed)
        sidebar.mode_changed.connect(self._on_mode_changed)
        sidebar.mask_paint_toggled.connect(self._on_mask_paint_toggled)
        sidebar.mask_clear_requested.connect(self._on_mask_clear)
        sidebar.step_exclude_changed.connect(self._on_step_exclude_changed)

    # ── Algorithmic step-edge exclusion ───────────────────────────────────────

    def _build_exclude_mask(self):
        """Compute the step-edge band when the sidebar toggle is on, else None.

        Returns a full-resolution boolean mask (or None). Computed by the panel
        from the RAW scan plane so the painted mask never reads as a fake step.
        """
        params = self._sidebar.step_exclude_params()
        if not params.get("enabled"):
            self._panel.clear_step_mask()
            return None
        invert = bool(self._sidebar.particles_params().get("invert", False))
        return self._panel.compute_step_mask(
            threshold_deg=params["threshold_deg"],
            molecule_diameter_m=params["molecule_diameter_m"],
            dilate_m=params["dilate_m"],
            min_step_height_m=params["min_step_height_m"],
            suppress_dark=invert,
        )

    def _on_step_exclude_changed(self) -> None:
        """Recompute/clear the step band overlay and refresh the live preview."""
        self._build_exclude_mask()
        self._sidebar.preview_requested.emit()

    # ── Mode / classify params ────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str) -> None:
        self._panel.set_mode(mode)
        if mode == "classify" and self._panel.get_particles():
            self._panel.set_sample_selection_armed(True)
            self._sidebar.set_status(
                "Click any particle on the image to label it, then press ▶ Run.")
        elif mode != "classify":
            self._panel.set_sample_selection_armed(False)

    def _on_classify_params_changed(self) -> None:
        self._panel.clear_sample_labels()
        self._sidebar.set_status(
            "Segmentation parameters changed — sample labels cleared.")

    # ── Mask ─────────────────────────────────────────────────────────────────

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

    # ── Clear actions (Phase 1 / Phase 2) ────────────────────────────────────

    def _on_clear_segmentation(self) -> None:
        """'Remove Segmentation' — clears overlay, stays in Phase 1."""
        self._panel.clear_particles()
        self._sidebar.set_status("Segmentation cleared.")
        self._status_cb("Segmentation cleared.")

    def _on_clear_classification(self) -> None:
        """'Remove Classification' — drops labels, restores particle contour view."""
        self._panel.clear_classifications()
        self._sidebar.set_status(
            "Classification cleared — particle contours restored.")
        self._status_cb("Classification cleared.")

    # ── Live preview (slider drag) ────────────────────────────────────────────

    def _on_preview(self) -> None:
        """Run segmentation silently while sliders are dragged.

        Performance strategy
        --------------------
        * Step-slice the array to ≤ 512 px on the longest axis.  Physical nm²
          coordinates are preserved so contour overlays land at the correct
          positions on the full-resolution image.
        * A dedicated 1-thread pool prevents stale workers from piling up.
          If a preview worker is still *pending* when a newer one arrives,
          ``tryTake`` cancels it.
        * A generation counter discards results from overtaken workers.
        """
        arr = self._panel.get_analysis_array()
        if arr is None:
            return
        px_m = self._panel.current_pixel_size()
        px_x_m, px_y_m = self._panel.current_pixel_sizes()
        if px_m <= 0:
            return
        params = self._sidebar.particles_params()
        min_pct = params.pop("min_area_pct", 0.001)
        max_pct = params.pop("max_area_pct", 0.0)
        params["min_area_nm2"] = _pct_to_nm2(min_pct, arr, px_x_m, px_y_m)
        params["max_area_nm2"] = (
            _pct_to_nm2(max_pct, arr, px_x_m, px_y_m) if max_pct > 0 else None
        )
        full_mask = self._build_exclude_mask()   # full-res, or None

        step = max(1, max(arr.shape) // 512)
        if step > 1:
            arr_prev  = arr[::step, ::step]
            px_x_prev = px_x_m * step
            px_y_prev = px_y_m * step
            px_prev   = float(np.sqrt(px_x_prev * px_y_prev))
            # Slice the mask identically so it matches the step-sliced array.
            params["exclude_mask"] = None if full_mask is None else full_mask[::step, ::step]
        else:
            arr_prev, px_x_prev, px_y_prev, px_prev = arr, px_x_m, px_y_m, px_m
            params["exclude_mask"] = full_mask

        self._preview_generation += 1
        gen = self._preview_generation

        if self._pending_preview_worker is not None:
            self._preview_pool.tryTake(self._pending_preview_worker)
            self._pending_preview_worker = None

        worker = _FeaturesWorker(
            "particles", arr_prev, px_prev, px_x_prev, px_y_prev, params)
        worker.signals.finished.connect(
            lambda mode, result, error, g=gen:
                self._on_preview_finished(mode, result, error, g))
        self._pending_preview_worker = worker
        self._preview_pool.start(worker)

    def _on_preview_finished(
        self, mode: str, result, error: str, gen: int
    ) -> None:
        if gen != self._preview_generation or mode != "particles" or error:
            return
        self._panel.set_particles(result)
        self._sidebar.set_status(
            f"Preview: {len(result)} particle(s) — press 'Apply Settings' to confirm.")

    # ── Phase 1: Segment ──────────────────────────────────────────────────────

    def _on_segment(self) -> None:
        """Apply Segmentation — full-res run, stays in Phase 1."""
        arr = self._panel.get_analysis_array()
        if arr is None:
            self._sidebar.set_status("Load a scan first.")
            return
        px_m = self._panel.current_pixel_size()
        px_x_m, px_y_m = self._panel.current_pixel_sizes()
        if px_m <= 0:
            self._sidebar.set_status("Scan has no physical pixel size.")
            return
        params = self._sidebar.particles_params()
        min_pct = params.pop("min_area_pct", 0.001)
        max_pct = params.pop("max_area_pct", 0.0)
        params["min_area_nm2"] = _pct_to_nm2(min_pct, arr, px_x_m, px_y_m)
        params["max_area_nm2"] = (
            _pct_to_nm2(max_pct, arr, px_x_m, px_y_m) if max_pct > 0 else None
        )
        params["exclude_mask"] = self._build_exclude_mask()
        self._sidebar.set_status("Segmenting…")
        worker = _FeaturesWorker("particles", arr, px_m, px_x_m, px_y_m, params)
        worker.signals.finished.connect(self._on_segment_finished)
        self._pool.start(worker)

    def _on_segment_finished(self, mode: str, result, error: str) -> None:
        if error:
            self._sidebar.set_status(f"Segmentation failed: {error}")
            self._status_cb(f"Segmentation failed: {error}")
            return
        self._panel.set_particles(result)
        n = len(result)
        self._sidebar.set_status(
            f"Found {n} particle{'s' if n != 1 else ''} — "
            "adjust sliders or click 'Move to Phase 2 →' to continue.")
        self._status_cb(f"Segmentation: {n} particle(s)")

    def _on_advance_phase2(self) -> None:
        """Move to Phase 2 — auto-disarms mask painting."""
        particles = self._panel.get_particles()
        if not particles:
            self._sidebar.set_status(
                "No particles found yet — click 'Apply Segmentation' first.")
            return
        self._sidebar.stop_mask_painting()
        self._sidebar.set_segment_count(len(particles))
        if self._sidebar.current_mode() == "classify":
            self._panel.set_mode("classify")
            self._panel.set_sample_selection_armed(True)
            self._sidebar.set_status(
                f"{len(particles)} particle(s). "
                "Click any particle to label it, then press ▶ Run.")

    # ── Phase 2: Run ──────────────────────────────────────────────────────────

    def _on_run(self, mode: str) -> None:
        arr = self._panel.get_analysis_array()
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
            min_pct = params.pop("min_area_pct", 0.001)
            max_pct = params.pop("max_area_pct", 0.0)
            params["min_area_nm2"] = _pct_to_nm2(min_pct, arr, px_x_m, px_y_m)
            params["max_area_nm2"] = (
                _pct_to_nm2(max_pct, arr, px_x_m, px_y_m) if max_pct > 0 else None
            )
            params["exclude_mask"] = self._build_exclude_mask()
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
            params = {
                "particles":        particles,
                "samples":          samples,
                "use_sharpness":    run_p.get("use_sharpness",    False),
                "threshold_method": run_p.get("threshold_method", "gmm"),
                "manual_threshold": run_p.get("manual_threshold", 0.5),
                "encoder":          run_p.get("encoder",          "raw"),
                "rotate_augment":   run_p.get("rotate_augment",   False),
            }
        else:
            self._sidebar.set_status(f"Unknown mode {mode!r}")
            return

        self._sidebar.set_status(f"Running {mode}…")
        worker = _FeaturesWorker(mode, arr, px_m, px_x_m, px_y_m, params)
        worker.signals.finished.connect(self._on_finished)
        self._pool.start(worker)

    # ── Results ───────────────────────────────────────────────────────────────

    def _on_finished(self, mode: str, result, error: str) -> None:
        if error:
            self._sidebar.set_status(f"{mode} failed: {error}")
            self._status_cb(f"{mode} failed: {error}")
            return

        if mode == "particles":
            self._panel.set_particles(result)
            self._sidebar.set_segment_count(len(result))
            if self._sidebar.current_mode() == "classify":
                self._panel.set_mode("classify")
                self._panel.set_sample_selection_armed(True)
                self._sidebar.set_status(
                    f"Found {len(result)} particle(s). "
                    "Click any particle to label it, then press ▶ Run.")
            else:
                self._sidebar.set_status(f"Found {len(result)} particle(s).")
            self._status_cb(f"Segmentation: {len(result)} particle(s)")

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
            summary = _classify_summary(result)
            self._panel.set_classifications(result)
            self._sidebar.set_status(
                f"Classified {len(result)} particle(s) — {summary}")
            self._status_cb(f"Classify: {summary}")

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export(self, mode: str) -> None:
        # ``write_json`` records full provenance (scan_range_m, pixel sizes,
        # plane names/units, processing state) into the JSON ``meta`` block when
        # passed a ``scan=``.  The panel carries the source Scan when one was
        # available at load time, so GUI exports now match the CLI's provenance;
        # we fall back to just the source path when loaded without a Scan (e.g.
        # an array handed over from the image viewer).
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

        if mode == "particles":
            items = self._panel.get_particles()
            kind, extra = "particles", {}
        elif mode == "template":
            items = self._panel.get_detections()
            kind, extra = "detections", {}
        elif mode == "lattice":
            lat = self._panel.get_lattice()
            items = [lat] if lat is not None else []
            kind, extra = "lattice", {}
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
        scan = self._panel.current_scan()
        # The Scan, when present, supplies source_path/source_format and the
        # physical metadata via write_json; only fall back to a bare source path
        # when we have no Scan, so we never lose that pointer.
        if scan is None and mode != "classify" and entry:
            extra["source"] = str(entry.path)

        if not items:
            self._sidebar.set_status("Nothing to export — run an analysis first.")
            return

        suggested = Path.home() / f"{entry.stem if entry else 'features'}_{kind}.json"
        out_path, _ = QFileDialog.getSaveFileName(
            self._parent_widget,
            f"Export {kind} JSON",
            str(suggested),
            "JSON (*.json)",
        )
        if not out_path:
            return
        try:
            from probeflow.io.writers.json import write_json
            write_json(out_path, items, kind=kind, scan=scan, extra_meta=extra)
            self._sidebar.set_status(f"Exported → {out_path}")
            self._status_cb(f"Exported {kind} → {out_path}")
        except Exception as exc:
            self._sidebar.set_status(f"Export failed: {exc}")
