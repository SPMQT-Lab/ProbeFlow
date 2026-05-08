"""Spectroscopy position overlay controller extracted from ImageViewerDialog."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probeflow.gui.models import SxmFile


class SpecOverlayController:
    """Loads, stores, and displays spectroscopy position markers on a scan canvas.

    The controller owns the marker list and the parallel ROISet.  The caller
    retains the ``spec_image_map`` (a shared mutable dict) so that multiple
    dialogs can synchronise their mapping state.
    """

    def __init__(self, zoom_lbl, spec_image_map: dict):
        self._zoom_lbl = zoom_lbl
        self._spec_image_map = spec_image_map
        self._markers: list[dict] = []
        self._roi_set = None

    # ── Public read-only properties ───────────────────────────────────────────

    @property
    def markers(self) -> list[dict]:
        return self._markers

    @property
    def roi_set(self):
        return self._roi_set

    # ── Marker loading ────────────────────────────────────────────────────────

    def load(
        self,
        entry: "SxmFile",
        scan_range_m,
        scan_shape,
        scan_format: str,
        scan_header: dict,
        *,
        show: bool = False,
    ) -> None:
        """Reload spec markers for *entry* and optionally push them to the canvas.

        Markers are only generated for specs explicitly mapped to this image
        via ``spec_image_map``; coordinate-based auto-matching is intentionally
        absent (it was removed because it produced false associations when scan
        windows overlapped).
        """
        self._markers = []
        self._roi_set = None
        self._zoom_lbl.set_markers([])

        if scan_range_m is None or scan_shape is None:
            return

        from probeflow.io.file_type import FileType, sniff_file_type
        from probeflow.io.spectroscopy import read_spec_file
        from probeflow.analysis.spec_plot import spec_position_to_pixel, _parse_sxm_offset
        from probeflow.gui.models import VertFile

        try:
            folder = entry.path.parent
            assigned = {
                spec_stem
                for spec_stem, img_stem in self._spec_image_map.items()
                if img_stem == entry.stem
            }
            if not assigned:
                return

            spec_types = (FileType.CREATEC_SPEC, FileType.NANONIS_SPEC)
            candidates = [
                f for f in sorted(folder.iterdir())
                if f.is_file()
                   and f.stem in assigned
                   and sniff_file_type(f) in spec_types
            ]

            if scan_format == "sxm" and scan_header:
                scan_offset_m = _parse_sxm_offset(scan_header)
                raw_angle = scan_header.get("SCAN_ANGLE", "0").strip()
                try:
                    scan_angle_deg = float(raw_angle) if raw_angle else 0.0
                except ValueError:
                    scan_angle_deg = 0.0
            else:
                scan_offset_m = (0.0, 0.0)
                scan_angle_deg = 0.0

            markers: list[dict] = []
            for spec_path in candidates:
                try:
                    spec = read_spec_file(spec_path)
                    x_m, y_m = spec.position
                    result = spec_position_to_pixel(
                        x_m, y_m,
                        scan_shape=scan_shape,
                        scan_range_m=scan_range_m,
                        scan_offset_m=scan_offset_m,
                        scan_angle_deg=scan_angle_deg,
                    )
                    frac_x, frac_y = result if result is not None else (0.5, 0.5)
                    markers.append({
                        "frac_x": frac_x,
                        "frac_y": frac_y,
                        "entry": VertFile(
                            path=spec_path,
                            stem=spec_path.stem,
                            sweep_type=spec.metadata.get("sweep_type", "unknown"),
                            bias_mv=spec.metadata.get("bias_mv"),
                        ),
                    })
                except Exception:
                    continue

            self._roi_set = _build_spec_roi_set(entry, markers, scan_shape)
            self._markers = markers
            if show:
                self._zoom_lbl.set_markers(markers)

        except Exception:
            pass

    # ── Visibility ────────────────────────────────────────────────────────────

    def apply_visibility(self, show: bool) -> None:
        """Push or clear markers on the canvas according to *show*."""
        self._zoom_lbl.set_markers(self._markers if show else [])

    # ── Dialogs ───────────────────────────────────────────────────────────────

    def open_map_dialog(self, entry: "SxmFile", parent) -> tuple[bool, int]:
        """Open the per-image spec→image mapping dialog.

        Returns ``(accepted, n_specs_mapped_to_this_image)``.
        The caller is responsible for reloading markers after a successful mapping.
        """
        from probeflow.io.file_type import FileType, sniff_file_type
        from probeflow.gui.models import VertFile
        from probeflow.gui.dialogs import ViewerSpecMappingDialog
        from PySide6.QtWidgets import QDialog

        try:
            spec_paths = sorted(
                f for f in entry.path.parent.iterdir()
                if f.is_file() and sniff_file_type(f) in (
                    FileType.CREATEC_SPEC, FileType.NANONIS_SPEC)
            )
        except Exception:
            spec_paths = []

        if not spec_paths:
            return False, 0

        vert_entries = [VertFile(path=p, stem=p.stem) for p in spec_paths]
        dlg = ViewerSpecMappingDialog(entry.stem, vert_entries, self._spec_image_map, parent)
        if dlg.exec() != QDialog.Accepted:
            return False, 0

        new_map = dlg.updated_map()
        self._spec_image_map.clear()
        self._spec_image_map.update(new_map)
        n_for_this = sum(1 for v in new_map.values() if v == entry.stem)
        return True, n_for_this

    def open_spec_viewer(self, entry, t: dict, parent) -> None:
        """Open a SpecViewerDialog for the given spec *entry*."""
        from probeflow.gui.dialogs import SpecViewerDialog
        dlg = SpecViewerDialog(entry, t, parent)
        dlg.exec()


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_spec_roi_set(entry, markers: list[dict], scan_shape):
    """Build a ROISet with one point ROI per marker; returns None on error."""
    try:
        from probeflow.core.roi import ROI, ROISet
        roi_set = ROISet(image_id=str(entry.path))
        shape = scan_shape or (1, 1)
        for m in markers:
            frac_x = float(m.get("frac_x", 0.5))
            frac_y = float(m.get("frac_y", 0.5))
            vert = m.get("entry")
            stem = getattr(vert, "stem", None) or "spectrum"
            linked = str(getattr(vert, "path", "") or "")
            px_x = frac_x * (shape[1] - 1)
            px_y = frac_y * (shape[0] - 1)
            roi_set.add(ROI.new(
                "point", {"x": px_x, "y": px_y},
                name=f"spectrum_{stem}",
                linked_file=linked or None,
            ))
        return roi_set
    except Exception:
        return None
