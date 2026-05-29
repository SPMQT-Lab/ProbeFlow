"""Qt-free processed-image and provenance export helpers.

All functions accept plain Python values and return plain Python values.
No Qt dependency — fully testable without a running QApplication.

These mirror the domain blocks in:
  ImageViewerDialog._processed_scan_for_export
  ImageViewerDialog._on_save_processed_image  (the non-dialog part)
  ImageViewerDialog._on_save_provenance        (the non-dialog part)
  ImageViewerDialog._processed_export_provenance
  ImageViewerDialog._write_processed_export_sidecar
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from probeflow.core.scan_model import Scan
    from probeflow.provenance.records import ExportRecord, ProcessingHistory


def build_processed_scan_for_export(
    path: Path,
    channel_idx: int,
    display_arr: Optional[np.ndarray],
    processing_gui_state: dict,
    *,
    scan_range_m: Optional[tuple[float, float]] = None,
) -> tuple["Scan", int]:
    """Load scan from *path*, inject *display_arr*, record processing state.

    ``display_arr`` is the viewer's current (possibly processed) array.  If
    ``None``, the raw plane from disk is used.  The returned Scan has the
    caller-supplied array in its plane slot and the canonical processing state
    recorded on it — ready for ``scan.save_*()``.

    ``scan_range_m`` is the post-processing physical extent.  When the
    processing pipeline includes a shape-changing step (``rotate_arbitrary``,
    ``shear``, ``affine_lattice_correction`` with canvas expansion), the raw
    scan's ``scan_range_m`` no longer matches the injected array's shape and
    must be updated so PNG scale bars, FFT k-axes, and feature pixel→nm
    conversions stay correct (review image-proc #4).  Compute this via
    :func:`probeflow.processing.state.apply_processing_state_with_calibration`.
    When omitted, the raw scan's ``scan_range_m`` is preserved — safe only
    when the pipeline did not change the array shape.

    Raises ``ValueError`` if there is no image data and ``display_arr`` is
    also ``None``.
    """
    from probeflow.core.scan_loader import load_scan
    from probeflow.processing.gui_adapter import processing_state_from_gui

    scan = load_scan(path)
    idx = max(0, min(channel_idx, scan.n_planes - 1))

    if display_arr is None:
        if scan.n_planes == 0:
            raise ValueError("No image data loaded.")
        arr = scan.planes[idx]
    else:
        arr = display_arr

    scan.planes[idx] = np.asarray(arr, dtype=np.float64).copy()
    if scan_range_m is not None:
        scan.scan_range_m = (float(scan_range_m[0]), float(scan_range_m[1]))
    state = processing_state_from_gui(processing_gui_state or {})
    if state.steps:
        scan.record_processing_state(state)
    return scan, idx


def build_processed_export_provenance(
    scan: "Scan",
    out_path: Path,
    plane_idx: int,
    display_settings: dict,
    roi_set=None,
    processing_history: Optional[dict] = None,
):
    """Build a provenance record for a processed-image export.

    ``display_settings`` should be a plain dict (e.g. from
    ``DisplayRangeController.to_dict()`` merged with colormap/scalebar keys).
    """
    from probeflow.provenance.export import build_scan_export_provenance

    suffix = out_path.suffix.lower().lstrip(".") or "export"
    channel_name = (
        scan.plane_names[plane_idx]
        if plane_idx < len(scan.plane_names) else None
    )
    return build_scan_export_provenance(
        scan,
        channel_index=plane_idx,
        channel_name=channel_name,
        processing_state=scan.processing_state,
        display_state=display_settings,
        export_kind=f"viewer_{suffix}",
        output_path=out_path,
        roi_set=roi_set,
        processing_history=processing_history,
    )


def write_processed_export_sidecar(
    scan: "Scan",
    out_path: Path,
    plane_idx: int,
    display_settings: dict,
    roi_set=None,
    processing_history: Optional[dict] = None,
    provenance=None,
) -> None:
    """Write a .probeflow.json sidecar for non-PNG/SXM processed exports."""
    from probeflow.provenance.export import write_provenance_sidecars

    suffix = out_path.suffix.lower().lstrip(".") or "export"
    if suffix == "png":
        return
    prov = provenance
    if prov is None:
        prov = build_processed_export_provenance(
            scan, out_path, plane_idx, display_settings,
            roi_set=roi_set, processing_history=processing_history,
        )
    write_provenance_sidecars(
        out_path, prov, legacy=False, probeflow=True, export_format=suffix,
    )


def save_processed_image(
    scan: "Scan",
    plane_idx: int,
    out_path: Path,
    *,
    colormap: str = "gray",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    display_settings: Optional[dict] = None,
    roi_set=None,
    processing_history: Optional[dict] = None,
    include_provenance: bool = True,
    add_scalebar: bool = True,
) -> str:
    """Write *scan* plane *plane_idx* to *out_path* in the format implied by suffix.

    Supported suffixes: .png, .pdf, .csv, .gwy, .sxm

    Returns a status string (success or error message) — the caller shows it
    in the UI.  No Qt dependency.
    """
    from probeflow.provenance.export import check_provenance_sidecar_collisions

    suffix = out_path.suffix.lower()
    try:
        provenance = None
        if include_provenance and suffix != ".sxm" and display_settings is not None:
            provenance = build_processed_export_provenance(
                scan, out_path, plane_idx, display_settings,
                roi_set=roi_set, processing_history=processing_history,
            )
            if suffix != ".png":
                check_provenance_sidecar_collisions(
                    out_path, legacy=False, probeflow=True,
                )

        if suffix == ".png":
            scan.save_png(
                out_path, plane_idx=plane_idx,
                colormap=colormap, clip_low=clip_low, clip_high=clip_high,
                add_scalebar=add_scalebar,
                provenance=provenance,
            )
        elif suffix == ".pdf":
            scan.save_pdf(
                out_path, plane_idx=plane_idx,
                colormap=colormap, clip_low=clip_low, clip_high=clip_high,
                show_scalebar=add_scalebar,
                provenance=provenance,
            )
        elif suffix == ".csv":
            scan.save_csv(out_path, plane_idx=plane_idx, provenance=provenance)
        elif suffix == ".gwy":
            scan.save_gwy(
                out_path, plane_idx=plane_idx,
                include_provenance=include_provenance,
                include_meta=include_provenance,
                provenance=provenance,
            )
        elif suffix == ".sxm":
            if scan.processing_state.steps and scan.n_planes > 1:
                return (
                    "Processed SXM export is blocked because SXM writes all planes. "
                    "Use PNG/CSV/PDF/GWY for the selected plane until per-plane "
                    "SXM processing provenance is supported."
                )
            scan.save_sxm(out_path)
        else:
            return "Unsupported processed image format. Use .sxm, .png, .csv, .pdf, or .gwy."

        return f"Saved processed image -> {out_path.name}"
    except Exception as exc:
        return f"Save processed image error: {exc}"


def save_provenance_json(
    processing_history: "ProcessingHistory",
    out_path: Path,
    display_settings: dict,
) -> tuple[str, "ExportRecord"]:
    """Write a provenance JSON to *out_path*.

    Returns ``(status_message, export_record)``.  The caller updates the UI
    with the message and caches the record.

    Raises on failure (caller wraps in try/except).
    """
    from probeflow.provenance.records import build_export_record

    record = build_export_record(
        processing_history,
        export_path=out_path,
        export_format="provenance_json",
        display_settings=display_settings,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(record.to_json(indent=2, default=str), encoding="utf-8")
    return f"Saved provenance -> {out_path.name}", record


__all__ = [
    "build_processed_scan_for_export",
    "build_processed_export_provenance",
    "write_processed_export_sidecar",
    "save_processed_image",
    "save_provenance_json",
]
