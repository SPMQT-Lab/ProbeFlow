"""ROI sidecar load/save helpers extracted from ImageViewerDialog."""

from __future__ import annotations

from pathlib import Path


def load_roi_set(image_path: Path | str):
    """Load a ROISet from the sidecar next to *image_path*, or return an empty one.

    Returns ``(roi_set, error_message)``. Never raises — a missing sidecar is
    normal (empty set, ``None``), but a *corrupt* one returns the error string
    so the caller can tell the user their saved ROIs did not load (review: a
    damaged sidecar was completely invisible at viewer open).
    """
    from probeflow.core.roi import ROISet
    from probeflow.io.roi_sidecar import load_roi_set_sidecar

    image_path = Path(image_path)
    try:
        loaded, _sidecar = load_roi_set_sidecar(image_path, missing_ok=True)
    except Exception as exc:
        return (ROISet(image_id=str(image_path)),
                f"Could not load ROI sidecar: {exc}")

    return (loaded or ROISet(image_id=str(image_path))), None


def save_roi_set(roi_set, image_path: Path | str) -> str | None:
    """Persist *roi_set* to its sidecar file next to *image_path*.

    Returns an error message string on failure, or ``None`` on success.
    """
    from probeflow.io.roi_sidecar import save_roi_set_sidecar

    try:
        save_roi_set_sidecar(roi_set, Path(image_path))
        return None
    except Exception as exc:
        return f"Could not save ROI sidecar: {exc}"
