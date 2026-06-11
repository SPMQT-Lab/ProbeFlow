"""Mask sidecar load/save helpers for ImageViewerDialog.

Mirror of :mod:`probeflow.gui.viewer.roi_sidecar` for the active-mask layer.
"""

from __future__ import annotations

from pathlib import Path


def load_mask_set(image_path: Path | str):
    """Load a MaskSet from the sidecar next to *image_path*, or return an empty one.

    Returns ``(mask_set, error_message)``. Never raises — a missing sidecar
    is normal (empty set, ``None``), but a *corrupt* one returns the error
    string so the caller can tell the user their saved masks did not load
    (review: a damaged sidecar was completely invisible at viewer open).
    """
    from probeflow.core.mask import MaskSet
    from probeflow.io.mask_sidecar import load_mask_set_sidecar

    image_path = Path(image_path)
    try:
        loaded, _sidecar = load_mask_set_sidecar(image_path, missing_ok=True)
    except Exception as exc:
        return (MaskSet(image_id=str(image_path)),
                f"Could not load mask sidecar: {exc}")

    return (loaded or MaskSet(image_id=str(image_path))), None


def save_mask_set(mask_set, image_path: Path | str) -> str | None:
    """Persist *mask_set* to its sidecar file next to *image_path*.

    Returns an error message string on failure, or ``None`` on success.
    """
    from probeflow.io.mask_sidecar import save_mask_set_sidecar

    try:
        save_mask_set_sidecar(mask_set, Path(image_path))
        return None
    except Exception as exc:
        return f"Could not save mask sidecar: {exc}"
