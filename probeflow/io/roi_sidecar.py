"""ROI sidecar loading and saving helpers.

The ROI model lives in :mod:`probeflow.core.roi`; this module only owns the
small JSON sidecar convention shared by the GUI and CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from probeflow.core.roi import ROI, ROISet


def default_roi_sidecar_path(scan_path: str | Path) -> Path:
    """Return the canonical GUI ROI sidecar path for *scan_path*."""
    path = Path(scan_path)
    return path.with_suffix("").with_suffix(".rois.json")


def roi_sidecar_candidates(scan_path: str | Path) -> tuple[Path, ...]:
    """Return sidecars checked when a CLI command needs persisted ROIs.

    The canonical GUI sidecar is first.  The provenance sidecar fallback keeps
    older exports and hand-authored sidecars usable.
    """
    path = Path(scan_path)
    stem_path = path.with_suffix("")
    candidates = (
        stem_path.with_suffix(".rois.json"),
        stem_path.with_suffix(".provenance.json"),
        path.parent / f"{path.stem}.rois.json",
        path.parent / f"{path.stem}.provenance.json",
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return tuple(out)


def _roi_set_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a ROISet dict from a direct sidecar or provenance payload."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("rois"), dict):
        return data["rois"]
    if data.get("image_id") is not None and isinstance(data.get("rois"), list):
        return data
    return None


def load_roi_set_sidecar(
    scan_path: str | Path,
    *,
    sidecar: str | Path | None = None,
    missing_ok: bool = False,
) -> tuple[ROISet | None, Path]:
    """Load a ROISet from *sidecar* or the default sidecar search path.

    Returns ``(roi_set, path_used)``.  If no sidecar exists and ``missing_ok``
    is true, ``roi_set`` is ``None`` and ``path_used`` is the canonical
    ``.rois.json`` path.
    """
    scan = Path(scan_path)
    if sidecar is None:
        candidates = roi_sidecar_candidates(scan)
        chosen = next((p for p in candidates if p.exists()), candidates[0])
    else:
        chosen = Path(sidecar)

    if not chosen.exists():
        if missing_ok:
            return None, chosen
        tried = ", ".join(str(p) for p in roi_sidecar_candidates(scan))
        raise FileNotFoundError(
            f"No ROI/provenance sidecar found for {scan} (tried {tried})"
        )

    try:
        data = json.loads(chosen.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read ROI sidecar {chosen}: {exc}") from exc

    payload = _roi_set_payload(data)
    if payload is None:
        raise ValueError(f"Sidecar {chosen} contains no ROISet data")

    try:
        return ROISet.from_dict(payload), chosen
    except Exception as exc:
        raise ValueError(f"Could not deserialise ROIs from {chosen}: {exc}") from exc


def save_roi_set_sidecar(
    roi_set: ROISet,
    scan_path: str | Path,
    *,
    sidecar: str | Path | None = None,
) -> Path:
    """Write *roi_set* to the canonical ``.rois.json`` sidecar.

    Uses a write-to-temp-then-rename strategy so a crash or full disk during
    the write never leaves a partially-written (corrupt) sidecar.
    """
    import tempfile
    target = (
        Path(sidecar)
        if sidecar is not None
        else default_roi_sidecar_path(scan_path)
    )
    payload = json.dumps(roi_set.to_dict(), indent=2)
    # Write to a sibling temp file, then atomically replace the target.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=target.parent, prefix=target.name + ".tmp", suffix=".json"
    )
    try:
        with open(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        Path(tmp_path).replace(target)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def find_roi_in_sidecar(
    scan_path: str | Path,
    name_or_id: str,
    *,
    sidecar: str | Path | None = None,
) -> tuple[ROI | None, ROISet, Path]:
    """Load a sidecar and return the ROI matching *name_or_id* by id or name."""
    roi_set, path_used = load_roi_set_sidecar(scan_path, sidecar=sidecar)
    assert roi_set is not None
    roi = roi_set.get(str(name_or_id)) or roi_set.get_by_name(str(name_or_id))
    return roi, roi_set, path_used
