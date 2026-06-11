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
    """Return the canonical GUI ROI sidecar path for *scan_path*.

    Uses ``f"{path.stem}.rois.json"`` (which strips only the *last* extension)
    so dotted filenames like ``A250320.191933.dat`` produce the per-scan
    sidecar ``A250320.191933.rois.json`` instead of the date-prefix-collapsed
    ``A250320.rois.json`` that the old ``with_suffix`` chain produced.  This
    matters for Createc ``Aymmdd.HHmmss.dat`` files: the old path collided
    across every scan saved on the same date and silently overwrote prior
    ROIs.  (Review IO #2, fixed 2026-05-28.)
    """
    path = Path(scan_path)
    return path.parent / f"{path.stem}.rois.json"


def roi_sidecar_candidates(scan_path: str | Path) -> tuple[Path, ...]:
    """Return sidecars checked when a CLI command needs persisted ROIs.

    The canonical GUI sidecar (per-scan, dotted-filename-safe) is first.
    The provenance sidecar fallback keeps older exports and hand-authored
    sidecars usable.  Legacy ``with_suffix``-style paths are kept as final
    fallbacks so any sidecars saved before review IO #2 was fixed remain
    discoverable.
    """
    path = Path(scan_path)
    stem_legacy = path.with_suffix("")  # for legacy buggy paths
    candidates = (
        # Canonical (correct) paths — preferred for new reads.
        path.parent / f"{path.stem}.rois.json",
        path.parent / f"{path.stem}.probeflow.json",
        path.parent / f"{path.stem}.provenance.json",
        # Legacy paths from the pre-fix ``with_suffix`` chain — still
        # discoverable for any sidecars written before 2026-05-28.  For
        # non-dotted filenames these collapse to the canonical paths
        # above and are removed by the dedup loop.
        stem_legacy.with_suffix(".rois.json"),
        stem_legacy.with_suffix(".probeflow.json"),
        stem_legacy.with_suffix(".provenance.json"),
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
    if sidecar is not None:
        chosen = Path(sidecar)
        if not chosen.exists():
            if missing_ok:
                return None, chosen
            raise FileNotFoundError(f"No ROI sidecar found at {chosen}")
        try:
            data = json.loads(chosen.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not read ROI sidecar {chosen}: {exc}") from exc
        payload = _roi_set_payload(data)
        if payload is None:
            raise ValueError(f"Sidecar {chosen} contains no ROISet data")
        return _deserialise_roi_payload(payload, chosen), chosen

    # Candidate search: the first existing candidate *with a usable payload*
    # wins. A candidate that exists but holds no ROI data — typically a
    # processing-only provenance export with ``"rois": null`` — is not an
    # error; it simply serves a different purpose, and a later candidate (or
    # missing_ok) may still provide the ROIs. Corrupt/unparseable files DO
    # raise: silently substituting a stale fallback for a damaged canonical
    # sidecar would hide data loss.
    candidates = roi_sidecar_candidates(scan)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(
                f"Could not read ROI sidecar {candidate}: {exc}"
            ) from exc
        payload = _roi_set_payload(data)
        if payload is None:
            continue
        return _deserialise_roi_payload(payload, candidate), candidate

    if missing_ok:
        return None, candidates[0]
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"No ROI/provenance sidecar found for {scan} (tried {tried})"
    )


def _deserialise_roi_payload(payload: dict[str, Any], source: Path) -> ROISet:
    try:
        return ROISet.from_dict(payload)
    except Exception as exc:
        raise ValueError(f"Could not deserialise ROIs from {source}: {exc}") from exc


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
