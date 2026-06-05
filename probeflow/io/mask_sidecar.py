"""Mask sidecar loading and saving helpers.

The mask model lives in :mod:`probeflow.core.mask`; this module owns the small
JSON sidecar convention, mirroring :mod:`probeflow.io.roi_sidecar`.  Masks are
a new on-disk format, so there are no legacy-path fallbacks to carry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from probeflow.core.mask import MaskSet


def default_mask_sidecar_path(scan_path: str | Path) -> Path:
    """Return the canonical mask sidecar path: ``{stem}.masks.json``.

    Uses ``f"{path.stem}.masks.json"`` (strips only the last extension) so
    dotted Createc filenames like ``A250320.191933.dat`` get a per-scan sidecar
    rather than a date-prefix-collapsed one — the same convention as
    :func:`probeflow.io.roi_sidecar.default_roi_sidecar_path`.
    """
    path = Path(scan_path)
    return path.parent / f"{path.stem}.masks.json"


def _mask_set_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a MaskSet dict from a sidecar payload."""
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("masks"), dict):
        return data["masks"]
    if data.get("image_id") is not None and isinstance(data.get("masks"), list):
        return data
    return None


def load_mask_set_sidecar(
    scan_path: str | Path,
    *,
    sidecar: str | Path | None = None,
    missing_ok: bool = False,
) -> tuple[MaskSet | None, Path]:
    """Load a MaskSet from *sidecar* or the default ``.masks.json`` path.

    Returns ``(mask_set, path_used)``.  If no sidecar exists and ``missing_ok``
    is true, ``mask_set`` is ``None`` and ``path_used`` is the canonical path.
    """
    scan = Path(scan_path)
    chosen = Path(sidecar) if sidecar is not None else default_mask_sidecar_path(scan)

    if not chosen.exists():
        if missing_ok:
            return None, chosen
        raise FileNotFoundError(f"No mask sidecar found for {scan} (tried {chosen})")

    try:
        data = json.loads(chosen.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read mask sidecar {chosen}: {exc}") from exc

    payload = _mask_set_payload(data)
    if payload is None:
        raise ValueError(f"Sidecar {chosen} contains no MaskSet data")

    try:
        return MaskSet.from_dict(payload), chosen
    except Exception as exc:
        raise ValueError(f"Could not deserialise masks from {chosen}: {exc}") from exc


def save_mask_set_sidecar(
    mask_set: MaskSet,
    scan_path: str | Path,
    *,
    sidecar: str | Path | None = None,
) -> Path:
    """Write *mask_set* to the canonical ``.masks.json`` sidecar.

    Uses a write-to-temp-then-rename strategy so a crash or full disk during
    the write never leaves a partially-written (corrupt) sidecar — mirroring
    :func:`probeflow.io.roi_sidecar.save_roi_set_sidecar`.
    """
    import tempfile
    target = (
        Path(sidecar)
        if sidecar is not None
        else default_mask_sidecar_path(scan_path)
    )
    payload = json.dumps(mask_set.to_dict(), indent=2)
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
