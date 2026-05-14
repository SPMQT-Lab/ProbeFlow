"""Lightweight folder indexing for supported scan and spectroscopy files.

Public API
----------
index_folder(folder, *, recursive=False, include_errors=True) -> list[ProbeFlowItem]
    Walk a folder and return a list of recognised items, one per file.

ProbeFlowItem
    Frozen dataclass summarising one recognised file without holding any
    image arrays or full spectroscopy data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from probeflow.core.common import _f
from probeflow.core.file_type import FileType, sniff_file_type

# Folder names to always skip when walking.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".probeflow", ".git", "__pycache__", "output", "processed",
})

_FORMAT_MAP: dict[FileType, tuple[str, str]] = {
    FileType.CREATEC_IMAGE: ("createc_dat",          "scan"),
    FileType.NANONIS_IMAGE:  ("nanonis_sxm",           "scan"),
    FileType.RHK_SM4_IMAGE:  ("rhk_sm4",               "scan"),
    FileType.CREATEC_SPEC:   ("createc_vert",          "spectrum"),
    FileType.NANONIS_SPEC:   ("nanonis_dat_spectrum",  "spectrum"),
}


# ── ProbeFlowItem ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeFlowItem:
    """Lightweight, immutable summary of one recognised file in a folder.

    image arrays and full spectroscopy data are never stored here.
    """

    path: Path
    display_name: str
    source_format: str                          # see _FORMAT_MAP above
    item_type: str                              # "scan" | "spectrum"
    shape: Optional[tuple[int, int]] = None     # (Ny, Nx) for scans
    channels: tuple[str, ...] = ()              # plane / channel names
    units: tuple[str, ...] = ()
    scan_range: Optional[tuple[float, float]] = None  # (width_m, height_m)
    bias: Optional[float] = None               # V
    setpoint: Optional[float] = None           # A
    comment: Optional[str] = None
    acquisition_datetime: Optional[str] = None
    mtime_ns: Optional[int] = None
    size_bytes: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    load_error: Optional[str] = None


# ── index_folder ──────────────────────────────────────────────────────────────

def index_folder(
    folder,
    *,
    recursive: bool = False,
    include_errors: bool = True,
) -> list[ProbeFlowItem]:
    """Return a sorted list of recognised Createc/Nanonis files in *folder*.

    Parameters
    ----------
    folder:
        Path to the directory to scan.
    recursive:
        If True, walk all subdirectories (skipping hidden and output dirs).
    include_errors:
        If True, files that are recognised but fail to parse are included with
        ``load_error`` set.  If False, they are silently dropped.

    Returns
    -------
    list[ProbeFlowItem]
        Sorted by acquisition_datetime then by filename.
    """
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    items: list[ProbeFlowItem] = []

    for path in _iter_files(folder, recursive=recursive):
        ft = sniff_file_type(path)
        if ft not in _FORMAT_MAP:
            continue
        source_format, item_type = _FORMAT_MAP[ft]
        item = _build_item(path, source_format, item_type)
        if item.load_error is not None and not include_errors:
            continue
        items.append(item)

    items.sort(key=lambda it: (it.acquisition_datetime or "", it.path.name))
    return items


# ── File iteration ────────────────────────────────────────────────────────────

def _iter_files(folder: Path, *, recursive: bool):
    if not recursive:
        for p in sorted(folder.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                yield p
        return

    for root, dirs, files in os.walk(folder):
        # Prune hidden and output dirs in-place so os.walk doesn't descend.
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".") and d not in _SKIP_DIRS
        )
        root_path = Path(root)
        for name in sorted(files):
            if not name.startswith("."):
                yield root_path / name


# ── Item builders ─────────────────────────────────────────────────────────────

def _file_stat(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None, None


def _build_item(path: Path, source_format: str, item_type: str) -> ProbeFlowItem:
    mtime_ns, size_bytes = _file_stat(path)
    try:
        if item_type == "scan":
            return _item_from_scan(path, source_format, mtime_ns, size_bytes)
        else:
            return _item_from_spec(path, source_format, mtime_ns, size_bytes)
    except Exception as exc:
        return ProbeFlowItem(
            path=path,
            display_name=path.stem,
            source_format=source_format,
            item_type=item_type,
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
            load_error=str(exc),
        )


def _item_from_scan(
    path: Path,
    source_format: str,
    mtime_ns: Optional[int],
    size_bytes: Optional[int],
) -> ProbeFlowItem:
    from probeflow.core.metadata import read_scan_metadata
    meta = read_scan_metadata(path)
    extra = dict(meta.raw_header)
    extra["experiment_metadata"] = dict(meta.experiment_metadata)
    return ProbeFlowItem(
        path=path,
        display_name=meta.display_name or path.stem,
        source_format=source_format,
        item_type="scan",
        shape=meta.shape,
        channels=meta.plane_names,
        units=meta.units,
        scan_range=meta.scan_range,
        bias=meta.bias,
        setpoint=meta.setpoint,
        comment=meta.comment,
        acquisition_datetime=meta.acquisition_datetime,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
        metadata=extra,
    )


def _item_from_spec(
    path: Path,
    source_format: str,
    mtime_ns: Optional[int],
    size_bytes: Optional[int],
) -> ProbeFlowItem:
    from probeflow.io.spectroscopy import read_spec_metadata, spec_channel_to_dict
    meta = read_spec_metadata(path)
    n_pts = meta.metadata.get("n_points")
    extra: dict[str, Any] = {
        "sweep_type": meta.metadata.get("sweep_type"),
        "measurement_family": meta.metadata.get("measurement_family"),
        "feedback_mode": meta.metadata.get("feedback_mode"),
        "derivative_label": meta.metadata.get("derivative_label"),
        "height_channel": meta.metadata.get("height_channel"),
        "height_source_channel": meta.metadata.get("height_source_channel"),
        "z_command_channel": meta.metadata.get("z_command_channel"),
        "measurement_confidence": meta.metadata.get("measurement_confidence"),
        "measurement_evidence": meta.metadata.get("measurement_evidence"),
        "channel_info": [
            spec_channel_to_dict(channel)
            for channel in meta.channel_info
        ],
        "channel_roles": meta.metadata.get("channel_roles"),
        "source_channels": meta.metadata.get("source_channels"),
        "n_points": n_pts,
        "position_m": meta.position,
        "spec_freq_hz": _f(meta.metadata.get("spec_freq_hz")),
        "bias_mv": _f(meta.metadata.get("bias_mv")),
    }
    return ProbeFlowItem(
        path=path,
        display_name=path.stem,
        source_format=source_format,
        item_type="spectrum",
        channels=meta.channels,
        units=meta.units,
        bias=meta.bias,
        comment=meta.comment,
        acquisition_datetime=meta.acquisition_datetime,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
        metadata=extra,
    )


# ── Pure filtering helpers (testable without Qt) ─────────────────────────────

def split_indexed_items(
    items: list[ProbeFlowItem],
) -> tuple[list[ProbeFlowItem], list[ProbeFlowItem], list[ProbeFlowItem]]:
    """Split items into (scans, spectra, errors) for the GUI or CLI.

    Errored items are excluded from scans and spectra regardless of their
    item_type, so callers can ignore them by default or handle them separately.
    """
    scans   = [it for it in items if it.item_type == "scan"     and not it.load_error]
    spectra = [it for it in items if it.item_type == "spectrum" and not it.load_error]
    errors  = [it for it in items if it.load_error]
    return scans, spectra, errors


def image_browser_items(items: list[ProbeFlowItem]) -> list[ProbeFlowItem]:
    """Return only non-errored scan items — what the image browser should show."""
    return [it for it in items if it.item_type == "scan" and not it.load_error]


# ── Shallow (folder-by-folder) browsing ───────────────────────────────────────

@dataclass(frozen=True)
class SubfolderEntry:
    """Lightweight summary of an immediate subfolder for the browse grid."""

    path: Path
    name: str
    n_scans: int                          # scan files found within peek depth
    n_specs: int                          # spectroscopy files found within peek depth
    sample_scan_paths: tuple[Path, ...]   # up to 3 paths for preview thumbnails


@dataclass(frozen=True)
class ShallowFolderIndex:
    """Result of a non-recursive folder index: files at this level + subfolders."""

    folder: Path
    files: list[ProbeFlowItem]
    subfolders: list[SubfolderEntry]


def _peek_subfolder(
    folder: Path,
    *,
    max_samples: int = 3,
    peek_depth: int = 2,
) -> SubfolderEntry:
    """Briefly scan *folder* (BFS, capped by peek_depth) for counts and samples.

    Bounded so users get a meaningful preview even for nested experiment trees,
    without paying for a full recursive walk.
    """
    n_scans = 0
    n_specs = 0
    samples: list[Path] = []
    queue: list[tuple[Path, int]] = [(folder, 0)]

    while queue:
        current, depth = queue.pop(0)
        try:
            entries = sorted(current.iterdir())
        except (OSError, PermissionError):
            continue
        for p in entries:
            if p.name.startswith(".") or p.name in _SKIP_DIRS:
                continue
            try:
                is_file = p.is_file()
                is_dir = p.is_dir()
            except OSError:
                continue
            if is_file:
                ft = sniff_file_type(p)
                if ft in (FileType.CREATEC_IMAGE, FileType.NANONIS_IMAGE):
                    n_scans += 1
                    if len(samples) < max_samples:
                        samples.append(p)
                elif ft in (FileType.CREATEC_SPEC, FileType.NANONIS_SPEC):
                    n_specs += 1
            elif is_dir and depth < peek_depth:
                queue.append((p, depth + 1))

    return SubfolderEntry(
        path=folder,
        name=folder.name,
        n_scans=n_scans,
        n_specs=n_specs,
        sample_scan_paths=tuple(samples),
    )


def index_folder_shallow(
    folder,
    *,
    include_errors: bool = True,
) -> ShallowFolderIndex:
    """Return immediate files + immediate subfolders of *folder* (no recursion).

    Each subfolder is summarised with counts and up to 3 sample scan paths
    (gathered via a depth-limited peek) so a folder card can show preview
    thumbnails of what's inside without fully indexing the subtree.
    """
    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder}")

    files: list[ProbeFlowItem] = []
    subfolders: list[SubfolderEntry] = []

    try:
        entries = sorted(folder.iterdir())
    except (OSError, PermissionError):
        entries = []

    for p in entries:
        if p.name.startswith(".") or p.name in _SKIP_DIRS:
            continue
        try:
            is_file = p.is_file()
            is_dir = p.is_dir()
        except OSError:
            continue
        if is_file:
            ft = sniff_file_type(p)
            if ft not in _FORMAT_MAP:
                continue
            source_format, item_type = _FORMAT_MAP[ft]
            item = _build_item(p, source_format, item_type)
            if item.load_error is not None and not include_errors:
                continue
            files.append(item)
        elif is_dir:
            subfolders.append(_peek_subfolder(p))

    files.sort(key=lambda it: (it.acquisition_datetime or "", it.path.name))
    subfolders.sort(key=lambda s: s.name.lower())
    return ShallowFolderIndex(folder=folder, files=files, subfolders=subfolders)
