"""Lightweight GUI data models and folder-index adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from probeflow.io.createc_interpretation import scan_mode_label, spec_measurement_label

# ── Data model ────────────────────────────────────────────────────────────────
PLANE_NAMES = ["Z fwd", "Z bwd", "I fwd", "I bwd"]


@dataclass
class SxmFile:
    path:          Path
    stem:          str
    Nx:            int            = 512
    Ny:            int            = 512
    bias_mv:       Optional[float] = None
    current_pa:    Optional[float] = None
    scan_nm:       Optional[float] = None
    source_format: str            = "sxm"
    acquisition_label: Optional[str] = None
    experiment_metadata: dict     = field(default_factory=dict)

    @classmethod
    def from_index_item(cls, item) -> "SxmFile":
        """Build the legacy GUI scan entry from a package-level ProbeFlowItem."""
        fmt = {"createc_dat": "dat", "nanonis_sxm": "sxm"}.get(
            item.source_format, item.source_format)
        experiment = dict(item.metadata.get("experiment_metadata") or {})
        label = scan_mode_label(experiment)
        if item.load_error or item.shape is None:
            return cls(
                path=item.path,
                stem=item.path.stem,
                source_format=fmt,
                acquisition_label=label,
                experiment_metadata=experiment,
            )
        Ny, Nx = item.shape
        return cls(
            path=item.path,
            stem=item.path.stem,
            Nx=Nx,
            Ny=Ny,
            bias_mv=item.bias * 1000 if item.bias is not None else None,
            current_pa=item.setpoint * 1e12 if item.setpoint is not None else None,
            scan_nm=item.scan_range[0] * 1e9 if item.scan_range else None,
            source_format=fmt,
            acquisition_label=label,
            experiment_metadata=experiment,
        )


@dataclass
class VertFile:
    path:         Path
    stem:         str
    sweep_type:   str            = "unknown"
    n_points:     int            = 0
    bias_mv:      Optional[float] = None
    spec_freq_hz: Optional[float] = None
    measurement_family: str       = "unknown"
    measurement_label: Optional[str] = None

    @classmethod
    def from_index_item(cls, item) -> "VertFile":
        """Build the legacy GUI spectroscopy entry from a ProbeFlowItem."""
        if item.load_error:
            return cls(path=item.path, stem=item.path.stem)
        measurement = {
            "measurement_family": item.metadata.get("measurement_family"),
            "derivative_label": item.metadata.get("derivative_label"),
        }
        return cls(
            path=item.path,
            stem=item.path.stem,
            sweep_type=str(item.metadata.get("sweep_type") or "unknown"),
            n_points=int(item.metadata.get("n_points") or 0),
            bias_mv=item.bias * 1000 if item.bias is not None else None,
            spec_freq_hz=item.metadata.get("spec_freq_hz"),
            measurement_family=str(item.metadata.get("measurement_family") or "unknown"),
            measurement_label=spec_measurement_label(measurement),
        )


def _card_meta_str(entry: SxmFile) -> str:
    """Format key physical parameters for the thumbnail card label.

    Labels V and I explicitly so a missing setpoint reads as ``I: ?`` rather
    than silently disappearing — that ambiguity confused users who couldn't
    tell whether a low-current value was missing or just zero.
    """
    line1 = "  |  ".join(filter(None, [
        f"{entry.Nx}×{entry.Ny}" if entry.Nx > 0 else "",
        f"{entry.scan_nm:.1f} nm" if entry.scan_nm is not None else "",
    ]))
    v_str = f"V: {entry.bias_mv:.0f} mV"    if entry.bias_mv    is not None else "V: ?"
    i_str = f"I: {entry.current_pa:.0f} pA" if entry.current_pa is not None else "I: ?"
    line2 = f"{v_str}  |  {i_str}"
    return "\n".join(filter(None, [line1, entry.acquisition_label, line2]))


def _scan_items_to_sxm(items) -> list[SxmFile]:
    """Convert ProbeFlowItem scan entries to SxmFile for the existing GUI.

    Preserves the stem-deduplication behaviour of the old scan_image_folder:
    if two files share a stem, only the first (by index_folder sort order)
    is kept.
    """
    seen: set[str] = set()
    result: list[SxmFile] = []
    for item in items:
        if item.item_type != "scan" or item.path.stem in seen:
            continue
        seen.add(item.path.stem)
        result.append(SxmFile.from_index_item(item))
    return result


def _spec_items_to_vert(items) -> list[VertFile]:
    """Convert ProbeFlowItem spectrum entries to VertFile for the existing GUI."""
    result: list[VertFile] = []
    for item in items:
        if item.item_type != "spectrum":
            continue
        result.append(VertFile.from_index_item(item))
    return result


def scan_image_folder(root: Path) -> list[SxmFile]:
    """Find all supported scan files under root and return lightweight SxmFile entries.

    Delegates to :func:`~probeflow.core.indexing.index_folder` for discovery and
    sniffing, then converts the resulting :class:`~probeflow.core.indexing.ProbeFlowItem`
    objects to the GUI-internal :class:`SxmFile` type.
    """
    from probeflow.core.indexing import index_folder
    items = index_folder(Path(root), recursive=True, include_errors=True)
    return _scan_items_to_sxm(items)


def scan_vert_folder(root: Path) -> list[VertFile]:
    """Find all spectroscopy files under root and return lightweight VertFile entries.

    Delegates to :func:`~probeflow.core.indexing.index_folder` for discovery and
    sniffing, then converts the resulting :class:`~probeflow.core.indexing.ProbeFlowItem`
    objects to the GUI-internal :class:`VertFile` type.
    """
    from probeflow.core.indexing import index_folder
    items = index_folder(Path(root), recursive=True, include_errors=True)
    return _spec_items_to_vert(items)
