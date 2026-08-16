"""Plain-data request and result contracts for processed-image export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from probeflow.core.scan_model import Scan
    from probeflow.provenance.export import ExportProvenance


@dataclass(frozen=True)
class ProcessedExportRequest:
    """Everything needed to write one processed scan artifact."""

    scan: "Scan"
    destination: Path
    plane_idx: int = 0
    processed_plane: np.ndarray | None = field(default=None, repr=False, compare=False)
    scan_range_m: tuple[float, float] | None = None
    processing_state: Any | None = field(default=None, repr=False, compare=False)
    display_state: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    roi_set: Any | None = field(default=None, repr=False, compare=False)
    mask_set: Any | None = field(default=None, repr=False, compare=False)
    processing_history: Any | None = field(default=None, repr=False, compare=False)
    provenance: Any | None = field(default=None, repr=False, compare=False)
    export_kind: str | None = None
    warnings: tuple[str, ...] = ()
    include_provenance: bool = True
    build_provenance: bool | None = None
    overwrite: bool = False
    overwrite_sidecars: bool = False
    writer_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "destination", Path(self.destination))
        object.__setattr__(self, "plane_idx", int(self.plane_idx))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        object.__setattr__(
            self,
            "writer_options",
            MappingProxyType(dict(self.writer_options)),
        )
        if self.display_state is not None:
            object.__setattr__(
                self,
                "display_state",
                MappingProxyType(dict(self.display_state)),
            )
        if self.scan_range_m is not None:
            width_m, height_m = self.scan_range_m
            object.__setattr__(
                self,
                "scan_range_m",
                (float(width_m), float(height_m)),
            )


@dataclass(frozen=True)
class ProcessedExportResult:
    """Description of a successfully written processed artifact."""

    destination: Path
    export_format: str
    plane_idx: int
    provenance: "ExportProvenance | None" = field(
        default=None,
        repr=False,
        compare=False,
    )


__all__ = ["ProcessedExportRequest", "ProcessedExportResult"]
