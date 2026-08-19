"""Immutable catalog of ProbeFlow's existing processing operations."""

from __future__ import annotations

from probeflow.core.operations.catalog import OperationCatalog
from probeflow.core.operations.frequency import FREQUENCY_OPERATION_SPECS
from probeflow.core.operations.geometry import GEOMETRY_OPERATION_SPECS
from probeflow.core.operations.scoped import SCOPED_OPERATION_SPECS
from probeflow.core.operations.spatial import SPATIAL_OPERATION_SPECS


BUILTIN_OPERATIONS = OperationCatalog(
    SPATIAL_OPERATION_SPECS
    + FREQUENCY_OPERATION_SPECS
    + SCOPED_OPERATION_SPECS
    + GEOMETRY_OPERATION_SPECS
)


__all__ = ["BUILTIN_OPERATIONS"]
