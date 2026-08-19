"""Public facade for kernel-free processing-operation contracts."""

from probeflow.core.operations import (
    OperationCatalog,
    OperationGroup,
    OperationScope,
    OperationSpec,
    RangePolicy,
    ShapePolicy,
)
from probeflow.core.operations.builtins import BUILTIN_OPERATIONS

__all__ = [
    "BUILTIN_OPERATIONS",
    "OperationCatalog",
    "OperationGroup",
    "OperationScope",
    "OperationSpec",
    "RangePolicy",
    "ShapePolicy",
]
