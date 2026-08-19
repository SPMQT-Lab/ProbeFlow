"""Immutable catalog for built-in processing-operation contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from probeflow.core.operations.model import OperationGroup, OperationSpec


@dataclass(frozen=True)
class OperationCatalog:
    specs: tuple[OperationSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", tuple(self.specs))
        canonical = [spec.operation_id for spec in self.specs]
        if len(set(canonical)) != len(canonical):
            raise ValueError("Operation IDs must be unique")
        identifiers = [
            identifier
            for spec in self.specs
            for identifier in (spec.operation_id, *spec.aliases)
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Operation IDs and aliases must be globally unique")

    @property
    def operation_ids(self) -> frozenset[str]:
        return frozenset(spec.operation_id for spec in self.specs)

    @property
    def roi_eligible_ids(self) -> frozenset[str]:
        return frozenset(
            spec.operation_id for spec in self.specs if "roi" in spec.allowed_scopes
        )

    def by_id(self, operation_id: str) -> OperationSpec | None:
        """Look up a canonical ID; aliases are intentionally not accepted."""
        return next(
            (spec for spec in self.specs if spec.operation_id == operation_id),
            None,
        )

    def resolve(self, identifier: str) -> OperationSpec | None:
        """Resolve a canonical ID or a compatibility alias."""
        return next(
            (
                spec
                for spec in self.specs
                if identifier == spec.operation_id or identifier in spec.aliases
            ),
            None,
        )

    def for_group(self, group: OperationGroup) -> tuple[OperationSpec, ...]:
        return tuple(spec for spec in self.specs if spec.group == group)

    def default_for(self, operation_id: str, parameter: str) -> Any:
        """Return a copied execution default for an interface adapter."""
        spec = self.by_id(operation_id)
        if spec is None:
            raise KeyError(f"Unknown processing operation: {operation_id!r}")
        if parameter not in spec.default_params:
            raise KeyError(
                f"Operation {operation_id!r} has no default for {parameter!r}"
            )
        return deepcopy(spec.default_params[parameter])
