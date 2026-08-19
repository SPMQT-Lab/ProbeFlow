"""Kernel-free contract for one processing operation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping


OperationGroup = Literal["spatial", "frequency", "scoped", "geometry"]
OperationScope = Literal["global", "roi", "mask"]
ShapePolicy = Literal["preserve", "swap_axes", "resize"]
RangePolicy = Literal["preserve", "swap_axes", "scale_with_shape"]


@dataclass(frozen=True)
class OperationSpec:
    """Declarative behaviour shared by interfaces, replay, and provenance.

    The specification deliberately contains no numerical callable. Unknown
    parameters are retained because historical processing states accepted and
    round-tripped them even when a handler ignored them.
    """

    operation_id: str
    group: OperationGroup
    default_params: Mapping[str, Any] = field(default_factory=dict)
    optional_params: frozenset[str] = frozenset()
    required_params: frozenset[str] = frozenset()
    aliases: frozenset[str] = frozenset()
    allowed_scopes: frozenset[OperationScope] = frozenset({"global"})
    shape_policy: ShapePolicy = "preserve"
    calibration_inputs: frozenset[str] = frozenset()
    range_policy: RangePolicy = "preserve"
    handler_key: str | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        if not self.operation_id or not self.operation_id.strip():
            raise ValueError("operation_id must be a non-empty canonical name")
        if self.group not in {"spatial", "frequency", "scoped", "geometry"}:
            raise ValueError(f"Unsupported operation group: {self.group!r}")
        if self.shape_policy not in {"preserve", "swap_axes", "resize"}:
            raise ValueError(f"Unsupported shape policy: {self.shape_policy!r}")
        if self.range_policy not in {"preserve", "swap_axes", "scale_with_shape"}:
            raise ValueError(f"Unsupported range policy: {self.range_policy!r}")

        defaults = dict(deepcopy(dict(self.default_params)))
        optional = frozenset(self.optional_params)
        required = frozenset(self.required_params)
        aliases = frozenset(self.aliases)
        scopes = frozenset(self.allowed_scopes)
        calibration = frozenset(self.calibration_inputs)
        if required & defaults.keys():
            raise ValueError("A required parameter cannot also have a default")
        if required & optional:
            raise ValueError("A required parameter cannot also be optional")
        if not scopes or not scopes <= {"global", "roi", "mask"}:
            raise ValueError(f"Unsupported operation scopes: {sorted(scopes)!r}")
        if self.operation_id in aliases:
            raise ValueError("The canonical operation ID cannot also be an alias")

        object.__setattr__(self, "default_params", MappingProxyType(defaults))
        object.__setattr__(self, "optional_params", optional)
        object.__setattr__(self, "required_params", required)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "allowed_scopes", scopes)
        object.__setattr__(self, "calibration_inputs", calibration)
        object.__setattr__(self, "handler_key", self.handler_key or self.operation_id)
        object.__setattr__(
            self,
            "display_name",
            self.display_name or self.operation_id.replace("_", " ").title(),
        )

    @property
    def parameter_names(self) -> frozenset[str]:
        return frozenset(self.default_params) | self.optional_params | self.required_params

    def params_with_defaults(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return execution parameters without changing serialized state."""
        resolved = deepcopy(dict(self.default_params))
        resolved.update(deepcopy(dict(params or {})))
        return resolved
