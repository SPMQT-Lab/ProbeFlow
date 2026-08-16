"""Contracts for the kernel-free processing-operation catalog."""

from __future__ import annotations

import pytest

from probeflow.core.operations import OperationCatalog, OperationSpec


def test_operation_spec_copies_defaults_and_preserves_unknown_parameters():
    defaults = {"sigma": 1.0, "points": [1, 2]}
    spec = OperationSpec("smooth", "spatial", default_params=defaults)
    defaults["points"].append(3)

    first = spec.params_with_defaults({"future_parameter": True})
    first["points"].append(4)
    second = spec.params_with_defaults(None)

    assert second == {"sigma": 1.0, "points": [1, 2]}
    assert first["future_parameter"] is True


def test_catalog_keeps_canonical_lookup_separate_from_alias_resolution():
    rotate = OperationSpec(
        "rotate_90_cw",
        "geometry",
        aliases=frozenset({"rot90_cw"}),
        shape_policy="swap_axes",
        range_policy="swap_axes",
    )
    catalog = OperationCatalog((rotate,))

    assert catalog.by_id("rot90_cw") is None
    assert catalog.resolve("rot90_cw") is rotate
    assert catalog.resolve("rotate_90_cw") is rotate


def test_catalog_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="globally unique"):
        OperationCatalog(
            (
                OperationSpec("first", "spatial", aliases=frozenset({"old"})),
                OperationSpec("second", "spatial", aliases=frozenset({"old"})),
            )
        )
