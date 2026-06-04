"""Consistency guards for the centralized op vocabulary (Phase 1).

These lock ``processing/op_vocab.py`` as a faithful single source of truth for
the geometric-operation names/aliases before the call sites in core/roi.py,
processing/state.py, processing/gui_adapter.py and the CLI are pointed at it.
"""

from __future__ import annotations

from probeflow.core.processing_state import _SUPPORTED_OPS
from probeflow.processing import op_vocab as ov


def test_alias_maps_are_inverse():
    assert ov.SHORT_TO_LONG == {s: long for long, s in ov.LONG_TO_SHORT.items()}
    # Bijective: no short name collides.
    assert len(ov.SHORT_TO_LONG) == len(ov.LONG_TO_SHORT)


def test_to_short_to_long_roundtrip():
    for long, short in ov.LONG_TO_SHORT.items():
        assert ov.to_short(long) == short
        assert ov.to_long(short) == long
        assert ov.to_long(ov.to_short(long)) == long
    # Identity for names with no alias.
    for op in ("flip_horizontal", "flip_vertical", "shear", "not_an_op"):
        assert ov.to_short(op) == op
        assert ov.to_long(op) == op


def test_dimension_swapping_is_subset_of_lossless():
    # Rotations that transpose dims are still exact pixel transforms.
    assert ov.DIMENSION_SWAPPING_OPS <= ov.LOSSLESS_OPS


def test_lossless_ops_are_known_short_names():
    valid_short = set(ov.SHORT_TO_LONG) | {"flip_horizontal", "flip_vertical"}
    assert ov.LOSSLESS_OPS <= valid_short


def test_simple_geometric_ops_are_long_and_supported():
    # Every simple geometric op must be in long form (i.e. not a short alias)
    # and present in the canonical supported-op set.
    assert ov.SIMPLE_GEOMETRIC_OPS <= _SUPPORTED_OPS
    assert not (ov.SIMPLE_GEOMETRIC_OPS & set(ov.SHORT_TO_LONG))


def test_alias_keys_are_supported_long_ops():
    # The long side of every alias is a real supported op; the short side is not
    # itself a separately-supported op (it's purely an internal convenience).
    assert set(ov.LONG_TO_SHORT) <= _SUPPORTED_OPS
    assert not (set(ov.SHORT_TO_LONG) & _SUPPORTED_OPS)
