"""Phase 0 safety-net: lock the geometric-op alias vocabulary.

The long↔short geometric-op alias maps (``rotate_90_cw`` ↔ ``rot90_cw`` …) are
currently duplicated across ``core/roi.py``, ``processing/state.py``,
``processing/gui_adapter.py`` and ``cli/*``.  The core de-risking plan
(docs/core_derisk_plan.md) Phase 1 centralises them into a single source of
truth.  These behavioural invariants guard that refactor: if the long and short
forms ever stop being treated identically — at any dispatch site — a test fails.

Pure backend; no Qt.
"""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.core.roi import ROI
from probeflow.processing.state import apply_geometric_op_to_scan

# (long form accepted by the GUI/CLI, canonical short form used internally)
_ALIAS_PAIRS = [
    ("rotate_90_cw", "rot90_cw"),
    ("rotate_180", "rot180"),
    ("rotate_270_cw", "rot270_cw"),
]

# Operations whose name is identical in both vocabularies (no alias).
_SHARED_OPS = ["flip_horizontal", "flip_vertical"]


class _StubScan:
    """Minimal duck-typed Scan: a plane list + range, as the function documents."""

    def __init__(self, planes, scan_range_m):
        self.planes = list(planes)
        self.scan_range_m = scan_range_m
        self.plane_names: list[str] = []


def _rect() -> ROI:
    return ROI(
        id="r1", name="r", kind="rectangle",
        geometry={"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
    )


@pytest.mark.parametrize("long_op,short_op", _ALIAS_PAIRS)
def test_roi_transform_alias_equivalence(long_op, short_op):
    """ROI.transform must treat the long and short names identically."""
    shape = (10, 12)  # (Ny, Nx)
    a = _rect().transform(long_op, {}, shape)
    b = _rect().transform(short_op, {}, shape)
    assert a is not None and b is not None
    assert a.kind == b.kind
    assert a.geometry == b.geometry


@pytest.mark.parametrize("long_op,short_op", _ALIAS_PAIRS)
def test_scan_geometric_alias_equivalence(long_op, short_op):
    """apply_geometric_op_to_scan must treat the long and short names identically."""
    img = np.arange(120, dtype=float).reshape(10, 12)
    s_long = _StubScan([img.copy()], (12e-9, 10e-9))
    s_short = _StubScan([img.copy()], (12e-9, 10e-9))

    apply_geometric_op_to_scan(s_long, long_op, {}, None)
    apply_geometric_op_to_scan(s_short, short_op, {}, None)

    np.testing.assert_array_equal(s_long.planes[0], s_short.planes[0])
    assert s_long.scan_range_m == s_short.scan_range_m


@pytest.mark.parametrize("op", _SHARED_OPS)
def test_shared_ops_apply_on_both_surfaces(op):
    """Flips have no alias; confirm they still apply cleanly on both surfaces
    (anchors the shared-name set so a future alias change can't silently drop
    one of them)."""
    shape = (10, 12)
    assert _rect().transform(op, {}, shape) is not None

    img = np.arange(120, dtype=float).reshape(10, 12)
    scan = _StubScan([img.copy()], (12e-9, 10e-9))
    apply_geometric_op_to_scan(scan, op, {}, None)
    assert scan.planes[0].shape == img.shape


def test_unknown_geometric_op_rejected():
    """A bogus op must raise, not silently no-op, on the scan surface."""
    img = np.zeros((4, 4), dtype=float)
    scan = _StubScan([img], (4e-9, 4e-9))
    with pytest.raises(ValueError):
        apply_geometric_op_to_scan(scan, "rotate_42_cw", {}, None)
