"""Phase 0 safety-net: characterize the ``Scan`` save/history surface.

``Scan`` currently reaches up into ``io.writers`` and ``processing.history`` via
function-local imports (the god-object entanglement targeted by the core
de-risking plan, Phase 2).  These tests lock the current behaviour of those
methods so the Phase-2 "thin delegation" refactor is guarded end-to-end.

Pure backend; no Qt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.core.scan_model import Scan


def _scan() -> Scan:
    plane = np.linspace(0.0, 1e-9, 64 * 64).reshape(64, 64).astype(np.float64)
    return Scan(
        planes=[plane],
        plane_names=["Z (fwd)"],
        plane_units=["m"],
        plane_synthetic=[False],
        header={},
        scan_range_m=(10e-9, 10e-9),
        source_path=Path("synthetic.sxm"),
        source_format="sxm",
    )


# NOTE (finding for the de-risk plan): ``save_sxm`` is intentionally NOT
# round-tripped from a synthetic Scan here.  ``write_sxm`` reads the *source*
# .sxm file on disk to reuse its header cushion, so it raises FileNotFoundError
# for an in-memory Scan whose ``source_path`` does not exist.  The real
# SXM-sourced write path is covered by the IO test suite with on-disk fixtures;
# the hidden source-file dependency is recorded as a Phase-2 candidate.


def test_save_png_smoke(tmp_path):
    out = tmp_path / "out.png"
    _scan().save_png(out, plane_idx=0, overwrite=True, overwrite_sidecars=True)
    assert out.exists() and out.stat().st_size > 0


def test_save_csv_smoke(tmp_path):
    out = tmp_path / "out.csv"
    _scan().save_csv(out, plane_idx=0)
    assert out.exists() and out.stat().st_size > 0


def test_processing_history_roundtrip_is_idempotent():
    """history getter ∘ setter must be a fixed point (the exact code Phase 2.1
    will turn into a statically-checkable delegation)."""
    scan = _scan()
    entries = [{"op": "align_rows", "params": {"method": "median"}}]
    scan.processing_history = entries
    once = scan.processing_history
    assert once and once[0]["op"] == "align_rows"

    # Feeding the derived history back in must not change it.
    scan.processing_history = once
    twice = scan.processing_history
    assert [e["op"] for e in twice] == [e["op"] for e in once]


def test_processing_history_empty_by_default():
    assert _scan().processing_history == []
