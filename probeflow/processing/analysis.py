"""Backward-compat re-exports — measurement helpers moved to ``analysis/grains.py``.

The original ``gmm_autoclip`` / ``detect_grains`` / ``measure_periodicity``
helpers produce measurements, not transformed arrays, so they were moved to
:mod:`probeflow.analysis.grains` (review arch-backend #15).  This module
remains as a thin shim because production callers (CLI, GUI, tests) reach
them via ``probeflow.processing.detect_grains`` / ``.gmm_autoclip`` /
``.measure_periodicity`` — which resolve via
``processing/__init__.py`` -> ``processing/image.py``.

New code should import directly from :mod:`probeflow.analysis.grains`.
"""

from __future__ import annotations

from probeflow.analysis.grains import (  # noqa: F401
    detect_grains,
    gmm_autoclip,
    measure_periodicity,
)

__all__ = ["gmm_autoclip", "detect_grains", "measure_periodicity"]
