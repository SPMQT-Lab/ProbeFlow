"""Specialized GUI tools for the ProbeFlow Features workspace.

This package is intentionally separate from :mod:`probeflow.gui`.

The Browse tab should stay focused on file selection, thumbnails, display
scale, and lightweight thumbnail corrections. The Viewer should stay focused
on canonical image-processing/export operations. Tools in this package are
different: they are feature analyses or specialized one-off transforms that
act on a selected scan after the user explicitly loads it into the Features
workspace.

Future Codex/Claude/readthrough note:
    Keep particle counting, template counting, lattice extraction, and future
    TV-denoise/background-removal panels here (or in sibling Features modules),
    not in Browse/Viewer, unless the tool becomes a normal canonical processing
    operation. This boundary prevents optional feature dependencies and more
    experimental workflows from creating odd dependencies in basic browsing,
    conversion, thumbnail rendering, or standard image manipulation.

Module map
----------
- ``panel.py``      — FeaturesPanel / FeaturesSidebar widgets and their workers
- ``controller.py`` — FeatureCountingController (worker dispatch, run/export flow)
- ``window.py``     — floating FeatureCountingWindow that hosts panel + sidebar
- ``tv.py``         — TV-denoise panel/sidebar/worker (main-window workspace page)
"""

from __future__ import annotations

from probeflow.gui.features.panel import (  # noqa: F401
    PLANE_NAMES,
    FeaturesPanel,
    FeaturesSidebar,
    _FeaturesWorker,
    _FeaturesWorkerSignals,
    _FeatureView,
)

__all__ = [
    "PLANE_NAMES",
    "FeaturesPanel",
    "FeaturesSidebar",
]
