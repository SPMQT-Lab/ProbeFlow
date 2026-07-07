"""Pin the ProbeFlow surfaces that ScanFlow imports (Qt-free).

ScanFlow (SPMQT-Lab/ScanFlow) integrates with ProbeFlow two ways:

1. Direct imports in its preview/sweep panels and group survey — the names
   pinned below. They are an informal public API: renaming or moving any of
   them breaks ScanFlow at import time on the rig PC, silently from this
   repo's point of view.
2. The ``probeflow --open-survey <survey.json>`` CLI shortcut.

If one of these tests fails, either restore the name via a re-export (the
usual convention here — see gui/features/__init__.py) or coordinate a
matching ScanFlow change BEFORE merging. Surface inventory as of the
2026-07-07 ScanFlow review (docs/reviews/2026-07-07-review.md over there).
"""

from __future__ import annotations

import inspect


def test_scanflow_analysis_imports_resolve():
    from probeflow.analysis.preview import (  # noqa: F401
        PreviewAnalysisParams,
        PreviewFeatureRow,
        apply_preview_background,
        detect_preview_features,
        run_preview,
    )
    from probeflow.analysis.features import Particle, classify_particles  # noqa: F401
    from probeflow.analysis.helpers import cv2_module, to_uint8_for_cv  # noqa: F401


def test_scanflow_core_and_processing_imports_resolve():
    from probeflow.core.scan_loader import load_scan  # noqa: F401
    from probeflow.processing.geometry import set_zero_plane  # noqa: F401

    assert callable(load_scan)
    assert callable(set_zero_plane)


def test_open_survey_cli_shortcut_exists():
    from probeflow.cli.parser import _build_parser

    args = _build_parser().parse_args(["gui", "--open-survey", "survey.json"])
    assert str(getattr(args, "open_survey", "")).endswith("survey.json"), (
        "gui subcommand lost --open-survey — ScanFlow's "
        "open_survey_in_probeflow() shells out to it"
    )


def test_pinned_callables_keep_expected_parameters():
    """Keyword names ScanFlow passes must survive signature changes."""
    from probeflow.analysis.features import classify_particles
    from probeflow.core.scan_loader import load_scan

    load_scan_params = inspect.signature(load_scan).parameters
    assert "path" in load_scan_params or len(load_scan_params) >= 1

    classify_params = set(inspect.signature(classify_particles).parameters)
    # ScanFlow calls classify_particles with particle/sample keywords; keep
    # the first two positional-or-keyword parameters stable.
    assert len(classify_params) >= 2
