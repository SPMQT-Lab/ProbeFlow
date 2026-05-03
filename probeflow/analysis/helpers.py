"""Shared helpers for optional analysis modules."""

from __future__ import annotations

import sys

import numpy as np


def missing_extra_message(pkg: str, import_name: str, purpose: str) -> str:
    """Build a diagnostic message for missing optional analysis dependencies."""
    return (
        f"{pkg} is required for {purpose} but `import {import_name}` failed.\n"
        f"  Active interpreter: {sys.executable}\n"
        f"  Python version:     {sys.version.split()[0]}\n"
        f"Install into THIS interpreter with:\n"
        f"  {sys.executable} -m pip install 'probeflow[features]'\n"
        f"(Plain `pip install ...` may target a different environment.)"
    )


def cv2_module(purpose: str):
    """Import OpenCV lazily with a ProbeFlow-specific optional-extra hint."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - deps guard
        raise ImportError(missing_extra_message("OpenCV", "cv2", purpose)) from exc
    return cv2


def to_uint8_for_cv(
    arr: np.ndarray,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> np.ndarray:
    """Percentile-clip and rescale an analysis array to uint8 for OpenCV."""
    from probeflow.processing.display import array_to_uint8

    try:
        return array_to_uint8(arr, clip_percentiles=(clip_low, clip_high))
    except ValueError:
        return np.zeros(np.asarray(arr).shape, dtype=np.uint8)
