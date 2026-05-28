"""Backward-compat re-exports — spectroscopy kernels moved to ``spectroscopy/_kernels.py``.

The numerical kernels (``smooth_spectrum``, ``numeric_derivative``,
``normalize``, ``crop``, ``average_spectra``, ``current_histogram``) live in
:mod:`probeflow.spectroscopy._kernels` (review arch-backend #8).  This shim
preserves the historical ``probeflow.processing.spectroscopy.*`` import path
that CLI commands, tests, and `analysis/spec_plot.py` already use.

New code should import directly from :mod:`probeflow.spectroscopy._kernels`
(or rely on the convenience exports in :mod:`probeflow.spectroscopy`).
"""

from __future__ import annotations

from probeflow.spectroscopy._kernels import (  # noqa: F401
    average_spectra,
    crop,
    current_histogram,
    normalize,
    numeric_derivative,
    smooth_spectrum,
)

__all__ = [
    "average_spectra",
    "crop",
    "current_histogram",
    "normalize",
    "numeric_derivative",
    "smooth_spectrum",
]
