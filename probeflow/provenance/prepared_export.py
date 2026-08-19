"""Compatibility import for the prepared PNG export workflow."""

from __future__ import annotations

from probeflow.workflows.prepared_export import write_prepared_png as _write_prepared_png


def write_prepared_png(*args, **kwargs) -> None:
    """Delegate to :mod:`probeflow.workflows.prepared_export`."""
    _write_prepared_png(*args, **kwargs)


__all__ = ["write_prepared_png"]
