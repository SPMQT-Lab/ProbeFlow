"""Channel unit resolution helpers extracted from ImageViewerDialog."""

from __future__ import annotations

import numpy as np


def resolve_channel_unit(
    plane_units: list[str],
    plane_names: list[str],
    channel_idx: int,
    channel_name: str,
    data_arr: np.ndarray | None,
) -> tuple[float, str, str]:
    """Return ``(scale, unit_label, axis_label)`` for the given channel index.

    * ``scale`` converts the raw SI value to the display unit.
    * ``unit_label`` is the short display unit string (e.g. ``"nm"``).
    * ``axis_label`` strips the trailing ``" forward"`` / ``" backward"``
      direction suffix that Createc and Nanonis add, giving a clean axis title.
    """
    from probeflow.analysis.spec_plot import choose_display_unit

    unit = plane_units[channel_idx] if channel_idx < len(plane_units) else ""
    name = (
        plane_names[channel_idx]
        if channel_idx < len(plane_names)
        else channel_name
    )
    scale, unit_label = choose_display_unit(unit, data_arr)
    axis_label = (
        name.rsplit(" ", 1)[0]
        if name.endswith((" forward", " backward"))
        else name
    )
    return scale, unit_label, axis_label
