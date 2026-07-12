"""CLI helpers for ROI arguments and persisted ROI lookup."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from probeflow.core.roi import ROI
from probeflow.io.roi_sidecar import find_roi_in_sidecar


def load_named_roi(
    input_path: Path,
    name_or_id: str,
    sidecar: Path | None = None,
    *,
    logger: logging.Logger | None = None,
):
    """Load a named / UUID ROI from a scan ROI or provenance sidecar."""
    log = logger or logging.getLogger(__name__)
    try:
        roi, roi_set, sidecar_path = find_roi_in_sidecar(
            input_path,
            name_or_id,
            sidecar=sidecar,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return None

    if roi is None:
        log.error(
            "ROI %r not found in sidecar %s (available: %s)",
            name_or_id,
            sidecar_path,
            ", ".join(r.name for r in roi_set.rois) or "(none)",
        )
        return None
    return roi


def resolve_inline_roi(
    args: Any,
    *,
    allow_line: bool = False,
    logger: logging.Logger | None = None,
):
    """Parse CLI ROI flags into an ROI object.

    Returns ``(roi_obj | None, error: bool)``.  If ``error`` is true, the
    diagnostic has already been logged.
    """
    log = logger or logging.getLogger(__name__)

    has_rect = getattr(args, "roi_rect", None) is not None
    has_poly = getattr(args, "roi_polygon", None) is not None
    has_line = allow_line and getattr(args, "roi_line", None) is not None
    has_named = getattr(args, "roi", None) is not None

    specified = sum([has_rect, has_poly, has_line, has_named])
    if specified > 1:
        log.error(
            "Specify at most one of --roi-rect, --roi-polygon, --roi-line, --roi"
        )
        return None, True
    if specified == 0:
        return None, False

    inline_values = None
    if has_rect:
        inline_values = args.roi_rect
    elif has_poly:
        inline_values = args.roi_polygon
    elif has_line:
        inline_values = args.roi_line
    if inline_values is not None and not all(
        math.isfinite(float(value)) for value in inline_values
    ):
        log.error("ROI coordinates must all be finite numbers")
        return None, True

    if has_named:
        roi = load_named_roi(
            args.input,
            args.roi,
            getattr(args, "sidecar", None),
            logger=log,
        )
        if roi is None:
            return None, True
        return roi, False

    if has_rect:
        x0, y0, x1, y1 = args.roi_rect
        roi = ROI.new("rectangle", {
            "x": float(min(x0, x1)),
            "y": float(min(y0, y1)),
            "width": float(abs(x1 - x0)),
            "height": float(abs(y1 - y0)),
        })
        return roi, False

    if has_poly:
        coords = list(args.roi_polygon)
        if len(coords) < 6 or len(coords) % 2 != 0:
            log.error(
                "--roi-polygon requires an even number of coordinates "
                "(at least 6 for 3 vertices)"
            )
            return None, True
        vertices = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]
        roi = ROI.new("polygon", {"vertices": vertices})
        return roi, False

    if has_line:
        x1_c, y1_c, x2_c, y2_c = args.roi_line
        roi = ROI.new("line", {
            "x1": float(x1_c), "y1": float(y1_c),
            "x2": float(x2_c), "y2": float(y2_c),
        })
        return roi, False

    return None, False
