"""Scan processing CLI commands: pipeline, prepare-png, plane-bg."""

from __future__ import annotations

import logging

import numpy as np

from probeflow import processing as _proc
from probeflow.io.common import setup_logging
from probeflow.core.scan_loader import load_scan
from probeflow.cli.processing_ops import (
    _cmd_single_op,
    _ensure_output_available,
    _load_named_roi,
    _Op,
    _parse_processing_steps,
    _processing_state_from_ops,
    _record_op,
    _write_output,
)

log = logging.getLogger(__name__)


def _cmd_pipeline(args) -> int:
    """Apply a sequence of processing steps in order."""
    setup_logging(args.verbose)
    if not args.steps:
        log.error("--steps is required, e.g. "
                  "--steps align-rows:median plane-bg:1 smooth:1.5")
        return 2
    try:
        ops = _parse_processing_steps(args.steps)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    scan = load_scan(args.input)
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)", args.plane, scan.n_planes)
        return 1
    for op in ops:
        scan.planes[args.plane] = op(scan.planes[args.plane])
        _record_op(scan, op.name, op.params)
    try:
        _write_output(args, scan, default_suffix=".sxm")
    except Exception as exc:
        log.error("%s", exc)
        return 1
    return 0


def _cmd_prepare_png(args) -> int:
    """Write a downstream-analysis PNG with a provenance sidecar."""
    setup_logging(args.verbose)
    try:
        ops = _parse_processing_steps(args.steps or [])
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)", args.plane, scan.n_planes)
        return 1

    state = _processing_state_from_ops(ops)
    if state.steps:
        from probeflow.processing.state import apply_processing_state
        scan.planes[args.plane] = apply_processing_state(scan.planes[args.plane], state)
        scan.record_processing_state(state)

    from probeflow.provenance.prepared_export import write_prepared_png
    force = bool(getattr(args, "force", False))
    _ensure_output_available(args.output, force=force)
    write_prepared_png(
        scan,
        args.output,
        plane_idx=args.plane,
        processing_state=state,
        colormap=args.colormap,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        add_scalebar=not args.no_scalebar,
        overwrite=force,
        overwrite_sidecars=force,
    )
    log.info("[OK] prepared PNG → %s", args.output)
    return 0


def _cmd_plane_bg(args) -> int:
    """Subtract a polynomial plane background with optional ROI parameters."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)", args.plane, scan.n_planes)
        return 1

    arr = scan.planes[args.plane]

    fit_roi_count = sum([
        args.fit_roi is not None,
        getattr(args, "fit_roi_rect", None) is not None,
        getattr(args, "fit_roi_invert", None) is not None,
        getattr(args, "fit_roi_union", None) is not None,
    ])
    if fit_roi_count > 1:
        log.error("Specify at most one of --fit-roi, --fit-roi-rect, "
                  "--fit-roi-invert, --fit-roi-union")
        return 1

    from probeflow.core.roi import ROI, invert as _invert, combine as _combine
    fit_roi = None
    if args.fit_roi is not None:
        fit_roi = _load_named_roi(args.input, args.fit_roi, args.sidecar)
        if fit_roi is None:
            return 1
    elif getattr(args, "fit_roi_rect", None) is not None:
        x0, y0, x1, y1 = args.fit_roi_rect
        fit_roi = ROI.new("rectangle", {
            "x": float(min(x0, x1)), "y": float(min(y0, y1)),
            "width": float(abs(x1 - x0)), "height": float(abs(y1 - y0)),
        })
    elif getattr(args, "fit_roi_invert", None) is not None:
        base = _load_named_roi(args.input, args.fit_roi_invert, args.sidecar)
        if base is None:
            return 1
        try:
            fit_roi = _invert(base, arr.shape)
        except ValueError as exc:
            log.error("--fit-roi-invert failed: %s", exc)
            return 1
    elif getattr(args, "fit_roi_union", None) is not None:
        names = [n.strip() for n in args.fit_roi_union.split(",")]
        rois = [_load_named_roi(args.input, n, args.sidecar) for n in names]
        if any(r is None for r in rois):
            return 1
        try:
            fit_roi = _combine(rois, "union")
        except ValueError as exc:
            log.error("--fit-roi-union failed: %s", exc)
            return 1

    apply_roi = None
    if args.apply_roi is not None:
        apply_roi = _load_named_roi(args.input, args.apply_roi, args.sidecar)
        if apply_roi is None:
            return 1
    elif getattr(args, "apply_roi_rect", None) is not None:
        x0, y0, x1, y1 = args.apply_roi_rect
        apply_roi = ROI.new("rectangle", {
            "x": float(min(x0, x1)), "y": float(min(y0, y1)),
            "width": float(abs(x1 - x0)), "height": float(abs(y1 - y0)),
        })

    exclude_roi = None
    if args.exclude_roi is not None:
        exclude_roi = _load_named_roi(args.input, args.exclude_roi, args.sidecar)
        if exclude_roi is None:
            return 1
    elif getattr(args, "exclude_roi_rect", None) is not None:
        x0, y0, x1, y1 = args.exclude_roi_rect
        exclude_roi = ROI.new("rectangle", {
            "x": float(min(x0, x1)), "y": float(min(y0, y1)),
            "width": float(abs(x1 - x0)), "height": float(abs(y1 - y0)),
        })

    params_hist: dict = {"order": args.order}
    if args.step_tolerance:
        params_hist["step_tolerance"] = True
    if fit_roi is not None:
        params_hist["fit_roi"] = getattr(fit_roi, "name", "inline")
    if apply_roi is not None:
        params_hist["apply_roi"] = getattr(apply_roi, "name", "inline")
    if exclude_roi is not None:
        params_hist["exclude_roi"] = getattr(exclude_roi, "name", "inline")

    def _bg_op(a: np.ndarray) -> np.ndarray:
        return _proc.subtract_background(
            a,
            order=args.order,
            fit_roi=fit_roi,
            apply_roi=apply_roi,
            exclude_roi=exclude_roi,
            step_tolerance=args.step_tolerance,
        )

    op = _Op("plane_bg", params_hist, fn=_bg_op)
    try:
        return _cmd_single_op(args, op)
    except ValueError as exc:
        log.error("plane-bg failed: %s", exc)
        return 1


__all__ = ["_cmd_pipeline", "_cmd_plane_bg", "_cmd_prepare_png", "_cmd_single_op"]
