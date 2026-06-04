"""CLI processing operation helpers shared across multiple command runners."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from probeflow.io.common import setup_logging
from probeflow.processing.state import ProcessingState, ProcessingStep
from probeflow.core.scan_loader import load_scan
from probeflow.core.scan_model import Scan

log = logging.getLogger(__name__)


# ─── Processing-op wrapper ───────────────────────────────────────────────────

class _Op:
    """A plane-level processing step bundled with its name and params.

    Acts as a plain ``Callable[[np.ndarray], np.ndarray]`` so it can be
    passed anywhere an op function is expected, but also carries the
    metadata needed to write a ``processing_history`` entry.
    """
    __slots__ = ("name", "params", "_fn", "state")

    def __init__(self, name: str, params: dict,
                 fn: Callable[[np.ndarray], np.ndarray] | None = None,
                 state: ProcessingState | None = None) -> None:
        self.name = name
        self.params = params
        self._fn = fn
        self.state = state

    @classmethod
    def from_step(cls, step: ProcessingStep) -> "_Op":
        return cls(
            step.op,
            dict(step.params),
            state=ProcessingState(steps=[step]),
        )

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        if self.state is not None:
            from probeflow.processing.state import apply_processing_state
            return apply_processing_state(arr, self.state)
        if self._fn is None:
            raise TypeError(f"Processing operation {self.name!r} has no executor")
        return self._fn(arr)


def _record_op(scan: Scan, name: str, params: dict) -> None:
    """Record one canonical processing step on *scan*."""
    state = ProcessingState(steps=[ProcessingStep(name, dict(params))])
    scan.record_processing_state(state)


# ─── Shared argument helpers ─────────────────────────────────────────────────

def _add_common_io(sub: argparse.ArgumentParser, *, out_suffix: str) -> None:
    sub.add_argument("input", type=Path, help="Input .sxm file")
    sub.add_argument(
        "-o", "--output", type=Path, default=None,
        help=f"Output path (default: <input-stem>{out_suffix} next to input)",
    )
    sub.add_argument(
        "--png", action="store_true",
        help="Write a colorised PNG instead of a modified .sxm",
    )
    sub.add_argument(
        "--plane", type=int, default=0,
        help="Plane index to process for --png mode (0=Z-fwd, 1=Z-bwd, "
             "2=I-fwd, 3=I-bwd; default 0)",
    )
    sub.add_argument("--colormap", default="gray", help="Matplotlib colormap name")
    sub.add_argument("--clip-low",  type=float, default=1.0,
                     help="Lower percentile for contrast clipping (PNG mode)")
    sub.add_argument("--clip-high", type=float, default=99.0,
                     help="Upper percentile for contrast clipping (PNG mode)")
    sub.add_argument("--no-scalebar", action="store_true",
                     help="Disable scale bar on PNG output")
    sub.add_argument("--scalebar-unit", choices=("nm", "Å", "pm"), default="nm")
    sub.add_argument("--scalebar-pos",
                     choices=("bottom-right", "bottom-left"), default="bottom-right")
    sub.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file and provenance sidecars",
    )
    sub.add_argument("--verbose", action="store_true", help="Debug logging")
    sub.set_defaults(default_output_suffix=out_suffix)


def _derive_output(args: argparse.Namespace, suffix: str) -> Path:
    """Resolve the output path from CLI args, using a sensible default."""
    if args.output is not None:
        return args.output
    stem = args.input.stem
    parent = args.input.parent
    return parent / f"{stem}{suffix}"


def _default_output_suffix(args: argparse.Namespace, fallback: str) -> str:
    return getattr(args, "default_output_suffix", fallback) or fallback


def _png_output_suffix(suffix: str) -> str:
    if suffix == ".sxm":
        return ".png"
    if suffix.endswith(".sxm"):
        return f"{suffix[:-4]}.png"
    return suffix


def _ensure_output_available(out_path: Path, *, force: bool = False) -> None:
    if out_path.exists() and not force:
        raise ValueError(
            f"Output path already exists: {out_path}. "
            "Choose a different -o/--output path or pass --force."
        )


def _apply_to_plane(
    input_path: Path,
    plane_idx: int,
    op: Callable[[np.ndarray], np.ndarray],
) -> Scan:
    """Load a scan, apply ``op`` to one plane in place, return the Scan.

    Accepts ``.sxm`` *or* ``.dat`` input — dispatch happens in ``load_scan``.
    Updates ``scan.scan_range_m`` when ``op`` is a canvas-expanding step
    (rotate_arbitrary / shear / affine_lattice_correction), so the exported
    PNG scale bar and SXM scan range match the post-op array shape (review
    image-proc #4).
    """
    from probeflow.processing.state import _update_scan_range_for_op

    scan = load_scan(input_path)
    if plane_idx >= scan.n_planes:
        raise ValueError(
            f"Plane {plane_idx} not present — file has {scan.n_planes} plane(s)"
        )
    old_shape = scan.planes[plane_idx].shape
    scan.planes[plane_idx] = op(scan.planes[plane_idx])
    new_shape = scan.planes[plane_idx].shape
    if isinstance(op, _Op):
        if new_shape != old_shape:
            new_range = _update_scan_range_for_op(
                op.name, scan.scan_range_m, old_shape, new_shape,
            )
            if new_range is not None:
                scan.scan_range_m = new_range
        _record_op(scan, op.name, op.params)
    return scan


def _write_output(
    args: argparse.Namespace,
    scan: Scan,
    default_suffix: str,
) -> Path:
    """Write either an .sxm (all planes) or a colorised PNG (selected plane)."""
    default_suffix = _default_output_suffix(args, default_suffix)
    force = bool(getattr(args, "force", False))
    if args.png:
        out_path = _derive_output(args, _png_output_suffix(default_suffix))
        _ensure_output_available(out_path, force=force)
        provenance = _cli_png_provenance(scan, args.plane, args, out_path, "cli_png")
        scan.save_png(
            out_path,
            plane_idx=args.plane,
            colormap=args.colormap,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            add_scalebar=not args.no_scalebar,
            scalebar_unit=args.scalebar_unit,
            scalebar_pos=args.scalebar_pos,
            provenance=provenance,
            overwrite=force,
            overwrite_sidecars=force,
        )
    else:
        out_path = _derive_output(args, default_suffix)
        state = getattr(scan, "processing_state", None)
        if getattr(state, "steps", None) and scan.n_planes > 1:
            raise ValueError(
                "Refusing to write selected-plane processing as an all-plane SXM. "
                "Use --png for selected-plane export until per-plane SXM "
                "processing provenance is supported."
            )
        _ensure_output_available(out_path, force=force)
        scan.save_sxm(out_path, overwrite=force, overwrite_sidecars=force)
    log.info("[OK] %s → %s", args.input.name, out_path)
    return out_path


def _cli_png_provenance(scan: Scan, plane_idx: int, args, out_path, export_kind: str):
    """Build standard provenance for CLI PNG-style exports."""
    from probeflow.provenance.export import build_scan_export_provenance, png_display_state

    clip_low = getattr(args, "clip_low", 1.0)
    clip_high = getattr(args, "clip_high", 99.0)
    display_state = png_display_state(
        clip_low=float(1.0 if clip_low is None else clip_low),
        clip_high=float(99.0 if clip_high is None else clip_high),
        colormap=getattr(args, "colormap", None),
        add_scalebar=not bool(getattr(args, "no_scalebar", False)),
        scalebar_unit=getattr(args, "scalebar_unit", None),
        scalebar_pos=getattr(args, "scalebar_pos", None),
    )
    return build_scan_export_provenance(
        scan,
        channel_index=plane_idx,
        display_state=display_state,
        export_kind=export_kind,
        output_path=out_path,
    )


# ─── Pipeline atoms (each returns an _Op) ────────────────────────────────────

def _op_plane_bg(order: int) -> _Op:
    return _Op.from_step(ProcessingStep("plane_bg", {"order": order}))


def _op_align_rows(method: str) -> _Op:
    return _Op.from_step(ProcessingStep("align_rows", {"method": method}))


def _op_remove_bad_lines(mad: float) -> _Op:
    return _Op.from_step(ProcessingStep(
        "remove_bad_lines",
        {"threshold_mad": mad},
    ))


def _op_facet_level(deg: float) -> _Op:
    return _Op.from_step(ProcessingStep(
        "facet_level",
        {"threshold_deg": deg},
    ))


def _op_smooth(sigma: float) -> _Op:
    return _Op.from_step(ProcessingStep("smooth", {"sigma_px": sigma}))


def _op_edge(method: str, sigma: float, sigma2: float) -> _Op:
    return _Op.from_step(ProcessingStep(
        "edge_detect",
        {"method": method, "sigma": sigma, "sigma2": sigma2},
    ))


def _op_fft(mode: str, cutoff: float, window: str) -> _Op:
    return _Op.from_step(ProcessingStep(
        "fourier_filter",
        {"mode": mode, "cutoff": cutoff, "window": window},
    ))


def _op_flip_horizontal() -> _Op:
    return _Op.from_step(ProcessingStep("flip_horizontal", {}))


def _op_flip_vertical() -> _Op:
    return _Op.from_step(ProcessingStep("flip_vertical", {}))


def _op_rotate_90_cw() -> _Op:
    return _Op.from_step(ProcessingStep("rotate_90_cw", {}))


def _op_rotate_180() -> _Op:
    return _Op.from_step(ProcessingStep("rotate_180", {}))


def _op_rotate_270_cw() -> _Op:
    return _Op.from_step(ProcessingStep("rotate_270_cw", {}))


def _op_rotate_arbitrary(angle_degrees: float, order: int = 1) -> _Op:
    return _Op.from_step(ProcessingStep(
        "rotate_arbitrary",
        {"angle_degrees": angle_degrees, "order": order},
    ))


def _parse_processing_steps(steps_spec: list[str] | tuple[str, ...] | None) -> list[_Op]:
    """Parse CLI ``--steps`` entries into canonical processing operations."""
    if not steps_spec:
        return []

    ops: list[_Op] = []
    for raw in steps_spec:
        name, _, params = raw.partition(":")
        name = name.strip()
        parts = params.split(",") if params else []

        if name == "align-rows":
            method = parts[0] if parts else "median"
            ops.append(_op_align_rows(method))
        elif name == "remove-bad-lines":
            mad = float(parts[0]) if parts else 5.0
            ops.append(_op_remove_bad_lines(mad))
        elif name == "plane-bg":
            order = int(parts[0]) if parts else 1
            if order not in (1, 2, 3, 4):
                raise ValueError(f"plane-bg order must be 1-4, got {order}")
            ops.append(_op_plane_bg(order))
        elif name == "facet-level":
            deg = float(parts[0]) if parts else 3.0
            ops.append(_op_facet_level(deg))
        elif name == "smooth":
            sigma = float(parts[0]) if parts else 1.0
            ops.append(_op_smooth(sigma))
        elif name == "edge":
            method = parts[0] if parts else "laplacian"
            sigma = float(parts[1]) if len(parts) > 1 else 1.0
            sigma2 = float(parts[2]) if len(parts) > 2 else 2.0
            ops.append(_op_edge(method, sigma, sigma2))
        elif name == "fft":
            mode = parts[0] if parts else "low_pass"
            cutoff = float(parts[1]) if len(parts) > 1 else 0.1
            window = parts[2] if len(parts) > 2 else "hanning"
            ops.append(_op_fft(mode, cutoff, window))
        elif name == "flip-h":
            ops.append(_op_flip_horizontal())
        elif name == "flip-v":
            ops.append(_op_flip_vertical())
        elif name == "rotate-90":
            ops.append(_op_rotate_90_cw())
        elif name == "rotate-180":
            ops.append(_op_rotate_180())
        elif name == "rotate-270":
            ops.append(_op_rotate_270_cw())
        elif name == "rotate":
            angle = float(parts[0]) if parts else 0.0
            order = int(parts[1]) if len(parts) > 1 else 1
            ops.append(_op_rotate_arbitrary(angle, order))
        else:
            raise ValueError(f"Unknown pipeline step: {name!r}")
    return ops


def _processing_state_from_ops(ops: list[_Op]) -> ProcessingState:
    """Build the canonical state represented by parsed CLI operations."""
    return ProcessingState(
        steps=[ProcessingStep(op.name, dict(op.params)) for op in ops]
    )


# ─── Shared command utilities ─────────────────────────────────────────────────

def _cmd_single_op(args, op: Callable[[np.ndarray], np.ndarray]) -> int:
    setup_logging(args.verbose)
    scan = _apply_to_plane(args.input, args.plane, op)
    try:
        _write_output(args, scan, default_suffix=".sxm")
    except Exception as exc:
        log.error("%s", exc)
        return 1
    return 0


def _load_plane_for_analysis(args):
    """Load one plane from an .sxm or .dat input; return the numpy array or None."""
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return None
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return None
    return scan.planes[args.plane]


def _pixel_size_m_from_scan(scan) -> float:
    """Geometric mean pixel size — used as a single-number proxy."""
    dx_m, dy_m = _pixel_sizes_m_from_scan(scan)
    if dx_m <= 0 or dy_m <= 0:
        return 0.0
    return float(np.sqrt(dx_m * dy_m))


def _pixel_sizes_m_from_scan(scan) -> tuple[float, float]:
    """Return physical ``(dx, dy)`` pixel sizes in metres."""
    w_m, h_m = scan.scan_range_m
    Nx, Ny = scan.dims
    if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
        return 0.0, 0.0
    return float(w_m / Nx), float(h_m / Ny)


def _load_named_roi(input_path: Path, name_or_id: str, sidecar: "Path | None" = None):
    """Load a named / UUID ROI from the scan's ROI or provenance sidecar.

    Returns the ROI object or None (error already logged).
    """
    from probeflow.cli.roi_args import load_named_roi
    return load_named_roi(input_path, name_or_id, sidecar, logger=log)


def _resolve_inline_roi(args, allow_line: bool = False):
    """Parse --roi-rect / --roi-polygon / --roi-line from CLI args into an ROI.

    Returns (roi_obj | None, error: bool).  If error is True, error is already
    logged and the caller should return 1.
    """
    from probeflow.cli.roi_args import resolve_inline_roi
    return resolve_inline_roi(args, allow_line=allow_line, logger=log)


__all__ = [
    "_Op",
    "_add_common_io",
    "_apply_to_plane",
    "_cli_png_provenance",
    "_cmd_single_op",
    "_default_output_suffix",
    "_derive_output",
    "_ensure_output_available",
    "_load_named_roi",
    "_load_plane_for_analysis",
    "_op_align_rows",
    "_op_edge",
    "_op_facet_level",
    "_op_fft",
    "_op_flip_horizontal",
    "_op_flip_vertical",
    "_op_plane_bg",
    "_op_remove_bad_lines",
    "_op_rotate_90_cw",
    "_op_rotate_180",
    "_op_rotate_270_cw",
    "_op_rotate_arbitrary",
    "_op_smooth",
    "_parse_processing_steps",
    "_pixel_size_m_from_scan",
    "_pixel_sizes_m_from_scan",
    "_png_output_suffix",
    "_processing_state_from_ops",
    "_record_op",
    "_resolve_inline_roi",
    "_write_output",
]
