"""
ProbeFlow unified command-line interface.

NOTE — this is the working CLI implementation, not deprecated code.
The `_legacy` suffix reflects an in-progress refactor: command parsers
and runners will be moved out of this file into `cli/commands/` and
`cli/parser.py` opportunistically as commands are touched. Until that
work completes, the bulk of the CLI lives here. New commands should be
added in their proper submodule under `cli/commands/`; only edits to
existing parsers belong in this file.

Every GUI processing capability is available from the shell so that
pipelines can be scripted:

    probeflow <command> [options]

Commands fall into four groups:

  Conversion
    dat2sxm           Createc .dat → Nanonis .sxm
    dat2png           Createc .dat → PNG previews
    sxm2png           Nanonis  .sxm → colorised PNG (with optional scale bar)

  Processing (.sxm in → .sxm out, or .sxm in → .png out via ``--png``)
    plane-bg          Subtract polynomial plane background (order 1 / 2)
    align-rows        Per-row median / mean / linear offset correction
    remove-bad-lines  Interpolate outlier scan lines
    facet-level       Plane fit using only flat-terrace pixels
    smooth            Isotropic Gaussian smoothing
    edge              Laplacian / LoG / DoG edge detection
    fft               Low-pass or high-pass FFT filter
    grains            Threshold-based grain / island detection (prints stats)
    autoclip          GMM-suggested clip percentiles for display
    periodicity       Dominant spatial periodicities via FFT power spectrum

  Pipeline
    pipeline          Chain several of the above steps in one invocation
    prepare-png       Prepared PNG handoff with provenance sidecar

  Inspection / GUI
    info              Print header metadata of an .sxm file
    gui               Launch the ProbeFlow graphical interface

Run ``probeflow <command> --help`` for the options of any subcommand.
"""

from __future__ import annotations

# Layout cleanup note: this module preserves current CLI behaviour while command
# runners and parser helpers are moved into cli/commands, parser.py, and
# processing_ops.py. New domain models, graph nodes, GUI widgets, and numerical
# kernels belong in their canonical packages.

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from probeflow import processing as _proc
from probeflow.io.common import setup_logging
from probeflow.processing.gui_adapter import processing_history_entries_from_state
from probeflow.processing.state import ProcessingState, ProcessingStep
from probeflow.core.scan_loader import load_scan
from probeflow.core.scan_model import Scan
from probeflow.io.sxm_io import (
    parse_sxm_header,
    read_all_sxm_planes,
    read_sxm_plane,
    sxm_dims,
    sxm_scan_range,
    write_sxm_with_planes,
)

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


def _record_op(scan: "Scan", name: str, params: dict) -> None:
    """Append one history entry to *scan.processing_history*."""
    state = ProcessingState(steps=[ProcessingStep(name, dict(params))])
    scan.processing_history.extend(processing_history_entries_from_state(state))


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
    sub.add_argument("--verbose", action="store_true", help="Debug logging")


def _derive_output(args: argparse.Namespace, suffix: str) -> Path:
    """Resolve the output path from CLI args, using a sensible default."""
    if args.output is not None:
        return args.output
    stem = args.input.stem
    parent = args.input.parent
    return parent / f"{stem}{suffix}"


def _apply_to_plane(
    input_path: Path,
    plane_idx: int,
    op: Callable[[np.ndarray], np.ndarray],
) -> Scan:
    """Load a scan, apply ``op`` to one plane in place, return the Scan.

    Accepts ``.sxm`` *or* ``.dat`` input — dispatch happens in ``load_scan``.
    """
    scan = load_scan(input_path)
    if plane_idx >= scan.n_planes:
        raise ValueError(
            f"Plane {plane_idx} not present — file has {scan.n_planes} plane(s)"
        )
    scan.planes[plane_idx] = op(scan.planes[plane_idx])
    if isinstance(op, _Op):
        _record_op(scan, op.name, op.params)
    return scan


def _write_output(
    args: argparse.Namespace,
    scan: Scan,
    default_suffix: str,
) -> Path:
    """Write either an .sxm (all planes) or a colorised PNG (selected plane)."""
    if args.png:
        out_path = _derive_output(args, ".png" if default_suffix == ".sxm"
                                   else default_suffix)
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
        )
    else:
        out_path = _derive_output(args, default_suffix)
        scan.save_sxm(out_path)
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


# ─── Per-command runners ─────────────────────────────────────────────────────

def _cmd_single_op(args, op: Callable[[np.ndarray], np.ndarray]) -> int:
    setup_logging(args.verbose)
    scan = _apply_to_plane(args.input, args.plane, op)
    _write_output(args, scan, default_suffix=".sxm")
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


def _cmd_grains(args) -> int:
    setup_logging(args.verbose)
    arr = _load_plane_for_analysis(args)
    if arr is None:
        return 1
    label_map, n, stats = _proc.detect_grains(
        arr,
        threshold_pct=args.threshold,
        above=not args.below,
        min_grain_px=args.min_px,
    )
    print(f"Grains detected: {n}")
    if args.json:
        out = {"n_grains": n, **stats}
        print(json.dumps(out, indent=2))
    else:
        for i, (area, centroid, height) in enumerate(zip(
                stats.get("areas_px", []),
                stats.get("centroids", []),
                stats.get("mean_heights", [])), start=1):
            cx, cy = centroid
            print(f"  #{i:3d}  area={area:6d} px  centroid=({cx:7.1f},{cy:7.1f})"
                  f"  mean_height={height: .3e}")
    if args.save_mask:
        Image.fromarray(((label_map > 0).astype(np.uint8) * 255), mode="L") \
            .save(str(args.save_mask))
        log.info("[OK] grain mask → %s", args.save_mask)
    return 0


def _cmd_autoclip(args) -> int:
    setup_logging(args.verbose)
    arr = _load_plane_for_analysis(args)
    if arr is None:
        return 1
    low, high = _proc.gmm_autoclip(arr)
    if args.json:
        print(json.dumps({"clip_low": low, "clip_high": high}, indent=2))
    else:
        print(f"clip_low  = {low:.3f}")
        print(f"clip_high = {high:.3f}")
    return 0


def _cmd_periodicity(args) -> int:
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    Nx, Ny = scan.dims
    w_m, h_m = scan.scan_range_m
    if w_m <= 0 or h_m <= 0 or Nx <= 0 or Ny <= 0:
        log.error("Invalid scan range / pixel dims in %s", args.input)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return 1
    arr = scan.planes[args.plane]
    peaks = _proc.measure_periodicity(
        arr,
        pixel_size_x_m=w_m / Nx,
        pixel_size_y_m=h_m / Ny,
        n_peaks=args.n_peaks,
    )
    if args.json:
        print(json.dumps(peaks, indent=2))
    else:
        for i, p in enumerate(peaks, start=1):
            print(f"#{i}  period={p['period_m']*1e9:8.3f} nm  "
                  f"angle={p['angle_deg']:7.2f} deg  "
                  f"strength={p['strength']:.3e}")
    return 0


def _cmd_sxm2png(args) -> int:
    """Render any supported scan (.sxm or .dat) to a PNG."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return 1
    out = args.output or args.input.with_suffix(".png")
    provenance = _cli_png_provenance(scan, args.plane, args, out, "cli_sxm2png")
    scan.save_png(
        out, plane_idx=args.plane,
        colormap=args.colormap,
        clip_low=args.clip_low, clip_high=args.clip_high,
        add_scalebar=not args.no_scalebar,
        scalebar_unit=args.scalebar_unit,
        scalebar_pos=args.scalebar_pos,
        provenance=provenance,
    )
    log.info("[OK] %s → %s", args.input.name, out)
    return 0


def _cmd_info(args) -> int:
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    Nx, Ny = scan.dims
    w_m, h_m = scan.scan_range_m
    hdr = scan.header
    if args.json:
        print(json.dumps({
            "file": str(args.input),
            "format": scan.source_format,
            "Nx": Nx, "Ny": Ny,
            "n_planes": scan.n_planes,
            "plane_names": scan.plane_names,
            "plane_synthetic": scan.plane_synthetic,
            "scan_range_m": [w_m, h_m],
            "header": hdr,
        }, indent=2))
        return 0
    print(f"file      : {args.input}")
    print(f"format    : {scan.source_format}")
    print(f"pixels    : {Nx} x {Ny}")
    print(f"scan size : {w_m*1e9:.3f} nm × {h_m*1e9:.3f} nm")
    print(f"planes    : {scan.n_planes}")
    if any(scan.plane_synthetic):
        synth_idx = [i for i, s in enumerate(scan.plane_synthetic) if s]
        print(f"synthetic : {synth_idx}")
    # Format-specific header highlights.
    if scan.source_format == "sxm":
        keys = ("REC_DATE", "REC_TIME", "BIAS", "SCAN_DIR",
                "SCAN_ANGLE", "SCAN_OFFSET", "COMMENT")
    else:  # dat
        keys = ("Titel", "Biasvolt[mV]", "SetPoint", "ScanYDirec",
                "DAC-Type", "T_AUXADC6[K]")
    for key in keys:
        if key in hdr and hdr[key]:
            print(f"{key:14s}: {hdr[key]}")
    return 0


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
    _write_output(args, scan, default_suffix=".sxm")
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
        from probeflow.processing.gui_adapter import processing_history_entries_from_state
        from probeflow.processing.state import apply_processing_state
        scan.planes[args.plane] = apply_processing_state(scan.planes[args.plane], state)
        scan.processing_history.extend(processing_history_entries_from_state(state))

    from probeflow.provenance.prepared_export import write_prepared_png
    write_prepared_png(
        scan,
        args.output,
        plane_idx=args.plane,
        processing_state=state,
        colormap=args.colormap,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        add_scalebar=not args.no_scalebar,
    )
    log.info("[OK] prepared PNG → %s", args.output)
    return 0


def _cmd_gui(_args) -> int:
    from probeflow.gui import main as _gui_main
    _gui_main()
    return 0


def _cmd_convert(args) -> int:
    """Suffix-driven any-in/any-out topography conversion."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1

    out = Path(args.output)
    extra: dict = {}
    # Build a small kwargs set shared by the image writers.
    if args.colormap is not None:
        extra["colormap"] = args.colormap
    if args.clip_low is not None:
        extra["clip_low"] = args.clip_low
    if args.clip_high is not None:
        extra["clip_high"] = args.clip_high
    if out.suffix.lower() == ".png":
        extra["provenance"] = _cli_png_provenance(
            scan, args.plane, args, out, "cli_convert_png")

    try:
        scan.save(out, plane_idx=args.plane, **extra)
    except Exception as exc:
        log.error("Could not write %s: %s", out, exc)
        return 1
    log.info("[OK] %s → %s", args.input.name, out)
    return 0


def _pixel_size_m_from_scan(scan) -> float:
    """Geometric mean pixel size — used as a single-number proxy."""
    w_m, h_m = scan.scan_range_m
    Nx, Ny = scan.dims
    if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
        return 0.0
    return float(np.sqrt((w_m / Nx) * (h_m / Ny)))


def _cmd_particles(args) -> int:
    """Segment bright (or dark, with --invert) particles and print / export."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return 1
    px_m = _pixel_size_m_from_scan(scan)
    if px_m <= 0:
        log.error("Scan has no physical pixel size — cannot segment.")
        return 1

    from probeflow.analysis.features import segment_particles
    particles = segment_particles(
        scan.planes[args.plane],
        pixel_size_m=px_m,
        threshold=args.threshold,
        manual_value=args.manual_value,
        invert=args.invert,
        min_area_nm2=args.min_area,
        max_area_nm2=args.max_area,
        size_sigma_clip=None if args.no_sigma_clip else args.sigma_clip,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )

    if args.output:
        from probeflow.io.writers.json import write_json
        write_json(args.output, particles, kind="particles", scan=scan,
                   extra_meta={"plane": args.plane, "threshold": args.threshold})
        log.info("[OK] %d particles → %s", len(particles), args.output)
    if args.json:
        import json as _json
        print(_json.dumps([p.to_dict() for p in particles], indent=2))
    else:
        print(f"Detected {len(particles)} particles")
        for p in particles[:args.limit]:
            print(f"  #{p.index:4d}  area={p.area_nm2:8.2f} nm²  "
                  f"centroid=({p.centroid_x_m * 1e9:7.2f},"
                  f" {p.centroid_y_m * 1e9:7.2f}) nm  "
                  f"mean_h={p.mean_height: .3e}")
        if len(particles) > args.limit:
            print(f"  ... ({len(particles) - args.limit} more)")
    return 0


def _cmd_count(args) -> int:
    """Count features by cross-correlating with a template image."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return 1
    px_m = _pixel_size_m_from_scan(scan)
    if px_m <= 0:
        log.error("Scan has no physical pixel size.")
        return 1

    # Load the template: either a PNG or another scan file.
    if args.template.suffix.lower() == ".png":
        tmpl = np.asarray(Image.open(args.template).convert("L"),
                          dtype=np.float64)
    else:
        tscan = load_scan(args.template)
        tmpl = tscan.planes[0]

    from probeflow.analysis.features import count_features
    dets = count_features(
        scan.planes[args.plane], tmpl,
        pixel_size_m=px_m,
        min_correlation=args.min_corr,
        min_distance_m=args.min_distance * 1e-9 if args.min_distance else None,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )

    if args.output:
        from probeflow.io.writers.json import write_json
        write_json(args.output, dets, kind="detections", scan=scan,
                   extra_meta={"template": str(args.template),
                               "min_correlation": args.min_corr})
        log.info("[OK] %d detections → %s", len(dets), args.output)
    if args.json:
        import json as _json
        print(_json.dumps([d.to_dict() for d in dets], indent=2))
    else:
        print(f"Detected {len(dets)} features")
        mean_corr = float(np.mean([d.correlation for d in dets])) if dets else 0.0
        print(f"Mean correlation: {mean_corr:.3f}")
    return 0


def _cmd_tv_denoise(args) -> int:
    """Apply total-variation denoising and write a new .sxm (or PNG)."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return 1
    scan.planes[args.plane] = _proc.tv_denoise(
        scan.planes[args.plane],
        method=args.method,
        lam=args.lam,
        alpha=args.alpha,
        tau=args.tau,
        max_iter=args.max_iter,
        nabla_comp=args.nabla_comp,
    )
    _record_op(scan, "tv_denoise", {
        "method": args.method, "lam": args.lam, "alpha": args.alpha,
        "tau": args.tau, "max_iter": args.max_iter, "nabla_comp": args.nabla_comp,
    })
    _write_output(args, scan, default_suffix="_tv.sxm")
    return 0


def _cmd_lattice(args) -> int:
    """Extract primitive lattice vectors and (optionally) write a PDF report."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return 1
    px_m = _pixel_size_m_from_scan(scan)
    if px_m <= 0:
        log.error("Scan has no physical pixel size.")
        return 1

    from probeflow.analysis.lattice import (
        LatticeParams, extract_lattice, write_lattice_pdf,
    )
    params = LatticeParams(
        contrast_threshold=args.contrast_threshold,
        sigma=args.sigma,
        cluster_kp_low=args.cluster_kp_low,
        cluster_kp_high=args.cluster_kp_high,
        cluster_kNN_low=args.cluster_knn_low,
        cluster_kNN_high=args.cluster_knn_high,
    )
    try:
        res = extract_lattice(scan.planes[args.plane], pixel_size_m=px_m,
                              params=params)
    except Exception as exc:
        log.error("Lattice extraction failed: %s", exc)
        return 1

    if args.output:
        suffix = args.output.suffix.lower()
        if suffix == ".pdf":
            write_lattice_pdf(scan, res, args.output, plane_idx=args.plane,
                              colormap=args.colormap,
                              clip_low=args.clip_low, clip_high=args.clip_high)
        else:
            from probeflow.io.writers.json import write_json
            write_json(args.output, [res], kind="lattice", scan=scan,
                       extra_meta={"plane": args.plane})
        log.info("[OK] lattice result → %s", args.output)

    if args.json:
        import json as _json
        print(_json.dumps(res.to_dict(), indent=2))
    else:
        print(f"|a| = {res.a_length_m * 1e9:7.3f} nm")
        print(f"|b| = {res.b_length_m * 1e9:7.3f} nm")
        print(f" γ  = {res.gamma_deg:7.2f} °")
        print(f"Keypoints: {res.n_keypoints}  (primary cluster: "
              f"{res.n_keypoints_used})")
    return 0


def _cmd_classify(args) -> int:
    """Classify segmented particles against labelled samples in a JSON file."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)",
                  args.plane, scan.n_planes)
        return 1
    px_m = _pixel_size_m_from_scan(scan)
    if px_m <= 0:
        log.error("Scan has no physical pixel size.")
        return 1

    from probeflow.analysis.features import (
        Particle, segment_particles, classify_particles,
    )
    arr = scan.planes[args.plane]

    particles = segment_particles(
        arr, pixel_size_m=px_m,
        min_area_nm2=args.min_area,
        size_sigma_clip=None if args.no_sigma_clip else args.sigma_clip,
    )

    # Samples file: JSON produced from `probeflow particles` (or hand-crafted).
    import json as _json
    samples_data = _json.loads(Path(args.samples).read_text(encoding="utf-8"))
    if isinstance(samples_data, dict) and "items" in samples_data:
        samples_data = samples_data["items"]
    samples: list[tuple[str, Particle]] = []
    for entry in samples_data:
        name = entry.get("class_name") or entry.get("label") or "sample"
        p = Particle(**{k: v for k, v in entry.items()
                        if k in Particle.__dataclass_fields__})
        samples.append((name, p))

    classifs = classify_particles(
        arr, particles, samples,
        encoder=args.encoder,
        threshold_method=args.threshold_method,
    )

    if args.output:
        from probeflow.io.writers.json import write_json
        write_json(args.output, classifs, kind="classifications", scan=scan,
                   extra_meta={"encoder": args.encoder,
                               "threshold_method": args.threshold_method})
        log.info("[OK] %d classifications → %s", len(classifs), args.output)

    # Summary counts per class
    counts: dict = {}
    for c in classifs:
        counts[c.class_name] = counts.get(c.class_name, 0) + 1
    if args.json:
        print(_json.dumps(counts, indent=2))
    else:
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:20s}  {n}")
    return 0


def _load_named_roi(input_path: "Path", name_or_id: str, sidecar: "Path | None" = None):
    """Load a named / UUID ROI from the scan's ROI or provenance sidecar.

    Returns the ROI object or None (error already logged).
    """
    import json as _json
    if sidecar is None:
        stem_path = input_path.with_suffix("")
        candidates = [
            stem_path.with_suffix(".rois.json"),
            stem_path.with_suffix(".provenance.json"),
            input_path.parent / f"{input_path.stem}.rois.json",
            input_path.parent / f"{input_path.stem}.provenance.json",
        ]
        sidecar = next((p for p in candidates if p.exists()), candidates[0])
    if not sidecar.exists():
        log.error("No ROI/provenance sidecar found for %s (tried %s)", input_path, sidecar)
        return None
    try:
        data = _json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Could not read sidecar %s: %s", sidecar, exc)
        return None
    roi_set_data = data if isinstance(data.get("rois"), list) else data.get("rois")
    if not roi_set_data:
        log.error("Sidecar %s contains no ROI data", sidecar)
        return None
    from probeflow.core.roi import ROISet
    try:
        roi_set = ROISet.from_dict(roi_set_data)
    except Exception as exc:
        log.error("Could not deserialise ROIs from sidecar: %s", exc)
        return None
    roi = roi_set.get(name_or_id) or roi_set.get_by_name(name_or_id)
    if roi is None:
        log.error("ROI %r not found in sidecar (available: %s)",
                  name_or_id,
                  ", ".join(r.name for r in roi_set.rois) or "(none)")
        return None
    return roi


def _resolve_inline_roi(args, allow_line: bool = False):
    """Parse --roi-rect / --roi-polygon / --roi-line from CLI args into an ROI.

    Returns (roi_obj | None, error: bool).  If error is True, error is already
    logged and the caller should return 1.
    """
    from probeflow.core.roi import ROI

    has_rect = getattr(args, "roi_rect", None) is not None
    has_poly = getattr(args, "roi_polygon", None) is not None
    has_line = allow_line and getattr(args, "roi_line", None) is not None
    has_named = getattr(args, "roi", None) is not None

    specified = sum([has_rect, has_poly, has_line, has_named])
    if specified > 1:
        log.error("Specify at most one of --roi-rect, --roi-polygon, "
                  "--roi-line, --roi")
        return None, True
    if specified == 0:
        return None, False

    if has_named:
        sidecar = getattr(args, "sidecar", None)
        roi = _load_named_roi(args.input, args.roi, sidecar)
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
            log.error("--roi-polygon requires an even number of coordinates "
                      "(at least 6 for 3 vertices)")
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

    # ── Resolve fit ROI ──────────────────────────────────────────────────────
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

    # ── Resolve apply ROI ─────────────────────────────────────────────────────
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

    # ── Resolve exclude ROI ───────────────────────────────────────────────────
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

    # ── Build _Op for history recording and execution ─────────────────────────
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


def _cmd_histogram(args) -> int:
    """Pixel-value histogram, optionally restricted to an ROI."""
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

    roi, err = _resolve_inline_roi(args)
    if err:
        return 1

    from probeflow.processing.display import histogram_from_array
    try:
        counts, edges = histogram_from_array(arr, roi=roi, bins=args.bins)
    except ValueError as exc:
        log.error("Histogram failed: %s", exc)
        return 1

    if args.output is None:
        print("# bin_centre\tcount")
        centres = 0.5 * (edges[:-1] + edges[1:])
        for c, n in zip(centres, counts):
            print(f"{c:.6e}\t{n}")
        return 0

    suffix = args.output.suffix.lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        with args.output.open("w", encoding="utf-8") as f:
            f.write("# bin_edge_low,bin_edge_high,count\n")
            for lo, hi, n in zip(edges[:-1], edges[1:], counts):
                f.write(f"{lo:.6e},{hi:.6e},{n}\n")
    elif suffix == ".png":
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, counts, width=(edges[1] - edges[0]), color="steelblue",
               edgecolor="none")
        ax.set_xlabel(f"{scan.plane_names[args.plane]} ({scan.plane_units[args.plane]})")
        ax.set_ylabel("Count")
        roi_desc = f" — ROI: {args.roi}" if getattr(args, "roi", None) else ""
        ax.set_title(f"{scan.source_path.name} plane {args.plane}{roi_desc}")
        fig.tight_layout()
        fig.savefig(str(args.output))
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    else:
        log.error("Unsupported output suffix %r — use .csv or .png", suffix)
        return 1
    log.info("[OK] histogram → %s", args.output)
    return 0


def _cmd_fft_spectrum(args) -> int:
    """Compute the 2-D FFT magnitude spectrum of a scan plane or ROI."""
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
    Ny, Nx = arr.shape
    w_m, h_m = scan.scan_range_m
    px_x = w_m / Nx if Nx > 0 and w_m > 0 else 1.0
    px_y = h_m / Ny if Ny > 0 and h_m > 0 else 1.0

    roi, err = _resolve_inline_roi(args)
    if err:
        return 1

    try:
        mag, qx, qy = _proc.fft_magnitude(
            arr,
            roi,
            pixel_size_x_m=px_x,
            pixel_size_y_m=px_y,
            window=args.window,
            window_param=args.window_param,
            log_scale=args.log_scale,
        )
    except ValueError as exc:
        log.error("fft-spectrum failed: %s", exc)
        return 1

    if args.output is None:
        # Print the top-5 peaks by magnitude
        flat = mag.ravel()
        top_idx = flat.argsort()[::-1][:5]
        print("# qx (nm⁻¹)  qy (nm⁻¹)  magnitude")
        for idx in top_idx:
            iy, ix = divmod(int(idx), mag.shape[1])
            print(f"{float(qx[ix]):.4f}\t{float(qy[iy]):.4f}\t{float(flat[idx]):.4e}")
        return 0

    suffix = args.output.suffix.lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from probeflow.processing.display import clip_range_from_array
        vmin, vmax = clip_range_from_array(mag, 1.0, 99.9)
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        ax.imshow(
            mag, cmap=args.colormap, origin="upper",
            vmin=vmin, vmax=vmax,
            extent=[float(qx[0]), float(qx[-1]), float(qy[-1]), float(qy[0])],
        )
        ax.set_xlabel("qx (nm⁻¹)")
        ax.set_ylabel("qy (nm⁻¹)")
        roi_desc = f" — ROI: {args.roi}" if getattr(args, "roi", None) else ""
        ax.set_title(f"FFT spectrum — {scan.source_path.name}{roi_desc}")
        fig.tight_layout()
        fig.savefig(str(args.output))
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    else:
        log.error("Unsupported output suffix %r — use .png", suffix)
        return 1
    log.info("[OK] FFT spectrum → %s", args.output)
    return 0


def _cmd_profile(args) -> int:
    """Sample z-values along a straight segment and write a CSV / PNG / JSON."""
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
    Ny, Nx = arr.shape
    w_m, h_m = scan.scan_range_m
    px_x = w_m / Nx if Nx > 0 and w_m > 0 else 1e-10
    px_y = h_m / Ny if Ny > 0 and h_m > 0 else 1e-10

    # Endpoints: --p0/--p1 in pixels, --p0-nm/--p1-nm in nm, OR --roi-line / --roi.
    roi_line = getattr(args, "roi_line", None)
    roi_name = getattr(args, "roi", None)

    roi_obj = None
    if roi_name is not None:
        roi_obj = _load_named_roi(args.input, roi_name,
                                  getattr(args, "sidecar", None))
        if roi_obj is None:
            return 1  # error already logged
        if roi_obj.kind != "line":
            log.error("ROI %r has kind=%r; profile requires a line ROI",
                      roi_name, roi_obj.kind)
            return 1

    if roi_obj is not None:
        # Check for conflicting explicit endpoints
        if any(x is not None for x in (
            args.p0, args.p1,
            getattr(args, "p0_nm", None), getattr(args, "p1_nm", None),
            roi_line,
        )):
            log.error("Cannot combine --roi with --p0/--p1/--roi-line")
            return 1
        s_m, z = _proc.line_profile(
            arr, roi=roi_obj,
            pixel_size_x_m=px_x, pixel_size_y_m=px_y,
            n_samples=args.n_samples,
            width_px=args.width,
            interp=args.interp,
        )
        p0 = (roi_obj.geometry["x1"], roi_obj.geometry["y1"])
        p1 = (roi_obj.geometry["x2"], roi_obj.geometry["y2"])
    elif roi_line is not None:
        x1, y1, x2, y2 = roi_line
        p0, p1 = (x1, y1), (x2, y2)
        s_m, z = _proc.line_profile(
            arr, p0, p1,
            pixel_size_x_m=px_x, pixel_size_y_m=px_y,
            n_samples=args.n_samples,
            width_px=args.width,
            interp=args.interp,
        )
    elif getattr(args, "p0_nm", None) is not None and getattr(args, "p1_nm", None) is not None:
        p0 = (args.p0_nm[0] * 1e-9 / px_x, args.p0_nm[1] * 1e-9 / px_y)
        p1 = (args.p1_nm[0] * 1e-9 / px_x, args.p1_nm[1] * 1e-9 / px_y)
        s_m, z = _proc.line_profile(
            arr, p0, p1,
            pixel_size_x_m=px_x, pixel_size_y_m=px_y,
            n_samples=args.n_samples,
            width_px=args.width,
            interp=args.interp,
        )
    elif args.p0 is not None and args.p1 is not None:
        p0 = tuple(args.p0)
        p1 = tuple(args.p1)
        s_m, z = _proc.line_profile(
            arr, p0, p1,
            pixel_size_x_m=px_x, pixel_size_y_m=px_y,
            n_samples=args.n_samples,
            width_px=args.width,
            interp=args.interp,
        )
    else:
        log.error("Provide one of: --p0/--p1, --p0-nm/--p1-nm, --roi-line, or --roi")
        return 1

    if args.output is None:
        for s, zi in zip(s_m, z):
            print(f"{s:.6e}\t{zi:.6e}")
        return 0

    suffix = args.output.suffix.lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        with args.output.open("w", encoding="utf-8") as f:
            f.write("# distance_m\tz\n")
            for s, zi in zip(s_m, z):
                f.write(f"{s:.6e}\t{zi:.6e}\n")
    elif suffix == ".json":
        from probeflow.io.writers.json import write_json

        class _Sample:
            __dataclass_fields__ = {"distance_m": None, "z": None}

            def __init__(self, distance_m, z):
                self.distance_m = float(distance_m)
                self.z = float(z)

            def to_dict(self):
                return {"distance_m": self.distance_m, "z": self.z}

        items = [_Sample(s, zi) for s, zi in zip(s_m, z)]
        write_json(args.output, items, kind="line_profile", scan=scan,
                   extra_meta={"plane": args.plane,
                               "p0_px": list(p0), "p1_px": list(p1),
                               "width_px": args.width})
    elif suffix == ".png":
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
        ax.plot(s_m * 1e9, z, lw=1.2)
        ax.set_xlabel("Distance along profile (nm)")
        ax.set_ylabel(f"{scan.plane_names[args.plane]} ({scan.plane_units[args.plane]})")
        ax.set_title(f"{scan.source_path.name} — plane {args.plane}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(args.output))
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    else:
        log.error("Unsupported output suffix %r — use .csv / .json / .png", suffix)
        return 1
    log.info("[OK] %d samples → %s", len(s_m), args.output)
    return 0


def _cmd_unit_cell(args) -> int:
    """Extract lattice, then average all unit cells into one canonical motif."""
    setup_logging(args.verbose)
    try:
        scan = load_scan(args.input)
    except Exception as exc:
        log.error("Could not load %s: %s", args.input, exc)
        return 1
    if args.plane >= scan.n_planes:
        log.error("Plane %d not present (file has %d)", args.plane, scan.n_planes)
        return 1
    px_m = _pixel_size_m_from_scan(scan)
    if px_m <= 0:
        log.error("Scan has no physical pixel size.")
        return 1

    from probeflow.analysis.lattice import (
        LatticeParams, extract_lattice, average_unit_cell,
    )
    arr = scan.planes[args.plane]
    try:
        lat = extract_lattice(arr, pixel_size_m=px_m, params=LatticeParams())
    except Exception as exc:
        log.error("Lattice extraction failed: %s", exc)
        return 1
    try:
        cell = average_unit_cell(arr, lat,
                                 oversample=args.oversample,
                                 border_margin_px=args.border_margin)
    except Exception as exc:
        log.error("Unit-cell averaging failed: %s", exc)
        return 1

    print(f"Averaged {cell.n_cells} unit cell(s)")
    print(f"Cell size:  {cell.cell_size_px[1]} × {cell.cell_size_px[0]} px  "
          f"({cell.cell_size_m[1] * 1e9:.3f} × {cell.cell_size_m[0] * 1e9:.3f} nm)")
    print(f"|a|={lat.a_length_m * 1e9:.3f} nm   "
          f"|b|={lat.b_length_m * 1e9:.3f} nm   γ={lat.gamma_deg:.2f}°")

    if args.output is None:
        return 0
    suffix = args.output.suffix.lower()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        finite = cell.avg_cell[np.isfinite(cell.avg_cell)]
        vmin = float(np.percentile(finite, args.clip_low)) if finite.size else 0.0
        vmax = float(np.percentile(finite, args.clip_high)) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        fig, ax = plt.subplots(figsize=(4, 4), dpi=180)
        ax.imshow(cell.avg_cell, cmap=args.colormap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", origin="upper")
        ax.set_axis_off()
        ax.set_title(f"avg of {cell.n_cells} cells", fontsize=9)
        fig.tight_layout()
        fig.savefig(str(args.output))
        import matplotlib.pyplot as _plt
        _plt.close(fig)
    elif suffix in (".npy",):
        np.save(str(args.output), cell.avg_cell)
    else:
        log.error("Unsupported output suffix %r — use .png or .npy", suffix)
        return 1
    log.info("[OK] unit cell → %s", args.output)
    return 0


def _cmd_diag_z(args) -> int:
    """Diagnose Z-scale interpretation for a Createc .dat file."""
    from probeflow.io.readers.createc_dat import read_createc_dat_report
    from probeflow.io.common import _f, find_hdr, get_dac_bits, v_per_dac, z_scale_m_per_dac

    path = args.input
    try:
        report = read_createc_dat_report(path, include_raw=True)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    hdr = report.header
    bits = get_dac_bits(hdr)

    # 1. Header values
    print(f"File: {path.name}")
    print()
    print("Header values:")
    for key in ["Dacto[A]z", "GainZ", "ZPiezoconst", "DAC-Type",
                "Channels", "Channelselectval", "Length x[A]", "Length y[A]"]:
        val = find_hdr(hdr, key, None)
        print(f"  {key:<20} = {val if val is not None else 'MISSING'}")

    # 2. Dacto[A]z parsing
    raw_dacto_str = find_hdr(hdr, "Dacto[A]z", None)
    parsed_dacto = _f(raw_dacto_str)
    print()
    print("Dacto[A]z parsing:")
    print(f"  find_hdr result = {raw_dacto_str!r}")
    print(f"  _f(...)         = {parsed_dacto!r}")

    # 3. v_per_dac conventions
    vpd_half = 10.0 / (2 ** bits)
    vpd_full = 20.0 / (2 ** bits)
    print()
    print(f"v_per_dac (bits={bits}):")
    print(f"  vpd_half  code convention  (10/2^bits) = {vpd_half:.6e} V/DAC")
    print(f"  vpd_full  bipolar ±V_ref   (20/2^bits) = {vpd_full:.6e} V/DAC")

    # Parse scalars needed for candidates
    gainz = _f(find_hdr(hdr, "GainZ", None))
    zpiezo = _f(find_hdr(hdr, "ZPiezoconst", None))

    # 4. Five candidate Z scales in Å/DAC.
    # Dacto[A]z label is Ångstrom → candidates use it as-is in Å/DAC.
    # ZPiezoconst treated here as Å/V (its header label unit) for the zp candidates.
    candidates = [
        ("cand_dacto",          parsed_dacto),
        ("cand_dacto_div_gain", (parsed_dacto / gainz)
                                if (parsed_dacto is not None and gainz) else None),
        ("cand_dacto_mul_gain", (parsed_dacto * gainz)
                                if (parsed_dacto is not None and gainz is not None) else None),
        ("cand_zp_vpdhalf",     (zpiezo * vpd_half) if zpiezo is not None else None),
        ("cand_zp_vpdfull",     (zpiezo * vpd_full) if zpiezo is not None else None),
    ]

    # 5. What the code actually returns, converted to Å/DAC for comparison.
    code_scale_m = z_scale_m_per_dac(hdr, v_per_dac(bits))
    code_scale_a = code_scale_m * 1e10  # m/DAC → Å/DAC

    if parsed_dacto is not None:
        code_branch = "Dacto[A]z branch  (dz * 1e-9 m/DAC, treating dz as nm/DAC)"
    else:
        code_branch = "fallback branch   (2 * ZPiezoconst * vpd * 1e-9, treating ZPiezoconst as nm/V)"

    # 6. Raw DAC count range from the Z-forward plane.
    z_fwd_native = 0
    for info in report.channel_info:
        if info.semantic == "z" and info.direction == "forward":
            z_fwd_native = info.native_index
            break

    raw_z = report.raw_channels_dac[z_fwd_native]
    raw_min = float(np.nanmin(raw_z))
    raw_max = float(np.nanmax(raw_z))
    raw_ptp = raw_max - raw_min

    print()
    print(f"Z-forward raw DAC counts (channel native_index={z_fwd_native}):")
    print(f"  min           = {raw_min:+.6e}")
    print(f"  max           = {raw_max:+.6e}")
    print(f"  peak-to-peak  = {raw_ptp:.6e}")

    # Candidate table
    print()
    print(f"{'Candidate':<25}  {'Scale (Å/DAC)':>14}   Z range")
    print(f"{'-'*25}  {'-'*14}   -------")
    for name, scale in candidates:
        if scale is None:
            print(f"{name:<25}  {'MISSING':>14}   n/a")
        else:
            z_range_a = abs(scale) * raw_ptp
            print(f"{name:<25}  {scale:>14.4e}   {z_range_a:.2f} Å")

    print()
    print("Code result (z_scale_m_per_dac):")
    print(f"  scale       = {code_scale_m:.6e} m/DAC")
    print(f"              = {code_scale_a:.6e} Å/DAC")
    print(f"  branch      : {code_branch}")
    print(f"  Z range     = {code_scale_a * raw_ptp:.2f} Å")
    return 0


def _cmd_dat2sxm(args) -> int:
    from probeflow.io.converters.createc_dat_to_sxm import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-sxm"] + forwarded
    _main()
    return 0


def _cmd_dat2png(args) -> int:
    from probeflow.io.converters.createc_dat_to_png import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-png"] + forwarded
    _main()
    return 0


# ─── Parser construction ─────────────────────────────────────────────────────

def _cmd_spec_info(args) -> int:
    setup_logging(args.verbose)
    from probeflow.io.spectroscopy import read_spec_file, spec_channel_to_dict
    spec = read_spec_file(args.input)
    channels = list(spec.channel_order) if spec.channel_order else list(spec.channels.keys())
    if args.json:
        import json as _json
        out = {
            "file": str(args.input),
            "sweep_type": spec.metadata["sweep_type"],
            "measurement_family": spec.metadata.get("measurement_family"),
            "feedback_mode": spec.metadata.get("feedback_mode"),
            "derivative_label": spec.metadata.get("derivative_label"),
            "measurement_confidence": spec.metadata.get("measurement_confidence"),
            "measurement_evidence": spec.metadata.get("measurement_evidence"),
            "n_points": spec.metadata["n_points"],
            "channels": channels,
            "channel_info": [
                spec_channel_to_dict(spec.channel_info[ch])
                for ch in channels
                if ch in spec.channel_info
            ],
            "x_label": spec.x_label,
            "x_unit": spec.x_unit,
            "position_m": list(spec.position),
            "metadata": spec.metadata,
        }
        print(_json.dumps(out, indent=2))
    else:
        print(f"file        : {args.input}")
        print(f"sweep type  : {spec.metadata['sweep_type']}")
        if spec.metadata.get("measurement_family"):
            print(f"measurement : {spec.metadata['measurement_family']}")
        if spec.metadata.get("feedback_mode"):
            print(f"feedback    : {spec.metadata['feedback_mode']}")
        if spec.metadata.get("derivative_label"):
            print(f"derivative  : {spec.metadata['derivative_label']}")
        print(f"n_points    : {spec.metadata['n_points']}")
        print(f"channels    : {', '.join(channels)}")
        print(f"x_axis      : {spec.x_label}")
        x = spec.x_array
        print(f"x_range     : {x.min():.4g} to {x.max():.4g} {spec.x_unit}")
        px, py = spec.position
        print(f"position    : ({px*1e9:.3f}, {py*1e9:.3f}) nm")
        for key in ("bias_mv", "spec_freq_hz", "gain_pre_exp", "fb_log", "title"):
            if key in spec.metadata:
                print(f"{key:12s}: {spec.metadata[key]}")
    return 0


def _cmd_spec_plot(args) -> int:
    setup_logging(args.verbose)
    import matplotlib
    matplotlib.use("Agg" if args.output else "TkAgg", force=False)
    import matplotlib.pyplot as plt
    from probeflow.io.spectroscopy import read_spec_file
    from probeflow.analysis.spec_plot import plot_spectrum

    spec = read_spec_file(args.input)
    fig, ax = plt.subplots()
    plot_spectrum(spec, channel=args.channel, ax=ax)
    ax.set_title(Path(args.input).stem)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        log.info("[OK] %s → %s", args.input.name, args.output)
    else:
        plt.show()
    return 0


def _cmd_spec_overlay(args) -> int:
    setup_logging(args.verbose)
    import matplotlib
    matplotlib.use("Agg" if args.output else "TkAgg", force=False)
    import matplotlib.pyplot as plt
    from probeflow.io.spectroscopy import read_spec_file
    from probeflow.analysis.spec_plot import plot_spectra
    from probeflow.processing.spectroscopy import average_spectra

    specs = [read_spec_file(p) for p in args.inputs]
    fig, ax = plt.subplots()
    plot_spectra(specs, channel=args.channel, offset=args.offset, ax=ax)

    if args.average:
        ch_data = [s.channels[args.channel] for s in specs]
        avg = average_spectra(ch_data)
        ax.plot(specs[0].x_array, avg, "k--", linewidth=2, label="average")

    ax.legend(fontsize=7)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        log.info("[OK] overlay → %s", args.output)
    else:
        plt.show()
    return 0


def _cmd_spec_positions(args) -> int:
    setup_logging(args.verbose)
    import matplotlib
    matplotlib.use("Agg" if args.output else "TkAgg", force=False)
    import matplotlib.pyplot as plt
    from probeflow.io.spectroscopy import read_spec_file
    from probeflow.analysis.spec_plot import plot_spec_positions

    specs = [read_spec_file(p) for p in args.inputs]
    fig, ax = plt.subplots()
    plot_spec_positions(str(args.image), specs, ax=ax)
    ax.set_title(Path(args.image).stem)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        log.info("[OK] positions → %s", args.output)
    else:
        plt.show()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="probeflow",
        description="ProbeFlow — STM browser, processor, and converter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  probeflow gui\n"
            "  probeflow info scan.sxm\n"
            "  probeflow plane-bg scan.sxm --order 2 -o scan_bg.sxm\n"
            "  probeflow align-rows scan.sxm --method median --png -o scan.png\n"
            "  probeflow smooth scan.sxm --sigma 1.5 --png\n"
            "  probeflow pipeline scan.sxm \\\n"
            "      --steps align-rows:median plane-bg:1 smooth:1.5 \\\n"
            "      --png -o scan_processed.png\n"
            "  probeflow periodicity scan.sxm --n-peaks 3 --json\n"
            "  probeflow dat2sxm -- --input-dir data/scans --output-dir out/sxm\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    # ── conversion ──
    dat2sxm = sub.add_parser("dat2sxm",
        help="Createc .dat → Nanonis .sxm (delegates to dat-sxm)")
    dat2sxm.add_argument("rest", nargs=argparse.REMAINDER,
        help="Arguments forwarded to dat-sxm (prefix with '--')")
    dat2sxm.set_defaults(func=_cmd_dat2sxm)

    dat2png = sub.add_parser("dat2png",
        help="Createc .dat → PNG previews (delegates to dat-png)")
    dat2png.add_argument("rest", nargs=argparse.REMAINDER,
        help="Arguments forwarded to dat-png (prefix with '--')")
    dat2png.set_defaults(func=_cmd_dat2png)

    sxm2png_p = sub.add_parser("sxm2png",
        help="Export a plane of an .sxm to a colorised PNG")
    sxm2png_p.add_argument("input", type=Path)
    sxm2png_p.add_argument("-o", "--output", type=Path, default=None)
    sxm2png_p.add_argument("--plane", type=int, default=0)
    sxm2png_p.add_argument("--colormap", default="gray")
    sxm2png_p.add_argument("--clip-low",  type=float, default=1.0)
    sxm2png_p.add_argument("--clip-high", type=float, default=99.0)
    sxm2png_p.add_argument("--no-scalebar", action="store_true")
    sxm2png_p.add_argument("--scalebar-unit", choices=("nm", "Å", "pm"), default="nm")
    sxm2png_p.add_argument("--scalebar-pos",
                           choices=("bottom-right", "bottom-left"), default="bottom-right")
    sxm2png_p.add_argument("--verbose", action="store_true")
    sxm2png_p.set_defaults(func=_cmd_sxm2png)

    # ── processing ──
    plane_bg = sub.add_parser("plane-bg",
        help="Subtract a polynomial plane background (ROI-aware)")
    _add_common_io(plane_bg, out_suffix="_bg.sxm")
    plane_bg.add_argument("--order", type=int, default=1, choices=(1, 2, 3, 4),
        help="Polynomial order (1=plane, 2=quadratic, 3=cubic, 4=quartic)")
    plane_bg.add_argument("--step-tolerance", action="store_true",
        help="Exclude step-edge pixels from the polynomial fit")
    # Fit ROI
    plane_bg.add_argument("--fit-roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Fit background to pixels within this persisted ROI only")
    plane_bg.add_argument("--fit-roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Inline rectangular fit-ROI (pixel coordinates)")
    plane_bg.add_argument("--fit-roi-invert", type=str, default=None, metavar="NAME_OR_ID",
        help="Fit background to the complement of the named ROI")
    plane_bg.add_argument("--fit-roi-union", type=str, default=None, metavar="NAME[,NAME,...]",
        help="Fit background to the union of the named ROIs (comma-separated)")
    # Apply ROI
    plane_bg.add_argument("--apply-roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Apply the fitted background only within this persisted ROI")
    plane_bg.add_argument("--apply-roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Inline rectangular apply-ROI (pixel coordinates)")
    # Exclude ROI
    plane_bg.add_argument("--exclude-roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Exclude this persisted ROI from the polynomial fit")
    plane_bg.add_argument("--exclude-roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Inline rectangular region to exclude from the fit")
    # Sidecar for named ROI lookup
    plane_bg.add_argument("--sidecar", type=Path, default=None,
        help="Explicit ROI/provenance sidecar .json path (default: <input>.rois.json, then provenance)")
    plane_bg.set_defaults(func=_cmd_plane_bg)

    align = sub.add_parser("align-rows",
        help="Fix per-row offsets (median / mean / linear)")
    _add_common_io(align, out_suffix="_aligned.sxm")
    align.add_argument("--method", choices=("median", "mean", "linear"),
                       default="median")
    align.set_defaults(func=lambda a: _cmd_single_op(a, _op_align_rows(a.method)))

    bad = sub.add_parser("remove-bad-lines",
        help="Interpolate outlier scan lines (MAD-based)")
    _add_common_io(bad, out_suffix="_clean.sxm")
    bad.add_argument("--threshold-mad", type=float, default=5.0)
    bad.set_defaults(func=lambda a: _cmd_single_op(a,
        _op_remove_bad_lines(a.threshold_mad)))

    facet = sub.add_parser("facet-level",
        help="Plane-level using only flat-terrace pixels (stepped surfaces)")
    _add_common_io(facet, out_suffix="_facet.sxm")
    facet.add_argument("--threshold-deg", type=float, default=3.0,
        help="Max slope angle (degrees) treated as 'flat'")
    facet.set_defaults(func=lambda a: _cmd_single_op(a,
        _op_facet_level(a.threshold_deg)))

    smooth = sub.add_parser("smooth",
        help="Isotropic Gaussian smoothing")
    _add_common_io(smooth, out_suffix="_smooth.sxm")
    smooth.add_argument("--sigma", type=float, default=1.0,
        help="Standard deviation in pixels")
    smooth.set_defaults(func=lambda a: _cmd_single_op(a, _op_smooth(a.sigma)))

    edge = sub.add_parser("edge",
        help="Edge detection (Laplacian / LoG / DoG)")
    _add_common_io(edge, out_suffix="_edge.sxm")
    edge.add_argument("--method", choices=("laplacian", "log", "dog"),
                      default="laplacian")
    edge.add_argument("--sigma",  type=float, default=1.0)
    edge.add_argument("--sigma2", type=float, default=2.0)
    edge.set_defaults(func=lambda a: _cmd_single_op(a,
        _op_edge(a.method, a.sigma, a.sigma2)))

    fft = sub.add_parser("fft",
        help="FFT low-pass / high-pass filter")
    _add_common_io(fft, out_suffix="_fft.sxm")
    fft.add_argument("--mode", choices=("low_pass", "high_pass"),
                     default="low_pass")
    fft.add_argument("--cutoff", type=float, default=0.1,
        help="Cutoff fraction of Nyquist (0–1)")
    fft.add_argument("--window", choices=("hanning", "hamming", "rect"),
                     default="hanning")
    fft.set_defaults(func=lambda a: _cmd_single_op(a,
        _op_fft(a.mode, a.cutoff, a.window)))

    flip_h = sub.add_parser("flip-h",
        help="Flip scan left-to-right (mirror about vertical axis)")
    _add_common_io(flip_h, out_suffix="_fliph.sxm")
    flip_h.set_defaults(func=lambda a: _cmd_single_op(a, _op_flip_horizontal()))

    flip_v = sub.add_parser("flip-v",
        help="Flip scan top-to-bottom (mirror about horizontal axis)")
    _add_common_io(flip_v, out_suffix="_flipv.sxm")
    flip_v.set_defaults(func=lambda a: _cmd_single_op(a, _op_flip_vertical()))

    rot90 = sub.add_parser("rotate-90",
        help="Rotate scan 90° clockwise")
    _add_common_io(rot90, out_suffix="_rot90.sxm")
    rot90.set_defaults(func=lambda a: _cmd_single_op(a, _op_rotate_90_cw()))

    rot180 = sub.add_parser("rotate-180",
        help="Rotate scan 180°")
    _add_common_io(rot180, out_suffix="_rot180.sxm")
    rot180.set_defaults(func=lambda a: _cmd_single_op(a, _op_rotate_180()))

    rot270 = sub.add_parser("rotate-270",
        help="Rotate scan 270° clockwise (90° counter-clockwise)")
    _add_common_io(rot270, out_suffix="_rot270.sxm")
    rot270.set_defaults(func=lambda a: _cmd_single_op(a, _op_rotate_270_cw()))

    rotate = sub.add_parser("rotate",
        help="Rotate scan by an arbitrary angle (CCW positive, canvas expands)")
    _add_common_io(rotate, out_suffix="_rotated.sxm")
    rotate.add_argument("--angle", type=float, default=0.0,
        help="Rotation angle in degrees (positive = counter-clockwise)")
    rotate.add_argument("--order", type=int, default=1, choices=(0, 1, 2, 3),
        help="Interpolation order: 0=nearest, 1=bilinear (default), 2=quad, 3=bicubic")
    rotate.set_defaults(func=lambda a: _cmd_single_op(a,
        _op_rotate_arbitrary(a.angle, a.order)))

    grains = sub.add_parser("grains",
        help="Detect grains / islands by threshold and print statistics")
    grains.add_argument("input", type=Path)
    grains.add_argument("--plane", type=int, default=0)
    grains.add_argument("--threshold", type=float, default=50.0,
        help="Percentile of data used as threshold")
    grains.add_argument("--below", action="store_true",
        help="Detect depressions (below threshold) instead of islands")
    grains.add_argument("--min-px", type=int, default=5)
    grains.add_argument("--save-mask", type=Path, default=None,
        help="Also save a binary PNG mask of grain pixels")
    grains.add_argument("--json", action="store_true")
    grains.add_argument("--verbose", action="store_true")
    grains.set_defaults(func=_cmd_grains)

    autoclip = sub.add_parser("autoclip",
        help="Compute GMM-based auto-clip percentiles for display")
    autoclip.add_argument("input", type=Path)
    autoclip.add_argument("--plane", type=int, default=0)
    autoclip.add_argument("--json", action="store_true")
    autoclip.add_argument("--verbose", action="store_true")
    autoclip.set_defaults(func=_cmd_autoclip)

    period = sub.add_parser("periodicity",
        help="Find dominant spatial periodicities via power spectrum")
    period.add_argument("input", type=Path)
    period.add_argument("--plane", type=int, default=0)
    period.add_argument("--n-peaks", type=int, default=5)
    period.add_argument("--json", action="store_true")
    period.add_argument("--verbose", action="store_true")
    period.set_defaults(func=_cmd_periodicity)

    # ── Optional feature commands: counting / lattice / denoise / classify ──
    particles = sub.add_parser("particles",
        help="Segment bright (or dark) particles / molecules on a scan plane")
    particles.add_argument("input", type=Path)
    particles.add_argument("-o", "--output", type=Path, default=None,
        help="Optional .json output with full particle list + scan provenance")
    particles.add_argument("--plane", type=int, default=0)
    particles.add_argument("--threshold", choices=("otsu", "manual", "adaptive"),
                           default="otsu")
    particles.add_argument("--manual-value", type=float, default=None,
        help="0-255 byte cutoff when --threshold=manual")
    particles.add_argument("--invert", action="store_true",
        help="Segment depressions instead of bright features")
    particles.add_argument("--min-area", type=float, default=0.5,
        help="Minimum particle area (nm²; default 0.5)")
    particles.add_argument("--max-area", type=float, default=None,
        help="Maximum particle area (nm²; default: no limit)")
    particles.add_argument("--sigma-clip", type=float, default=2.0,
        help="Drop particles more than this many σ from the mean area")
    particles.add_argument("--no-sigma-clip", action="store_true",
        help="Disable σ-clipping of particle areas")
    particles.add_argument("--clip-low", type=float, default=1.0)
    particles.add_argument("--clip-high", type=float, default=99.0)
    particles.add_argument("--limit", type=int, default=20,
        help="Max particles printed to stdout (table mode)")
    particles.add_argument("--json", action="store_true")
    particles.add_argument("--verbose", action="store_true")
    particles.set_defaults(func=_cmd_particles)

    count = sub.add_parser("count",
        help="Count features by template matching (AiSurf atom_counting)")
    count.add_argument("input", type=Path)
    count.add_argument("--template", type=Path, required=True,
        help="Template image — PNG or another scan file")
    count.add_argument("-o", "--output", type=Path, default=None,
        help="Optional .json output with all detections")
    count.add_argument("--plane", type=int, default=0)
    count.add_argument("--min-corr", type=float, default=0.5,
        help="Minimum normalised cross-correlation (0.4-0.6 typical)")
    count.add_argument("--min-distance", type=float, default=None,
        help="Minimum feature separation (nm); default = half template side")
    count.add_argument("--clip-low", type=float, default=1.0)
    count.add_argument("--clip-high", type=float, default=99.0)
    count.add_argument("--json", action="store_true")
    count.add_argument("--verbose", action="store_true")
    count.set_defaults(func=_cmd_count)

    tv = sub.add_parser("tv-denoise",
        help="Total-variation denoising (Huber-ROF / TV-L1)")
    _add_common_io(tv, out_suffix="_tv.sxm")
    tv.add_argument("--method", choices=("huber_rof", "tv_l1"),
                    default="huber_rof")
    tv.add_argument("--lam", type=float, default=0.05,
                    help="Data-fidelity weight (higher = closer to input)")
    tv.add_argument("--alpha", type=float, default=0.05,
                    help="Huber smoothing parameter (huber_rof only)")
    tv.add_argument("--tau", type=float, default=0.25)
    tv.add_argument("--max-iter", type=int, default=500)
    tv.add_argument("--nabla-comp", choices=("both", "x", "y"),
                    default="both",
                    help="'x' removes vertical scratches; 'y' removes horizontal")
    tv.set_defaults(func=_cmd_tv_denoise)

    lat = sub.add_parser("lattice",
        help="SIFT-based primitive lattice vector extraction")
    lat.add_argument("input", type=Path)
    lat.add_argument("-o", "--output", type=Path, default=None,
        help="Optional output — .pdf for a report, .json for raw numbers")
    lat.add_argument("--plane", type=int, default=0)
    lat.add_argument("--contrast-threshold", type=float, default=0.003)
    lat.add_argument("--sigma", type=float, default=4.0)
    lat.add_argument("--cluster-kp-low", type=int, default=2)
    lat.add_argument("--cluster-kp-high", type=int, default=12)
    lat.add_argument("--cluster-knn-low", type=int, default=6)
    lat.add_argument("--cluster-knn-high", type=int, default=24)
    lat.add_argument("--colormap", default="gray")
    lat.add_argument("--clip-low", type=float, default=1.0)
    lat.add_argument("--clip-high", type=float, default=99.0)
    lat.add_argument("--json", action="store_true")
    lat.add_argument("--verbose", action="store_true")
    lat.set_defaults(func=_cmd_lattice)

    classify = sub.add_parser("classify",
        help="Few-shot classify particles against labelled samples")
    classify.add_argument("input", type=Path)
    classify.add_argument("--samples", type=Path, required=True,
        help="JSON file with sample particles (each object must include "
             "'class_name' / 'label' and all Particle fields)")
    classify.add_argument("-o", "--output", type=Path, default=None)
    classify.add_argument("--plane", type=int, default=0)
    classify.add_argument("--encoder", choices=("raw", "pca_kmeans"),
                          default="raw")
    classify.add_argument("--threshold-method",
                          choices=("gmm", "otsu", "distribution"),
                          default="gmm")
    classify.add_argument("--min-area", type=float, default=0.5)
    classify.add_argument("--sigma-clip", type=float, default=2.0)
    classify.add_argument("--no-sigma-clip", action="store_true")
    classify.add_argument("--json", action="store_true")
    classify.add_argument("--verbose", action="store_true")
    classify.set_defaults(func=_cmd_classify)

    # ── line profile ──
    profile = sub.add_parser("profile",
        help="Sample z along a straight segment (CSV / JSON / PNG output)")
    profile.add_argument("input", type=Path)
    profile.add_argument("-o", "--output", type=Path, default=None,
        help="Output suffix selects format: .csv | .json | .png. "
             "Omit for tab-separated stdout.")
    profile.add_argument("--plane", type=int, default=0)
    profile.add_argument("--p0", type=float, nargs=2, metavar=("X", "Y"),
        help="Start point in pixel coordinates")
    profile.add_argument("--p1", type=float, nargs=2, metavar=("X", "Y"),
        help="End point in pixel coordinates")
    profile.add_argument("--p0-nm", type=float, nargs=2, metavar=("X", "Y"),
        help="Start point in nanometres (alternative to --p0)")
    profile.add_argument("--p1-nm", type=float, nargs=2, metavar=("X", "Y"),
        help="End point in nanometres (alternative to --p1)")
    profile.add_argument("--n-samples", type=int, default=None,
        help="Sample count (default: ceil of pixel length + 1)")
    profile.add_argument("--width", type=float, default=1.0,
        help="Perpendicular swath width in pixels (averages across; default 1)")
    profile.add_argument("--interp", choices=("linear", "nearest"),
        default="linear")
    profile.add_argument("--roi-line", type=float, nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Line ROI as pixel coordinates (alternative to --p0/--p1)")
    profile.add_argument("--roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Use a persisted line ROI from the scan sidecar by name or UUID")
    profile.add_argument("--verbose", action="store_true")
    profile.set_defaults(func=_cmd_profile)

    # ── ROI-aware histogram ──
    hist = sub.add_parser("histogram",
        help="Pixel-value histogram (optionally restricted to an ROI)")
    hist.add_argument("input", type=Path)
    hist.add_argument("-o", "--output", type=Path, default=None,
        help="Output: .csv for (counts, edges) table, .png for a plot. "
             "Omit for tab-separated stdout.")
    hist.add_argument("--plane", type=int, default=0)
    hist.add_argument("--bins", type=int, default=256,
        help="Number of histogram bins (default 256)")
    hist.add_argument("--roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Rectangular ROI as pixel coordinates (x0 y0 x1 y1)")
    hist.add_argument("--roi-polygon", type=float, nargs="+",
        metavar="X Y",
        help="Polygon ROI as alternating x y pixel coordinates (at least 3 vertices)")
    hist.add_argument("--roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Use a persisted ROI from the scan sidecar by name or UUID")
    hist.add_argument("--sidecar", type=Path, default=None,
        help="Explicit ROI/provenance sidecar .json path (default: <input>.rois.json, then provenance)")
    hist.add_argument("--verbose", action="store_true")
    hist.set_defaults(func=_cmd_histogram)

    # ── ROI-aware FFT magnitude spectrum ──
    fft_spec = sub.add_parser("fft-spectrum",
        help="2-D FFT magnitude spectrum of a scan plane or ROI")
    fft_spec.add_argument("input", type=Path)
    fft_spec.add_argument("-o", "--output", type=Path, default=None,
        help="Output PNG for the magnitude image. Omit to print peak summary.")
    fft_spec.add_argument("--plane", type=int, default=0)
    fft_spec.add_argument("--window",
        choices=("hann", "tukey", "none"), default="hann",
        help="Spatial window applied before the DFT (default: hann)")
    fft_spec.add_argument("--window-param", type=float, default=0.25,
        help="Tukey plateau fraction [0,1] (ignored for other windows)")
    fft_spec.add_argument("--no-log", dest="log_scale", action="store_false",
        help="Disable log1p scaling of the magnitude (linear output)")
    fft_spec.add_argument("--roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Rectangular ROI as pixel coordinates")
    fft_spec.add_argument("--roi-polygon", type=float, nargs="+",
        metavar="X Y",
        help="Polygon ROI as alternating x y pixel coordinates (≥3 vertices)")
    fft_spec.add_argument("--roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Use a persisted ROI from the scan sidecar by name or UUID")
    fft_spec.add_argument("--sidecar", type=Path, default=None,
        help="Explicit ROI/provenance sidecar .json path (default: <input>.rois.json, then provenance)")
    fft_spec.add_argument("--colormap", default="gray")
    fft_spec.add_argument("--verbose", action="store_true")
    fft_spec.set_defaults(func=_cmd_fft_spectrum)

    # ── unit-cell averaging ──
    ucell = sub.add_parser("unit-cell",
        help="Extract lattice and average all unit cells into a canonical motif")
    ucell.add_argument("input", type=Path)
    ucell.add_argument("-o", "--output", type=Path, default=None,
        help="Output suffix selects format: .png (image) or .npy (raw array)")
    ucell.add_argument("--plane", type=int, default=0)
    ucell.add_argument("--oversample", type=float, default=1.5,
        help="Output pixel count is oversample × max(|a|, |b|) per side")
    ucell.add_argument("--border-margin", type=int, default=4,
        help="Skip lattice sites within this many pixels of the image border")
    ucell.add_argument("--colormap", default="gray")
    ucell.add_argument("--clip-low", type=float, default=1.0)
    ucell.add_argument("--clip-high", type=float, default=99.0)
    ucell.add_argument("--verbose", action="store_true")
    ucell.set_defaults(func=_cmd_unit_cell)

    # ── pipeline ──
    pipe = sub.add_parser("pipeline",
        help="Apply a chain of processing steps in one call")
    pipe.add_argument("input", type=Path)
    pipe.add_argument("-o", "--output", type=Path, default=None)
    pipe.add_argument("--plane", type=int, default=0)
    pipe.add_argument("--png", action="store_true")
    pipe.add_argument("--steps", nargs="+", required=True, metavar="STEP",
        help=("Space-separated pipeline steps, each 'name[:params]' with "
              "params comma-separated. See examples."))
    pipe.add_argument("--colormap", default="gray")
    pipe.add_argument("--clip-low",  type=float, default=1.0)
    pipe.add_argument("--clip-high", type=float, default=99.0)
    pipe.add_argument("--no-scalebar", action="store_true")
    pipe.add_argument("--scalebar-unit", choices=("nm", "Å", "pm"), default="nm")
    pipe.add_argument("--scalebar-pos",
                      choices=("bottom-right", "bottom-left"), default="bottom-right")
    pipe.add_argument("--verbose", action="store_true")
    pipe.set_defaults(func=_cmd_pipeline)

    # ── prepared downstream handoff ──
    prep = sub.add_parser("prepare-png",
        help="Write a downstream-analysis PNG plus provenance sidecar")
    prep.add_argument("input", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--plane", type=int, default=0)
    prep.add_argument("--steps", nargs="*", default=[], metavar="STEP",
        help=("Optional processing steps before export, using pipeline syntax. "
              "If omitted, the sidecar warns that no background/line correction "
              "is recorded."))
    prep.add_argument("--colormap", default="gray")
    prep.add_argument("--clip-low", type=float, default=1.0)
    prep.add_argument("--clip-high", type=float, default=99.0)
    prep.add_argument("--no-scalebar", action="store_true", default=True,
        help="Disable scale bar on the prepared PNG (default; PNG handoffs often prefer raw image content)")
    prep.add_argument("--with-scalebar", dest="no_scalebar", action="store_false",
        help="Include a visual scale bar in the PNG")
    prep.add_argument("--verbose", action="store_true")
    prep.set_defaults(func=_cmd_prepare_png)

    # ── info / gui ──
    info = sub.add_parser("info", help="Print .sxm header metadata")
    info.add_argument("input", type=Path)
    info.add_argument("--json", action="store_true")
    info.add_argument("--verbose", action="store_true")
    info.set_defaults(func=_cmd_info)

    diag_z = sub.add_parser("diag-z",
        help="Diagnose Z-scale candidates for a Createc .dat file")
    diag_z.add_argument("input", type=Path,
        help="Createc .dat file to inspect")
    diag_z.set_defaults(func=_cmd_diag_z)

    gui = sub.add_parser("gui", help="Launch the ProbeFlow graphical interface")
    gui.set_defaults(func=_cmd_gui)

    # ── any-in/any-out convert ──
    convert = sub.add_parser("convert",
        help=("Convert any supported scan (.sxm, .dat) "
              "to any supported output (.sxm, .png, .pdf, .csv)"))
    convert.add_argument("input", type=Path,
        help="Input scan (format auto-detected from content)")
    convert.add_argument("output", type=Path,
        help="Output file (format auto-detected from suffix)")
    convert.add_argument("--plane", type=int, default=0,
        help="Plane index for single-plane outputs (default 0)")
    convert.add_argument("--colormap", default=None,
        help="Matplotlib colormap (for PNG / PDF)")
    convert.add_argument("--clip-low", type=float, default=None,
        help="Lower percentile clip (default 1.0)")
    convert.add_argument("--clip-high", type=float, default=None,
        help="Upper percentile clip (default 99.0)")
    convert.add_argument("--verbose", action="store_true")
    convert.set_defaults(func=_cmd_convert)

    # ── spectroscopy ──
    spec_info = sub.add_parser("spec-info",
        help="Print metadata from a Createc .VERT spectroscopy file")
    spec_info.add_argument("input", type=Path, help="Path to a .VERT file")
    spec_info.add_argument("--json", action="store_true",
        help="Output as JSON")
    spec_info.add_argument("--verbose", action="store_true")
    spec_info.set_defaults(func=_cmd_spec_info)

    spec_plot = sub.add_parser("spec-plot",
        help="Quick plot of a single .VERT spectrum")
    spec_plot.add_argument("input", type=Path, help="Path to a .VERT file")
    spec_plot.add_argument("--channel", default="Z",
        help="Data channel to plot: I, Z, or V (default: Z)")
    spec_plot.add_argument("-o", "--output", type=Path, default=None,
        help="Save plot to this path instead of showing it interactively")
    spec_plot.add_argument("--verbose", action="store_true")
    spec_plot.set_defaults(func=_cmd_spec_plot)

    spec_overlay = sub.add_parser("spec-overlay",
        help="Overlay multiple .VERT spectra on one axes")
    spec_overlay.add_argument("inputs", nargs="+", type=Path,
        help="Two or more .VERT files")
    spec_overlay.add_argument("--channel", default="Z",
        help="Data channel to plot (default: Z)")
    spec_overlay.add_argument("--offset", type=float, default=0.0,
        help="Vertical offset between curves for waterfall display")
    spec_overlay.add_argument("--average", action="store_true",
        help="Also plot the mean of all spectra")
    spec_overlay.add_argument("-o", "--output", type=Path, default=None,
        help="Save plot to this path instead of showing it interactively")
    spec_overlay.add_argument("--verbose", action="store_true")
    spec_overlay.set_defaults(func=_cmd_spec_overlay)

    spec_pos = sub.add_parser("spec-positions",
        help="Show tip positions of .VERT files overlaid on an .sxm topography")
    spec_pos.add_argument("image", type=Path, help="Path to the .sxm topography file")
    spec_pos.add_argument("inputs", nargs="+", type=Path,
        help="One or more .VERT files whose positions to mark")
    spec_pos.add_argument("-o", "--output", type=Path, default=None,
        help="Save plot to this path instead of showing it interactively")
    spec_pos.add_argument("--verbose", action="store_true")
    spec_pos.set_defaults(func=_cmd_spec_positions)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
