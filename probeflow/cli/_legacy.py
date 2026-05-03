"""
ProbeFlow unified command-line interface.

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

    # Endpoints: --p0 and --p1 in pixels, OR --p0-nm / --p1-nm in nanometres.
    if args.p0_nm is not None and args.p1_nm is not None:
        p0 = (args.p0_nm[0] * 1e-9 / px_x, args.p0_nm[1] * 1e-9 / px_y)
        p1 = (args.p1_nm[0] * 1e-9 / px_x, args.p1_nm[1] * 1e-9 / px_y)
    elif args.p0 is not None and args.p1 is not None:
        p0 = tuple(args.p0)
        p1 = tuple(args.p1)
    else:
        log.error("Provide either --p0/--p1 (px) or --p0-nm/--p1-nm")
        return 1

    s_m, z = _proc.line_profile(
        arr, p0, p1,
        pixel_size_x_m=px_x, pixel_size_y_m=px_y,
        n_samples=args.n_samples,
        width_px=args.width,
        interp=args.interp,
    )

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
        help="Subtract a polynomial plane background")
    _add_common_io(plane_bg, out_suffix="_bg.sxm")
    plane_bg.add_argument("--order", type=int, default=1, choices=(1, 2, 3, 4),
        help="Polynomial order (1=plane, 2=quadratic, 3=cubic, 4=quartic)")
    plane_bg.set_defaults(func=lambda a: _cmd_single_op(a, _op_plane_bg(a.order)))

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
    profile.add_argument("--verbose", action="store_true")
    profile.set_defaults(func=_cmd_profile)

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
