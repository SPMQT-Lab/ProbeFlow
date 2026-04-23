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

  Inspection / GUI
    info              Print header metadata of an .sxm file
    gui               Launch the ProbeFlow graphical interface

Run ``probeflow <command> --help`` for the options of any subcommand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from probeflow import processing as _proc
from probeflow.common import setup_logging
from probeflow.sxm_io import (
    parse_sxm_header,
    read_all_sxm_planes,
    read_sxm_plane,
    sxm_dims,
    sxm_scan_range,
    write_sxm_with_planes,
)

log = logging.getLogger(__name__)


# ─── Colormap helpers (no PySide6 dependency) ────────────────────────────────

def _lut_from_matplotlib(name: str) -> np.ndarray:
    """Return a (256, 3) uint8 LUT from a matplotlib colormap name."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)  # avoid starting Qt from the CLI
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(name)
    except Exception:
        # Fallback to gray
        return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)
    rgba = (cmap(np.linspace(0, 1, 256))[:, :3] * 255.0).astype(np.uint8)
    return rgba


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
    sxm_path: Path,
    plane_idx: int,
    op: Callable[[np.ndarray], np.ndarray],
) -> Tuple[dict, List[np.ndarray], np.ndarray]:
    """Apply ``op`` to one plane, return (hdr, all_planes_modified, processed)."""
    hdr, planes = read_all_sxm_planes(sxm_path, orient=True)
    if plane_idx >= len(planes):
        raise ValueError(
            f"Plane {plane_idx} not present — file has {len(planes)} plane(s)"
        )
    processed = op(planes[plane_idx])
    # Only rewrite the forward-plane half to keep forward/backward self-consistent
    planes[plane_idx] = processed
    return hdr, planes, processed


def _write_output(
    args: argparse.Namespace,
    hdr: dict,
    planes: List[np.ndarray],
    processed: np.ndarray,
    default_suffix: str,
) -> Path:
    """Write either an .sxm (all planes) or a colorised PNG (selected plane)."""
    if args.png:
        out_path = _derive_output(args, ".png" if default_suffix == ".sxm"
                                   else default_suffix)
        w_m, h_m = sxm_scan_range(hdr)
        _proc.export_png(
            processed, out_path, args.colormap,
            args.clip_low, args.clip_high,
            lut_fn=_lut_from_matplotlib,
            scan_range_m=(w_m, h_m),
            add_scalebar=not args.no_scalebar,
            scalebar_unit=args.scalebar_unit,
            scalebar_pos=args.scalebar_pos,
        )
    else:
        out_path = _derive_output(args, default_suffix)
        write_sxm_with_planes(args.input, out_path, planes)
    log.info("[OK] %s → %s", args.input.name, out_path)
    return out_path


# ─── Pipeline atoms (each returns a new ndarray) ─────────────────────────────

def _op_plane_bg(order: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: _proc.subtract_background(a, order=order)


def _op_align_rows(method: str) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: _proc.align_rows(a, method=method)


def _op_remove_bad_lines(mad: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: _proc.remove_bad_lines(a, threshold_mad=mad)


def _op_facet_level(deg: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: _proc.facet_level(a, threshold_deg=deg)


def _op_smooth(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: _proc.gaussian_smooth(a, sigma_px=sigma)


def _op_edge(method: str, sigma: float,
             sigma2: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: _proc.edge_detect(a, method=method, sigma=sigma, sigma2=sigma2)


def _op_fft(mode: str, cutoff: float,
            window: str) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: _proc.fourier_filter(a, mode=mode, cutoff=cutoff, window=window)


# ─── Per-command runners ─────────────────────────────────────────────────────

def _cmd_single_op(args, op: Callable[[np.ndarray], np.ndarray]) -> int:
    setup_logging(args.verbose)
    hdr, planes, processed = _apply_to_plane(args.input, args.plane, op)
    _write_output(args, hdr, planes, processed, default_suffix=".sxm")
    return 0


def _cmd_grains(args) -> int:
    setup_logging(args.verbose)
    arr = read_sxm_plane(args.input, plane_idx=args.plane)
    if arr is None:
        log.error("Could not read plane %d of %s", args.plane, args.input)
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
    arr = read_sxm_plane(args.input, plane_idx=args.plane)
    if arr is None:
        log.error("Could not read plane %d of %s", args.plane, args.input)
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
    hdr = parse_sxm_header(args.input)
    Nx, Ny = sxm_dims(hdr)
    w_m, h_m = sxm_scan_range(hdr)
    if w_m <= 0 or h_m <= 0 or Nx <= 0 or Ny <= 0:
        log.error("Invalid SCAN_RANGE / SCAN_PIXELS in %s", args.input)
        return 1
    arr = read_sxm_plane(args.input, plane_idx=args.plane)
    if arr is None:
        log.error("Could not read plane %d of %s", args.plane, args.input)
        return 1
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
    setup_logging(args.verbose)
    arr = read_sxm_plane(args.input, plane_idx=args.plane)
    if arr is None:
        log.error("Could not read plane %d of %s", args.plane, args.input)
        return 1
    hdr = parse_sxm_header(args.input)
    w_m, h_m = sxm_scan_range(hdr)
    out = args.output or args.input.with_suffix(".png")
    _proc.export_png(
        arr, out, args.colormap,
        args.clip_low, args.clip_high,
        lut_fn=_lut_from_matplotlib,
        scan_range_m=(w_m, h_m),
        add_scalebar=not args.no_scalebar,
        scalebar_unit=args.scalebar_unit,
        scalebar_pos=args.scalebar_pos,
    )
    log.info("[OK] %s → %s", args.input.name, out)
    return 0


def _cmd_info(args) -> int:
    setup_logging(args.verbose)
    hdr = parse_sxm_header(args.input)
    Nx, Ny = sxm_dims(hdr)
    w_m, h_m = sxm_scan_range(hdr)
    if args.json:
        print(json.dumps({
            "file": str(args.input),
            "Nx": Nx, "Ny": Ny,
            "scan_range_m": [w_m, h_m],
            "header": hdr,
        }, indent=2))
        return 0
    print(f"file      : {args.input}")
    print(f"pixels    : {Nx} x {Ny}")
    print(f"scan size : {w_m*1e9:.3f} nm × {h_m*1e9:.3f} nm")
    for key in ("REC_DATE", "REC_TIME", "BIAS", "SCAN_DIR",
                "SCAN_ANGLE", "SCAN_OFFSET", "COMMENT"):
        if key in hdr and hdr[key]:
            print(f"{key:10s}: {hdr[key]}")
    return 0


def _cmd_pipeline(args) -> int:
    """Apply a sequence of processing steps in order."""
    setup_logging(args.verbose)
    steps_spec = args.steps
    if not steps_spec:
        log.error("--steps is required, e.g. "
                  "--steps align-rows:median plane-bg:1 smooth:1.5")
        return 2

    ops: List[Callable[[np.ndarray], np.ndarray]] = []
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
            ops.append(_op_plane_bg(order))
        elif name == "facet-level":
            deg = float(parts[0]) if parts else 3.0
            ops.append(_op_facet_level(deg))
        elif name == "smooth":
            sigma = float(parts[0]) if parts else 1.0
            ops.append(_op_smooth(sigma))
        elif name == "edge":
            method = parts[0] if parts else "laplacian"
            sigma  = float(parts[1]) if len(parts) > 1 else 1.0
            sigma2 = float(parts[2]) if len(parts) > 2 else 2.0
            ops.append(_op_edge(method, sigma, sigma2))
        elif name == "fft":
            mode   = parts[0] if parts else "low_pass"
            cutoff = float(parts[1]) if len(parts) > 1 else 0.1
            window = parts[2] if len(parts) > 2 else "hanning"
            ops.append(_op_fft(mode, cutoff, window))
        else:
            log.error("Unknown pipeline step: %r", name)
            return 2

    def _compose(a: np.ndarray) -> np.ndarray:
        for op in ops:
            a = op(a)
        return a

    hdr, planes, processed = _apply_to_plane(args.input, args.plane, _compose)
    _write_output(args, hdr, planes, processed, default_suffix=".sxm")
    return 0


def _cmd_gui(_args) -> int:
    from probeflow.gui import main as _gui_main
    _gui_main()
    return 0


def _cmd_dat2sxm(args) -> int:
    from probeflow.dat_sxm import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-sxm"] + forwarded
    _main()
    return 0


def _cmd_dat2png(args) -> int:
    from probeflow.dat_png import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-png"] + forwarded
    _main()
    return 0


# ─── Parser construction ─────────────────────────────────────────────────────

def _cmd_spec_info(args) -> int:
    setup_logging(args.verbose)
    from probeflow.spec_io import read_spec_file
    spec = read_spec_file(args.input)
    if args.json:
        import json as _json
        out = {
            "file": str(args.input),
            "sweep_type": spec.metadata["sweep_type"],
            "n_points": spec.metadata["n_points"],
            "channels": list(spec.channels.keys()),
            "x_label": spec.x_label,
            "x_unit": spec.x_unit,
            "position_m": list(spec.position),
            "metadata": spec.metadata,
        }
        print(_json.dumps(out, indent=2))
    else:
        print(f"file        : {args.input}")
        print(f"sweep type  : {spec.metadata['sweep_type']}")
        print(f"n_points    : {spec.metadata['n_points']}")
        print(f"channels    : {', '.join(spec.channels)}")
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
    from probeflow.spec_io import read_spec_file
    from probeflow.spec_plot import plot_spectrum

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
    from probeflow.spec_io import read_spec_file
    from probeflow.spec_plot import plot_spectra
    from probeflow.spec_processing import average_spectra

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
    from probeflow.spec_io import read_spec_file
    from probeflow.spec_plot import plot_spec_positions

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
    plane_bg.add_argument("--order", type=int, default=1, choices=(1, 2),
        help="Polynomial order (1=plane, 2=quadratic)")
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

    # ── info / gui ──
    info = sub.add_parser("info", help="Print .sxm header metadata")
    info.add_argument("input", type=Path)
    info.add_argument("--json", action="store_true")
    info.add_argument("--verbose", action="store_true")
    info.set_defaults(func=_cmd_info)

    gui = sub.add_parser("gui", help="Launch the ProbeFlow graphical interface")
    gui.set_defaults(func=_cmd_gui)

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
