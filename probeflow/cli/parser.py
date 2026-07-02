"""ProbeFlow CLI argument parser and entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from probeflow.cli.processing_ops import (
    _add_common_io,
    _cmd_single_op,
    _op_align_rows,
    _op_edge,
    _op_facet_level,
    _op_fft,
    _op_flip_horizontal,
    _op_flip_vertical,
    _op_rotate_90_cw,
    _op_rotate_180,
    _op_rotate_270_cw,
    _op_rotate_arbitrary,
    _op_remove_bad_lines,
    _op_smooth,
)
from probeflow.cli.commands.analysis import (
    _cmd_autoclip,
    _cmd_classify,
    _cmd_count,
    _cmd_fft_spectrum,
    _cmd_grains,
    _cmd_histogram,
    _cmd_lattice,
    _cmd_particles,
    _cmd_periodicity,
    _cmd_profile,
    _cmd_tv_denoise,
    _cmd_unit_cell,
)
from probeflow.cli.commands.conversion import _cmd_dat2npy, _cmd_dat2png, _cmd_dat2sxm
from probeflow.cli.commands.gui import _cmd_gui
from probeflow.cli.commands.processing import (
    _cmd_pipeline,
    _cmd_plane_bg,
    _cmd_prepare_png,
)
from probeflow.cli.commands.scan import (
    _cmd_convert,
    _cmd_diag_z,
    _cmd_info,
    _cmd_sxm2png,
)
from probeflow.cli.commands.spectroscopy import (
    _cmd_spec_info,
    _cmd_spec_overlay,
    _cmd_spec_plot,
    _cmd_spec_positions,
)


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
            "  probeflow dat2npy -- --input-dir data/scans --output-dir out/npy\n"
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

    dat2npy = sub.add_parser("dat2npy",
        help="Createc .dat -> NumPy bundles with provenance sidecars (delegates to dat-npy)")
    dat2npy.add_argument("rest", nargs=argparse.REMAINDER,
        help="Arguments forwarded to dat-npy (prefix with '--')")
    dat2npy.set_defaults(func=_cmd_dat2npy)

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
    sxm2png_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file and provenance sidecars",
    )
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
    plane_bg.add_argument("--fit-roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Fit background to pixels within this persisted ROI only")
    plane_bg.add_argument("--fit-roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Inline rectangular fit-ROI (pixel coordinates)")
    plane_bg.add_argument("--fit-roi-invert", type=str, default=None, metavar="NAME_OR_ID",
        help="Fit background to the complement of the named ROI")
    plane_bg.add_argument("--fit-roi-union", type=str, default=None, metavar="NAME[,NAME,...]",
        help="Fit background to the union of the named ROIs (comma-separated)")
    plane_bg.add_argument("--apply-roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Apply the fitted background only within this persisted ROI")
    plane_bg.add_argument("--apply-roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Inline rectangular apply-ROI (pixel coordinates)")
    plane_bg.add_argument("--exclude-roi", type=str, default=None, metavar="NAME_OR_ID",
        help="Exclude this persisted ROI from the polynomial fit")
    plane_bg.add_argument("--exclude-roi-rect", type=float, nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Inline rectangular region to exclude from the fit")
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
    grains.add_argument("--threshold", type=float, default=90.0,
        help="Percentile of data used as threshold (default 90: islands on a "
             "flat terrace; lower it for dense/rough scans)")
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
    # Reproducible step-edge exclusion (analysis.step_edges.step_edge_mask)
    particles.add_argument("--exclude-step-edges", action="store_true",
        help="Drop molecules sitting on substrate step edges (algorithmic, "
             "reproducible alternative to hand-painting a mask)")
    particles.add_argument("--step-angle", type=float, default=20.0,
        help="Step slope angle in degrees (default 20)")
    particles.add_argument("--step-molecule-size", type=float, default=1.0,
        help="Molecule diameter in nm, suppressed before step detection (default 1.0)")
    particles.add_argument("--step-margin", type=float, default=0.3,
        help="Extra margin grown around the step band, nm (default 0.3)")
    particles.add_argument("--step-min-height", type=float, default=0.0,
        help="Only exclude at steps at least this tall, nm (0 = any steep edge)")
    particles.add_argument("--step-max-overlap", type=float, default=0.25,
        help="Reject a particle when more than this fraction overlaps the step band")
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
    pipe.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file and provenance sidecars",
    )
    pipe.add_argument("--verbose", action="store_true")
    pipe.set_defaults(func=_cmd_pipeline, default_output_suffix="_pipeline.sxm")

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
    prep.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file and provenance sidecars",
    )
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
    gui.add_argument("--open-survey", type=Path, default=None, metavar="SURVEY_JSON",
                     help="Pre-load a ScanFlow survey manifest into Survey mode")
    gui.add_argument("--browse", type=Path, default=None, metavar="FOLDER",
                     help="Open this folder in the Browse tab on startup "
                          "(used internally by the Restart action)")
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
    if argv is None:
        argv = sys.argv[1:]
    # Top-level shortcut: `probeflow --open-survey PATH` → `probeflow gui --open-survey PATH`.
    if argv and (argv[0] == "--open-survey" or argv[0].startswith("--open-survey=")):
        argv = ["gui"] + list(argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    return rc if isinstance(rc, int) else 0


__all__ = ["_build_parser", "main"]
