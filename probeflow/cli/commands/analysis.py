"""Analysis CLI commands: grains, autoclip, periodicity, particles, count,
tv-denoise, lattice, classify, profile, histogram, fft-spectrum, unit-cell."""

from __future__ import annotations

import json
import logging

import numpy as np
from PIL import Image

from probeflow import processing as _proc
from probeflow.io.common import setup_logging
from probeflow.core.scan_loader import load_scan
from probeflow.cli.processing_ops import (
    _load_plane_for_analysis,
    _load_named_roi,
    _pixel_sizes_m_from_scan,
    _record_op,
    _resolve_inline_roi,
    _write_output,
)

log = logging.getLogger(__name__)


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
    try:
        _write_output(args, scan, default_suffix="_tv.sxm")
    except Exception as exc:
        log.error("%s", exc)
        return 1
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
    dx_m, dy_m = _pixel_sizes_m_from_scan(scan)
    px_m = float(np.sqrt(dx_m * dy_m)) if dx_m > 0 and dy_m > 0 else 0.0
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
        res = extract_lattice(
            scan.planes[args.plane],
            pixel_size_m=px_m,
            pixel_size_x_m=dx_m,
            pixel_size_y_m=dy_m,
            params=params,
        )
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
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"|a| = {res.a_length_m * 1e9:7.3f} nm")
        print(f"|b| = {res.b_length_m * 1e9:7.3f} nm")
        print(f" γ  = {res.gamma_deg:7.2f} °")
        print(f"Keypoints: {res.n_keypoints}  (primary cluster: "
              f"{res.n_keypoints_used})")
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

    roi_line = getattr(args, "roi_line", None)
    roi_name = getattr(args, "roi", None)

    roi_obj = None
    if roi_name is not None:
        roi_obj = _load_named_roi(args.input, roi_name,
                                  getattr(args, "sidecar", None))
        if roi_obj is None:
            return 1
        if roi_obj.kind != "line":
            log.error("ROI %r has kind=%r; profile requires a line ROI",
                      roi_name, roi_obj.kind)
            return 1

    if roi_obj is not None:
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
    dx_m, dy_m = _pixel_sizes_m_from_scan(scan)
    px_m = float(np.sqrt(dx_m * dy_m)) if dx_m > 0 and dy_m > 0 else 0.0
    if px_m <= 0:
        log.error("Scan has no physical pixel size.")
        return 1

    from probeflow.analysis.lattice import (
        LatticeParams, extract_lattice, average_unit_cell,
    )
    arr = scan.planes[args.plane]
    try:
        lat = extract_lattice(
            arr,
            pixel_size_m=px_m,
            pixel_size_x_m=dx_m,
            pixel_size_y_m=dy_m,
            params=LatticeParams(),
        )
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


__all__ = [
    "_cmd_autoclip",
    "_cmd_fft_spectrum",
    "_cmd_grains",
    "_cmd_histogram",
    "_cmd_lattice",
    "_cmd_periodicity",
    "_cmd_profile",
    "_cmd_tv_denoise",
    "_cmd_unit_cell",
]
