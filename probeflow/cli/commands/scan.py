"""Scan inspection and conversion CLI commands: info, sxm2png, convert, diag-z."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from probeflow.io.common import setup_logging
from probeflow.core.scan_loader import load_scan
from probeflow.cli.processing_ops import (
    _cli_png_provenance,
    _ensure_output_available,
)

log = logging.getLogger(__name__)


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
    force = bool(getattr(args, "force", False))
    _ensure_output_available(out, force=force)
    provenance = _cli_png_provenance(scan, args.plane, args, out, "cli_sxm2png")
    scan.save_png(
        out, plane_idx=args.plane,
        colormap=args.colormap,
        clip_low=args.clip_low, clip_high=args.clip_high,
        add_scalebar=not args.no_scalebar,
        scalebar_unit=args.scalebar_unit,
        scalebar_pos=args.scalebar_pos,
        provenance=provenance,
        overwrite=force,
        overwrite_sidecars=force,
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
    if scan.source_format == "sxm":
        keys = ("REC_DATE", "REC_TIME", "BIAS", "SCAN_DIR",
                "SCAN_ANGLE", "SCAN_OFFSET", "COMMENT")
    else:
        keys = ("Titel", "Biasvolt[mV]", "SetPoint", "ScanYDirec",
                "DAC-Type", "T_AUXADC6[K]")
    for key in keys:
        if key in hdr and hdr[key]:
            print(f"{key:14s}: {hdr[key]}")
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

    print(f"File: {path.name}")
    print()
    print("Header values:")
    for key in ["Dacto[A]z", "GainZ", "ZPiezoconst", "DAC-Type",
                "Channels", "Channelselectval", "Length x[A]", "Length y[A]"]:
        val = find_hdr(hdr, key, None)
        print(f"  {key:<20} = {val if val is not None else 'MISSING'}")

    raw_dacto_str = find_hdr(hdr, "Dacto[A]z", None)
    parsed_dacto = _f(raw_dacto_str)
    print()
    print("Dacto[A]z parsing:")
    print(f"  find_hdr result = {raw_dacto_str!r}")
    print(f"  _f(...)         = {parsed_dacto!r}")

    vpd_half = 10.0 / (2 ** bits)
    vpd_full = 20.0 / (2 ** bits)
    print()
    print(f"v_per_dac (bits={bits}):")
    print(f"  vpd_half  code convention  (10/2^bits) = {vpd_half:.6e} V/DAC")
    print(f"  vpd_full  bipolar ±V_ref   (20/2^bits) = {vpd_full:.6e} V/DAC")

    gainz = _f(find_hdr(hdr, "GainZ", None))
    zpiezo = _f(find_hdr(hdr, "ZPiezoconst", None))

    candidates = [
        ("cand_dacto",          parsed_dacto),
        ("cand_dacto_div_gain", (parsed_dacto / gainz)
                                if (parsed_dacto is not None and gainz) else None),
        ("cand_dacto_mul_gain", (parsed_dacto * gainz)
                                if (parsed_dacto is not None and gainz is not None) else None),
        ("cand_zp_vpdhalf",     (zpiezo * vpd_half) if zpiezo is not None else None),
        ("cand_zp_vpdfull",     (zpiezo * vpd_full) if zpiezo is not None else None),
    ]

    code_scale_m = z_scale_m_per_dac(hdr, v_per_dac(bits))
    code_scale_a = code_scale_m * 1e10

    if parsed_dacto is not None:
        code_branch = "Dacto[A]z branch  (dz * 1e-9 m/DAC, treating dz as nm/DAC)"
    else:
        code_branch = "fallback branch   (2 * ZPiezoconst * vpd * 1e-9, treating ZPiezoconst as nm/V)"

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


__all__ = ["_cmd_convert", "_cmd_diag_z", "_cmd_info", "_cmd_sxm2png"]
