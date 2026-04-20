"""Convert Nanonis .dat files to preview PNG images."""

import argparse
import json
import logging
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (
    _f, _i,
    detect_channels,
    find_hdr,
    get_dac_bits,
    i_scale_a_per_dac,
    parse_header,
    percentile_clip,
    sanitize,
    setup_logging,
    to_uint8,
    trim_stack,
    v_per_dac,
    z_scale_m_per_dac,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "sample_input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "output_png"


def dat_to_hdr_imgs(
    dat_path: Path,
    out_dir: Path,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> dict:
    """Convert a single .dat file to a header text file and PNG images."""
    raw = dat_path.read_bytes()

    if b"DATA" not in raw:
        raise ValueError(
            f"{dat_path.name}: missing DATA marker — not a valid Nanonis .dat file"
        )

    hb, comp = raw.split(b"DATA", 1)
    hdr = parse_header(hb)

    Nx = _i(find_hdr(hdr, "Num.X", 0), 0)
    Ny = _i(find_hdr(hdr, "Num.Y", 0), 0)
    if Nx <= 0 or Ny <= 0:
        raise ValueError(
            f"{dat_path.name}: invalid pixel dimensions Nx={Nx}, Ny={Ny}"
        )

    if _i(find_hdr(hdr, "ScanmodeSine", 0), 0) != 0:
        raise NotImplementedError(
            f"{dat_path.name}: sine scan mode is not supported"
        )

    log.debug("%s: Nx=%d, Ny=%d", dat_path.name, Nx, Ny)

    try:
        payload = zlib.decompress(comp)
    except zlib.error as exc:
        raise ValueError(
            f"{dat_path.name}: zlib decompression failed — {exc}"
        ) from exc

    stack, num_chan = detect_channels(payload, Ny, Nx)
    log.info("%s: %d channels detected", dat_path.name, num_chan)

    stack, Ny = trim_stack(stack)

    bits = get_dac_bits(hdr)
    vpd = v_per_dac(bits)
    zs = z_scale_m_per_dac(hdr, vpd)
    is_ = i_scale_a_per_dac(hdr, vpd)

    log.debug(
        "%s: DAC bits=%d, V/DAC=%.3e, Z=%.3e m/DAC, I=%.3e A/DAC",
        dat_path.name, bits, vpd, zs, is_,
    )

    # Apply physical scaling in-place: even indices = Z, odd = current
    for k in range(num_chan):
        stack[k] = (stack[k] * (zs if k % 2 == 0 else is_)).astype(np.float32)

    origin_upper = str(find_hdr(hdr, "ScanYDirec", "1")).strip() == "1"

    # Channel layout from Nanonis .dat: [FT, FC, BT, BC]
    if num_chan == 4:
        imgs = [
            ("Z",       "m", "forward",  stack[0]),
            ("Z",       "m", "backward", stack[2]),
            ("Current", "A", "forward",  stack[1]),
            ("Current", "A", "backward", stack[3]),
        ]
    else:
        imgs = [
            ("Z",       "m", "forward", stack[0]),
            ("Current", "A", "forward", stack[1]),
        ]

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "hdr.txt", "w", encoding="utf-8") as f:
        for line in hb.splitlines():
            if b"=" in line:
                key, val = line.split(b"=", 1)
                field = key.decode("ascii", "ignore").split("/")[-1].strip()
                f.write(f"{field}: {val.decode('ascii', 'ignore').strip()}\n")

    png_dir = out_dir / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    for k, (nm, _un, dr, arr_) in enumerate(imgs):
        disp = np.fliplr(arr_) if dr == "backward" else arr_.copy()
        vmin, vmax = percentile_clip(disp, clip_low, clip_high)
        u8 = to_uint8(disp, vmin, vmax)
        if origin_upper:
            u8 = np.flipud(u8)
        fname = f"img_{k:02d}_{sanitize(nm)}_{dr}.png"
        Image.fromarray(u8, mode="L").save(png_dir / fname)
        log.debug("Saved %s (vmin=%.3e, vmax=%.3e)", fname, vmin, vmax)

    return {
        "Nx": Nx,
        "Ny": Ny,
        "num_channels": num_chan,
        "z_scale_m_per_dac": zs,
        "i_scale_a_per_dac": is_,
        "out_dir": str(out_dir),
        "png_dir": str(png_dir),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Nanonis .dat files to preview PNGs",
        epilog=(
            "Examples:\n"
            "  dat-png\n"
            "  dat-png --input-dir data/my_scans --output-dir out/pngs\n"
            "  dat-png --input-dir scan.dat --clip-low 2 --clip-high 98 --verbose"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", dest="input_dir", default=None,
        help="Path to a .dat file or directory of .dat files (default: data/sample_input)",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Output directory for per-file PNG folders (default: data/output_png)",
    )
    p.add_argument(
        "--clip-low", dest="clip_low", type=float, default=1.0,
        help="Lower percentile for contrast clipping (default: 1.0)",
    )
    p.add_argument(
        "--clip-high", dest="clip_high", type=float, default=99.0,
        help="Upper percentile for contrast clipping (default: 99.0)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main(
    src=None,
    out_root=None,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    verbose: bool = False,
) -> None:
    if src is None and out_root is None:
        args = parse_args()
        src = args.input_dir or DEFAULT_INPUT_DIR
        out_root = args.output_dir or DEFAULT_OUTPUT_DIR
        clip_low = args.clip_low
        clip_high = args.clip_high
        verbose = args.verbose

    setup_logging(verbose)

    SRC = Path(src)
    OUT_ROOT = Path(out_root)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    errors: dict = {}

    def _run_one(dat_path: Path) -> None:
        rep = dat_to_hdr_imgs(
            dat_path, OUT_ROOT / dat_path.stem,
            clip_low=clip_low, clip_high=clip_high,
        )
        log.info(
            "[OK] %s → %dx%d  Z=%.3e m/DAC  I=%.3e A/DAC",
            dat_path.name, rep["Nx"], rep["Ny"],
            rep["z_scale_m_per_dac"], rep["i_scale_a_per_dac"],
        )

    if SRC.is_file():
        if SRC.suffix.lower() != ".dat":
            raise ValueError(f"Expected a .dat file, got: {SRC}")
        _run_one(SRC)
    else:
        files = sorted(SRC.glob("*.dat"))
        if not files:
            log.warning("No .dat files found in %s", SRC)
            return
        log.info("Found %d .dat file(s) to process", len(files))
        for i, p in enumerate(files, 1):
            log.info("[%d/%d] Processing %s ...", i, len(files), p.name)
            try:
                _run_one(p)
            except Exception as exc:
                log.error("FAILED %s: %s", p.name, exc)
                errors[p.name] = str(exc)

    if errors:
        err_path = OUT_ROOT / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        log.warning("%d file(s) failed. Error log: %s", len(errors), err_path)
    else:
        log.info("All files processed successfully.")

    log.info("Outputs in: %s", OUT_ROOT)


if __name__ == "__main__":
    main()
