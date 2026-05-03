"""Convert Createc .dat scan files to preview PNG images."""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from probeflow.io.common import (
    get_dac_bits,
    i_scale_a_per_dac,
    sanitize,
    setup_logging,
    v_per_dac,
    z_scale_m_per_dac,
)
from probeflow.processing.display import array_to_uint8
from probeflow.core.scan_loader import load_scan

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "sample_input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "output_png"


def dat_to_hdr_imgs(
    dat_path: Path,
    out_dir: Path,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> dict:
    """Convert a single .dat file to a header text file and PNG images."""
    dat_path = Path(dat_path)
    try:
        scan = load_scan(dat_path)
    except ValueError as exc:
        if dat_path.suffix.lower() == ".dat" and b"DATA" not in dat_path.read_bytes():
            raise ValueError(
                f"{dat_path.name}: missing DATA marker — not a valid Createc .dat file"
            ) from exc
        raise
    hdr = scan.header
    Nx, Ny = scan.dims
    synthetic = list(getattr(scan, "plane_synthetic", []) or [])
    num_chan = 2 if scan.n_planes >= 4 and any(synthetic) else scan.n_planes
    bits = get_dac_bits(hdr)
    vpd = v_per_dac(bits)
    zs = z_scale_m_per_dac(hdr, vpd)
    is_ = i_scale_a_per_dac(hdr, vpd)
    log.debug(
        "%s: DAC bits=%d, V/DAC=%.3e, Z=%.3e m/DAC, I=%.3e A/DAC",
        dat_path.name, bits, vpd, zs, is_,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "hdr.txt", "w", encoding="utf-8") as f:
        for key, val in hdr.items():
            f.write(f"{key}: {val}\n")

    png_dir = out_dir / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    if num_chan == 2 and scan.n_planes >= 4:
        plane_indices = [0, 2]
    else:
        plane_indices = list(range(scan.n_planes))

    for k, plane_idx in enumerate(plane_indices):
        arr = scan.planes[plane_idx]
        name = scan.plane_names[plane_idx] if plane_idx < len(scan.plane_names) else f"Plane {plane_idx}"
        parts = name.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].lower() in {"forward", "backward"}:
            nm, dr = parts[0], parts[1].lower()
        else:
            nm, dr = name, "forward"
        u8 = array_to_uint8(arr, clip_percentiles=(clip_low, clip_high))
        fname = f"img_{k:02d}_{sanitize(nm)}_{dr}.png"
        Image.fromarray(u8, mode="L").save(png_dir / fname)
        log.debug("Saved %s", fname)

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
