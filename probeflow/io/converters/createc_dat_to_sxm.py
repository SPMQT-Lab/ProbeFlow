"""Convert Createc .dat image files to .sxm format."""

import argparse
import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from probeflow.io.common import _f, setup_logging
from probeflow.io.readers.createc_dat import (
    has_canonical_stm_four_channel_layout,
    has_legacy_stm_two_channel_layout,
    read_createc_dat_report,
)
from probeflow.core.scan_loader import load_scan
from probeflow.io.writers.sxm import write_sxm

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "sample_input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "output_sxm"
DEFAULT_CUSHION_DIR = REPO_ROOT / "src" / "file_cushions"

# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nums(txt: str, n: Optional[int] = None) -> List[float]:
    xs = [float(t) for t in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", txt or "")]
    return xs if n is None else xs[:n]


def _E(x: float, prec: int) -> str:
    s = f"{float(x):.{prec}E}"
    return re.sub(r"E([+-])0?(\d)(?!\d)", r"E\1\2", s)


def to_f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Layout / cushion loaders  (no globals — paths are explicit parameters)
# ─────────────────────────────────────────────────────────────────────────────

def load_sxm_layout(cushion_dir: Path) -> dict:
    """Load binary cushion files that describe the .sxm file structure."""
    cushion_dir = Path(cushion_dir)
    files = {
        "post_end_bytes":   cushion_dir / "post_end_bytes.bin",
        "pre_payload_bytes": cushion_dir / "pre_payload_bytes.bin",
        "pad_len":          cushion_dir / "pad_len.txt",
        "data_offset":      cushion_dir / "data_offset.txt",
        "tail_bytes":       cushion_dir / "tail_bytes.bin",
    }
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing layout files in {cushion_dir}: {', '.join(missing)}"
        )

    post_end_bytes   = files["post_end_bytes"].read_bytes()
    pre_payload_bytes = files["pre_payload_bytes"].read_bytes()
    pad_len          = int(files["pad_len"].read_text(encoding="utf-8").strip())
    data_offset      = int(files["data_offset"].read_text(encoding="utf-8").strip())
    tail_bytes       = files["tail_bytes"].read_bytes()

    if len(pre_payload_bytes) != pad_len:
        raise ValueError(
            f"pad_len ({pad_len}) != len(pre_payload_bytes) ({len(pre_payload_bytes)}). "
            "Re-run boundary capture."
        )

    return {
        "post_end_bytes":   post_end_bytes,
        "pre_payload_bytes": pre_payload_bytes,
        "pad_len":          pad_len,
        "data_offset":      data_offset,
        "tail_bytes":       tail_bytes,
    }


def load_header_format(fmt_path: Path) -> dict:
    """Load and validate the header_format.json descriptor."""
    fmt_path = Path(fmt_path)
    obj = json.loads(fmt_path.read_text(encoding="utf-8"))

    required = ["marker", "line_ending", "block_order", "between_blocks", "key_case", "filler"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"header_format.json missing keys: {', '.join(missing)}")

    le = obj["line_ending"]
    if le in (r"\r\n", "\\r\\n"):
        obj["line_ending"] = "\r\n"
    elif le in (r"\n", "\\n"):
        obj["line_ending"] = "\n"

    obj.setdefault("numeric_hints", {})
    obj.setdefault("data_info", {"columns": [], "delimiter": "\\t", "include_header_row": True})
    obj["marker"].setdefault("newline_after_marker", True)

    _PAD_DEFAULTS = {
        "REC_TEMP_LPAD": 6, "ACQ_LPAD": 7,
        "PIX_LPAD": 7,  "PIX_SEP": 7,
        "E3_LPAD": 13,  "E3_SEP": 13,
        "E6_LPAD": 11,  "E6_SEP": 11,
        "OFF_LPAD": 13, "OFF_SEP": 9,
        "ANGLE_LPAD": 12, "BIAS_LPAD": 12,
        "TYPE1": 14, "TYPE2": 12,
        "DATE_LPAD": 1,
    }
    obj["left_pads"] = {**_PAD_DEFAULTS, **obj.get("left_pads", {})}
    return obj


def load_layout_and_format(cushion_dir: Path) -> Tuple[dict, dict]:
    layout = load_sxm_layout(cushion_dir)
    header_format = load_header_format(Path(cushion_dir) / "header_format.json")
    return layout, header_format


# ─────────────────────────────────────────────────────────────────────────────
# Header construction
# ─────────────────────────────────────────────────────────────────────────────

def parse_dat_timestamp(fname: str) -> datetime:
    """Parse timestamp from filename format AyyMMdd.HHmmss.dat."""
    name = Path(fname).name
    try:
        return datetime.strptime(name, "A%y%m%d.%H%M%S.dat")
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse timestamp from filename '{name}'. "
            "Expected format: AyyMMdd.HHmmss.dat (e.g. A250320.191933.dat)"
        ) from exc


def construct_hdr(
    dat_hdr: Dict[str, str],
    dat_path: Path,
    num_chan: int,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> Dict[str, str]:
    """Build the .sxm header dict from .dat metadata."""

    def sci(x, dec=6):
        s = f"{float(x):.{dec}E}".upper()
        return re.sub(r"E([+-])0?(\d+)$", lambda m: f"E{m.group(1)}{int(m.group(2))}", s)

    dt = parse_dat_timestamp(dat_path.name)
    date = dt.strftime("%d.%m.%Y")
    time_str = dt.strftime("%H:%M:%S")

    temperature = dat_hdr.get("T_AUXADC6[K]", "0")

    numx = dat_hdr.get("Num.X", "512")
    numy = dat_hdr.get("Num.Y", "512")

    lx_a = _f(dat_hdr.get("Length x[A]", "0"), 0.0)
    ly_a = _f(dat_hdr.get("Length y[A]", "0"), 0.0)
    lx_m = sci(lx_a * 1e-10, 6)
    ly_m = sci(ly_a * 1e-10, 6)

    total_time = dat_hdr.get("Image:", "0")
    dur_line = _f(dat_hdr.get("line:", "0"), 0.0)
    dur_sci = sci(dur_line, 3)

    ox_dac = _f(dat_hdr.get("OffsetX", "0"), 0.0)
    oy_dac = _f(dat_hdr.get("OffsetY", "0"), 0.0)
    dac_to_nm = _f(dat_hdr.get("Dacto[A]xy", "1"), 1.0)
    # Createc labels this Dacto field with "[A]", but lateral self-consistency
    # against Length x[A] and Delta X [Dac] shows the value is nm/DAC. Use the
    # same convention as z_scale_m_per_dac so spec-marker offsets do not inherit
    # the historical factor-of-10 ambiguity.
    ox_m = sci(ox_dac * dac_to_nm * 1e-9, 6)
    oy_m = sci(oy_dac * dac_to_nm * 1e-9, 6)

    bias_mv = _f(dat_hdr.get("Biasvolt[mV]", "0"), 0.0)
    bias_str = sci(bias_mv * 1e-3, 3)

    scan_dir = "down" if str(dat_hdr.get("ScanYDirec", "1")).strip() == "1" else "up"
    angle_rad = math.radians(_f(str(dat_hdr.get("Rotation", "0")).replace(",", "."), 0.0))
    angle_str = sci(angle_rad, 3)

    comment = (dat_hdr.get("Titel", "") or "Empty").strip()

    z_name = "log Current" if str(dat_hdr.get("FBLog", "0")).strip() == "1" else "Current"
    z_on   = "0" if str(dat_hdr.get("FBOff", "0")).strip() == "1" else "1"
    setp   = _f(str(dat_hdr.get("SetPoint", "0")).replace(",", "."), 0.0)
    setp_str = sci(setp, 3) + " A"

    hdr: Dict[str, str] = {
        "NANONIS_VERSION": "2",
        "SCANIT_TYPE":     "FLOAT            MSBFIRST",
        "REC_DATE":        date,
        "REC_TIME":        time_str,
        "REC_TEMP":        str(temperature),
        "ACQ_TIME":        str(total_time),
        "SCAN_PIXELS":     f"{numx}       {numy}",
        "SCAN_FILE":       str(dat_path),
        "SCAN_TIME":       f"{dur_sci}             {dur_sci}",
        "SCAN_RANGE":      f"{lx_m}           {ly_m}",
        "SCAN_OFFSET":     f"{ox_m}         {oy_m}",
        "SCAN_ANGLE":      angle_str,
        "SCAN_DIR":        scan_dir,
        "BIAS":            bias_str,
        "Z-CONTROLLER": (
            "Name\ton\tSetpoint\tP-gain\tI-gain\tT-const\n"
            f"\t{z_name}\t{z_on}\t{setp_str}\t1.000E+0 m\t1.000E+0 m/s\t0.000E+0 s"
        ),
        "COMMENT": comment,
        "DATA_INFO": (
            "Channel\tName\tUnit\tDirection\tCalibration\tOffset\n"
            "\t14\tZ\tm\tboth\t1.000E+0\t0.000E+0\n"
            "\t0\tCurrent\tA\tboth\t1.000E+0\t0.000E+0"
        ),
        "Clip_percentile_Lower":  str(clip_low),
        "Clip_percentile_Higher": str(clip_high),
    }
    hdr.update(dat_hdr)
    return hdr


# ─────────────────────────────────────────────────────────────────────────────
# Header emitters
# ─────────────────────────────────────────────────────────────────────────────

def make_emitters(header_format: dict) -> Tuple[dict, str]:
    PAD = header_format["left_pads"]
    EOL = header_format.get("line_ending", "\n")
    if EOL in (r"\r\n", "\\r\\n"):
        EOL = "\r\n"
    if EOL in (r"\n", "\\n"):
        EOL = "\n"

    def emit_SCANIT_TYPE(val: str) -> str:
        toks = (val or "").split()
        c1 = toks[0] if toks else "FLOAT"
        c2 = toks[1] if len(toks) > 1 else "LSBFIRST"
        return " " * PAD["TYPE1"] + c1 + " " * PAD["TYPE2"] + c2

    def emit_REC_DATE(val: str) -> str:
        return " " * PAD["DATE_LPAD"] + str(val).strip()

    def emit_REC_TIME(val: str) -> str:
        parts = re.findall(r"\d{1,2}", str(val).strip())
        return (
            f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
            if len(parts) >= 3 else str(val).strip()
        )

    def emit_REC_TEMP(val: str) -> str:
        xs = _nums(val, 1)
        return " " * PAD["REC_TEMP_LPAD"] + f"{(xs[0] if xs else 0.0):.10f}"

    def emit_ACQ_TIME(val: str) -> str:
        xs = _nums(val, 1)
        return " " * PAD["ACQ_LPAD"] + f"{(xs[0] if xs else 0.0):.1f}"

    def emit_SCAN_PIXELS(val: str) -> str:
        Nx, Ny = map(int, _nums(val, 2))
        return " " * PAD["PIX_LPAD"] + f"{Nx}" + " " * PAD["PIX_SEP"] + f"{Ny}"

    def emit_SCAN_TIME(val: str) -> str:
        a, b = (_nums(val, 2) + [0.0, 0.0])[:2]
        return " " * PAD["E3_LPAD"] + _E(a, 3) + " " * PAD["E3_SEP"] + _E(b, 3)

    def emit_SCAN_RANGE(val: str) -> str:
        a, b = (_nums(val, 2) + [0.0, 0.0])[:2]
        return " " * PAD["E6_LPAD"] + _E(a, 6) + " " * PAD["E6_SEP"] + _E(b, 6)

    def emit_SCAN_OFFSET(val: str) -> str:
        a, b = (_nums(val, 2) + [0.0, 0.0])[:2]
        return " " * PAD["OFF_LPAD"] + _E(a, 6) + " " * PAD["OFF_SEP"] + _E(b, 6)

    def emit_SCAN_ANGLE(val: str) -> str:
        xs = _nums(val, 1)
        return " " * PAD["ANGLE_LPAD"] + _E(xs[0] if xs else 0.0, 3)

    def emit_BIAS(val: str) -> str:
        xs = _nums(val, 1)
        return " " * PAD["BIAS_LPAD"] + _E(xs[0] if xs else 0.0, 3)

    def emit_Z_CONTROLLER(val: str) -> str:
        lines = (val or "").splitlines()
        if lines and not lines[0].startswith("\t"):
            lines[0] = "\t" + lines[0]
        return EOL.join(ln.rstrip(" ") for ln in lines)

    def emit_DATA_INFO(val: str) -> str:
        lines = (val or "").splitlines()
        fixed = []
        for ln in lines:
            if ln and not ln.startswith("\t"):
                ln = "\t" + ln
            fixed.append(ln.rstrip(" "))
        return EOL.join(fixed) + EOL

    special = {
        "SCANIT_TYPE":  emit_SCANIT_TYPE,
        "REC_DATE":     emit_REC_DATE,
        "REC_TIME":     emit_REC_TIME,
        "REC_TEMP":     emit_REC_TEMP,
        "ACQ_TIME":     emit_ACQ_TIME,
        "SCAN_PIXELS":  emit_SCAN_PIXELS,
        "SCAN_TIME":    emit_SCAN_TIME,
        "SCAN_RANGE":   emit_SCAN_RANGE,
        "SCAN_OFFSET":  emit_SCAN_OFFSET,
        "SCAN_ANGLE":   emit_SCAN_ANGLE,
        "BIAS":         emit_BIAS,
        "Z-CONTROLLER": emit_Z_CONTROLLER,
        "DATA_INFO":    emit_DATA_INFO,
    }
    return special, EOL


# ─────────────────────────────────────────────────────────────────────────────
# Image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_dat(
    dat: Path,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> Tuple[Dict[str, str], List[Tuple[str, str, str, np.ndarray]], int]:
    """Parse .dat, scale to SI, trim, clip, and return header + image planes."""
    report = read_createc_dat_report(dat, include_raw=True)
    dat_hdr = dict(report.header)
    raw_stack = report.raw_channels_dac
    if raw_stack is None:
        raise ValueError(f"{dat.name}: internal decode report has no image data")
    if not (
        has_canonical_stm_four_channel_layout(report)
        or has_legacy_stm_two_channel_layout(report)
    ):
        raise ValueError(
            f"{dat.name}: .dat→.sxm reconstruction supports only canonical "
            "STM Z/current layouts"
        )
    stack = np.array(raw_stack, copy=True)
    num_chan = report.detected_channel_count
    log.info("%s: %d channels detected", dat.name, num_chan)

    bits = int(report.scale_factors["dac_bits"])
    vpd = float(report.scale_factors["v_per_dac"])
    zs = float(report.scale_factors["z_m_per_dac"])
    is_ = float(report.scale_factors["current_a_per_dac"])

    log.debug(
        "%s: DAC bits=%d, V/DAC=%.3e, Z=%.3e m/DAC, I=%.3e A/DAC",
        dat.name, bits, vpd, zs, is_,
    )

    # Scale to physical units
    for k in range(num_chan):
        stack[k] = (stack[k] * (zs if k % 2 == 0 else is_)).astype(np.float32)

    FT = stack[0]
    FC = stack[1]
    if num_chan == 2:
        # SXM stores backward scan rows in acquisition order.  The SXM reader
        # flips them back to display order, so this round-trips to read_dat().
        BT = np.fliplr(stack[0])
        BC = np.fliplr(stack[1])
        log.warning(
            "%s: 2-channel input — backward planes synthesised by mirroring forward",
            dat.name,
        )
    elif num_chan == 4:
        BT = np.fliplr(stack[2])
        BC = np.fliplr(stack[3])
    else:
        raise ValueError(
            f"{dat.name}: .dat→.sxm reconstruction expects STM 2- or 4-channel "
            f"data, got {num_chan} channel(s)"
        )

    hdr = construct_hdr(dat_hdr, dat, num_chan, clip_low, clip_high)

    Ny2, Nx2 = FT.shape
    hdr["SCAN_PIXELS"] = f"{Nx2}{' ' * 7}{Ny2}"

    imgs = [
        ("Z",       "m", "forward",  to_f32(FT)),
        ("Z",       "m", "backward", to_f32(BT)),
        ("Current", "A", "forward",  to_f32(FC)),
        ("Current", "A", "backward", to_f32(BC)),
    ]
    return hdr, imgs, num_chan


# ─────────────────────────────────────────────────────────────────────────────
# .sxm binary reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_from_hdr_imgs(
    hdr: dict,
    imgs: List[Tuple[str, str, str, np.ndarray]],
    header_format: dict,
    post_end_bytes: bytes,
    pre_payload_bytes: bytes,
    out_path: Path,
    tail_bytes: bytes = b"",
    force_data_offset: Optional[int] = None,
    filler_char: bytes = b" ",
) -> Tuple[int, int]:
    """Assemble and write the binary .sxm file. Returns (data_offset, payload_len)."""
    special, EOL = make_emitters(header_format)

    block_order = header_format.get("block_order") or [
        k for k in hdr if not k.startswith("__") and k != "DATA_INFO_PARSED"
    ]

    out_lines: List[str] = []
    emitted: set = set()

    def _emit(key: str) -> None:
        val = hdr[key]
        body = special[key](val) if key in special else str(val)
        if body and not body.endswith(EOL):
            body = body.rstrip(" ")
        out_lines.append(f":{key}:{EOL}{body}{EOL}")

    for key in block_order:
        if key not in hdr or key == "DATA_INFO_PARSED" or re.fullmatch(r"\d+", key):
            continue
        _emit(key)
        emitted.add(key)

    for key in hdr:
        if key in emitted or key.startswith("__") or key == "DATA_INFO_PARSED" or re.fullmatch(r"\d+", key):
            continue
        _emit(key)

    header_core = "".join(out_lines)
    while not header_core.endswith(EOL * 2):
        header_core += EOL

    hdr_bytes = header_core.encode("latin-1", "ignore")

    if force_data_offset is not None:
        target_len = (
            int(force_data_offset)
            - len(b":SCANIT_END:")
            - len(post_end_bytes)
            - len(pre_payload_bytes)
        )
        fill = target_len - len(hdr_bytes)
        if fill:
            trailer = (EOL * 2).encode("latin1")
            body_core = hdr_bytes[: -len(trailer)]
            one_eol = EOL.encode("latin1")
            if isinstance(filler_char, str):
                filler_char = filler_char.encode("latin1", "ignore")
            hdr_bytes = body_core + one_eol + (filler_char * fill) + one_eol + one_eol

    header_bytes = hdr_bytes + b":SCANIT_END:" + post_end_bytes + pre_payload_bytes
    data_offset = len(header_bytes)

    toks = (hdr.get("SCANIT_TYPE") or "").split()
    endian = ">" if (len(toks) >= 2 and toks[1].strip().upper() == "MSBFIRST") else "<"
    dt = np.dtype(endian + "f4")
    Nx, Ny = map(int, _nums(hdr.get("SCAN_PIXELS", ""), 2))

    arrs = [item[3] if isinstance(item, tuple) else item for item in imgs]

    for i, a in enumerate(arrs):
        if a.shape != (Ny, Nx):
            raise ValueError(f"Plane {i} shape {a.shape} != expected {(Ny, Nx)}")

    payload = b"".join(
        np.asarray(a, dtype=dt, order="C").tobytes(order="C") for a in arrs
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header_bytes + payload + (tail_bytes or b""))

    return data_offset, len(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file entry point
# ─────────────────────────────────────────────────────────────────────────────

def convert_dat_to_sxm(
    dat: Path,
    out_dir: Path,
    cushion_dir: Path,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> None:
    """Convert a single .dat file to .sxm and write it to out_dir.

    The public conversion path goes through ``load_scan()`` so DAT inputs get
    the same reader dispatch and validation as the rest of ProbeFlow, then uses
    the shared SXM writer for binary reconstruction.
    """
    dat = Path(dat)
    out_dir = Path(out_dir)
    out_path = out_dir / (dat.stem + ".sxm")

    scan = load_scan(dat)
    write_sxm(
        scan,
        out_path,
        cushion_dir=cushion_dir,
        clip_low=clip_low,
        clip_high=clip_high,
    )
    log.info("[OK] %s → %s", dat.name, out_path.name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Nanonis .dat files to .sxm format",
        epilog=(
            "Examples:\n"
            "  dat-sxm\n"
            "  dat-sxm --input-dir data/scans --output-dir out/sxm\n"
            "  dat-sxm --input-dir scan.dat --clip-low 2 --clip-high 98 --verbose"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", dest="input_dir", default=None,
        help="Path to a .dat file or directory of .dat files (default: data/sample_input)",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Output directory for .sxm files (default: data/output_sxm)",
    )
    p.add_argument(
        "--cushion-dir", dest="cushion_dir", default=None,
        help="Directory containing file cushion layout files (default: src/file_cushions)",
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


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    src        = Path(args.input_dir  or DEFAULT_INPUT_DIR)
    out_dir    = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    cushion_dir = Path(args.cushion_dir or DEFAULT_CUSHION_DIR)
    clip_low   = args.clip_low
    clip_high  = args.clip_high

    out_dir.mkdir(parents=True, exist_ok=True)

    errors: dict = {}

    def _run_one(dat_path: Path) -> None:
        convert_dat_to_sxm(dat_path, out_dir, cushion_dir, clip_low, clip_high)

    if src.is_file():
        if src.suffix.lower() != ".dat":
            raise ValueError(f"Expected a .dat file or directory, got: {src}")
        _run_one(src)
    else:
        files = sorted(src.glob("*.dat"))
        if not files:
            log.warning("No .dat files found in %s", src)
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
        err_path = out_dir / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        log.warning("%d file(s) failed. Error log: %s", len(errors), err_path)
    else:
        log.info("All files processed successfully.")

    log.info("Outputs in: %s", out_dir)


if __name__ == "__main__":
    main()
