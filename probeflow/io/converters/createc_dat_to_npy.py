"""Convert Createc ``.dat`` image files into NumPy bundle directories."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from probeflow.io.common import check_output_available, setup_logging
from probeflow.io.readers.createc_dat import read_createc_dat_report, scale_channels_for_scan
from probeflow.io.readers.createc_scan import (
    createc_public_planes_from_report,
)
from probeflow.io.readers.createc_dat import decoded_scan_range_m
from probeflow.provenance.export import write_provenance_sidecars

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "test_data" / "sample_input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "test_data" / "output_npy"
DATA_SUBDIR_NAME = "npy"


@dataclass(frozen=True)
class CreatecNpyPlane:
    """One exported NumPy plane plus its metadata."""

    public_index: int
    file_name: str
    public_name: str
    public_unit: str
    saved_unit: str
    source_native_index: int | None
    source_name: str
    source_unit: str
    semantic: str | None
    direction: str | None
    synthetic: bool
    scale_factor: float
    dtype: str
    shape: tuple[int, int]
    array: np.ndarray

    def to_dict(self) -> dict:
        return {
            "public_index": self.public_index,
            "file_name": self.file_name,
            "public_name": self.public_name,
            "public_unit": self.public_unit,
            "saved_unit": self.saved_unit,
            "source_native_index": self.source_native_index,
            "source_name": self.source_name,
            "source_unit": self.source_unit,
            "semantic": self.semantic,
            "direction": self.direction,
            "synthetic": self.synthetic,
            "scale_factor": self.scale_factor,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class CreatecNpyBundle:
    """Fully resolved Createc NumPy export bundle."""

    source_path: Path
    bundle_dir: Path
    data_dir: Path
    basis: str
    report: object
    planes: tuple[CreatecNpyPlane, ...]
    header_text: str
    provenance: object


@dataclass(frozen=True)
class _BundleProvenance:
    """Bundle-level provenance payload written to ProbeFlow sidecars."""

    artifact_type: str
    source_path: str
    source_format: str
    basis: str
    bundle_dir: str
    bundle_name: str
    raw_definition: str
    scan_pixels: tuple[int, int]
    scan_range_m: tuple[float, float]
    original_shape: tuple[int, int]
    decoded_shape: tuple[int, int]
    detected_channel_count: int
    public_plane_count: int
    first_column_removed: bool
    ignored_tail_float_count: int
    parser_warnings: tuple[str, ...]
    planes: tuple[dict, ...]
    source_header: dict

    def to_dict(self) -> dict:
        return {
            "artifact_type": self.artifact_type,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "basis": self.basis,
            "bundle_dir": self.bundle_dir,
            "bundle_name": self.bundle_name,
            "raw_definition": self.raw_definition,
            "scan_pixels": list(self.scan_pixels),
            "scan_range_m": list(self.scan_range_m),
            "original_shape": list(self.original_shape),
            "decoded_shape": list(self.decoded_shape),
            "detected_channel_count": self.detected_channel_count,
            "public_plane_count": self.public_plane_count,
            "first_column_removed": self.first_column_removed,
            "ignored_tail_float_count": self.ignored_tail_float_count,
            "parser_warnings": list(self.parser_warnings),
            "planes": [dict(p) for p in self.planes],
            "source_header": dict(self.source_header),
        }


def _basis_note(basis: str) -> str:
    if basis == "raw":
        return (
            "RAW means decoded Createc numerical arrays after ProbeFlow's parser "
            "cleanup, before physical scaling or PNG normalization. It is not a "
            "byte-exact copy of the compressed file payload."
        )
    return (
        "Physical arrays are converted with ProbeFlow's Createc calibration "
        "logic and match the public Scan representation."
    )


def _bundle_dir_name(dat_path: Path, basis: str) -> str:
    return f"{dat_path.stem}_{basis}_npy"


def _saved_unit_for_basis(public_unit: str, basis: str) -> str:
    return "DAC" if basis == "raw" else public_unit


def _format_raw_header(report) -> str:
    lines: list[str] = []
    for key, val in report.header.items():
        lines.append(f"{key}: {val}")
    return "\n".join(lines) + ("\n" if lines else "")


def _resolve_report_arrays(report, basis: str) -> list[np.ndarray]:
    raw = report.decoded_channels_dac
    if raw is None:
        raise ValueError("decoded channel arrays are unavailable")
    if basis == "raw":
        arrays = [np.asarray(raw[i]) for i in range(raw.shape[0])]
    elif basis == "physical":
        arrays = scale_channels_for_scan(report)
    else:
        raise ValueError(f"Unknown basis {basis!r}")
    return arrays


def build_createc_dat_npy_bundle(
    dat_path: Path,
    out_root: Path,
    *,
    basis: str,
) -> CreatecNpyBundle:
    """Build an in-memory bundle description for one Createc ``.dat`` file."""

    dat_path = Path(dat_path)
    out_root = Path(out_root)
    report = read_createc_dat_report(dat_path, include_raw=True)
    arrays = _resolve_report_arrays(report, basis)
    public_planes = createc_public_planes_from_report(report, arrays)
    bundle_dir = out_root / _bundle_dir_name(dat_path, basis)
    data_dir = bundle_dir / DATA_SUBDIR_NAME

    planes: list[CreatecNpyPlane] = []
    for idx, plane in enumerate(public_planes, 1):
        file_name = f"{dat_path.stem}_{idx}.npy"
        saved_unit = _saved_unit_for_basis(plane.public_unit, basis)
        array = np.ascontiguousarray(plane.array)
        planes.append(
            CreatecNpyPlane(
                public_index=idx,
                file_name=file_name,
                public_name=plane.public_name,
                public_unit=plane.public_unit,
                saved_unit=saved_unit,
                source_native_index=plane.source_native_index,
                source_name=plane.source_name,
                source_unit=plane.source_unit,
                semantic=plane.semantic,
                direction=plane.direction,
                synthetic=plane.synthetic,
                scale_factor=plane.scale_factor,
                dtype=str(array.dtype),
                shape=(int(array.shape[0]), int(array.shape[1])),
                array=array,
            )
        )

    bundle = CreatecNpyBundle(
        source_path=dat_path.resolve(),
        bundle_dir=bundle_dir,
        data_dir=data_dir,
        basis=basis,
        report=report,
        planes=tuple(planes),
        header_text=_format_raw_header(report),
        provenance=None,
    )

    provenance = _BundleProvenance(
        artifact_type="createc_dat_npy_bundle",
        source_path=str(dat_path.resolve()),
        source_format="dat",
        basis=basis,
        bundle_dir=str(bundle_dir),
        bundle_name=bundle_dir.name,
        raw_definition=_basis_note(basis),
        scan_pixels=(int(report.decoded_Nx), int(report.decoded_Ny)),
        # Matches scan_pixels: the decoded planes' extent, not the programmed frame.
        scan_range_m=decoded_scan_range_m(report),
        original_shape=(int(report.original_Nx), int(report.original_Ny)),
        decoded_shape=(int(report.decoded_Nx), int(report.decoded_Ny)),
        detected_channel_count=int(report.detected_channel_count),
        public_plane_count=len(planes),
        first_column_removed=bool(report.first_column_removed),
        ignored_tail_float_count=int(report.ignored_tail_float_count),
        parser_warnings=tuple(report.warnings),
        planes=tuple(plane.to_dict() for plane in planes),
        source_header=dict(report.header),
    )
    return CreatecNpyBundle(
        source_path=bundle.source_path,
        bundle_dir=bundle.bundle_dir,
        data_dir=bundle.data_dir,
        basis=bundle.basis,
        report=bundle.report,
        planes=bundle.planes,
        header_text=bundle.header_text,
        provenance=provenance,
    )


def write_createc_dat_npy_bundle(
    bundle: CreatecNpyBundle,
    *,
    overwrite: bool = False,
    overwrite_sidecars: bool = False,
) -> None:
    """Write a previously built Createc NumPy bundle to disk."""

    bundle_dir = Path(bundle.bundle_dir)
    check_output_available(bundle_dir, overwrite=overwrite)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    header_path = bundle_dir / "hdr.txt"
    check_output_available(header_path, overwrite=overwrite)
    header_path.write_text(bundle.header_text, encoding="utf-8")

    data_dir = Path(bundle.data_dir)
    check_output_available(data_dir, overwrite=overwrite)
    data_dir.mkdir(parents=True, exist_ok=True)

    for plane in bundle.planes:
        out_path = data_dir / plane.file_name
        check_output_available(out_path, overwrite=overwrite)
        with open(out_path, "wb") as fh:
            np.save(fh, plane.array, allow_pickle=False)

    write_provenance_sidecars(
        bundle_dir,
        bundle.provenance,
        legacy=True,
        probeflow=True,
        export_format="npy",
        overwrite=overwrite_sidecars,
    )


def export_createc_dat_npy(
    dat_path: Path,
    out_root: Path,
    *,
    basis: str,
    overwrite: bool = False,
    overwrite_sidecars: bool = False,
) -> Path:
    """Build and write one Createc NumPy bundle, returning the output dir."""

    bundle = build_createc_dat_npy_bundle(dat_path, out_root, basis=basis)
    write_createc_dat_npy_bundle(
        bundle,
        overwrite=overwrite,
        overwrite_sidecars=overwrite_sidecars,
    )
    log.info("[OK] %s → %s", dat_path.name, bundle.bundle_dir.name)
    return bundle.bundle_dir


def _run_one(
    dat_path: Path,
    out_root: Path,
    *,
    basis: str,
    force: bool,
) -> None:
    export_createc_dat_npy(
        dat_path,
        out_root,
        basis=basis,
        overwrite=force,
        overwrite_sidecars=force,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Createc .dat files to NumPy bundle directories",
        epilog=(
            "Examples:\n"
            "  dat-npy\n"
            "  dat-npy --input-dir data/scans --output-dir out/npy\n"
            "  dat-npy --basis raw --input-dir scan.dat --force"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", dest="input_dir", default=None,
        help="Path to a .dat file or directory of .dat files (default: data/sample_input)",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Output directory for per-file NumPy bundle folders (default: data/output_npy)",
    )
    p.add_argument(
        "--basis",
        choices=("raw", "physical", "both"),
        default="both",
        help="Export raw arrays, physical arrays, or both (default: both)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing bundle directories and provenance sidecars",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main(
    src=None,
    out_root=None,
    *,
    basis: str = "both",
    force: bool = False,
    verbose: bool = False,
) -> int:
    if src is None and out_root is None and basis == "both" and not force and not verbose:
        args = parse_args()
        src = args.input_dir or DEFAULT_INPUT_DIR
        out_root = args.output_dir or DEFAULT_OUTPUT_DIR
        basis = args.basis
        force = bool(args.force)
        verbose = bool(args.verbose)

    # Programmatic callers may pass only some keywords; fall back to the same
    # defaults the CLI path uses instead of crashing on Path(None).
    if src is None:
        src = DEFAULT_INPUT_DIR
    if out_root is None:
        out_root = DEFAULT_OUTPUT_DIR

    setup_logging(verbose)

    src_path = Path(src)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    bases = ["raw", "physical"] if basis == "both" else [basis]
    errors: dict[str, str] = {}

    def _process_one(dat_path: Path) -> None:
        basis_errors: list[str] = []
        for b in bases:
            try:
                _run_one(dat_path, out_root_path, basis=b, force=force)
            except Exception as exc:
                basis_errors.append(f"{b}: {exc}")
        if basis_errors:
            raise RuntimeError("; ".join(basis_errors))

    if src_path.is_file():
        if src_path.suffix.lower() != ".dat":
            raise ValueError(f"Expected a .dat file or directory, got: {src_path}")
        try:
            _process_one(src_path)
        except Exception as exc:
            log.error("FAILED %s: %s", src_path.name, exc)
            return 1
    else:
        files = sorted(src_path.glob("*.dat"))
        if not files:
            log.warning("No .dat files found in %s", src_path)
            return 0
        log.info("Found %d .dat file(s) to process", len(files))
        for i, dat_path in enumerate(files, 1):
            log.info("[%d/%d] Processing %s ...", i, len(files), dat_path.name)
            try:
                _process_one(dat_path)
            except Exception as exc:
                log.error("FAILED %s: %s", dat_path.name, exc)
                errors[dat_path.name] = str(exc)

    if errors:
        err_path = out_root_path / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        log.warning("%d file(s) failed. Error log: %s", len(errors), err_path)
        return 1

    log.info("All files processed successfully.")
    log.info("Outputs in: %s", out_root_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
