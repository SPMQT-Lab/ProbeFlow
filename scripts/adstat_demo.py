#!/usr/bin/env python3
"""Generate a reproducible ProbeFlow-to-AdStat teaching run.

This script uses the same direct adapter path as the ProbeFlow image viewer:
synthetic ProbeFlow point source -> AdStat ParticleTable/region -> analysis
summary/comparison -> ResultViewSpec. It is intentionally small and filesystem
friendly so a user can inspect the generated CSV, JSON, and preview image.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from probeflow.analysis.adstat_adapter import compare_point_source_view_spec
from probeflow.gui.roi_context import PointSource


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a reproducible synthetic ProbeFlow-to-AdStat demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/probeflow_adstat_demo"),
        help="Directory for generated CSV, JSON, and PNG artifacts.",
    )
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--points", type=int, default=42)
    parser.add_argument("--width-nm", type=float, default=80.0)
    parser.add_argument("--height-nm", type=float, default=60.0)
    parser.add_argument("--width-px", type=int, default=320)
    parser.add_argument("--height-px", type=int, default=240)
    parser.add_argument("--n-simulations", type=int, default=19)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))
    source, scan = synthetic_probe_flow_point_source(
        seed=args.seed,
        n_points=args.points,
        width_nm=args.width_nm,
        height_nm=args.height_nm,
        width_px=args.width_px,
        height_px=args.height_px,
    )

    try:
        spec = compare_point_source_view_spec(
            source,
            scan=scan,
            image_shape=(args.height_px, args.width_px),
            scan_id="synthetic_adstat_demo",
            pair_bin_width_nm=2.0,
            pair_max_radius_nm=min(args.width_nm, args.height_nm) / 3.0,
            cluster_radius_nm=5.0,
            n_simulations=args.n_simulations,
            random_seed=args.seed,
        )
    except ImportError as exc:
        print(str(exc))
        print(
            "Install the optional AdStat dependency with 'pip install \"probeflow[adstat]\"' "
            "or put an AdStat checkout on PYTHONPATH."
        )
        return 2

    points_csv = args.output_dir / "synthetic_points.csv"
    result_json = args.output_dir / "adstat_result_view_spec.json"
    preview_png = args.output_dir / "synthetic_points_preview.png"

    write_points_csv(points_csv, source)
    write_view_spec_json(result_json, spec, args=args)
    write_preview_png(preview_png, source, width_nm=args.width_nm, height_nm=args.height_nm)
    print_demo_summary(
        spec,
        output_dir=args.output_dir,
        points_csv=points_csv,
        result_json=result_json,
        preview_png=preview_png,
    )
    return 0


def synthetic_probe_flow_point_source(
    *,
    seed: int,
    n_points: int,
    width_nm: float,
    height_nm: float,
    width_px: int,
    height_px: int,
) -> tuple[PointSource, SimpleNamespace]:
    """Return clustered-but-random ProbeFlow-shaped points and scan metadata."""

    if n_points < 4:
        raise ValueError("--points must be at least 4")
    rng = np.random.default_rng(seed)
    n_clustered = max(4, int(round(n_points * 0.72)))
    n_background = n_points - n_clustered
    centers = np.array(
        [
            [0.30 * width_nm, 0.35 * height_nm],
            [0.68 * width_nm, 0.62 * height_nm],
            [0.46 * width_nm, 0.76 * height_nm],
        ],
        dtype=float,
    )
    assignments = rng.integers(0, len(centers), size=n_clustered)
    clustered = centers[assignments] + rng.normal(
        loc=0.0,
        scale=[0.055 * width_nm, 0.06 * height_nm],
        size=(n_clustered, 2),
    )
    background = rng.uniform(
        [0.06 * width_nm, 0.06 * height_nm],
        [0.94 * width_nm, 0.94 * height_nm],
        size=(n_background, 2),
    )
    xy_nm = np.vstack((clustered, background))
    xy_nm[:, 0] = np.clip(xy_nm[:, 0], 0.5, width_nm - 0.5)
    xy_nm[:, 1] = np.clip(xy_nm[:, 1], 0.5, height_nm - 0.5)
    rng.shuffle(xy_nm)

    pixel_size_x_nm = width_nm / float(width_px)
    pixel_size_y_nm = height_nm / float(height_px)
    points_px = xy_nm / np.array([pixel_size_x_nm, pixel_size_y_nm])
    points_m = xy_nm * 1e-9
    source = PointSource(
        label="Synthetic clustered adsorbates",
        source_type="synthetic_demo",
        points_px=points_px,
        points_m=points_m,
        metadata={
            "seed": seed,
            "generator": "scripts/adstat_demo.py",
            "point_count": int(n_points),
            "pattern": "clustered background mixture",
        },
    )
    scan = SimpleNamespace(
        scan_range_m=(width_nm * 1e-9, height_nm * 1e-9),
        dims=(int(width_px), int(height_px)),
        source_path=Path("synthetic_adstat_demo.sxm"),
    )
    return source, scan


def write_points_csv(path: Path, source: PointSource) -> None:
    points_nm = np.asarray(source.points_m, dtype=float) * 1e9
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x_nm", "y_nm", "x_px", "y_px", "source"])
        for index, (xy_nm, xy_px) in enumerate(zip(points_nm, source.points_px)):
            writer.writerow(
                [
                    f"p{index:03d}",
                    f"{xy_nm[0]:.6g}",
                    f"{xy_nm[1]:.6g}",
                    f"{xy_px[0]:.6g}",
                    f"{xy_px[1]:.6g}",
                    source.label,
                ]
            )


def write_view_spec_json(path: Path, spec: Any, *, args: argparse.Namespace) -> None:
    payload = {
        "demo": {
            "script": "scripts/adstat_demo.py",
            "seed": args.seed,
            "points": args.points,
            "scan_nm": {"width": args.width_nm, "height": args.height_nm},
            "scan_px": {"width": args.width_px, "height": args.height_px},
            "n_simulations": args.n_simulations,
        },
        "result_view_spec": _plain(spec),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_preview_png(path: Path, source: PointSource, *, width_nm: float, height_nm: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a runtime dependency
        print(f"Skipping preview PNG; matplotlib could not be imported: {exc}")
        return

    points_nm = np.asarray(source.points_m, dtype=float) * 1e9
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    ax.scatter(points_nm[:, 0], points_nm[:, 1], s=34, c="#2f7ed8", edgecolors="white")
    ax.set_title("Synthetic ProbeFlow point collection")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_xlim(0.0, width_nm)
    ax.set_ylim(height_nm, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_demo_summary(
    spec: Any,
    *,
    output_dir: Path,
    points_csv: Path,
    result_json: Path,
    preview_png: Path,
) -> None:
    panels = tuple(_field(spec, "panels", ()) or ())
    verdict_rows = tuple(_field(spec, "verdict_rows", ()) or ())
    status_lines = tuple(_field(spec, "status_lines", ()) or ())
    print("ProbeFlow AdStat synthetic demo complete")
    print(f"Output directory: {output_dir}")
    print(f"Points CSV: {points_csv}")
    print(f"Result view-spec JSON: {result_json}")
    print(f"Preview PNG: {preview_png}")
    print("")
    print("Rendered panel contract:")
    for panel in panels:
        print(
            f"  - {str(_field(panel, 'kind', 'panel'))}: "
            f"{str(_field(panel, 'title', _field(panel, 'statistic', 'untitled')))}"
        )
    if verdict_rows:
        print("")
        print("Verdict rows:")
        for row in verdict_rows:
            print("  - " + " | ".join(str(item) for item in row))
    if status_lines:
        print("")
        print("Diagnostics:")
        for line in status_lines:
            print(f"  - {line}")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


if __name__ == "__main__":
    raise SystemExit(main())
