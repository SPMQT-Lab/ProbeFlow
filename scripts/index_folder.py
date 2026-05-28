#!/usr/bin/env python3
"""Dev script: print a one-line summary for every recognised file in a folder.

Usage:
    python scripts/index_folder.py path/to/folder
    python scripts/index_folder.py path/to/folder --recursive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probeflow.core.indexing import index_folder


def _format_item(item) -> str:
    parts = [f"{item.item_type:<8s}", f"{item.source_format:<25s}", f"{item.path.name:<45s}"]
    if item.item_type == "scan" and item.shape is not None:
        parts.append(f"shape={item.shape}")
    if item.channels:
        parts.append(f"channels={len(item.channels)}")
    if item.scan_range is not None:
        w_nm = item.scan_range[0] * 1e9
        parts.append(f"range={w_nm:.1f}nm")
    if item.item_type == "spectrum":
        n_pts = item.metadata.get("n_points")
        if n_pts is not None:
            parts.append(f"points={n_pts}")
    if item.load_error:
        parts.append(f"ERROR: {item.load_error}")
    return "  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="List recognised SPM files in a folder.")
    parser.add_argument("folder", help="Path to folder to index")
    parser.add_argument("--recursive", action="store_true", help="Walk subdirectories")
    args = parser.parse_args()

    try:
        items = index_folder(args.folder, recursive=args.recursive, include_errors=True)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not items:
        print("No recognised files found.")
        return

    for item in items:
        print(_format_item(item))

    scans = sum(1 for it in items if it.item_type == "scan")
    spectra = sum(1 for it in items if it.item_type == "spectrum")
    errors = sum(1 for it in items if it.load_error)
    print(f"\n{len(items)} items total: {scans} scans, {spectra} spectra, {errors} errors")


if __name__ == "__main__":
    main()
