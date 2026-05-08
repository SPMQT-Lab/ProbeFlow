"""Histogram export helper extracted from ImageViewerDialog."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QWidget


def export_histogram(
    flat_phys: np.ndarray | None,
    entry_stem: str,
    unit: str,
    channel_name: str,
    parent: QWidget | None = None,
) -> tuple[bool, str]:
    """Write the current histogram to a TSV file chosen by the user.

    Returns ``(success, status_message)``.  An empty message means the user
    cancelled the save dialog.
    """
    if flat_phys is None or flat_phys.size < 2:
        return False, "No histogram data to export."

    out_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Export histogram",
        str(Path.home() / f"{entry_stem}_histogram.txt"),
        "Text files (*.txt *.tsv *.csv)",
    )
    if not out_path:
        return True, ""

    try:
        n_bins = 256
        counts, edges = np.histogram(flat_phys, bins=n_bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("# ProbeFlow histogram export\n")
            fh.write(f"# source: {entry_stem}\n")
            fh.write(f"# channel: {channel_name}\n")
            fh.write(f"# n_samples: {flat_phys.size}\n")
            fh.write(f"# n_bins: {n_bins}\n")
            fh.write(f"# unit: {unit}\n")
            fh.write(f"bin_center_{unit}\tcount\n")
            for c, n in zip(centres, counts):
                fh.write(f"{c:.8g}\t{int(n)}\n")
        return True, f"Histogram → {out_path}"
    except Exception as exc:
        return False, f"Export error: {exc}"
