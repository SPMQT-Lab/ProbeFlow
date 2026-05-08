"""Line-profile CSV export helper extracted from ImageViewerDialog."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QWidget


def _parse_scan_header_meta(header: dict) -> tuple[float | None, float | None]:
    """Extract bias (mV) and setpoint current (A) from a raw scan header dict."""
    bias_mv: float | None = None
    current_a: float | None = None
    for k, v in header.items():
        kl = k.lower()
        if "biasvolt" in kl or "vgap" in kl:
            try:
                bias_mv = float(str(v).replace(",", "."))
            except (ValueError, TypeError):
                pass
        elif k == "Current[A]":
            try:
                current_a = float(str(v).replace(",", "."))
            except (ValueError, TypeError):
                pass
    return bias_mv, current_a


def export_line_profile(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_label: str,
    y_label: str,
    entry_stem: str,
    scan_header: dict,
    parent: QWidget | None = None,
) -> tuple[bool, str]:
    """Write the current line profile to a CSV file chosen by the user.

    Returns ``(success, status_message)``.  An empty message means the user
    cancelled the save dialog.
    """
    bias_mv, current_a = _parse_scan_header_meta(scan_header)

    parts = [entry_stem]
    if bias_mv is not None:
        parts.append(f"{bias_mv:.0f}mV")
    if current_a is not None:
        parts.append(f"{current_a * 1e12:.0f}pA")
    suggested_name = "_".join(parts) + "_lineprofile.csv"

    out_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Export line profile as CSV",
        str(Path.home() / suggested_name),
        "CSV files (*.csv)",
    )
    if not out_path:
        return True, ""

    try:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["# File", entry_stem])
            if bias_mv is not None:
                w.writerow(["# Bias (mV)", f"{bias_mv:.3f}"])
            if current_a is not None:
                w.writerow(["# Setpoint current (A)", f"{current_a:.3e}"])
            w.writerow([x_label, y_label])
            for x, y in zip(x_vals, y_vals):
                w.writerow([f"{float(x):.6g}", f"{float(y):.6g}"])
        return True, f"Profile → {Path(out_path).name}"
    except Exception as exc:
        return False, f"CSV export error: {exc}"
