"""Image Info dialog — shows acquisition metadata and processing history."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from probeflow.gui.typography import mono_font

if TYPE_CHECKING:
    from probeflow.core.metadata import ScanMetadata


def _fmt_or_dash(value, fmt: str = "{}") -> str:
    """Format *value* with *fmt* or return '—' if it is None."""
    if value is None:
        return "—"
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


def _source_dtype_description(source_format: str) -> str:
    """Return a human-readable description of the source file's native precision."""
    _MAP = {
        "nanonis_sxm": "32-bit float  (Nanonis SXM)",
        "createc_dat": "16/32-bit integer, scaled  (Createc DAT)",
        "rhk_sm4":     "16/32-bit integer, scaled  (RHK SM4)",
    }
    return _MAP.get(source_format, f"unknown  ({source_format})")


def _selectable_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lbl.setWordWrap(True)
    return lbl


class ImageInfoDialog(QDialog):
    """Modeless dialog showing image acquisition metadata and processing history.

    Parameters
    ----------
    metadata:
        A :class:`probeflow.core.metadata.ScanMetadata` instance, or ``None``
        for synthetic / unsaved images.
    processing_history_text:
        Pre-formatted processing history string (newline-separated lines).
    current_shape:
        ``(Ny, Nx)`` of the currently displayed (post-processing) array.
    parent:
        Qt parent widget.
    """

    def __init__(
        self,
        *,
        metadata: "ScanMetadata | None",
        processing_history_text: str,
        current_shape: tuple[int, int] | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        title = "Image Info"
        if metadata is not None and metadata.display_name:
            title = f"Image Info — {metadata.display_name}"
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(460, 400)

        tabs = QTabWidget(self)

        # ── Tab 1: Acquisition ────────────────────────────────────────────────
        acq_widget = QWidget()
        form = QFormLayout(acq_widget)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        def _row(field: str, text: str) -> None:
            lbl = _selectable_label(text)
            form.addRow(f"<b>{field}</b>", lbl)

        if metadata is not None:
            _row("Path", str(metadata.path))
            _row("Format", str(metadata.source_format))

            orig_shape = metadata.shape
            if orig_shape:
                _row("Original size", f"{orig_shape[1]} × {orig_shape[0]} px")
            else:
                _row("Original size", "—")

            if current_shape:
                _row("Current size", f"{current_shape[1]} × {current_shape[0]} px")
            else:
                _row("Current size", "—")

            scan_range = metadata.scan_range
            if scan_range:
                w_nm = scan_range[0] * 1e9
                h_nm = scan_range[1] * 1e9
                _row("Scan range", f"{w_nm:.4g} × {h_nm:.4g} nm")

                # Pixel size — use current_shape for the post-processing pixel size
                shape_for_px = current_shape or orig_shape
                if shape_for_px and shape_for_px[0] > 0 and shape_for_px[1] > 0:
                    dx_pm = (scan_range[0] / shape_for_px[1]) * 1e12
                    dy_pm = (scan_range[1] / shape_for_px[0]) * 1e12
                    _row("Pixel size", f"{dx_pm:.3g} × {dy_pm:.3g} pm/px")
            else:
                _row("Scan range", "—")

            if metadata.bias is not None:
                _row("Bias", f"{metadata.bias:.4g} V")
            else:
                _row("Bias", "—")

            if metadata.setpoint is not None:
                # Convert A → pA for display
                sp_pA = metadata.setpoint * 1e12
                _row("Setpoint", f"{sp_pA:.4g} pA")
            elif metadata.feedback_setpoint is not None:
                # Non-current feedback (e.g. constant-Δf AFM): report in native units.
                label = metadata.feedback_setpoint_label or "Feedback setpoint"
                unit = f" {metadata.feedback_setpoint_unit}" if metadata.feedback_setpoint_unit else ""
                _row(label, f"{metadata.feedback_setpoint:.4g}{unit}")
            else:
                _row("Setpoint", "—")

            _row("Acquired", _fmt_or_dash(metadata.acquisition_datetime))
            _row("Comment", _fmt_or_dash(metadata.comment))

            if metadata.plane_names:
                planes_str = ", ".join(
                    f"{n} ({u})" if u else n
                    for n, u in zip(metadata.plane_names, metadata.units or [""] * len(metadata.plane_names))
                )
                _row("Planes", planes_str)

            # ── Data precision ────────────────────────────────────────────────
            form.addRow(QLabel(""))   # visual spacer
            _row("Data type",   "float64  (64-bit double precision)")
            _row("Source file", _source_dtype_description(metadata.source_format))
            _row("Display",     "8-bit per channel  (256 levels)")
        else:
            if current_shape:
                _row("Current size", f"{current_shape[1]} × {current_shape[0]} px")
            form.addRow(_selectable_label("No acquisition metadata available."))

        tabs.addTab(acq_widget, "Acquisition")

        # ── Tab 2: Processing History ─────────────────────────────────────────
        hist_widget = QWidget()
        hist_lay = QVBoxLayout(hist_widget)
        hist_lay.setContentsMargins(4, 4, 4, 4)
        hist_text = QPlainTextEdit()
        hist_text.setReadOnly(True)
        hist_text.setFont(mono_font(9))
        hist_text.setPlainText(processing_history_text or "(No processing history)")
        hist_lay.addWidget(hist_text)
        tabs.addTab(hist_widget, "Processing History")

        # ── Layout ────────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.addWidget(tabs)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)
