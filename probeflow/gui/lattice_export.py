"""
Export helpers for the lattice/grid overlay tool.

Kept separate from lattice_grid_tool.py to avoid circular imports and to
make export logic independently testable.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probeflow.gui.lattice_grid_tool import LatticeGridItem


def export_grid(
    item: "LatticeGridItem",
    path: str,
    include_grid: bool = True,
    grid_only: bool = False,
) -> None:
    """
    Render the grid overlay to a file.

    Args:
        item:         The LatticeGridItem to render.
        path:         Destination file path (.png or .pdf).
        include_grid: If True, render the grid overlay.
        grid_only:    If True, render on a transparent background (PNG) or
                      white background (PDF) without the image data.
    """
    from PySide6.QtGui import QImage, QPainter, QColor
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QGraphicsPixmapItem, QStyleOptionGraphicsItem

    scene = item.scene()
    Nx = item._image_w
    Ny = item._image_h

    # Build target image
    img_format = QImage.Format_ARGB32 if grid_only else QImage.Format_RGB32
    img = QImage(Nx, Ny, img_format)
    img.fill(QColor(0, 0, 0, 0) if grid_only else QColor("black"))

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)

    if not grid_only and scene is not None:
        # Render background pixmap
        for scene_item in scene.items():
            if isinstance(scene_item, QGraphicsPixmapItem):
                pixmap = scene_item.pixmap()
                painter.drawPixmap(0, 0, Nx, Ny, pixmap)
                break

    if include_grid and not grid_only:
        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)
    elif grid_only:
        opt = QStyleOptionGraphicsItem()
        item.paint(painter, opt)

    painter.end()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        _save_pdf(img, path, Nx, Ny)
    else:
        if not img.save(path):
            raise IOError(f"Failed to save PNG to {path!r}")


def _save_pdf(img, path: str, width_px: int, height_px: int) -> None:
    from PySide6.QtGui import QPdfWriter, QPageSize, QMarginsF, QPainter
    from PySide6.QtCore import QSizeF

    dpi = 96
    writer = QPdfWriter(path)
    writer.setResolution(dpi)
    writer.setPageSize(
        QPageSize(QSizeF(width_px / dpi, height_px / dpi), QPageSize.Inch)
    )
    writer.setPageMargins(QMarginsF(0, 0, 0, 0))

    painter = QPainter(writer)
    painter.drawImage(0, 0, img)
    painter.end()
