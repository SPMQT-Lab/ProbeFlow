"""Borderless viewer-image PDF export.

This is intentionally distinct from :mod:`probeflow.io.writers.pdf`, which
creates an annotated publication figure with axes and a colorbar. The viewer's
"Save PDF copy" action promises the same image composition as its PNG action.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_image_pdf(
    arr: np.ndarray,
    out_path,
    colormap_key: str,
    clip_low: float,
    clip_high: float,
    lut_fn,
    scan_range_m: tuple,
    *,
    add_scalebar: bool = True,
    scalebar_unit: str = "nm",
    scalebar_pos: str = "bottom-right",
    vmin: float | None = None,
    vmax: float | None = None,
    provenance=None,
    overwrite: bool = False,
    overwrite_sidecars: bool = False,
) -> None:
    """Write the PNG renderer's exact image composition to a borderless PDF."""

    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure

    from probeflow.io.common import check_output_available
    from probeflow.processing.png_export import render_export_image

    out_path = Path(out_path)
    if provenance is not None:
        from probeflow.provenance.export import check_provenance_sidecar_collisions

        check_provenance_sidecar_collisions(
            out_path,
            legacy=hasattr(provenance, "to_dict"),
            overwrite=overwrite_sidecars,
        )
    check_output_available(out_path, overwrite=overwrite)

    image = render_export_image(
        arr,
        colormap_key,
        clip_low,
        clip_high,
        lut_fn,
        scan_range_m,
        add_scalebar=add_scalebar,
        scalebar_unit=scalebar_unit,
        scalebar_pos=scalebar_pos,
        vmin=vmin,
        vmax=vmax,
    )
    rgb = np.asarray(image)
    height_px, width_px = rgb.shape[:2]

    # One source pixel per PDF point gives an exact edge-to-edge MediaBox and
    # lets the PDF backend embed the raster without a plotting axes, margins,
    # resampling or a colorbar.
    figure = Figure(
        figsize=(width_px / 72.0, height_px / 72.0),
        dpi=72,
        frameon=False,
    )
    figure.figimage(rgb, xo=0, yo=0, origin="upper", resize=False)
    canvas = FigureCanvasPdf(figure)

    metadata = {
        "Title": out_path.stem,
        "Creator": "ProbeFlow",
    }
    if provenance is not None:
        try:
            from probeflow.provenance.export import human_summary_from_provenance

            metadata["Subject"] = human_summary_from_provenance(provenance)
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.print_pdf(str(out_path), metadata=metadata)

    if provenance is not None:
        from probeflow.provenance.export import write_provenance_sidecars

        write_provenance_sidecars(
            out_path,
            provenance,
            export_format="pdf",
            overwrite=overwrite_sidecars,
        )


__all__ = ["export_image_pdf"]
