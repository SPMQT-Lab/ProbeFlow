"""PNG export with optional scale bar and provenance metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ── Font path for scale-bar labels ────────────────────────────────────────────
_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

_NICE_STEPS_NM = [
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50,
    100, 200, 500, 1000, 2000, 5000, 10000,
]


def _pick_scalebar_length(width_m: float,
                          target_frac: float = 0.20,
                          unit: str = 'nm') -> tuple[float, str]:
    unit_factors = {'nm': 1e9, 'Å': 1e10, 'pm': 1e12}
    factor = unit_factors.get(unit, 1e9)

    target_m = width_m * target_frac
    target_u = target_m * factor

    best = _NICE_STEPS_NM[0]
    for s in _NICE_STEPS_NM:
        if abs(s - target_u) < abs(best - target_u):
            best = s
        if s > target_u * 2:
            break

    bar_m = best / factor

    if best == int(best):
        label = f"{int(best)} {unit}"
    else:
        label = f"{best:g} {unit}"

    return bar_m, label


def export_png(
    arr:           np.ndarray,
    out_path,
    colormap_key:  str,
    clip_low:      float,
    clip_high:     float,
    lut_fn,
    scan_range_m:  tuple,
    add_scalebar:  bool          = True,
    scalebar_unit: str           = 'nm',
    scalebar_pos:  str           = 'bottom-right',
    vmin:          float | None  = None,
    vmax:          float | None  = None,
    provenance                   = None,   # ExportProvenance | None
    overwrite: bool              = False,
    overwrite_sidecars: bool     = False,
) -> None:
    """
    Export a full-resolution colourised image with an optional scale bar.

    lut_fn(colormap_key) must return a (256, 3) uint8 LUT array.
    scan_range_m  — (width_m, height_m); scale bar is skipped when width ≤ 0.
    If *vmin*/*vmax* are provided, they override the percentile clip.
    If *provenance* is provided, a ``<stem>.provenance.json`` sidecar is written.
    """
    from PIL import Image as _Image, ImageDraw as _IDraw, ImageFont as _IFont
    from PIL.PngImagePlugin import PngInfo as _PngInfo

    from probeflow.processing.display import array_to_uint8 as _array_to_uint8, clip_range_from_array as _clip_range

    arr = arr.astype(np.float64, copy=True)
    if vmin is None or vmax is None:
        vmin, vmax = _clip_range(arr, clip_low, clip_high)  # raises ValueError if no finite values

    u8      = _array_to_uint8(arr, vmin=vmin, vmax=vmax)
    lut     = lut_fn(colormap_key)
    colored = lut[u8]
    img     = _Image.fromarray(colored, mode="RGB")

    width_m = scan_range_m[0] if len(scan_range_m) >= 1 else 0.0
    Ny, Nx  = arr.shape

    if add_scalebar and width_m > 0:
        bar_m, bar_label = _pick_scalebar_length(
            width_m, target_frac=0.20, unit=scalebar_unit)

        bar_px = int(round(bar_m / width_m * Nx))
        bar_px = max(4, min(bar_px, Nx - 20))

        font_size = max(12, Ny // 40)
        font = None
        if _FONT_PATH.exists():
            try:
                font = _IFont.truetype(str(_FONT_PATH), size=font_size)
            except Exception:
                pass
        if font is None:
            font = _IFont.load_default()

        MARGIN      = 10
        BAR_HEIGHT  = max(4, Ny // 80)
        TEXT_GAP    = 3

        dummy_img  = _Image.new("RGB", (1, 1))
        dummy_draw = _IDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), bar_label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if scalebar_pos == 'bottom-left':
            bar_x0 = MARGIN
        else:
            bar_x0 = Nx - MARGIN - bar_px

        bar_y0 = Ny - MARGIN - BAR_HEIGHT
        bar_x1 = bar_x0 + bar_px
        bar_y1 = bar_y0 + BAR_HEIGHT

        text_x = bar_x0 + (bar_px - text_w) // 2
        text_y = bar_y0 - TEXT_GAP - text_h

        draw = _IDraw.Draw(img)

        draw.rectangle([bar_x0 - 1, bar_y0 - 1, bar_x1 + 1, bar_y1 + 1],
                       fill=(0, 0, 0))
        draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(255, 255, 255))

        draw.text((text_x + 1, text_y + 1), bar_label, font=font,
                  fill=(0, 0, 0))
        draw.text((text_x, text_y), bar_label, font=font,
                  fill=(255, 255, 255))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if provenance is not None:
        from probeflow.provenance.export import check_provenance_sidecar_collisions

        check_provenance_sidecar_collisions(
            out_path,
            legacy=hasattr(provenance, "to_dict"),
            overwrite=overwrite_sidecars,
        )
    from probeflow.io.common import check_output_available

    check_output_available(out_path, overwrite=overwrite)
    pnginfo = None
    if provenance is not None:
        try:
            from probeflow.provenance.export import human_summary_from_provenance

            pnginfo = _PngInfo()
            pnginfo.add_text(
                "ProbeFlow provenance",
                human_summary_from_provenance(provenance),
            )
        except Exception:
            pnginfo = None
    if pnginfo is not None:
        img.save(str(out_path), format="PNG", pnginfo=pnginfo)
    else:
        img.save(str(out_path), format="PNG")

    if provenance is not None:
        from probeflow.provenance.export import write_provenance_sidecars

        write_provenance_sidecars(
            out_path,
            provenance,
            export_format="png",
            overwrite=overwrite_sidecars,
        )
