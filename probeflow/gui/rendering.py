"""Shared GUI rendering helpers for thumbnails, previews, and viewer rasters."""

from __future__ import annotations

import io
import re as _re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from probeflow import processing as _proc
from probeflow.processing.display import (
    array_to_uint8 as _array_to_uint8,
    clip_range_from_array as _clip_range_from_array,
)
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.core.scan_loader import load_scan

# ── Colormaps (25 — most cited in STM/SPM publications) ──────────────────────
STM_COLORMAPS: list[tuple[str, str]] = [
    # Topography
    ("AFM Hot",      "afmhot"),
    ("Hot",          "hot"),
    ("Gray",         "gray"),
    ("Copper",       "copper"),
    ("Bone",         "bone"),
    # Perceptually uniform
    ("Viridis",      "viridis"),
    ("Plasma",       "plasma"),
    ("Inferno",      "inferno"),
    ("Magma",        "magma"),
    ("Cividis",      "cividis"),
    # Diverging — dI/dV and STS maps
    ("Cool-Warm",    "coolwarm"),
    ("RdBu",         "RdBu_r"),
    ("Seismic",      "seismic"),
    ("BWR",          "bwr"),
    ("Spectral",     "Spectral_r"),
    ("PiYG",         "PiYG"),
    # Sequential
    ("YlOrRd",       "YlOrRd"),
    ("Blues",        "Blues"),
    ("Oranges",      "Oranges"),
    ("Greens",       "Greens"),
    # Legacy / full spectrum
    ("Jet",          "jet"),
    ("Turbo",        "turbo"),
    ("Rainbow",      "gist_rainbow"),
    # Cyclic — phase maps
    ("Twilight",     "twilight"),
    ("HSV",          "hsv"),
]

CMAP_NAMES = [label for label, _ in STM_COLORMAPS]
CMAP_KEY   = {label: key for label, key in STM_COLORMAPS}

_LUTS: dict[str, np.ndarray] = {}

DEFAULT_CMAP_LABEL = "Gray"
DEFAULT_CMAP_KEY   = "gray"


def _make_lut(mpl_name: str) -> np.ndarray:
    try:
        import matplotlib
        cmap = matplotlib.colormaps[mpl_name]
        x    = np.linspace(0, 1, 256)
        rgba = cmap(x)
        return (rgba[:, :3] * 255).astype(np.uint8)
    except Exception:
        pass
    # fallback: grayscale
    x = np.linspace(0, 1, 256)
    lut = (np.stack([x, x, x], axis=1) * 255).astype(np.uint8)
    return lut


def _get_lut(label_or_key: str) -> np.ndarray:
    key = CMAP_KEY.get(label_or_key, label_or_key)
    if key not in _LUTS:
        _LUTS[key] = _make_lut(key)
    return _LUTS[key]


def clip_range_from_arr(
    arr: Optional[np.ndarray],
    pct_lo: float = 1.0,
    pct_hi: float = 99.0,
) -> tuple[Optional[float], Optional[float]]:
    """Return (vmin, vmax) for display clipping, or (None, None) on failure.

    Thin adapter over :func:`probeflow.processing.display.clip_range_from_array` that
    preserves the GUI contract of returning (None, None) rather than raising.
    """
    if arr is None:
        return None, None
    try:
        return _clip_range_from_array(arr, pct_lo, pct_hi)
    except ValueError:
        return None, None


THUMBNAIL_CHANNEL_OPTIONS = (
    "Z",
    "Current",
    "Frequency shift",
    "Amplitude",
    "Drive",
    "Dissipation",
)
THUMBNAIL_CHANNEL_DEFAULT = "Z"


def _normalise_channel_name(name: str) -> str:
    text = _re.sub(r"[_\-]+", " ", str(name).lower())
    return _re.sub(r"\s+", " ", text).strip()


def _is_forward_channel_name(name: str) -> bool:
    norm = _normalise_channel_name(name)
    if any(tok in norm for tok in ("backward", "bwd")):
        return False
    if any(tok in norm for tok in ("forward", "fwd")):
        return True
    return True


def _matches_thumbnail_semantic(name: str, semantic: str) -> bool:
    norm = _normalise_channel_name(name)
    if not _is_forward_channel_name(norm):
        return False
    semantic = semantic if semantic in THUMBNAIL_CHANNEL_OPTIONS else THUMBNAIL_CHANNEL_DEFAULT
    if semantic == "Z":
        return (
            norm == "z"
            or norm.startswith("z ")
            or "topography" in norm
            or "height" in norm
        )
    if semantic == "Current":
        return norm == "i" or norm.startswith("i ") or "current" in norm
    if semantic == "Frequency shift":
        return (
            "freq" in norm
            or "frequency" in norm
            or "delta f" in norm
            or "df" in norm
        )
    if semantic == "Amplitude":
        return "amplitude" in norm or norm == "amp" or norm.startswith("amp ")
    if semantic == "Drive":
        return "drive" in norm or "excitation" in norm or norm == "exc" or norm.startswith("exc ")
    if semantic == "Dissipation":
        return "dissipation" in norm or norm == "diss" or norm.startswith("diss ")
    return False


def resolve_thumbnail_plane_index(
    plane_names: list[str] | tuple[str, ...],
    semantic: str = THUMBNAIL_CHANNEL_DEFAULT,
) -> int:
    """Return the best forward plane index for a browse-thumbnail semantic."""
    names = list(plane_names or [])
    if not names:
        return 0
    requested = semantic if semantic in THUMBNAIL_CHANNEL_OPTIONS else THUMBNAIL_CHANNEL_DEFAULT
    for idx, name in enumerate(names):
        if _matches_thumbnail_semantic(name, requested):
            return idx
    if requested != THUMBNAIL_CHANNEL_DEFAULT:
        for idx, name in enumerate(names):
            if _matches_thumbnail_semantic(name, THUMBNAIL_CHANNEL_DEFAULT):
                return idx
    return 0


def _apply_processing(
    arr: np.ndarray,
    processing: dict,
    roi_set=None,
) -> np.ndarray:
    """Apply the array-transform portion of the processing pipeline.

    Converts the GUI processing dict to a canonical ProcessingState and
    delegates to apply_processing_state().  Grain detection / colormap / clip
    settings are display-only and are silently ignored.  Returns a new float64
    array; the input is never modified.
    """
    from probeflow.processing.state import apply_processing_state
    return apply_processing_state(
        arr,
        processing_state_from_gui(processing or {}),
        roi_set=roi_set,
    )


def render_scan_thumbnail(
    scan_path: Path,
    plane_idx: int            = 0,
    colormap:  str            = "gray",
    clip_low:  float          = 1.0,
    clip_high: float          = 99.0,
    size:      tuple          = (148, 116),
    vmin:      Optional[float] = None,
    vmax:      Optional[float] = None,
    allow_upscale: bool       = False,
) -> Optional[Image.Image]:
    """Render a thumbnail of any supported scan format via the unified reader.

    If *vmin* and *vmax* are provided, they are used directly (manual mode).
    Otherwise *clip_low*/*clip_high* percentiles are applied (percentile mode).
    """
    return render_scan_image(
        scan_path=scan_path,
        plane_idx=plane_idx,
        colormap=colormap,
        clip_low=clip_low,
        clip_high=clip_high,
        size=size,
        vmin=vmin,
        vmax=vmax,
        allow_upscale=allow_upscale,
    )


def render_scan_image(
    scan_path: Optional[Path] = None,
    arr: Optional[np.ndarray] = None,
    plane_idx: int = 0,
    colormap: str = "gray",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    size: Optional[tuple[int, int]] = (148, 116),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    allow_upscale: bool = False,
    processing: Optional[dict] = None,
) -> Optional[Image.Image]:
    """Render a scan plane for all GUI image surfaces.

    Browse thumbnails, right-panel channel previews, and the full image viewer
    intentionally share this path so only sizing policy differs between them.
    """
    try:
        if arr is None:
            if scan_path is None:
                return None
            scan = load_scan(scan_path)
            if plane_idx >= scan.n_planes:
                return None
            arr = scan.planes[plane_idx]
        if arr is None:
            return None
        if processing:
            arr = _apply_processing(arr, processing)

        if vmin is None or vmax is None:
            vmin, vmax = clip_range_from_arr(arr, clip_low, clip_high)
        if vmin is None:
            return None

        u8 = _array_to_uint8(arr, vmin=vmin, vmax=vmax)
        colored = _get_lut(colormap)[u8]
        grain_thresh = (processing or {}).get("grain_threshold")
        if grain_thresh is not None:
            label_map, _, _ = _proc.detect_grains(
                arr,
                threshold_pct=float(grain_thresh),
                above=bool((processing or {}).get("grain_above", True)),
            )
            colored = colored.copy()
            grain_px = label_map > 0
            colored[grain_px, 0] = np.clip(
                colored[grain_px, 0].astype(int) // 2 + 128, 0, 255)
            colored[grain_px, 1] = colored[grain_px, 1] // 3
            colored[grain_px, 2] = colored[grain_px, 2] // 3
        img = Image.fromarray(colored, mode="RGB")
        if size:
            if allow_upscale:
                img = _fit_image_to_box(img, size)
            else:
                img.thumbnail(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def render_spec_thumbnail(
    vert_path: Path,
    size: tuple = (148, 116),
    dark: bool = True,
) -> Optional[Image.Image]:
    """Render a small matplotlib plot of a .VERT file as a PIL Image."""
    try:
        from probeflow.io.spectroscopy import read_spec_file
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        spec = read_spec_file(vert_path)
        # Prefer the first default channel; fall back to 'I' then first available
        ch = None
        if spec.default_channels:
            ch = spec.default_channels[0]
        elif "I" in spec.channels:
            ch = "I"
        elif spec.channels:
            ch = next(iter(spec.channels))
        y = spec.channels.get(ch) if ch else None
        if y is None or len(y) == 0:
            return None
        x = spec.x_array

        bg   = "#1e1e2e" if dark else "#ffffff"
        line = "#89b4fa" if dark else "#1e66f5"

        fig = Figure(figsize=(size[0] / 80, size[1] / 80), dpi=80)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.plot(x, y, linewidth=0.9, color=line)
        ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).convert("RGB").copy()
        img = img.resize(size, Image.LANCZOS)
        return img
    except Exception:
        return None


def render_with_processing(
    arr:        np.ndarray,
    colormap:   str,
    clip_low:   float,
    clip_high:  float,
    processing: dict,
    size:       Optional[tuple] = None,
    vmin:       Optional[float] = None,
    vmax:       Optional[float] = None,
    allow_upscale: bool = False,
) -> Optional[Image.Image]:
    """Apply the full processing pipeline to *arr* then render to a PIL Image.

    processing keys (all optional):
        remove_bad_lines : None | 'step' | 'mad'
        remove_bad_lines_threshold : float
        remove_bad_lines_polarity : 'bright' | 'dark'
        remove_bad_lines_min_segment_length_px : int
        remove_bad_lines_max_adjacent_bad_lines : int
        align_rows       : str | None  — 'median' | 'mean' | 'linear'
        stm_background   : dict | None
        smooth_sigma     : float | None  — sigma in pixels (Gaussian)
        edge_method      : str | None   — 'laplacian' | 'log' | 'dog'
        edge_sigma       : float
        edge_sigma2      : float
        fft_mode         : None | 'low_pass' | 'high_pass'
        fft_cutoff       : float  (0.01–0.50)
        fft_window       : str    ('hanning')
        grain_threshold  : float | None  — percentile for grain detection
        grain_above      : bool
    """
    return render_scan_image(
        arr=arr,
        colormap=colormap,
        clip_low=clip_low,
        clip_high=clip_high,
        size=size,
        vmin=vmin,
        vmax=vmax,
        allow_upscale=allow_upscale,
        processing=processing or {},
    )


def _fit_image_to_box(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize an image to fit inside ``size``, including upscaling.

    Thumbnail-card preview paths can opt into this for presentation sizing.
    The full scan viewer deliberately does not use it: measured image pixels
    should open at native raster size and zoom without interpolation.
    """
    max_w, max_h = int(size[0]), int(size[1])
    if max_w <= 0 or max_h <= 0 or img.width <= 0 or img.height <= 0:
        return img
    scale = min(max_w / img.width, max_h / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    if (new_w, new_h) == img.size:
        return img
    return img.resize((new_w, new_h), Image.LANCZOS)


# ── PIL → QPixmap ─────────────────────────────────────────────────────────────
def pil_to_pixmap(img: Image.Image):
    from PySide6.QtGui import QImage, QPixmap

    img  = img.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)
