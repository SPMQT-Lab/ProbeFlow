"""Background QRunnable workers used by the ProbeFlow GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QPixmap

from probeflow.gui.models import SxmFile, VertFile
from probeflow.gui.rendering import (
    THUMBNAIL_CHANNEL_DEFAULT,
    pil_to_pixmap,
    render_scan_image,
    render_spec_thumbnail,
    resolve_thumbnail_plane_index,
)
from probeflow.core.scan_loader import load_scan

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUSHION = REPO_ROOT / "src" / "file_cushions"

# ── Worker: thumbnail ─────────────────────────────────────────────────────────
class ThumbnailSignals(QObject):
    loaded = Signal(str, QPixmap, object)  # stem, pixmap, token


class ThumbnailLoader(QRunnable):
    def __init__(self, entry: SxmFile, colormap: str, token, w: int, h: int,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None,
                 thumbnail_channel: str = THUMBNAIL_CHANNEL_DEFAULT):
        super().__init__()
        self.setAutoDelete(True)
        self.signals    = ThumbnailSignals()
        self.entry      = entry
        self.colormap   = colormap
        self.token      = token
        self.w          = w
        self.h          = h
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}
        self.thumbnail_channel = thumbnail_channel

    def run(self):
        try:
            scan = load_scan(self.entry.path)
            plane_idx = resolve_thumbnail_plane_index(
                list(getattr(scan, "plane_names", []) or []),
                self.thumbnail_channel,
            )
            arr = scan.planes[plane_idx] if plane_idx < scan.n_planes else None
        except Exception:
            arr = None
        img = render_scan_image(
            arr=arr,
            colormap=self.colormap,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            size=(self.w, self.h),
            processing=self.processing or None,
        )
        if img is not None:
            self.signals.loaded.emit(self.entry.stem, pil_to_pixmap(img), self.token)


# ── Worker: spec thumbnail ────────────────────────────────────────────────────
class SpecThumbnailLoader(QRunnable):
    def __init__(self, entry: VertFile, token, w: int, h: int, dark: bool = True):
        super().__init__()
        self.setAutoDelete(True)
        self.signals = ThumbnailSignals()
        self.entry   = entry
        self.token   = token
        self.w       = w
        self.h       = h
        self.dark    = dark

    def run(self):
        img = render_spec_thumbnail(self.entry.path, size=(self.w, self.h),
                                    dark=self.dark)
        if img is not None:
            self.signals.loaded.emit(self.entry.stem, pil_to_pixmap(img), self.token)


# ── Worker: channel thumbnails ────────────────────────────────────────────────
class ChannelSignals(QObject):
    loaded = Signal(int, QPixmap, object)


class ChannelLoader(QRunnable):
    def __init__(self, entry: SxmFile, idx: int, colormap: str,
                 token, w: int, h: int, signals: ChannelSignals,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None,
                 arr: Optional[np.ndarray] = None):
        super().__init__()
        self.setAutoDelete(True)
        self.signals    = signals
        self.entry      = entry
        self.idx        = idx
        self.colormap   = colormap
        self.token      = token
        self.w          = w
        self.h          = h
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}
        self.arr        = arr

    def run(self):
        img = render_scan_image(
            scan_path=None if self.arr is not None else self.entry.path,
            arr=self.arr,
            plane_idx=self.idx,
            colormap=self.colormap,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            size=(self.w, self.h),
            processing=self.processing or None,
        )
        if img is not None:
            self.signals.loaded.emit(self.idx, pil_to_pixmap(img), self.token)


# ── Worker: full-size viewer image ────────────────────────────────────────────
class ViewerSignals(QObject):
    loaded = Signal(QPixmap, object)


class ViewerLoader(QRunnable):
    def __init__(self, entry: SxmFile, colormap: str, token,
                 size: Optional[tuple[int, int]] = None,
                 plane_idx: int = 0, clip_low: float = 1.0,
                 clip_high: float = 99.0, processing: dict = None,
                 vmin: Optional[float] = None, vmax: Optional[float] = None,
                 arr: Optional[np.ndarray] = None):
        super().__init__()
        self.setAutoDelete(True)
        self.signals    = ViewerSignals()
        self.entry      = entry
        self.colormap   = colormap
        self.token      = token
        self.size       = size
        self.plane_idx  = plane_idx
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}
        self.vmin       = vmin
        self.vmax       = vmax
        self.arr        = arr

    def run(self):
        img = render_scan_image(
            scan_path=None if self.arr is not None else self.entry.path,
            arr=self.arr,
            plane_idx=self.plane_idx,
            colormap=self.colormap,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
            size=self.size,
            vmin=self.vmin,
            vmax=self.vmax,
            allow_upscale=False,
            processing=None if self.arr is not None else (self.processing or None),
        )
        if img is not None:
            self.signals.loaded.emit(pil_to_pixmap(img), self.token)


# ── Worker: conversion ────────────────────────────────────────────────────────
class ConversionSignals(QObject):
    log_msg  = Signal(str, str)
    finished = Signal(str)


class ConversionWorker(QRunnable):
    def __init__(self, in_dir: str, out_dir: str,
                 do_png: bool, do_sxm: bool,
                 clip_low: float, clip_high: float):
        super().__init__()
        self.setAutoDelete(True)
        self.signals   = ConversionSignals()
        self.in_dir    = in_dir
        # if no custom output, use the input dir as base
        self.out_dir   = out_dir if out_dir else in_dir
        self.do_png    = do_png
        self.do_sxm    = do_sxm
        self.clip_low  = clip_low
        self.clip_high = clip_high

    def run(self):
        def _log(msg, tag="info"):
            self.signals.log_msg.emit(msg, tag)

        in_path  = Path(self.in_dir)
        out_path = Path(self.out_dir)
        try:
            if self.do_png:
                from probeflow.io.converters.createc_dat_to_png import main as png_main
                _log("── PNG conversion ──", "info")
                png_main(src=in_path, out_root=out_path / "png",
                         clip_low=self.clip_low, clip_high=self.clip_high,
                         verbose=False)
                _log("PNG done.", "ok")

            if self.do_sxm:
                from probeflow.io.converters.createc_dat_to_sxm import convert_dat_to_sxm
                _log("── SXM conversion ──", "info")
                files = sorted(in_path.glob("*.dat"))
                if not files:
                    _log(f"No .dat files found in {in_path}", "warn")
                else:
                    sxm_out = out_path / "sxm"
                    sxm_out.mkdir(parents=True, exist_ok=True)
                    errors: dict = {}
                    _log(f"Found {len(files)} .dat file(s)", "info")
                    for i, dat in enumerate(files, 1):
                        _log(f"[{i}/{len(files)}] {dat.name} …", "info")
                        try:
                            convert_dat_to_sxm(dat, sxm_out, DEFAULT_CUSHION,
                                               self.clip_low, self.clip_high)
                            _log(f"  [OK] {dat.name}", "ok")
                        except Exception as exc:
                            _log(f"  FAILED {dat.name}: {exc}", "err")
                            errors[dat.name] = str(exc)
                    if errors:
                        import json as _j
                        (sxm_out / "errors.json").write_text(_j.dumps(errors, indent=2))
                        _log(f"{len(errors)} file(s) failed — see errors.json", "warn")
                    else:
                        _log("All SXM files processed successfully.", "ok")
                    _log(f"Output: {sxm_out}", "info")
        except Exception as exc:
            _log(f"Unexpected error: {exc}", "err")
        finally:
            self.signals.finished.emit(self.out_dir)
