"""Background QRunnable workers used by the ProbeFlow GUI."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QPixmap

from probeflow.gui.models import SxmFile, VertFile, browse_entry_key
from probeflow.core.resources import FILE_CUSHIONS_DIR
from probeflow.core.scan_loader import SUPPORTED_SUFFIXES as _SCAN_SUFFIXES
from probeflow.gui.rendering import (
    THUMBNAIL_CHANNEL_DEFAULT,
    load_thumbnail_plane,
    pil_to_pixmap,
    render_scan_image,
    render_spec_thumbnail,
)

DEFAULT_CUSHION = FILE_CUSHIONS_DIR


def _file_identity(path) -> tuple[Optional[int], Optional[int]]:
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None, None


def _cached_thumbnail_pixmap(key: Optional[str]) -> Optional[QPixmap]:
    from probeflow.core import browse_cache

    data = browse_cache.get_thumbnail(key)
    if data is None:
        return None
    pm = QPixmap()
    if pm.loadFromData(data, b"PNG"):
        return pm
    return None


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _log_preview_failure(loader_name: str, action: str, path, exc: Exception) -> None:
    _log.warning("%s: failed to %s %s (%s)", loader_name, action, path, exc)
    _log.debug(
        "%s traceback while trying to %s %s",
        loader_name,
        action,
        path,
        exc_info=True,
    )


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
        from probeflow.core import browse_cache

        mtime_ns, size = _file_identity(self.entry.path)
        key = browse_cache.thumbnail_key(
            self.entry.path, mtime_ns, size,
            kind="scan", cm=self.colormap, ch=self.thumbnail_channel,
            cl=self.clip_low, chh=self.clip_high, w=self.w, h=self.h,
            proc=self.processing or None,
        )
        cached = _cached_thumbnail_pixmap(key)
        if cached is not None:
            self.signals.loaded.emit(browse_entry_key(self.entry), cached, self.token)
            return
        try:
            arr, _names = load_thumbnail_plane(self.entry.path, self.thumbnail_channel)
        except Exception as exc:
            _log_preview_failure("ThumbnailLoader", "load", self.entry.path, exc)
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
            browse_cache.put_thumbnail(key, _png_bytes(img))
            self.signals.loaded.emit(browse_entry_key(self.entry), pil_to_pixmap(img), self.token)


# ── Worker: folder preview thumbnails ─────────────────────────────────────────
class FolderThumbnailSignals(QObject):
    loaded = Signal(str, list, object)  # folder_key, list[QPixmap | None], token


class FolderThumbnailLoader(QRunnable):
    """Render a small set of preview thumbnails for a subfolder card."""

    def __init__(self, folder_key: str, sample_paths: list[Path],
                 colormap: str, token, w: int, h: int,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 thumbnail_channel: str = THUMBNAIL_CHANNEL_DEFAULT):
        super().__init__()
        self.setAutoDelete(True)
        self.signals       = FolderThumbnailSignals()
        self.folder_key    = folder_key
        self.sample_paths  = list(sample_paths)
        self.colormap      = colormap
        self.token         = token
        self.w             = w
        self.h             = h
        self.clip_low      = clip_low
        self.clip_high     = clip_high
        self.thumbnail_channel = thumbnail_channel

    def run(self):
        from probeflow.core import browse_cache

        pixmaps: list = []
        for path in self.sample_paths:
            if path.suffix.lower() not in _SCAN_SUFFIXES:
                _log.debug("FolderThumbnailLoader: skipping unsupported file %s", path)
                pixmaps.append(None)
                continue
            mtime_ns, size = _file_identity(path)
            key = browse_cache.thumbnail_key(
                path, mtime_ns, size,
                kind="folder", cm=self.colormap, ch=self.thumbnail_channel,
                cl=self.clip_low, chh=self.clip_high, w=self.w, h=self.h,
            )
            cached = _cached_thumbnail_pixmap(key)
            if cached is not None:
                pixmaps.append(cached)
                continue
            try:
                arr, _names = load_thumbnail_plane(path, self.thumbnail_channel)
            except Exception as exc:
                _log_preview_failure("FolderThumbnailLoader", "load", path, exc)
                arr = None
            img = render_scan_image(
                arr=arr,
                colormap=self.colormap,
                clip_low=self.clip_low,
                clip_high=self.clip_high,
                size=(self.w, self.h),
            )
            if img is not None:
                browse_cache.put_thumbnail(key, _png_bytes(img))
            pixmaps.append(pil_to_pixmap(img) if img is not None else None)
        self.signals.loaded.emit(self.folder_key, pixmaps, self.token)


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
        try:
            img = render_spec_thumbnail(self.entry.path, size=(self.w, self.h),
                                        dark=self.dark)
        except Exception as exc:
            _log_preview_failure("SpecThumbnailLoader", "render", self.entry.path, exc)
            img = None
        if img is not None:
            self.signals.loaded.emit(browse_entry_key(self.entry), pil_to_pixmap(img), self.token)


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
    failed = Signal(str, object)


class ViewerLoader(QRunnable):
    def __init__(self, entry: SxmFile, colormap: str, token,
                 size: Optional[tuple[int, int]] = None,
                 plane_idx: int = 0, clip_low: float = 1.0,
                 clip_high: float = 99.0, processing: dict = None,
                 vmin: Optional[float] = None, vmax: Optional[float] = None,
                 arr: Optional[np.ndarray] = None,
                 region_levels: Optional[list] = None):
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
        self.region_levels = region_levels

    def run(self):
        try:
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
                region_levels=self.region_levels,
            )
            if img is not None:
                self.signals.loaded.emit(pil_to_pixmap(img), self.token)
            else:
                self.signals.failed.emit("Image render returned no data.", self.token)
        except Exception as exc:
            self.signals.failed.emit(f"Image render failed: {exc}", self.token)


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
