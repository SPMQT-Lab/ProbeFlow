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
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from probeflow.gui.models import (
    SxmFile,
    VertFile,
    browse_entry_key,
    is_raw_channel_name,
)
from probeflow.core.resources import FILE_CUSHIONS_DIR
from probeflow.core.scan_loader import SUPPORTED_SUFFIXES as _SCAN_SUFFIXES
from probeflow.gui.rendering import (
    THUMBNAIL_CHANNEL_DEFAULT,
    load_thumbnail_plane,
    pil_to_qimage,
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


def _cached_thumbnail_image(key: Optional[str]) -> Optional[QImage]:
    from probeflow.core import browse_cache

    data = browse_cache.get_thumbnail(key)
    if data is None:
        return None
    img = QImage()
    # Let Qt auto-detect the format from the PNG header. Passing an explicit
    # positional format (b"PNG") trips PySide6's loadFromData overload
    # resolution on some versions ("wrong argument values"), which made every
    # cache hit fall back to a full re-render.
    if img.loadFromData(data):
        return img
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


class _PooledWorker(QRunnable):
    """Base for fire-and-forget ``QThreadPool`` workers that own their signals.

    QThreadPool auto-deletes a finished QRunnable on the *worker* thread. If the
    runnable is the sole owner of a parentless ``*Signals`` QObject — which
    carries cross-thread signal/slot connections — that QObject gets destroyed
    off the main thread, corrupting Qt's internals (observed as a hard SIGSEGV
    inside the app-level tooltip event filter). This base parents the signals to
    the main-thread ``QApplication`` so Shiboken never C++-destroys it from the
    worker thread, and ``deleteLater()``s it after ``work()`` so there is no
    per-run accumulation.

    Subclasses implement :meth:`work` (not ``run``) and pass their signals
    object to ``super().__init__()``. Workers whose signals are created and
    retained by the *caller* (e.g. ``ChannelLoader``) must not use this base —
    the caller owns that lifetime.
    """

    def __init__(self, signals: QObject) -> None:
        super().__init__()
        self.setAutoDelete(True)
        app = QApplication.instance()
        if app is not None:
            signals.setParent(app)
        self.signals = signals

    def run(self) -> None:
        try:
            self.work()
        finally:
            self.signals.deleteLater()

    def work(self) -> None:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError


# ── Worker: thumbnail ─────────────────────────────────────────────────────────
# All worker signals carry QImage, never QPixmap: QPixmap may only be created
# or copied on the GUI thread, so workers emit the thread-safe QImage and the
# main-thread receiver slot converts via QPixmap.fromImage().
class ThumbnailSignals(QObject):
    loaded = Signal(str, QImage, object)  # stem, image, token


class ThumbnailLoader(_PooledWorker):
    def __init__(self, entry: SxmFile, colormap: str, token, w: int, h: int,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None,
                 thumbnail_channel: str = THUMBNAIL_CHANNEL_DEFAULT):
        super().__init__(ThumbnailSignals())
        self.entry      = entry
        self.colormap   = colormap
        self.token      = token
        self.w          = w
        self.h          = h
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}
        self.thumbnail_channel = thumbnail_channel

    def work(self):
        from probeflow.core import browse_cache

        mtime_ns, size = _file_identity(self.entry.path)
        key = browse_cache.thumbnail_key(
            self.entry.path, mtime_ns, size,
            kind="scan", cm=self.colormap, ch=self.thumbnail_channel,
            cl=self.clip_low, chh=self.clip_high, w=self.w, h=self.h,
            proc=self.processing or None,
        )
        cached = _cached_thumbnail_image(key)
        if cached is not None:
            self.signals.loaded.emit(browse_entry_key(self.entry), cached, self.token)
            return
        try:
            arr, _names = load_thumbnail_plane(self.entry.path, self.thumbnail_channel)
        except Exception as exc:
            _log_preview_failure("ThumbnailLoader", "load", self.entry.path, exc)
            arr = None
        # The render must be guarded too: an exception escaping work() is
        # swallowed by QThreadPool with no emit, leaving the card on its
        # loading placeholder forever.
        try:
            img = render_scan_image(
                arr=arr,
                colormap=self.colormap,
                clip_low=self.clip_low,
                clip_high=self.clip_high,
                size=(self.w, self.h),
                processing=self.processing or None,
            )
        except Exception as exc:
            _log_preview_failure("ThumbnailLoader", "render", self.entry.path, exc)
            img = None
        if img is not None:
            browse_cache.put_thumbnail(key, _png_bytes(img))
            self.signals.loaded.emit(browse_entry_key(self.entry), pil_to_qimage(img), self.token)
        else:
            # Emit a null QImage so the card can show a failure placeholder
            # instead of silently never updating.
            self.signals.loaded.emit(browse_entry_key(self.entry), QImage(), self.token)


# ── Worker: folder preview thumbnails ─────────────────────────────────────────
class FolderThumbnailSignals(QObject):
    loaded = Signal(str, list, object)  # folder_key, list[QImage | None], token


class FolderThumbnailLoader(_PooledWorker):
    """Render a small set of preview thumbnails for a subfolder card."""

    def __init__(self, folder_key: str, sample_paths: list[Path],
                 colormap: str, token, w: int, h: int,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 thumbnail_channel: str = THUMBNAIL_CHANNEL_DEFAULT):
        super().__init__(FolderThumbnailSignals())
        self.folder_key    = folder_key
        self.sample_paths  = list(sample_paths)
        self.colormap      = colormap
        self.token         = token
        self.w             = w
        self.h             = h
        self.clip_low      = clip_low
        self.clip_high     = clip_high
        self.thumbnail_channel = thumbnail_channel

    def work(self):
        from probeflow.core import browse_cache

        images: list = []
        for path in self.sample_paths:
            if path.suffix.lower() not in _SCAN_SUFFIXES:
                _log.debug("FolderThumbnailLoader: skipping unsupported file %s", path)
                images.append(None)
                continue
            mtime_ns, size = _file_identity(path)
            key = browse_cache.thumbnail_key(
                path, mtime_ns, size,
                kind="folder", cm=self.colormap, ch=self.thumbnail_channel,
                cl=self.clip_low, chh=self.clip_high, w=self.w, h=self.h,
            )
            cached = _cached_thumbnail_image(key)
            if cached is not None:
                images.append(cached)
                continue
            try:
                arr, _names = load_thumbnail_plane(path, self.thumbnail_channel)
            except Exception as exc:
                _log_preview_failure("FolderThumbnailLoader", "load", path, exc)
                arr = None
            # Guard the render per sample path: one bad file must yield a
            # None slot, not kill the worker before the list is emitted.
            try:
                img = render_scan_image(
                    arr=arr,
                    colormap=self.colormap,
                    clip_low=self.clip_low,
                    clip_high=self.clip_high,
                    size=(self.w, self.h),
                )
            except Exception as exc:
                _log_preview_failure("FolderThumbnailLoader", "render", path, exc)
                img = None
            if img is not None:
                browse_cache.put_thumbnail(key, _png_bytes(img))
            images.append(pil_to_qimage(img) if img is not None else None)
        self.signals.loaded.emit(self.folder_key, images, self.token)


# ── Worker: spec thumbnail ────────────────────────────────────────────────────
class SpecThumbnailLoader(_PooledWorker):
    def __init__(self, entry: VertFile, token, w: int, h: int, dark: bool = True):
        super().__init__(ThumbnailSignals())
        self.entry   = entry
        self.token   = token
        self.w       = w
        self.h       = h
        self.dark    = dark

    def work(self):
        try:
            img = render_spec_thumbnail(self.entry.path, size=(self.w, self.h),
                                        dark=self.dark)
        except Exception as exc:
            _log_preview_failure("SpecThumbnailLoader", "render", self.entry.path, exc)
            img = None
        if img is not None:
            self.signals.loaded.emit(browse_entry_key(self.entry), pil_to_qimage(img), self.token)
        else:
            # Null QImage → card shows a failure placeholder instead of
            # silently never updating.
            self.signals.loaded.emit(browse_entry_key(self.entry), QImage(), self.token)


# ── Worker: browse channel previews + header (single off-thread load) ─────────
class ChannelPreviewSignals(QObject):
    meta_ready = Signal(list, dict, object)   # plane_names, header, token
    loaded     = Signal(int, QImage, object)  # plane idx, image, token
    failed     = Signal(str, object)          # message, token


class ChannelPreviewLoader(_PooledWorker):
    """Load a scan once off the GUI thread and render its channel previews.

    Replaces the browse info panel's synchronous double load — one full
    ``load_scan`` for the channel thumbnails plus a second one just for the
    header metadata table — which froze the GUI for two full file transfers
    per card click on a network drive.  Emits ``meta_ready`` (plane names +
    header) first so the panel can build its preview slots and metadata table,
    then one ``loaded`` per plane.
    """

    def __init__(self, entry: SxmFile, colormap: str, token, w: int, h: int,
                 clip_low: float = 1.0, clip_high: float = 99.0,
                 processing: dict = None):
        super().__init__(ChannelPreviewSignals())
        self.entry      = entry
        self.colormap   = colormap
        self.token      = token
        self.w          = w
        self.h          = h
        self.clip_low   = clip_low
        self.clip_high  = clip_high
        self.processing = processing or {}

    def work(self):
        from probeflow.core.scan_loader import load_scan

        try:
            scan = load_scan(self.entry.path)
        except Exception as exc:
            _log_preview_failure("ChannelPreviewLoader", "load", self.entry.path, exc)
            self.signals.failed.emit(str(exc), self.token)
            return
        names = list(scan.plane_names or []) or [
            f"Channel {i}" for i in range(scan.n_planes)
        ]
        # Hide generic 'Raw channel N' auxiliary planes (e.g. createc DAC slots
        # beyond the known signals) so the Browse preview shows only meaningful
        # channels. Keep all of them if none are named, so the grid is never
        # blank. ``visible`` pairs each kept plane's source index with its name;
        # previews are emitted by display position so they line up with the slots
        # the panel builds from the (filtered) names.
        visible = [
            (i, name) for i, name in enumerate(names)
            if not is_raw_channel_name(name)
        ]
        if not visible:
            visible = list(enumerate(names))
        header = dict(getattr(scan, "header", {}) or {})
        self.signals.meta_ready.emit([name for _, name in visible], header, self.token)
        for slot, (i, _name) in enumerate(visible):
            # Guard each plane independently: one unusual plane must emit a
            # null preview for its slot, not kill the worker mid-stream and
            # leave the remaining slots on their placeholders with no
            # failed signal.
            try:
                img = render_scan_image(
                    arr=scan.planes[i],
                    colormap=self.colormap,
                    clip_low=self.clip_low,
                    clip_high=self.clip_high,
                    size=(self.w, self.h),
                    processing=self.processing or None,
                )
                qimg = pil_to_qimage(img) if img is not None else QImage()
            except Exception as exc:
                _log_preview_failure(
                    "ChannelPreviewLoader", f"render plane {i} of",
                    self.entry.path, exc,
                )
                qimg = QImage()
            self.signals.loaded.emit(slot, qimg, self.token)


# ── Worker: channel thumbnails ────────────────────────────────────────────────
class ChannelSignals(QObject):
    loaded = Signal(int, QImage, object)


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
        # Same guard as the pooled loaders: a render exception must produce
        # a null-image emit, not escape run() with no signal at all.
        try:
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
        except Exception as exc:
            _log_preview_failure("ChannelLoader", "render", self.entry.path, exc)
            img = None
        if img is not None:
            self.signals.loaded.emit(self.idx, pil_to_qimage(img), self.token)
        else:
            # Null QImage → receiver shows a failure placeholder rather than
            # leaving the previous channel's preview silently in place.
            self.signals.loaded.emit(self.idx, QImage(), self.token)


# ── Worker: full-size viewer image ────────────────────────────────────────────
class ViewerSignals(QObject):
    loaded = Signal(QImage, object)
    failed = Signal(str, object)


class ViewerLoader(_PooledWorker):
    def __init__(self, entry: SxmFile, colormap: str, token,
                 size: Optional[tuple[int, int]] = None,
                 plane_idx: int = 0, clip_low: float = 1.0,
                 clip_high: float = 99.0, processing: dict = None,
                 vmin: Optional[float] = None, vmax: Optional[float] = None,
                 arr: Optional[np.ndarray] = None,
                 region_levels: Optional[list] = None):
        super().__init__(ViewerSignals())
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

    def work(self):
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
                self.signals.loaded.emit(pil_to_qimage(img), self.token)
            else:
                self.signals.failed.emit("Image render returned no data.", self.token)
        except Exception as exc:
            self.signals.failed.emit(f"Image render failed: {exc}", self.token)


# ── Worker: shallow folder index ──────────────────────────────────────────────
class FolderIndexSignals(QObject):
    indexed = Signal(object, object, object)  # path, ShallowFolderIndex, token
    failed  = Signal(object, str, object)     # path, message, token


class FolderIndexLoader(_PooledWorker):
    """Run ``index_folder_shallow`` off the GUI thread.

    A cold visit to a large folder on a network drive takes seconds even with
    the indexing layer's internal parallelism; doing it synchronously in
    ``ThumbnailGrid._navigate`` froze the whole UI for that long.  Stale
    results (the user navigated again) are dropped via the token.
    """

    def __init__(self, path, token) -> None:
        super().__init__(FolderIndexSignals())
        self.path = Path(path)
        self.token = token

    def work(self):
        from probeflow.core.indexing import index_folder_shallow

        try:
            index = index_folder_shallow(self.path, include_errors=True)
        except Exception as exc:
            _log.warning("FolderIndexLoader: failed to index %s (%s)", self.path, exc)
            self.signals.failed.emit(self.path, str(exc), self.token)
            return
        self.signals.indexed.emit(self.path, index, self.token)


# ── Worker: conversion ────────────────────────────────────────────────────────
class ConversionSignals(QObject):
    log_msg  = Signal(str, str)
    finished = Signal(str)


class ConversionWorker(_PooledWorker):
    def __init__(self, in_dir: str, out_dir: str,
                 do_png: bool, do_sxm: bool,
                 clip_low: float, clip_high: float,
                 do_npy_raw: bool = False,
                 do_npy_physical: bool = False):
        super().__init__(ConversionSignals())
        self.in_dir    = in_dir
        # if no custom output, use the input dir as base
        self.out_dir   = out_dir if out_dir else in_dir
        self.do_png    = do_png
        self.do_sxm    = do_sxm
        self.do_npy_raw = do_npy_raw
        self.do_npy_physical = do_npy_physical
        self.clip_low  = clip_low
        self.clip_high = clip_high

    def work(self):
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

            if self.do_npy_raw or self.do_npy_physical:
                from probeflow.io.converters.createc_dat_to_npy import main as npy_main
                _log("── NumPy conversion ──", "info")
                basis = (
                    "both"
                    if self.do_npy_raw and self.do_npy_physical
                    else "raw"
                    if self.do_npy_raw
                    else "physical"
                )
                npy_out = out_path / "npy"
                npy_main(
                    src=in_path,
                    out_root=npy_out,
                    basis=basis,
                    force=False,
                    verbose=False,
                )
                _log("NumPy done.", "ok")
                _log(f"Output: {npy_out}", "info")
        except Exception as exc:
            _log(f"Unexpected error: {exc}", "err")
        finally:
            self.signals.finished.emit(self.out_dir)
