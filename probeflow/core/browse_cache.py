"""Persistent on-disk cache for browse indexing metadata and thumbnails.

Browsing a folder on a high-latency network drive is dominated by (a) reading
each file's header to build its :class:`~probeflow.core.indexing.ProbeFlowItem`
and (b) reading the whole file again to render a thumbnail.  Re-opening the same
folder repeats all of it.  This module caches both results on the local disk so
a revisit needs only one ``scandir`` plus a cheap key comparison per file — no
file content is transferred when nothing changed.

Cache keys embed the file's ``(absolute_path, mtime_ns, size_bytes)``, so a
modified or replaced file simply misses (its stale entry is orphaned and later
evicted).  A cache-version tag in the directory name invalidates everything when
the serialised formats change.

The cache is best-effort: any I/O or deserialisation error is swallowed and
treated as a miss, so a corrupt or unreadable cache never breaks browsing.  Set
``PROBEFLOW_DISABLE_BROWSE_CACHE=1`` to turn it off, or ``PROBEFLOW_CACHE_DIR``
to relocate it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Optional

_log = logging.getLogger(__name__)

# Bump when the on-disk envelope or thumbnail params change shape.  Dataclass
# field changes are detected automatically via the schema tag (see _schema_tag),
# so this only needs bumping for structural changes the tag can't catch.
_CACHE_VERSION = "2"

# Soft cap on total cache size before least-recently-used entries are evicted.
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB
# Run eviction roughly once per this many writes (eviction scans the dir, so we
# do not want to pay for it on every single put).
_EVICT_EVERY = 256

_write_count = 0
_write_lock = threading.Lock()


def enabled() -> bool:
    """Return whether the browse cache is active (env opt-out honoured)."""
    return os.environ.get("PROBEFLOW_DISABLE_BROWSE_CACHE", "").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def _base_dir() -> Path:
    env = os.environ.get("PROBEFLOW_CACHE_DIR")
    if env:
        base = Path(env)
    else:
        try:
            import platformdirs

            base = Path(platformdirs.user_cache_dir("probeflow"))
        except Exception:
            base = Path.home() / ".cache" / "probeflow"
    return base / f"browse-v{_CACHE_VERSION}"


def _entry_path(kind: str, digest: str) -> Path:
    return _base_dir() / kind / f"{digest}.{ 'pkl' if kind == 'meta' else 'png'}"


def _digest(parts: str) -> str:
    return hashlib.sha1(parts.encode("utf-8", "replace")).hexdigest()


def _file_key(path, mtime_ns: int, size: Optional[int]) -> str:
    # os.path.abspath is a pure string op — no stat/realpath round-trip.
    return f"{os.path.abspath(str(path))}\0{mtime_ns}\0{size}"


# ── Metadata cache ────────────────────────────────────────────────────────────

def _schema_tag(item: Any) -> Optional[tuple]:
    """Return a fingerprint of a dataclass item's fields, or None.

    Stored alongside the pickled item so that adding/removing/renaming a field
    on the cached dataclass auto-invalidates stale entries — an old pickle would
    otherwise unpickle into an instance missing the new attribute and blow up
    only later, at access time.
    """
    if item is None or not dataclasses.is_dataclass(item):
        return None
    return tuple(f.name for f in dataclasses.fields(item))


def get_metadata(path, mtime_ns: Optional[int], size: Optional[int]):
    """Return ``(hit, value)`` for a cached ProbeFlowItem (value may be ``None``).

    ``value is None`` on a hit means "this file is not a recognised scan/spectrum"
    — a result worth caching so unrecognised files are not re-sniffed every visit.
    """
    if not enabled() or mtime_ns is None:
        return (False, None)
    fp = _entry_path("meta", _digest(_file_key(path, mtime_ns, size)))
    try:
        with open(fp, "rb") as fh:
            stored = pickle.load(fh)
    except FileNotFoundError:
        return (False, None)
    except Exception as exc:  # corrupt entry / unpicklable — treat as miss
        _log.debug("browse_cache: metadata read failed for %s (%s)", path, exc)
        return (False, None)

    # Envelope is (schema_tag, item); validate the dataclass shape still matches.
    if not (isinstance(stored, tuple) and len(stored) == 2):
        return (False, None)
    tag, item = stored
    if tag != _schema_tag(item):
        return (False, None)
    return (True, item)


def put_metadata(path, mtime_ns: Optional[int], size: Optional[int], item: Any) -> None:
    if not enabled() or mtime_ns is None:
        return
    fp = _entry_path("meta", _digest(_file_key(path, mtime_ns, size)))
    payload = (_schema_tag(item), item)
    _atomic_write(fp, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


# ── Thumbnail cache ─────────────────────────────────────────────────────────

def thumbnail_key(path, mtime_ns: Optional[int], size: Optional[int], **params: Any) -> Optional[str]:
    """Build a thumbnail cache key from the file identity and render params."""
    if mtime_ns is None:
        return None
    extra = "|".join(f"{k}={params[k]!r}" for k in sorted(params))
    return _digest(_file_key(path, mtime_ns, size) + "\0" + extra)


def get_thumbnail(key: Optional[str]) -> Optional[bytes]:
    if not enabled() or not key:
        return None
    fp = _entry_path("thumb", key)
    try:
        return fp.read_bytes()
    except FileNotFoundError:
        return None
    except Exception as exc:
        _log.debug("browse_cache: thumbnail read failed (%s)", exc)
        return None


def put_thumbnail(key: Optional[str], data: bytes) -> None:
    if not enabled() or not key or not data:
        return
    _atomic_write(_entry_path("thumb", key), data)


# ── Internals ─────────────────────────────────────────────────────────────────

def _atomic_write(fp: Path, data: bytes) -> None:
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_name(f".{fp.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, fp)
    except Exception as exc:  # never let a cache write break the caller
        _log.debug("browse_cache: write failed for %s (%s)", fp, exc)
        return
    _bump_and_maybe_evict()


def _bump_and_maybe_evict() -> None:
    global _write_count
    with _write_lock:
        _write_count += 1
        due = _write_count % _EVICT_EVERY == 0
    if due:
        try:
            evict(_DEFAULT_MAX_BYTES)
        except Exception as exc:
            _log.debug("browse_cache: eviction failed (%s)", exc)


def evict(max_bytes: int) -> None:
    """Delete least-recently-used entries until under ``max_bytes`` (best effort)."""
    base = _base_dir()
    files: list[tuple[float, int, Path]] = []
    total = 0
    for kind in ("meta", "thumb"):
        d = base / kind
        if not d.is_dir():
            continue
        for entry in os.scandir(d):
            try:
                st = entry.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, Path(entry.path)))
            total += st.st_size
    if total <= max_bytes:
        return
    files.sort(key=lambda t: t[0])  # oldest first
    target = int(max_bytes * 0.9)
    for _mtime, sz, fp in files:
        if total <= target:
            break
        try:
            fp.unlink()
            total -= sz
        except OSError:
            continue


def clear() -> None:
    """Remove the entire browse cache (used by tests and manual resets)."""
    import shutil

    shutil.rmtree(_base_dir(), ignore_errors=True)
