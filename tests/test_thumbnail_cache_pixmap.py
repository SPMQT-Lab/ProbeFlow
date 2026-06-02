"""Regression test for thumbnail-cache pixmap decoding.

`_cached_thumbnail_pixmap` previously called QPixmap.loadFromData(data, b"PNG"),
which trips PySide6's overload resolution on some versions and raised
ValueError on every cache hit (forcing a full re-render). The decoder must
load cached PNG bytes without an explicit positional format.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"QApplication unavailable: {exc}")


def _png_bytes(width: int, height: int) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray
    from PySide6.QtGui import QPixmap

    pm = QPixmap(width, height)
    pm.fill()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QBuffer.WriteOnly)
    pm.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def test_cached_thumbnail_pixmap_decodes_png(qapp, monkeypatch):
    from probeflow.core import browse_cache
    from probeflow.gui import workers

    data = _png_bytes(12, 9)
    monkeypatch.setattr(browse_cache, "get_thumbnail", lambda key: data)

    pm = workers._cached_thumbnail_pixmap("any-key")

    assert pm is not None
    assert not pm.isNull()
    assert (pm.width(), pm.height()) == (12, 9)


def test_cached_thumbnail_pixmap_returns_none_on_cache_miss(qapp, monkeypatch):
    from probeflow.core import browse_cache
    from probeflow.gui import workers

    monkeypatch.setattr(browse_cache, "get_thumbnail", lambda key: None)
    assert workers._cached_thumbnail_pixmap("missing") is None
