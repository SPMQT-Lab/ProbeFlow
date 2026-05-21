"""Browse-grid cards and thumbnail grid widgets for the ProbeFlow GUI."""

from __future__ import annotations

from probeflow.core.scan_loader import load_scan
from probeflow.gui.workers import (
    ChannelLoader,
    ChannelSignals,
    FolderThumbnailLoader,
    SpecThumbnailLoader,
    ThumbnailLoader,
)

from .breadcrumbs import _BreadcrumbBar
from .cards import FolderCard, ScanCard, SpecCard, _BrowseCard
from .helpers import (
    _CARD_SIZE_PRESETS,
    _card_compact_meta_str,
    _is_deleted_qt_runtime_error,
    _sep,
)
from .panels import BrowseInfoPanel, BrowseToolPanel
from .thumbnail_grid import ThumbnailGrid

_PUBLIC_MODULE = "probeflow.gui.browse"
for _cls in (
    BrowseToolPanel,
    BrowseInfoPanel,
    _BrowseCard,
    ScanCard,
    SpecCard,
    FolderCard,
    _BreadcrumbBar,
    ThumbnailGrid,
):
    _cls.__module__ = _PUBLIC_MODULE

del _cls

__all__ = [
    "BrowseToolPanel",
    "BrowseInfoPanel",
    "ThumbnailGrid",
    "ScanCard",
    "SpecCard",
    "FolderCard",
]
