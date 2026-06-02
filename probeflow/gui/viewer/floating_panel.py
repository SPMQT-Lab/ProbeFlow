"""Floating, dismissible tool panels that hover over the image canvas.

A :class:`FloatingPanel` is a frameless card — rounded corners, a soft drop
shadow, a header bar with a title and a close button, and an arbitrary content
widget.  It can be dragged by its header and is dismissed by the close button or
the Escape key.

:class:`FloatingPanelManager` owns a set of panels parented to the canvas host
widget, positions them in a cascading stack from the top-right of the canvas,
keeps them inside the host on resize, and provides ``summon`` / ``toggle`` /
``dismiss`` so callers can pop a tool open without permanently widening the
window.  The panels contain no processing logic — they only host widgets that
already do.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, Signal
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class FloatingPanel(QFrame):
    """A draggable, closable card that floats over its parent widget."""

    closed = Signal()

    _MARGIN = 12  # px kept between the panel and the host edges

    def __init__(self, title: str, content: QWidget, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("floatingPanel")
        self.setFrameShape(QFrame.NoFrame)
        # Paint our own background so the card is opaque over the image.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._drag_offset: QPoint | None = None

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(Qt.black)
        self.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(0)

        # ── header ──────────────────────────────────────────────────────────
        self._header = QWidget(self)
        self._header.setObjectName("floatingPanelHeader")
        self._header.setCursor(QCursor(Qt.OpenHandCursor))
        head_lay = QHBoxLayout(self._header)
        head_lay.setContentsMargins(10, 5, 5, 5)
        head_lay.setSpacing(6)

        self._title_lbl = QLabel(title, self._header)
        self._title_lbl.setObjectName("floatingPanelTitle")
        head_lay.addWidget(self._title_lbl, 1)

        self._close_btn = QToolButton(self._header)
        self._close_btn.setObjectName("floatingPanelClose")
        self._close_btn.setText("✕")
        self._close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._close_btn.setToolTip("Close (Esc)")
        self._close_btn.clicked.connect(self.close)
        head_lay.addWidget(self._close_btn, 0)

        # The header drives panel dragging.
        self._header.installEventFilter(self)
        root.addWidget(self._header)

        # ── content ─────────────────────────────────────────────────────────
        self._content = content
        content.setParent(self)
        body = QWidget(self)
        body.setObjectName("floatingPanelBody")
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(8, 8, 8, 8)
        body_lay.setSpacing(6)
        body_lay.addWidget(content)
        root.addWidget(body, 1)

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def content(self) -> QWidget:
        return self._content

    def set_title(self, title: str) -> None:
        self._title_lbl.setText(title)

    # ── dragging ──────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if obj is self._header:
            etype = event.type()
            if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_offset = event.globalPosition().toPoint() - self.pos()
                self._header.setCursor(QCursor(Qt.ClosedHandCursor))
                return True
            if etype == QEvent.MouseMove and self._drag_offset is not None:
                target = event.globalPosition().toPoint() - self._drag_offset
                self.move(self._clamp_to_parent(target))
                return True
            if etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._drag_offset = None
                self._header.setCursor(QCursor(Qt.OpenHandCursor))
                return True
        return super().eventFilter(obj, event)

    def _clamp_to_parent(self, pos: QPoint) -> QPoint:
        parent = self.parentWidget()
        if parent is None:
            return pos
        max_x = max(self._MARGIN, parent.width() - self.width() - self._MARGIN)
        max_y = max(self._MARGIN, parent.height() - self.height() - self._MARGIN)
        x = min(max(self._MARGIN, pos.x()), max_x)
        y = min(max(self._MARGIN, pos.y()), max_y)
        return QPoint(x, y)

    def keep_in_parent(self) -> None:
        """Re-clamp the panel inside its parent (call after the parent resizes)."""
        self.move(self._clamp_to_parent(self.pos()))

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class FloatingPanelManager:
    """Owns floating panels hovering over a single host widget (the canvas)."""

    _CASCADE = 28  # px offset between stacked panels
    _MARGIN = FloatingPanel._MARGIN

    def __init__(self, host: QWidget):
        self._host = host
        self._panels: dict[str, FloatingPanel] = {}
        host.installEventFilter(_HostResizeFilter(self))

    # ── queries ───────────────────────────────────────────────────────────────

    def is_open(self, key: str) -> bool:
        return key in self._panels

    def panel(self, key: str) -> "FloatingPanel | None":
        return self._panels.get(key)

    # ── open / close ──────────────────────────────────────────────────────────

    def summon(
        self,
        key: str,
        title: str,
        content_factory: Callable[[], QWidget],
    ) -> FloatingPanel:
        """Open the panel for *key* (or raise it if already open) and return it.

        ``content_factory`` is only called when a new panel must be created, so
        an already-open panel keeps its live widget/state.
        """
        existing = self._panels.get(key)
        if existing is not None:
            existing.show()
            existing.raise_()
            return existing

        panel = FloatingPanel(title, content_factory(), self._host)
        panel.closed.connect(lambda k=key: self._on_closed(k))
        self._panels[key] = panel
        panel.adjustSize()
        panel.move(self._next_position(panel))
        panel.show()
        panel.raise_()
        return panel

    def toggle(
        self,
        key: str,
        title: str,
        content_factory: Callable[[], QWidget],
    ) -> "FloatingPanel | None":
        """Close the panel if open, otherwise summon it."""
        if key in self._panels:
            self.dismiss(key)
            return None
        return self.summon(key, title, content_factory)

    def dismiss(self, key: str) -> None:
        panel = self._panels.get(key)
        if panel is not None:
            panel.close()

    def dismiss_all(self) -> None:
        for panel in list(self._panels.values()):
            panel.close()

    # ── internals ─────────────────────────────────────────────────────────────

    def _on_closed(self, key: str) -> None:
        panel = self._panels.pop(key, None)
        if panel is not None:
            panel.deleteLater()

    def _next_position(self, panel: FloatingPanel) -> QPoint:
        n = len(self._panels) - 1  # this panel is already registered
        base_x = max(self._MARGIN, self._host.width() - panel.width() - self._MARGIN)
        base_y = self._MARGIN
        offset = self._CASCADE * (n % 6)
        x = max(self._MARGIN, base_x - offset)
        y = base_y + offset
        return QPoint(x, y)

    def _reposition_all(self) -> None:
        for panel in self._panels.values():
            panel.keep_in_parent()


class _HostResizeFilter(QObject):
    """Lightweight event filter object that keeps panels inside the host."""

    def __init__(self, manager: FloatingPanelManager):
        super().__init__(manager._host)
        self._manager = manager

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Resize:
            self._manager._reposition_all()
        return False


class ModalOverlay(QWidget):
    """A dimmed full-window scrim that hosts a centered child widget (a tool dialog).

    Clicking the dimmed area (outside the hosted widget) or pressing Escape dismisses
    the overlay — the Claude-settings pattern.  The hosted widget is reused whole: it
    is reparented in as a plain child (``Qt.Widget`` flags) and centered; when it
    hides or closes itself (e.g. after Apply), the overlay dismisses too.  Because the
    scrim covers the whole host, the controls behind it cannot be clicked while a tool
    is open.
    """

    _PAD = 24  # min px gap between the hosted widget and the host edges

    dismissed = Signal()

    def __init__(self, host: QWidget, content: QWidget, *, persistent: bool = False):
        super().__init__(host)
        self.setObjectName("modalOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._dismissed = False
        # When persistent, the hosted widget is reused across opens: on dismiss it is
        # handed back to *host* (hidden) instead of being destroyed with the overlay.
        self._persistent = bool(persistent)
        self._host = host

        self._content = content

        # Card wrapper: rounded chrome + soft shadow + a universal close button so
        # the centred panel reads as a deliberate layer (not a bare rectangle).
        self._card = QFrame(self)
        self._card.setObjectName("overlayCard")
        card_lay = QVBoxLayout(self._card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 6, 6, 0)
        bar.addStretch(1)
        close_btn = QToolButton(self._card)
        close_btn.setObjectName("overlayCardClose")
        close_btn.setText("✕")
        close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        close_btn.setToolTip("Close (Esc)")
        close_btn.clicked.connect(self.dismiss)
        bar.addWidget(close_btn)
        card_lay.addLayout(bar)

        self._card_body = QWidget(self._card)
        self._card_body.setObjectName("overlayCardBody")
        body_lay = QVBoxLayout(self._card_body)
        body_lay.setContentsMargins(12, 4, 12, 12)
        body_lay.setSpacing(0)
        content.setParent(self._card_body)
        content.setWindowFlags(Qt.Widget)
        body_lay.addWidget(content)
        card_lay.addWidget(self._card_body, 1)

        shadow = QGraphicsDropShadowEffect(self._card)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 160))
        self._card.setGraphicsEffect(shadow)

        content.installEventFilter(self)
        self.setGeometry(host.rect())
        host.installEventFilter(self)

        self.show()
        self.raise_()
        self._card.show()
        content.show()
        self._center_content()
        self.setFocus(Qt.OtherFocusReason)

    # ── layout ────────────────────────────────────────────────────────────────

    def _center_content(self) -> None:
        card = self._card
        card.adjustSize()
        cw = min(card.width(), max(1, self.width() - self._PAD))
        ch = min(card.height(), max(1, self.height() - self._PAD))
        card.resize(cw, ch)
        card.move(max(0, (self.width() - cw) // 2), max(0, (self.height() - ch) // 2))

    # ── events ────────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if obj is self._content:
            if event.type() in (QEvent.Close, QEvent.Hide):
                self.dismiss()
        elif obj is self.parentWidget():
            if event.type() == QEvent.Resize:
                self.setGeometry(self.parentWidget().rect())
                self._center_content()
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        # Mouse events only reach the scrim when the click is outside the hosted
        # widget (the child consumes its own); any such click dismisses.
        self.dismiss()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.dismiss()
            return
        super().keyPressEvent(event)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def dismiss(self) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        if self._persistent and self._content is not None:
            # Hand the reusable widget back to the host (hidden) so deleting the
            # overlay does not destroy it; its state survives for the next open.
            self._content.removeEventFilter(self)
            self._content.hide()
            self._content.setParent(self._host)
        self.dismissed.emit()
        self.hide()
        self.deleteLater()
