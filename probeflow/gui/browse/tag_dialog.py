"""Small chooser for assigning and managing scan tags."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from probeflow.core.browse_tags import BrowseTag
from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class TagChooserDialog(QDialog):
    """Choose an existing tag or create one from a small colour palette."""

    PALETTE = (
        "#E9B949", "#E76F51", "#F28482", "#90BE6D",
        "#43AA8B", "#4D908E", "#577590", "#7B61A8",
    )

    def __init__(
        self,
        tags: Iterable[BrowseTag],
        delete_tag: Callable[[str], None],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Tag scan")
        self.setModal(True)
        self._delete_tag = delete_tag
        self._choice: tuple[str, str] | None = None
        self._rows: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(8)

        heading = QLabel("Choose an existing tag")
        heading.setFont(ui_font(10, weight=QFont.Bold))
        layout.addWidget(heading)

        self._existing_layout = QVBoxLayout()
        self._existing_layout.setSpacing(3)
        layout.addLayout(self._existing_layout)
        self._populate_existing(tuple(tags))

        new_heading = QLabel("Or create a new tag")
        new_heading.setFont(ui_font(10, weight=QFont.Bold))
        layout.addWidget(new_heading)
        palette = QGridLayout()
        palette.setSpacing(6)
        for index, color in enumerate(self.PALETTE):
            button = QToolButton()
            button.setFixedSize(30, 30)
            button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(f"Create a tag using {color}")
            button.setStyleSheet(
                f"QToolButton {{ background: {color}; border: 1px solid #ffffff; "
                "border-radius: 4px; }}"
                f"QToolButton:hover {{ border: 2px solid {color}; }}"
            )
            button.clicked.connect(lambda _=False, c=color: self._new_tag(c))
            palette.addWidget(button, index // 4, index % 4)
        palette.setAlignment(Qt.AlignLeft)
        layout.addLayout(palette)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.adjustSize()

    def _populate_existing(self, tags: tuple[BrowseTag, ...]) -> None:
        if not tags:
            empty = QLabel("No tags have been created yet.")
            empty.setFont(ui_font(9))
            self._existing_layout.addWidget(empty)
            return
        for tag in tags:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            choose = QPushButton(tag.name)
            choose.setFont(ui_font(9))
            choose.setCursor(Qt.PointingHandCursor)
            choose.setToolTip(f"Apply '{tag.name}' to this scan")
            choose.clicked.connect(
                lambda _=False, t=tag: self._choose(t.name, t.color)
            )
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: {tag.color}; border-radius: 6px;"
            )
            choose_row = QHBoxLayout()
            choose_row.setContentsMargins(0, 0, 0, 0)
            choose_row.setSpacing(6)
            choose_row.addWidget(swatch)
            choose_row.addWidget(choose, 1)

            delete = QToolButton()
            delete.setFixedSize(26, 26)
            delete.setCursor(Qt.PointingHandCursor)
            delete.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
            delete.setToolTip(f"Delete tag '{tag.name}'")
            delete.clicked.connect(lambda _=False, n=tag.name: self._confirm_delete(n))

            row_widget = QWidget()
            row_widget.setLayout(row)
            row.addLayout(choose_row, 1)
            row.addWidget(delete)
            self._existing_layout.addWidget(row_widget)
            self._rows[self._key(tag.name)] = row_widget

    @staticmethod
    def _key(name: str) -> str:
        return " ".join(str(name).split()).casefold()

    def _choose(self, name: str, color: str) -> None:
        self._choice = (name, color)
        self.accept()

    def _new_tag(self, color: str) -> None:
        name, accepted = QInputDialog.getText(
            self, "Name tag", "Tag name:",
        )
        name = " ".join(str(name).split()).strip()
        if accepted and name:
            self._choose(name, QColor(color).name(QColor.HexRgb))

    def _confirm_delete(self, name: str) -> None:
        answer = QMessageBox.question(
            self,
            "Delete tag",
            "Deleting this tag will remove it from all of its tagged images, "
            "are you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._delete_tag(name)
        row = self._rows.pop(self._key(name), None)
        if row is not None:
            row.deleteLater()

    def choice(self) -> tuple[str, str] | None:
        return self._choice
