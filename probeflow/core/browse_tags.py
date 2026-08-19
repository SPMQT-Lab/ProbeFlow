"""Persistent user tags for scans in a browsed folder."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HEX_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class BrowseTag:
    """A user-defined scan label and its display colour."""

    name: str
    color: str


class BrowseTagStore:
    """Store folder-local tag definitions and scan assignments.

    The sidecar is kept beside the selected browse root. Source scan files are
    never modified; assignments use paths relative to that root so nested
    folders share one tag collection.
    """

    FILENAME = ".probeflow_tags.json"

    def __init__(self, root: Optional[Path] = None):
        self.root: Optional[Path] = None
        self._tags: dict[str, BrowseTag] = {}
        self._assignments: dict[str, str] = {}
        if root is not None:
            self.load(root)

    @staticmethod
    def _name_key(name: str) -> str:
        return " ".join(str(name).split()).casefold()

    @staticmethod
    def _clean_name(name: str) -> str:
        return " ".join(str(name).split()).strip()

    @staticmethod
    def _clean_color(color: str) -> str:
        value = str(color).strip()
        if not _HEX_COLOUR.fullmatch(value):
            raise ValueError("Tag colours must be six-digit hex values")
        return value.upper()

    def load(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self._tags = {}
        self._assignments = {}
        path = self.root / self.FILENAME
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, dict):
            return

        for raw in payload.get("tags", []):
            if not isinstance(raw, dict):
                continue
            name = self._clean_name(raw.get("name", ""))
            color = str(raw.get("color", ""))
            if not name or not _HEX_COLOUR.fullmatch(color):
                continue
            self._tags[self._name_key(name)] = BrowseTag(
                name=name,
                color=color.upper(),
            )

        assignments = payload.get("assignments", {})
        if isinstance(assignments, dict):
            for path_key, tag_name in assignments.items():
                key = self._name_key(str(tag_name))
                if isinstance(path_key, str) and key in self._tags:
                    self._assignments[path_key] = key

    @property
    def sidecar_path(self) -> Optional[Path]:
        return self.root / self.FILENAME if self.root is not None else None

    def _path_key(self, path: Path) -> str:
        if self.root is None:
            raise RuntimeError("No browse root is loaded")
        path = Path(path).resolve()
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def _save(self) -> None:
        path = self.sidecar_path
        if path is None:
            return
        payload = {
            "version": 1,
            "tags": [tag.__dict__ for tag in sorted(
                self._tags.values(), key=lambda item: item.name.casefold()
            )],
            "assignments": dict(sorted(self._assignments.items())),
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def tags(self) -> tuple[BrowseTag, ...]:
        return tuple(sorted(self._tags.values(), key=lambda item: item.name.casefold()))

    def tag(self, name: str) -> Optional[BrowseTag]:
        return self._tags.get(self._name_key(name))

    def tag_for(self, path: Path) -> Optional[BrowseTag]:
        key = self._assignments.get(self._path_key(path))
        return self._tags.get(key) if key is not None else None

    def assign(self, path: Path, name: str, color: str) -> BrowseTag:
        clean_name = self._clean_name(name)
        if not clean_name:
            raise ValueError("Tag names cannot be empty")
        key = self._name_key(clean_name)
        tag = self._tags.get(key)
        if tag is None:
            tag = BrowseTag(clean_name, self._clean_color(color))
            self._tags[key] = tag
        self._assignments[self._path_key(path)] = key
        self._save()
        return tag

    def remove(self, path: Path) -> None:
        self._assignments.pop(self._path_key(path), None)
        self._save()

    def delete(self, name: str) -> None:
        key = self._name_key(name)
        self._tags.pop(key, None)
        self._assignments = {
            path: tag_key
            for path, tag_key in self._assignments.items()
            if tag_key != key
        }
        self._save()

    def options_for(self, paths: list[Path]) -> list[tuple[BrowseTag, int]]:
        counts: dict[str, int] = {}
        for path in paths:
            tag = self.tag_for(path)
            if tag is not None:
                key = self._name_key(tag.name)
                counts[key] = counts.get(key, 0) + 1
        return [
            (tag, counts.get(self._name_key(tag.name), 0))
            for tag in self.tags()
        ]
