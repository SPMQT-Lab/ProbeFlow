"""Typed description of one supported microscope file structure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal


FormatKind = Literal["scan", "spectrum"]
FormatReader = Callable[..., Any]
SignatureMatcher = Callable[[bytes], bool]


class FileType(Enum):
    """Stable content identities used by ProbeFlow's loading contract."""

    CREATEC_IMAGE = "createc_image"
    CREATEC_SPEC = "createc_spec"
    NANONIS_IMAGE = "nanonis_image"
    NANONIS_SPEC = "nanonis_spec"
    RHK_SM4_IMAGE = "rhk_sm4_image"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FormatDefinition:
    """One built-in format and the capabilities ProbeFlow currently supports."""

    file_type: FileType
    source_format: str
    kind: FormatKind
    suffixes: tuple[str, ...]
    matches_header: SignatureMatcher
    read_metadata: FormatReader
    read_full: FormatReader
    read_thumbnail: FormatReader | None = None
    export_formats: frozenset[str] = frozenset()
    aliases: frozenset[str] = frozenset()
    allow_suffix_fallback: bool = False

    def __post_init__(self) -> None:
        if self.file_type is FileType.UNKNOWN:
            raise ValueError("UNKNOWN cannot be registered as a supported format")
        if not self.source_format or not self.source_format.strip():
            raise ValueError("source_format must be a non-empty stable identifier")
        if self.kind not in {"scan", "spectrum"}:
            raise ValueError(f"Unsupported format kind: {self.kind!r}")
        if not self.suffixes:
            raise ValueError("At least one suffix is required")
        normalized = tuple(suffix.lower() for suffix in self.suffixes)
        if any(not suffix.startswith(".") for suffix in normalized):
            raise ValueError("Format suffixes must start with '.'")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Format suffixes must be unique within a definition")
        for name, reader in (
            ("matches_header", self.matches_header),
            ("read_metadata", self.read_metadata),
            ("read_full", self.read_full),
        ):
            if not callable(reader):
                raise TypeError(f"{name} must be callable")
        if self.read_thumbnail is not None and not callable(self.read_thumbnail):
            raise TypeError("read_thumbnail must be callable when provided")
        object.__setattr__(self, "suffixes", normalized)
        object.__setattr__(self, "aliases", frozenset(self.aliases))
        object.__setattr__(self, "export_formats", frozenset(self.export_formats))

    @property
    def identifiers(self) -> frozenset[str]:
        return frozenset({self.source_format, *self.aliases})
