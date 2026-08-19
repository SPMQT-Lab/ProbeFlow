"""Small immutable catalog for ProbeFlow's built-in formats."""

from __future__ import annotations

from dataclasses import dataclass

from probeflow.core.formats.model import FileType, FormatDefinition, FormatKind


@dataclass(frozen=True)
class FormatCatalog:
    """Validated collection of built-in format definitions.

    This is deliberately immutable. It is not a third-party plugin registry.
    """

    definitions: tuple[FormatDefinition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", tuple(self.definitions))
        file_types = [definition.file_type for definition in self.definitions]
        if len(set(file_types)) != len(file_types):
            raise ValueError("Each FileType must have exactly one definition")

        identifiers = [
            identifier
            for definition in self.definitions
            for identifier in definition.identifiers
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Format identifiers and aliases must be globally unique")

    def by_file_type(self, file_type: FileType) -> FormatDefinition | None:
        return next(
            (definition for definition in self.definitions if definition.file_type is file_type),
            None,
        )

    def by_identifier(self, identifier: str) -> FormatDefinition | None:
        return next(
            (
                definition
                for definition in self.definitions
                if identifier in definition.identifiers
            ),
            None,
        )

    @property
    def supported_suffixes(self) -> frozenset[str]:
        return frozenset(
            suffix
            for definition in self.definitions
            for suffix in definition.suffixes
        )

    def match_header(self, head: bytes) -> FormatDefinition | None:
        """Return the first exact match, then try conservative fallbacks."""
        for definition in self.definitions:
            if definition.matches_header(head):
                return definition
        for definition in self.definitions:
            if definition.matches_fallback is not None and definition.matches_fallback(head):
                return definition
        return None

    def for_suffix(
        self,
        suffix: str,
        *,
        kind: FormatKind | None = None,
    ) -> tuple[FormatDefinition, ...]:
        normalized = suffix.lower()
        return tuple(
            definition
            for definition in self.definitions
            if normalized in definition.suffixes and (kind is None or definition.kind == kind)
        )
