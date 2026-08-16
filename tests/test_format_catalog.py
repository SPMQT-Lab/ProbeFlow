"""Contract tests for the immutable built-in format catalog."""

from __future__ import annotations

import pytest

from probeflow.core.file_type import FileType as CompatibilityFileType
from probeflow.core.formats import BUILTIN_FORMATS, FileType, FormatCatalog, FormatDefinition


def _reader(*args, **kwargs):
    return args, kwargs


def _definition(
    file_type: FileType,
    format_id: str,
    *,
    suffixes: tuple[str, ...] = (".dat",),
    aliases: frozenset[str] = frozenset(),
) -> FormatDefinition:
    return FormatDefinition(
        file_type=file_type,
        format_id=format_id,
        kind="scan",
        suffixes=suffixes,
        matches_header=lambda head: bool(head),
        read_metadata=_reader,
        read_full=_reader,
        aliases=aliases,
    )


def test_file_type_keeps_its_existing_import_identity() -> None:
    assert CompatibilityFileType is FileType


def test_builtin_catalog_declares_only_the_five_supported_structures() -> None:
    assert {definition.format_id for definition in BUILTIN_FORMATS.definitions} == {
        "createc_dat",
        "nanonis_sxm",
        "rhk_sm4",
        "createc_vert",
        "nanonis_dat_spectrum",
    }
    assert {definition.kind for definition in BUILTIN_FORMATS.for_suffix(".dat")} == {
        "scan",
        "spectrum",
    }
    assert BUILTIN_FORMATS.by_identifier("dat").format_id == "createc_dat"
    assert BUILTIN_FORMATS.by_identifier("sxm").format_id == "nanonis_sxm"


def test_definition_normalizes_suffixes_and_freezes_sets() -> None:
    definition = _definition(
        FileType.CREATEC_IMAGE,
        "dat",
        suffixes=(".DAT",),
        aliases=frozenset({"createc_dat"}),
    )

    assert definition.suffixes == (".dat",)
    assert definition.identifiers == frozenset({"dat", "createc_dat"})
    assert definition.export_formats == frozenset()


def test_catalog_resolves_stable_ids_aliases_and_suffixes() -> None:
    createc = _definition(
        FileType.CREATEC_IMAGE,
        "dat",
        aliases=frozenset({"createc_dat"}),
    )
    nanonis = _definition(
        FileType.NANONIS_IMAGE,
        "sxm",
        suffixes=(".sxm",),
        aliases=frozenset({"nanonis_sxm"}),
    )
    catalog = FormatCatalog((createc, nanonis))

    assert catalog.by_file_type(FileType.CREATEC_IMAGE) is createc
    assert catalog.by_identifier("createc_dat") is createc
    assert catalog.by_identifier("nanonis_sxm") is nanonis
    assert catalog.for_suffix(".DAT") == (createc,)
    assert catalog.for_suffix(".sxm", kind="scan") == (nanonis,)


def test_catalog_rejects_duplicate_file_types_and_identifiers() -> None:
    first = _definition(FileType.CREATEC_IMAGE, "dat")
    duplicate_type = _definition(FileType.CREATEC_IMAGE, "other")
    duplicate_id = _definition(FileType.NANONIS_IMAGE, "sxm", aliases=frozenset({"dat"}))

    with pytest.raises(ValueError, match="FileType"):
        FormatCatalog((first, duplicate_type))
    with pytest.raises(ValueError, match="identifiers"):
        FormatCatalog((first, duplicate_id))


@pytest.mark.parametrize(
    "definition, message",
    [
        (
            lambda: _definition(FileType.UNKNOWN, "unknown"),
            "UNKNOWN",
        ),
        (
            lambda: _definition(FileType.CREATEC_IMAGE, ""),
            "format_id",
        ),
        (
            lambda: _definition(FileType.CREATEC_IMAGE, "dat", suffixes=("dat",)),
            "start with",
        ),
    ],
)
def test_definition_rejects_invalid_contracts(definition, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        definition()
