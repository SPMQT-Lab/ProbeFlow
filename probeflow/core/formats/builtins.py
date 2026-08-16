"""The five file structures currently supported by ProbeFlow."""

from __future__ import annotations

from probeflow.core.formats.catalog import FormatCatalog
from probeflow.core.formats.detection import (
    is_createc_image,
    is_createc_spec,
    is_legacy_createc_image,
    is_nanonis_image,
    is_nanonis_spec,
    is_rhk_sm4,
)
from probeflow.core.formats.model import FileType, FormatDefinition


def _read_createc_scan_metadata(path):
    from probeflow.io.readers.createc_scan import read_dat_metadata

    return read_dat_metadata(path)


def _read_createc_scan(path):
    from probeflow.io.readers.createc_scan import read_dat

    return read_dat(path)


def _read_nanonis_scan_metadata(path):
    from probeflow.io.readers.nanonis_sxm import read_sxm_metadata

    return read_sxm_metadata(path)


def _read_nanonis_scan(path):
    from probeflow.io.readers.nanonis_sxm import read_sxm

    return read_sxm(path)


def _read_rhk_scan_metadata(path):
    from probeflow.io.readers.rhk_sm4 import read_sm4_metadata

    return read_sm4_metadata(path)


def _read_rhk_scan(path):
    from probeflow.io.readers.rhk_sm4 import read_sm4

    return read_sm4(path)


def _read_rhk_thumbnail(path, resolve_index):
    from probeflow.io.readers.rhk_sm4 import read_sm4_thumbnail_plane

    return read_sm4_thumbnail_plane(path, resolve_index)


def _read_createc_spec_metadata(path, **options):
    from probeflow.io.spectroscopy import _read_createc_vert_metadata

    return _read_createc_vert_metadata(path, **options)


def _read_createc_spec(path, **options):
    from probeflow.io.spectroscopy import _read_createc_vert

    return _read_createc_vert(path, **options)


def _read_nanonis_spec_metadata(path, *, measurement_mode=None, **_options):
    from probeflow.io.readers.nanonis_spec import read_nanonis_spec_metadata
    from probeflow.io.spectroscopy import _apply_measurement_override

    metadata = read_nanonis_spec_metadata(path)
    _apply_measurement_override(metadata.metadata, measurement_mode)
    return metadata


def _read_nanonis_spec(path, *, measurement_mode=None, **_options):
    from probeflow.io.readers.nanonis_spec import read_nanonis_spec
    from probeflow.io.spectroscopy import _apply_measurement_override

    spec = read_nanonis_spec(path)
    _apply_measurement_override(spec.metadata, measurement_mode)
    return spec


BUILTIN_FORMATS = FormatCatalog(
    (
        FormatDefinition(
            file_type=FileType.RHK_SM4_IMAGE,
            format_id="rhk_sm4",
            kind="scan",
            suffixes=(".sm4",),
            matches_header=is_rhk_sm4,
            read_metadata=_read_rhk_scan_metadata,
            read_full=_read_rhk_scan,
            load_id="sm4",
            read_thumbnail=_read_rhk_thumbnail,
            export_formats=frozenset({"png", "pdf", "csv", "gwy"}),
            aliases=frozenset({"sm4"}),
        ),
        FormatDefinition(
            file_type=FileType.NANONIS_SPEC,
            format_id="nanonis_dat_spectrum",
            kind="spectrum",
            suffixes=(".dat",),
            matches_header=is_nanonis_spec,
            read_metadata=_read_nanonis_spec_metadata,
            read_full=_read_nanonis_spec,
            export_formats=frozenset({"csv", "json", "txt"}),
        ),
        FormatDefinition(
            file_type=FileType.CREATEC_SPEC,
            format_id="createc_vert",
            kind="spectrum",
            suffixes=(".vert",),
            matches_header=is_createc_spec,
            read_metadata=_read_createc_spec_metadata,
            read_full=_read_createc_spec,
            export_formats=frozenset({"csv", "json", "txt"}),
            allow_suffix_fallback=True,
        ),
        FormatDefinition(
            file_type=FileType.CREATEC_IMAGE,
            format_id="createc_dat",
            kind="scan",
            suffixes=(".dat",),
            matches_header=is_createc_image,
            matches_fallback=is_legacy_createc_image,
            read_metadata=_read_createc_scan_metadata,
            read_full=_read_createc_scan,
            load_id="dat",
            export_formats=frozenset({"npy", "png", "pdf", "csv", "gwy", "sxm"}),
            aliases=frozenset({"dat"}),
        ),
        FormatDefinition(
            file_type=FileType.NANONIS_IMAGE,
            format_id="nanonis_sxm",
            kind="scan",
            suffixes=(".sxm",),
            matches_header=is_nanonis_image,
            read_metadata=_read_nanonis_scan_metadata,
            read_full=_read_nanonis_scan,
            load_id="sxm",
            export_formats=frozenset({"png", "pdf", "csv", "gwy", "sxm"}),
            aliases=frozenset({"sxm"}),
            allow_suffix_fallback=True,
        ),
    )
)
