"""Declared package boundaries for ProbeFlow's static import tests."""

from __future__ import annotations

from dataclasses import dataclass


ROOT_PACKAGE = "<root>"

BACKEND_PACKAGES = frozenset(
    {
        "core",
        "io",
        "processing",
        "measurements",
        "analysis",
        "spectroscopy",
        "provenance",
        "workflows",
    }
)

# Same-package imports are always allowed and are omitted here. These are the
# normal cross-package directions. Existing back-edges belong in EXCEPTIONS so
# they cannot silently spread to another module.
ALLOWED_DEPENDENCIES: dict[str, frozenset[str]] = {
    ROOT_PACKAGE: frozenset({"core"}),
    "core": frozenset(),
    "io": frozenset({"core", "provenance"}),
    "processing": frozenset({"core"}),
    "measurements": frozenset({"core", "spectroscopy"}),
    "analysis": frozenset({"core", "measurements"}),
    "spectroscopy": frozenset({"core"}),
    "provenance": frozenset({"core", "processing"}),
    "workflows": BACKEND_PACKAGES - {"workflows"},
    "gui": BACKEND_PACKAGES | frozenset({ROOT_PACKAGE}),
    "cli": BACKEND_PACKAGES | frozenset({ROOT_PACKAGE}),
}


@dataclass(frozen=True)
class DependencyException:
    """A current back-edge that must not spread beyond its listed files."""

    source: str
    target: str
    files: frozenset[str]
    reason: str
    removal: str


EXCEPTIONS = (
    DependencyException(
        source="core",
        target=ROOT_PACKAGE,
        files=frozenset({"probeflow/core/processing_state.py"}),
        reason="Serialized processing state records the package version.",
        removal="Retain until version metadata has a lower-level owner.",
    ),
    DependencyException(
        source="core",
        target="io",
        files=frozenset(
            {
                "probeflow/core/indexing.py",
                "probeflow/core/metadata.py",
                "probeflow/core/scan_model.py",
                "probeflow/core/formats/builtins.py",
            }
        ),
        reason="Public loading facades and the format catalog delegate lazily to I/O.",
        removal="Retain as named public facades during the JOSS cycle.",
    ),
    DependencyException(
        source="io",
        target="processing",
        files=frozenset(
            {
                "probeflow/io/common.py",
                "probeflow/io/converters/createc_dat_to_png.py",
                "probeflow/io/writers/png.py",
            }
        ),
        reason="Rendered exports currently prepare display arrays inside I/O paths.",
        removal="Phase 5 moves preparation into the shared export workflow.",
    ),
    DependencyException(
        source="processing",
        target="analysis",
        files=frozenset({"probeflow/processing/analysis.py"}),
        reason="Historical analysis imports are forwarded from processing.",
        removal="Retain as a compatibility module until the public API is reviewed.",
    ),
    DependencyException(
        source="processing",
        target="spectroscopy",
        files=frozenset({"probeflow/processing/spectroscopy.py"}),
        reason="Historical spectroscopy imports are forwarded from processing.",
        removal="Retain as a compatibility module until the public API is reviewed.",
    ),
    DependencyException(
        source="processing",
        target="io",
        files=frozenset(
            {
                "probeflow/processing/pdf_export.py",
                "probeflow/processing/png_export.py",
            }
        ),
        reason="Compatibility export helpers currently call file writers.",
        removal="Phase 5 replaces their implementations with workflow delegates.",
    ),
    DependencyException(
        source="processing",
        target="provenance",
        files=frozenset(
            {
                "probeflow/processing/pdf_export.py",
                "probeflow/processing/png_export.py",
            }
        ),
        reason="Compatibility export helpers currently construct provenance.",
        removal="Phase 5 replaces their implementations with workflow delegates.",
    ),
    DependencyException(
        source="analysis",
        target="io",
        files=frozenset({"probeflow/analysis/spec_plot.py"}),
        reason="The legacy spectrum plot adapter consumes the decoded I/O model.",
        removal="Retain until the spectrum public API is reviewed.",
    ),
    DependencyException(
        source="analysis",
        target="processing",
        files=frozenset(
            {
                "probeflow/analysis/helpers.py",
                "probeflow/analysis/spec_plot.py",
            }
        ),
        reason="Legacy analysis helpers reuse existing processing utilities.",
        removal="Retain until the scientific package boundaries are reviewed by humans.",
    ),
    DependencyException(
        source="measurements",
        target=ROOT_PACKAGE,
        files=frozenset({"probeflow/measurements/export.py"}),
        reason="Measurement exports record the package version.",
        removal="Retain until version metadata has a lower-level owner.",
    ),
    DependencyException(
        source="provenance",
        target=ROOT_PACKAGE,
        files=frozenset(
            {
                "probeflow/provenance/export.py",
                "probeflow/provenance/records.py",
            }
        ),
        reason="Provenance records include the package version.",
        removal="Retain until version metadata has a lower-level owner.",
    ),
    DependencyException(
        source="provenance",
        target="workflows",
        files=frozenset({"probeflow/provenance/prepared_export.py"}),
        reason="The historical prepared-export path delegates to its workflow owner.",
        removal="Retain as a compatibility facade during the JOSS cycle.",
    ),
    DependencyException(
        source="cli",
        target="gui",
        files=frozenset({"probeflow/cli/commands/gui.py"}),
        reason="The explicit gui command launches the desktop application.",
        removal="Retain as the only CLI-to-GUI boundary.",
    ),
)
