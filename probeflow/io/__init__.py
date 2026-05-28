"""Parser and writer boundaries for external ProbeFlow data.

Architectural role
------------------
``io`` owns operations that cross the filesystem boundary.  Parsers turn
vendor files (SXM, DAT, SM4, VERT, …) into :class:`probeflow.core.Scan`
objects (or, for spectra, the matching spectroscopy structs).  Writers
turn those objects back into external artifacts (PNG, PDF, CSV, GWY, SXM)
together with their JSON-friendly
:class:`probeflow.provenance.records.ExportRecord` /
:class:`SourceRecord` sidecars.

Boundary rules
--------------
Keep vendor sniffing, readers, writers, and converters here.  Provenance
dataclasses live in ``probeflow.provenance``; do not add GUI widgets, CLI
command routing, numerical processing kernels, or measurement algorithms
to this package.
"""
