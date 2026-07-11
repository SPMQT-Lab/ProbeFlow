# The 2026-07 refocus — what was removed and why

> **Status: executed and merged to `main` (2026-07-11).** This file replaces
> the original 330-line planning document; the plan was carried out in full
> on the `refocus-pare` branch (≈32,500 lines removed, suite green
> throughout). It is kept as the record of *what* left the codebase and the
> test every future feature should pass. One planning-stage idea was
> superseded during execution: instead of a plugin-registry
> "advanced package", the non-core code lives on in a full private copy of
> the repository — deletion over abstraction.

## The mission test

ProbeFlow exists to **browse and process STM images while retaining their
physical meaning** — a fast multi-format alternative to Gwyddion / WSxM /
ImageJ-style tools that never falls back to calibration-losing bitmaps.

Every feature must answer yes to: *does it help someone browse or process an
STM image while keeping physical units?* Prefer deletion over abstraction;
the codebase should stay finishable and usable.

## Removed subsystems (2026-07-10 → 07-11)

All of the following were deleted from this public repository. A full private
copy retains them for lab-internal and student work; they are not maintained
here and nothing in this tree references them.

| Removed | What it was |
|---|---|
| Particle Statistics engine | Null-model hypothesis testing (AdStat), simulation sandbox, tutorial |
| Feature Counting | Segmentation / template matching / few-shot classification |
| ML classification | scikit-learn + CLIP-torch encoders, feature bank |
| Dataset Builder | ML-labelling workflow (queue / propose / correct / export) |
| Developer Terminal | In-app shell/Python console |
| ScanFlow integration | Survey mode + acquisition-sidecar plumbing for a separate, private instrument-control tool; the info-sharing pathway is no longer envisioned |
| step_edges analysis | Step-edge exclusion masks (its only consumer was Dataset Builder) |

What **stayed** and was extended instead: browsing, processing, ROIs and
measurements, the FFT toolset, spectroscopy display, conversion/export with
provenance — plus a small descriptive point-statistics tool (feature maxima →
g(r), nearest-neighbour spacing, density, CSV/JSON export) retained as the
dependency-light kernel of the particle-statistics work.

## Follow-through

The paring was verified by a dedicated multi-pass review (see
`docs/review_status.md`, "2026-07-11"): no dangling imports, registrations,
CLI commands, docs, or dead code remain from the removed subsystems. The
`features`/`adstat`/`clip` packaging extras were replaced by a single
`lattice` extra (OpenCV + scikit-learn, used only for SIFT lattice-vector
extraction).
