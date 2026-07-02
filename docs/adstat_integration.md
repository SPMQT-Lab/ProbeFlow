# AdStat Integration

ProbeFlow can hand curated STM point collections to AdStat without exporting an
intermediate file. ProbeFlow remains responsible for image loading, processing,
point detection, ROI and mask editing, and Qt presentation. AdStat receives
normalised point-pattern objects, runs the statistics and matched null models,
and returns GUI-free result/view specifications for ProbeFlow to render.

The adapter lives in `probeflow.analysis.adstat_adapter` and imports AdStat
lazily. Install the optional AdStat extra, or put an AdStat checkout on
`PYTHONPATH`, before using this path:

```bash
pip install "probeflow[adstat]"
```

The viewer's **Particle Statistics...** command opens a ProbeFlow-native Qt
shell powered by the AdStat engine. Its **Analyze scan points** mode calls
`compare_point_source_view_spec(...)` for live point-source records already
collected by the image viewer. The saved/session feature-set workflow builds
`point_set_record(...)` objects and runs `compare_point_set_record_view_spec(...)`
for one set or `compare_point_set_records_view_spec(...)` for pooled replicate
sets.

Points reach that shared feature-set pool from several sources:

- **Feature Finder** and **Feature Counting** both have a *Send to Particle
  Statistics* action. Feature Counting particles/detections are converted with
  `feature_counting_to_particle_table(...)` (via
  `point_table_io.feature_items_to_feature_set`) into a `FeatureSet`.
- **Load points from disk…** uses `point_table_io.sniff_point_table` /
  `load_point_table` to import CSV position tables and ProbeFlow JSON (Feature
  Counting exports and saved `FeatureSetStore` files).

`compare_particle_collection_view_spec(...)` remains the broadest single-call
entry point and also accepts independent feature layers.

## Single Image

1. Generate one point collection from any canonical ProbeFlow source:
   Feature Finder maxima/minima, Feature maxima, point ROIs, Feature Counting
   segmented particles, or template-match detections.
2. Choose that collection as the tested population. Each run analyses one
   population; mixed species should be analysed separately unless the scientific
   intent is an unlabelled merged population.
3. Choose the analysis region from an active area ROI, mask, or the full image.
   The same region must be used for observed statistics and every null
   simulation, so the adapter restricts the tested points to the region before
   analysis: points outside the ROI/mask are excluded from both the observed
   statistics and the simulated point counts, and the result reports how many
   points were kept.
4. Convert through AdStat `ImageCalibration`, preserving anisotropic pixel
   sizes from `Scan.scan_range_m` and `Scan.dims`.
5. Open **Measurements -> Features -> Particle Statistics...** and use
   **Analyze scan points** to pass
   `ParticleTable + AnalysisRegion + optional independent feature layers` to
   AdStat and render the returned `ResultViewSpec` with ProbeFlow's native Qt
   result-view widgets.

Measured feature layers, such as step traces or defect landmarks, must be
independent measurements. A layer derived from the same particle centroids being
tested is not a valid measured-inhomogeneity null.

The older **Pair correlation...** dialog remains available during migration. The
AdStat path uses matched simulation envelopes and verdict rows, so its plots are
not intended to be numerically identical to the older square-window
pair-correlation readout.

## Synthetic Demo Run

The repository includes a reproducible teaching script that generates a
clustered random point collection and runs it through the same direct adapter
path used by the viewer:

```bash
python scripts/adstat_demo.py --output-dir /tmp/probeflow_adstat_demo
```

If AdStat is checked out locally rather than installed as a package, put that
source tree on `PYTHONPATH` first, for example:

```bash
PYTHONPATH=/path/to/AdStat/src python scripts/adstat_demo.py
```

The script writes:

- `synthetic_points.csv` - the ProbeFlow-shaped point collection in nm and px.
- `adstat_result_view_spec.json` - the AdStat `ResultViewSpec` that ProbeFlow's
  Qt-native renderer consumes.
- `synthetic_points_preview.png` - a quick plot of the generated point pattern.

This demo teaches the data path and expected result panels. It does not replace
a GUI workflow: in the viewer, users still generate or curate a point
collection, choose an active ROI/mask if needed, and open
**Measurements -> Features -> Particle Statistics...**.

## Generated Teaching Modes

Two Particle Statistics modes use AdStat's synthetic sandbox backend rather than
the current scan's ROIs or detected features:

- **Learn with tutorial** — a guided walkthrough that stages generated patterns,
  models, and statistics step by step.
- **Model simulations** — free-play exploration of generated patterns, where the
  user picks the pattern, null model, particle count, seed, and simulation count
  directly and reruns at will.

Both pages are persistently labelled `TEST MODE - GENERATED DATA` and use
different point markers and colours from real-data analysis. Sandbox points are
deliberately isolated from real ProbeFlow data in v1: they do not become point
ROIs, measurements, processing provenance, or point-source records. Use
**Analyze scan points** for real scan data and either generated mode for
examples.

## Series

A multi-image collection is a list of scan records. Each record stores the scan
id, one normalised point set, calibration, the analysis ROI/mask, source
metadata, and an optional user-supplied series coordinate such as coverage or
temperature. ProbeFlow can run AdStat directly from those records and may also
export AdStat project/coverage-series JSON as provenance.
