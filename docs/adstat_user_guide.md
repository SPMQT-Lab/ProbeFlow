# Particle Statistics User Guide

ProbeFlow is the normal entry point for particle/point-pattern statistics. Open
**Measurements -> Features -> Particle Statistics...** to choose between real
scan analysis, a guided tutorial, and free-play model simulations. The
calculations are powered by the AdStat engine.

> **Maturity note — please read.** Particle Statistics is the newest and
> **least user-tested** part of ProbeFlow. It has unit tests and a worked
> tutorial, but it has had far less real-world use than the imaging, ROI, and
> Feature Finder tools, and it **may contain mistakes** — in the statistics, the
> automatic scale choices, or the interpretation language. Treat its output as
> *exploratory*: sanity-check verdicts against your own judgement and, for
> anything you intend to publish, against an independent point-pattern method you
> trust. If a result looks wrong, it may well be — please report it.

## What Particle Statistics Does

Particle Statistics compares a collection of points against spatial null models.
In plain language, it asks whether the points look consistent with random
placement, or whether they show signs of clustering, separation, or association
with another measured feature.

It does not prove a physical mechanism. The result depends on which points you
analyse, which image region you allow, and which null model you choose.

## How It Works

The core idea is a **null model** plus a **simulation envelope**:

1. You give it a set of point positions and an analysis region.
2. It measures one or more spatial statistics on your points.
3. It generates many random point patterns from the chosen null model (same
   region, same number of points) and measures the same statistic on each. The
   spread of those simulations is the **envelope** — the range expected by
   chance.
4. If your observed curve falls outside the envelope, the pattern is reported as
   *inconsistent with* the null model; if it stays inside, it is *consistent
   with* it.

**Null models** available for real data:

- **Homogeneous Poisson** — completely random placement (the usual baseline).
- **Hard-core random** — random but with a minimum spacing (points cannot sit
  closer than a set distance), a baseline for excluded-volume effects.
- **Measured-feature Poisson** — random but biased toward an *independently
  measured* feature layer (e.g. step edges), to test association. The feature
  layer must be a different measurement from the particles being tested —
  reusing the particles as their own feature is circular.

**Core statistics** (different lenses on the same points, always shown):

- **Pair correlation g(r)** — relative density of neighbours at distance *r*. A
  short-range bump means clustering; a dip near zero means avoidance/spacing.
- **Nearest-neighbour distribution** — how far each point is from its closest
  neighbour; the clearest first look at spacing vs clumping.
- **Ripley's L** — cumulative neighbour counts vs distance; sensitive to
  clustering or regularity across a range of scales.
- **Cluster sizes** — counts of connected groups within a linking distance.

**Local-order checks (opt-in).** Bond-orientational order **ψ4 / ψ6** and the
angular pair map **g(r, θ)** answer a *different* question — is there square or
triangular *lattice* order? — so they are **off by default** and not shown in a
plain randomness analysis. Tick **"Include local-order checks"** (or run the
ordered tutorial examples) to compute them. They are validated to behave
correctly — a triangular lattice rejects on ψ6, a square lattice on ψ4, and
random points stay consistent (see `tests/test_adstat_validation.py`) — but they
depend on a neighbour-distance cutoff and have no edge correction, so for sparse
patterns or particles near the region boundary treat them as suggestive of local
order, not proof of a crystal.

ProbeFlow chooses the distance scales (bin widths, maximum radii, hard-core and
cluster radii) automatically from the region size and point density when you do
not set them. These automatic choices are *teaching-quality defaults*, not tuned
parameters — see the maturity note above.

**Reading a verdict.** "Consistent with random" means the null model was *not
rejected* — it is not positive proof that nothing is going on (a small or noisy
sample simply may not have the power to detect an effect). "Inconsistent with
random" means the statistic departed from the envelope — evidence of structure,
but not proof of any particular physical mechanism. Pooling several independent
images of the same condition is the practical way to strengthen a conclusion.

## Analyze Scan Points

Use **Analyze scan points** for real ProbeFlow data.

1. Generate or curate a point collection:
   Feature Finder maxima/minima, feature maxima, or point ROIs are available in
   the current viewer workflow.
2. Open **Particle Statistics...** and stay on **Analyze scan points**.
3. Choose one point source as the tested population.
4. Choose the analysis region: active area ROI, active mask, or full image.
5. Choose a model and run the comparison.

For session feature-set workflows, click **Send to Particle Statistics** from
Feature Finder. The set is saved with its image calibration. Tick one saved set
to analyse that image, or tick multiple saved sets from independent scans of the
same condition to pool them.

The real-data UI exposes homogeneous Poisson, hard-core random, and
measured-feature Poisson models, plus simulation count and random seed.
Measured-feature Poisson uses one tested session set and a different,
independently measured Feature layer set.

## Getting Points In

Particle Statistics shares one feature-set pool across the whole session, so
points from any of these sources can be ticked together and pooled:

1. **Feature Finder → Send to Particle Statistics** — local maxima/minima from
   the open image.
2. **Feature Counting → Send to Particle Statistics** — segmented particle
   centroids (Particles mode) or template detections (Template mode) from the
   Feature Counting window.
3. **Point sources in the open image** — detected feature maxima and point ROIs
   appear directly in the *Analyze scan points* dropdown (point ROIs include any
   loaded from a `.rois.json` sidecar).
4. **Load points from disk…** — import an external position table. Accepted
   formats: CSV position tables (with or without a leading particle-number
   column; units inferred from `x_px` / `x_nm` / `x_m` / `x_phys` headers or
   chosen on import), ProbeFlow's own Feature Finder / measurements CSV, and
   ProbeFlow JSON (Feature Counting exports and saved feature-set files). For a
   file with no embedded calibration, a small dialog (prefilled from the file)
   asks for the position units and physical field size before the points become a
   feature set.

Use **Save feature sets…** to write the current pool to a JSON file; it can be
re-imported later with **Load points from disk…**.

## Exporting Results

After running a comparison, the top **Export** menu writes the results in simple
formats so you can reproduce the plots in another program:

- **Export curves + verdicts (CSV folder)…** — one CSV per statistic. Each curve
  file has a distance column plus the `observed` line and the model envelope
  (`model_low` / `model_central` / `model_high`), so g(r), the nearest-neighbour
  distribution, Ripley's L, cluster sizes, etc. can be re-plotted directly. A
  `…_verdicts.csv` holds the per-model/per-statistic verdict table. (Heatmap and
  real-space panels are not written as CSV; they are kept in the JSON export.)
- **Export full result (JSON)…** — the entire result (all panels, curves, and
  verdicts) in one JSON file, for archiving or scripted post-processing.

The input points themselves are exported separately via **Save feature sets…**
(above) or the source tools' own CSV/JSON exports.

## Learn With Tutorial

Use **Learn with tutorial** (the **Tutorial** card on the workflow start page)
for a guided walkthrough that builds up particle-pattern analysis one idea at a
time, using generated example data rather than the current image. It steps
through random placement, sample size, pooling, clustering, hard-core spacing,
and feature association, then hands you off to the real scan-points workflow.

## Model Simulations

Use **Model simulations** to experiment freely with generated patterns without
the tutorial's guidance. Choose the synthetic pattern, null model, particle
count, seed, and simulation count, then run the comparison or draw a fresh
pattern. This is the place to build intuition by changing one knob at a time.

Both generated modes are labelled `TEST MODE - GENERATED DATA` and use different
point markers and colours so they cannot be mistaken for real data. Generated
points stay isolated: they do not become point ROIs, measurements, processing
provenance, or active ProbeFlow point sources.

## Appropriate Data

Good tested populations are point-like features whose coordinates have a clear
meaning: atoms, adsorbates, defects, particle centroids, template detections, or
manually curated point ROIs.

Do not mix unrelated species unless the scientific question is explicitly about
the merged set. Independent feature layers, such as step traces or external
landmarks, must be measured separately from the particles being tested.

## Current Limitations

- **This is the least battle-tested area of ProbeFlow** (see the maturity note at
  the top): the statistics and especially the automatic scale defaults may still
  have rough edges. Verify important results independently.
- Feature sets live in one shared session pool; **Save feature sets…** /
  **Load points from disk…** persist and restore them as JSON, but they are not
  yet tied into a durable per-project record.
- Imported files with no embedded calibration require you to supply the field
  size; the image (pixel) dimensions are synthetic and only affect the
  pixel-resolution note, not the statistics.
- Series/project export workflows are still pending.

See [AdStat integration](adstat_integration.md) for the developer-facing API
contract between ProbeFlow and AdStat.
