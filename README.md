<p align="center">
  <img src="assets/logo.png" alt="ProbeFlow logo" width="100%"/>
</p>

<h1 align="center">ProbeFlow</h1>
<p align="center"><em>A focused Createc/Nanonis SPM browser, converter, and analysis workflow tool for STM images and point spectroscopy.</em></p>

---

## Why It Exists

A scanning-probe lab generates data faster than it can look at it.  A week of STM imaging leaves you with hundreds of scans, a stack of spectroscopy curves tied to unknown tip positions, and no routine way to go from raw file to a paper figure without manual clicking through several tools.  The interesting science — *did the molecules organise?  did the lattice shift?  how clean is the step?* — gets buried in file-format juggling.

**ProbeFlow collapses that pipeline.**  Point it at a folder; it reads every scan and spectrum file it recognises, renders thumbnails, lets you flatten / smooth / filter / measure in a GUI *or* from bash, runs feature detection (particles, lattices, unit cells, line profiles, grain counts), and exports to whatever format the next step of your workflow wants — publication PDF, CSV, or a plain PNG.  Every operation preserves physical units.  Every corrective operation is a one-liner you can script across a dataset.

It's not another viewer — it's the glue between your microscope, your analysis habits, and the figures you put in a paper.

## What's In It

* **Formats.** ProbeFlow reads Createc `.dat` (topography) and `.VERT` (spectroscopy), Nanonis `.sxm` (topography) and `.dat` (spectroscopy). It writes PNG, PDF, CSV, JSON, and Nanonis-compatible `.sxm`. Raw input files are never modified; all outputs go to separate files with clear provenance. PNG and JSON exports can include provenance describing the source file, selected channel, display state, and processing state.
* **Browse.** A thumbnail grid over an imaging session. Recognises Createc `.dat` scans, Nanonis `.sxm` scans, and both Createc (`.VERT`) and Nanonis (`.dat`) spectroscopy files in a single folder. Folder discovery runs through the shared indexing layer. The viewer uses shared display rendering for thumbnails, full-size images, histograms, and PNG export. Contrast can be set by percentile autoscale or by ImageJ-style manual histogram limits.
* **Process** — plane / facet flattening, row-offset alignment, bad-line interpolation, Gaussian / FFT / edge / TV denoising, grain detection, periodicity measurement.  All usable from the GUI, chainable as bash pipelines, and importable as plain Python functions (no Qt needed).
* **Analyse features** — particle / molecule segmentation, template-match counting, few-shot classification, SIFT lattice extraction, unit-cell averaging, line profiles.  Built for *discrete-object* STM work: molecular adsorbates, defect sites, coverage statistics.  Exports per-object JSON so your counts live alongside your raw scans in git or a lab notebook.
* **Point-measurement spectroscopy** — Createc `.VERT` and Nanonis `.dat` bias sweeps, time traces and Z spectroscopy read into a unit-aware model you can smooth, differentiate, overlay, waterfall, and map back onto a topography image to show *where* each spectrum came from.

## Why It Matters in the Lab

* **Fewer clicks per figure.**  "Open dat, flatten, apply colormap, add scale bar, save PNG" becomes `probeflow pipeline scan.dat --steps plane-bg:1 align-rows:median --png`.  When you're doing this for 200 scans, the difference is *hours*.
* **Consistent corrections across a dataset.**  Processing is parameterised in a way you can commit to git.  Two people analysing the same growth series apply the same flattening the same way.
* **Units aren't lost.**  Scale bars, colour-bar ticks, JSON exports — all carry metres / amperes / volts throughout, so a grain area in nm² today can be re-verified in a year.
* **Format is no longer a gatekeeper.**  A `.dat` straight off Createc is as analysable as a polished `.sxm`.  Conversion happens only when another tool (or a journal) asks for it — and it's one CLI call away.
* **Reproducibility by default.**  Every CLI invocation is a command you can paste into a methods section.  Every Python function has no Qt dependency, so lab notebooks run without PySide6 installed.

> **Status: beta.** The on-disk formats, the CLI surface, and the Python API are still subject to change between commits. Pin a commit hash if you depend on the current shape.

## Current backend status

ProbeFlow's current development focus is the boring but important backend layer: reliable Createc/Nanonis loading, validated scan objects, consistent display rendering, and traceable exports.

Recent backend work includes:

- content-sniffed loading for Createc `.dat`, Createc `.VERT`, Nanonis `.sxm`, and Nanonis spectroscopy `.dat` files;
- validation of loaded scan objects before GUI/export use;
- Createc first-column artifact handling in the reader path;
- lightweight folder indexing via `ProbeFlowItem` / `index_folder()`;
- shared display rendering for thumbnails, viewer images, histograms, and PNG export;
- ImageJ-style manual display limits from the interactive histogram, alongside percentile autoscale;
- canonical `ProcessingState` objects for GUI/export consistency;
- export provenance for PNG/JSON outputs, including source file, channel, display state, processing state, and ProbeFlow version.

Raw input files are not modified.

---

## Installation

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e .
```

Python 3.11+ is required. The install pulls in `numpy`, `scipy`, `pillow`, `matplotlib`, and `PySide6`.

**Optional extras** for feature-detection:

```bash
pip install probeflow[features]   # particles / lattice / count / classify  (cv2 + sklearn)
pip install probeflow[dev]        # pytest
```

---

## Image loading API

ProbeFlow exposes two levels for loading scan images:

```python
from probeflow import load_scan, read_scan_metadata

# Fully load image data (planes, physical units, processing-ready)
scan = load_scan("path/to/scan.dat")       # Createc .dat
scan = load_scan("path/to/scan.sxm")       # Nanonis .sxm

# Lightweight metadata only (no image arrays)
meta = read_scan_metadata("path/to/scan.dat")
print(meta.source_format)   # "createc_dat" or "nanonis_sxm"
print(meta.shape)            # (Ny, Nx) after any format corrections
print(meta.bias)             # V
print(meta.setpoint)         # A
print(meta.scan_range)       # (width_m, height_m)
```

`read_scan_metadata` is designed so that folder browsing can later retrieve
scan headers without loading pixel data, but both functions share the same
public `ScanMetadata` type today.

---

## Quick Start

### Launch the GUI

```bash
probeflow gui
```

### One-shot from the shell

```bash
# Look up what's in a scan (works for .dat and .sxm)
probeflow info some_scan.dat

# Flatten a scan straight off the microscope and export a publication-ready PDF
probeflow pipeline some_scan.dat \
    --steps align-rows:median plane-bg:1 smooth:1.2 \
    -o clean.sxm
probeflow convert clean.sxm figure.pdf --colormap inferno

# Or chain correction + image export in one go
probeflow pipeline some_scan.dat \
    --steps align-rows:median plane-bg:1 \
    --png --colormap viridis -o scan_clean.png

# Prepare a PNG handoff for AISurf-style tools, with provenance sidecar
probeflow prepare-png some_scan.dat aisurf_input.png \
    --steps align-rows:median plane-bg:1 --colormap gray

# Suggest a contrast window automatically
probeflow autoclip some_scan.sxm --json

# Convert between vendor formats when another tool asks for it
probeflow convert some_scan.dat some_scan.sxm      # Createc → Nanonis
probeflow convert some_scan.sxm some_scan.pdf --colormap inferno
```

---

## Commands

The top-level command is `probeflow`. Every subcommand accepts `--help`.

### Conversion

| Command   | Purpose                                                            |
|-----------|--------------------------------------------------------------------|
| `convert` | Suffix-driven any-in / any-out conversion (recommended)            |
| `dat2sxm` | Createc `.dat` → Nanonis `.sxm` (use `--` to pass through flags)   |
| `dat2png` | Createc `.dat` → preview PNGs                                      |
| `sxm2png` | Nanonis `.sxm` → colorised PNG with optional scale bar             |
| `prepare-png` | PNG handoff for downstream tools, with provenance sidecar      |

Legacy shortcuts `dat-sxm` and `dat-png` remain available for backward compatibility.

#### `probeflow convert`

Reads any supported scan format and writes any supported output, picking both ends from file content (for input) and suffixes (for output).

**Read:** `.sxm` · `.dat`
**Write:** `.sxm` · `.png` · `.pdf` · `.csv`

```bash
probeflow convert scan.dat scan.pdf --colormap inferno    # Createc .dat → publication-ready PDF
probeflow convert scan.dat scan.sxm                        # Createc → Nanonis
probeflow convert scan.sxm scan.png --colormap viridis     # Nanonis → PNG
probeflow convert scan.dat line0.csv --plane 0             # single plane → CSV grid
```

### Processing (scan in → `.sxm` or `.png` out)

Each of these reads a scan (`.sxm` or `.dat` — auto-detected), applies a single operation to the selected plane (0 = Z forward by default), and writes a new `.sxm` — or a PNG with `--png`.

| Command            | Operation                                                         |
|--------------------|-------------------------------------------------------------------|
| `plane-bg`         | Subtract polynomial plane background (`--order 1` or `2`)         |
| `align-rows`       | Per-row offset / slope removal (`--method median|mean|linear`)    |
| `remove-bad-lines` | MAD-based outlier-row interpolation                               |
| `facet-level`      | Plane fit using only flat-terrace pixels — good for stepped surfaces |
| `smooth`           | Isotropic Gaussian smoothing (`--sigma` in pixels)                |
| `edge`             | Laplacian / LoG / DoG edge detection                              |
| `fft`              | 2-D FFT low-pass or high-pass filter                              |
| `flip-h` / `flip-v` | Lossless horizontal / vertical flip                              |
| `rotate-90` / `rotate-180` / `rotate-270` | Lossless 90° rotations                       |
| `rotate`           | Arbitrary-angle rotation (bilinear, canvas-expanding)             |

Common options across the processing commands:

```
--plane N              # 0=Z-fwd, 1=Z-bwd, 2=I-fwd, 3=I-bwd  (default 0)
--png                  # write a colorised PNG instead of a new .sxm
--colormap NAME        # any matplotlib colormap name
--clip-low  P          # lower percentile for PNG contrast (default 1.0)
--clip-high P          # upper percentile for PNG contrast (default 99.0)
--no-scalebar
--scalebar-unit nm|Å|pm
--scalebar-pos  bottom-right|bottom-left
```

### Analysis / inspection

| Command        | Purpose                                                               |
|----------------|-----------------------------------------------------------------------|
| `grains`       | Detect islands or depressions and print per-grain area / centroid     |
| `autoclip`     | Suggest GMM-based clip percentiles for display                        |
| `periodicity`  | Find dominant spatial periods via the power spectrum                  |
| `info`         | Print header metadata (`--json` for machine-readable output)          |
| `profile`      | Sample z along a line (CSV / JSON / PNG; nm and px endpoints, swath averaging) |
| `histogram`    | Pixel-value histogram of a selected plane (CSV / PNG; ROI-aware)      |
| `fft-spectrum` | 2-D FFT magnitude spectrum of a plane or ROI (PNG; with peak summary) |
| `diag-z`       | Diagnostic dump for Z-scale calibration (Createc gain factor model)   |

### Feature detection (requires `probeflow[features]`)

| Command      | Purpose                                                                                |
|--------------|----------------------------------------------------------------------------------------|
| `particles`  | Segment bright (or `--invert` for dark) molecules / islands; areas in nm² + centroids  |
| `count`      | Count repeated motifs by NCC template matching (AiSurf `atom_counting`)                |
| `classify`   | Few-shot classify particles against labelled samples (raw or PCA encoders, no CLIP)    |
| `lattice`    | SIFT-based primitive lattice vectors `(a, b, γ)`; optional 4-panel PDF report          |
| `unit-cell`  | Run `lattice`, then average all interior unit cells into a single canonical motif      |
| `tv-denoise` | Edge-preserving Chambolle–Pock TV (`huber_rof` or `tv_l1`); axis-selective for scratches |

```bash
# Count molecules in an .sxm scan; export per-particle JSON.
probeflow particles scan.sxm --threshold otsu --min-area 0.5 -o particles.json

# Count atoms by template matching; --template can be a PNG crop or another scan.
probeflow count scan.sxm --template motif.png --min-corr 0.55 -o atoms.json

# SIFT lattice extraction with a 4-panel PDF report.
probeflow lattice scan.sxm -o lattice.pdf

# Average all unit cells into one clean motif.
probeflow unit-cell scan.sxm -o avg_cell.png --oversample 1.5

# Total-variation denoising (axis-selective gradient kills horizontal scratches).
probeflow tv-denoise scan.sxm --method huber_rof --lam 0.05 --nabla-comp y -o clean.sxm

# Line profile across an atomic step, with a 5-pixel swath average, in nm units.
probeflow profile scan.sxm --p0-nm 0 5 --p1-nm 30 5 --width 5 -o step.png
```

### Chain several steps: `pipeline`

`pipeline` runs any ordered sequence of processing atoms in a single invocation:

```bash
probeflow pipeline scan.sxm \
    --steps remove-bad-lines align-rows:median plane-bg:1 smooth:1.2 \
    -o scan_processed.sxm
```

Step syntax is `name[:param1,param2,…]`:

| Step               | Parameters                                      | Example                  |
|--------------------|-------------------------------------------------|--------------------------|
| `remove-bad-lines` | `mad_threshold` (default `5.0`)                 | `remove-bad-lines:4.0`   |
| `align-rows`       | `median` / `mean` / `linear`                    | `align-rows:linear`      |
| `plane-bg`         | `order` (`1` or `2`)                            | `plane-bg:2`             |
| `facet-level`      | `threshold_deg`                                 | `facet-level:2.0`        |
| `smooth`           | `sigma_px`                                      | `smooth:1.5`             |
| `edge`             | `method,sigma,sigma2`                           | `edge:log,1.0`           |
| `fft`              | `mode,cutoff,window`                            | `fft:low_pass,0.08`      |

Add `--png` to the `pipeline` command to skip `.sxm` output and write a colorised PNG directly.

### Prepared handoff PNGs

`prepare-png` is for sending a cleaned scan plane to tools that consume image files, such as AISurf-style PNG workflows. It writes the PNG plus `<name>.provenance.json`, recording the raw source, channel, scan size, units, processing state, display/export state, and a stable processing hash.

```bash
probeflow prepare-png scan.dat aisurf_input.png \
    --steps align-rows:median plane-bg:1 --colormap gray
```

If no background or line-leveling step is recorded, the provenance sidecar includes a warning so the handoff is visibly unprepared rather than quietly becoming an untracked screenshot.

### Spectroscopy (`.VERT` and Nanonis `.dat`)

ProbeFlow reads Createc `.VERT` and Nanonis `.dat` spectroscopy files and auto-detects the sweep type from the data:

| Sweep type   | X-axis         | Typical use                                    |
|--------------|----------------|------------------------------------------------|
| Bias sweep   | Bias (V)       | I(V) / Z(V) tunnelling spectroscopy            |
| Time trace   | Time (s)       | I(t) / Z(t) at fixed bias — telegraph noise    |
| Z spectroscopy | Z (m)        | Z-dependent spectroscopy (Nanonis `.dat`)       |

```bash
# Print header metadata from a .VERT file
probeflow spec-info spectrum.VERT

# Quick plot of the Z channel vs. bias or time
probeflow spec-plot spectrum.VERT --channel Z -o spectrum.png

# Overlay multiple spectra with a waterfall offset; also show the mean
probeflow spec-overlay *.VERT --channel Z --offset 1e-10 --average -o stack.png

# Mark tip positions of a set of spectra on a topography image
probeflow spec-positions scan.sxm *.VERT -o positions.png
```

Available channels per file: `I` (current, A), `Z` (tip-sample distance, m), `V` (bias, V).

**Programmatic API:**

```python
from probeflow.io.spectroscopy import read_spec_file
from probeflow.processing.spectroscopy import smooth_spectrum, numeric_derivative, crop
from probeflow.analysis.spec_plot import plot_spectrum, plot_spectra

spec = read_spec_file("spectrum.VERT")
print(spec.metadata["sweep_type"])  # "bias_sweep" or "time_trace"
print(spec.position)                # (x_m, y_m) tip position in metres

# Smooth the Z channel and compute dZ/dV
z_smooth = smooth_spectrum(spec.channels["Z"], method="savgol", window_length=21)
dzdv = numeric_derivative(spec.x_array, z_smooth)

# Crop to a sub-range and plot
x_crop, z_crop = crop(spec.x_array, z_smooth, x_min=-0.3, x_max=-0.05)

# Overlay multiple spectra
specs = [read_spec_file(p) for p in sorted(Path(".").glob("*.VERT"))]
ax = plot_spectra(specs, channel="Z", offset=5e-10)
```

### GUI

```bash
probeflow gui
```

Six tabs:

* **Browse** — point at a folder; the grid auto-detects supported Createc/Nanonis scan and spectroscopy files and renders thumbnails. An *All / Images / Spectra* toggle filters the visible cards. The full-size viewer has an interactive histogram with Auto percentile contrast and manual red/green display-limit bars, similar in spirit to ImageJ brightness/contrast controls. The viewer supports rectangle / ellipse / polygon / freehand ROI tools, geometric transforms (flip, rotate), live line profiles, and a side-by-side FFT viewer dialog. PNG export uses the same display limits as the viewer and writes provenance when available.
* **Convert** — folder-in / folder-out batch dat→sxm and dat→png with PNG / SXM checkboxes and clip-percentile controls.
* **FeatureCounting** — load the currently-selected Browse scan, choose a mode (*Particles* / *Template* / *Lattice*), tune parameters, hit *Run*. Results overlay on the canvas (contours, detection markers, primitive vectors + unit cell) and populate a sortable table. *Export JSON…* writes results with full scan provenance via `probeflow.io.writers.json`. Heavy analyses run on a background thread so the UI stays responsive.
* **TV-denoise** — dedicated panel for total-variation denoising (Chambolle–Pock `huber_rof` / `tv_l1`) with axis-selective gradient for line-noise scratches.
* **Dev** — embedded developer terminal (PySide6 `QProcess`) for quick shell access without leaving the GUI; auto-redirects PTY tools (`vim`, `ipython`, `claude`, …) to an external terminal.
* **Defs** — algorithm reference panel summarising the processing operations and their parameters.

Preferences (folders, theme, clip values, font size) are saved to `~/.probeflow_config.json`.

---

## Bash-driven workflows

### Batch-process a folder

```bash
for f in data/sxm/*.sxm; do
    probeflow pipeline "$f" \
        --steps align-rows:median plane-bg:1 \
        -o "processed/${f##*/}"
done
```

### Raw-to-figure: flatten a session and export for publication

Works on raw `.dat` or `.sxm` — no pre-conversion step required:

```bash
for s in data/session/*.{dat,sxm}; do
    [ -e "$s" ] || continue
    probeflow pipeline "$s" \
        --steps remove-bad-lines align-rows:median plane-bg:1 smooth:1.0 \
        --png --colormap inferno --scalebar-unit nm \
        -o "figures/$(basename "${s%.*}").png"
done
```

If a journal or collaborator specifically asks for `.sxm`:

```bash
for d in raw/*.dat; do
    probeflow convert "$d" "sxm/$(basename "${d%.dat}").sxm"
done
```

### Auto-suggest contrast across a dataset (JSON out)

```bash
for s in sxm/*.sxm; do
    echo -n "$s  "
    probeflow autoclip "$s" --json
done
```

### Machine-readable lattice periods

```bash
probeflow periodicity scan.sxm --n-peaks 3 --json \
    | jq '.[] | {period_nm: (.period_m*1e9), angle_deg}'
```

---

## Programmatic use

The package is importable without pulling in the GUI.  The primary entry point is `load_scan`, which returns a format-agnostic `Scan` object:

```python
from probeflow import load_scan, processing

# Works for .sxm and .dat — content-sniffed, not suffix-driven.
scan = load_scan("raw_scan.dat")

scan.planes[0] = processing.align_rows(scan.planes[0], method="median")
scan.planes[0] = processing.subtract_background(scan.planes[0], order=1)

# Export by file suffix — sxm / png / pdf / csv.
scan.save("figure.pdf", colormap="inferno")
scan.save("archive.sxm")
```

Lower-level primitives for when you need the full vendor header or raw byte layout:

```python
from probeflow.io.sxm_io import parse_sxm_header, read_all_sxm_planes

hdr, planes = read_all_sxm_planes("scan.sxm")
```

Spectroscopy is a different shape of data, so it has its own module:

```python
from probeflow.io.spectroscopy import read_spec_file
from probeflow.processing.spectroscopy import smooth_spectrum, numeric_derivative

spec = read_spec_file("spectrum.VERT")
z_smooth = smooth_spectrum(spec.channels["Z"], method="savgol")
dzdv = numeric_derivative(spec.x_array, z_smooth)
```

---

## Repository layout

```
probeflow/              # installable package
|-- __init__.py
|-- core/               # Scan model, loaders, metadata, indexing, ROI, validation
|-- io/                 # file sniffing, readers, writers, .sxm layout, converters
|-- processing/         # GUI-free image/display/spectroscopy processing + ProcessingState
|-- analysis/           # particles, lattice, spectroscopy plotting, xmgrace export
|-- provenance/         # export provenance now; scan-graph provenance later
|-- gui/                # PySide6 GUI package
|-- cli/                # unified "probeflow" command package
`-- plugins/            # future plugin API, registry, manifests, adapters

src/file_cushions/      # binary layout captured from a reference .sxm file
data/                   # sample input / output for manual runs + tests
tests/                  # pytest suite (conversion, processing, .sxm round-trip,
                        #               spectroscopy reader + processing,
                        #               ROI, geometric ops, layout compatibility)
assets/                 # logo artwork
```

The `src/file_cushions/` directory holds the byte-level layout used to reconstruct `.sxm` files (header padding, `:SCANIT_END:` marker position, tail bytes, fixed data offset). These were reverse-engineered once from a reference Nanonis file and should be regenerated only if a future Nanonis version shifts the binary layout.

### Current state vs roadmap

The directory tree above reflects the *intended* decomposition. In practice, the bulk of the GUI implementation still lives in `probeflow/gui/_legacy.py` (~6,200 LoC) and the CLI in `probeflow/cli/_legacy.py` (~2,200 LoC); the other modules in those packages are mostly thin re-exports today. The provenance graph (`probeflow/provenance/graph.py`) and plugin registry (`probeflow/plugins/`) are scaffolded but not yet implemented — see [`roadmap_architecture_proposal.md`](roadmap_architecture_proposal.md) for the architectural target. Decomposition is happening opportunistically; see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Tests

```bash
pip install -e '.[dev]'
pytest
```

Covers:

* Conversion (`.dat` → `.sxm` and `.dat` → PNG) against the bundled sample scans.
* Every public function in `probeflow.processing` (incl. `tv_denoise`, `line_profile`).
* `.sxm` header parsing, plane reading, and write-then-read round-trip.
* `.VERT` and Nanonis `.dat` spectroscopy header parsing, unit conversion, sweep-type detection, and error handling.
* Every public function in `probeflow.processing.spectroscopy`.
* Readers / writers (`sxm`, `dat`, `pdf`, `csv`).
* Feature-detection: `segment_particles`, `count_features`, `classify_particles`.
* SIFT lattice extraction and `average_unit_cell`.
* Line-profile sampling with sub-pixel interpolation and swath averaging.
* Content-sniffing file-type dispatcher (`probeflow.io.file_type`).
* Loading/indexing/display-state/export-provenance contracts for the Createc/Nanonis backend path.

---

## Processing Algorithms — Technical Reference

This section describes what each numerical processing step does and the scientific choices behind it.  Algorithms were validated against reference ImageJ STM plugins by Michael Schmid.

### remove_bad_lines

Detects scan lines whose values are statistical outliers and replaces them by interpolation from neighbouring rows.

| Parameter | Default | Description |
|---|---|---|
| `method` | `"mad"` | Detection strategy: `"mad"` (row-level Median Absolute Deviation) or `"step"` (column-level step detection) |
| `threshold_mad` | `5.0` | MAD multiplier; higher = more tolerant |

**`"mad"` method** — computes the median row offset (difference between each row's median and the global median). Rows whose offset exceeds `threshold_mad × MAD` are flagged as fully bad and replaced by linear interpolation from the nearest good rows above and below.  Robust to global tilt and step edges.

**`"step"` method** — scans each column independently for large vertical jumps. A column pixel is marked bad if it sits inside an upward step (entry) that has not yet been closed by a downward step of equal magnitude. The threshold is derived from the MAD of all inter-row differences across the image, scaled by `threshold_mad`. This can catch *partial* bad lines (stripe artifacts that affect only part of a row) that the MAD method misses, matching the ImageJ approach.

### align_rows

Removes the inter-row DC offset that accumulates during a raster scan.  Each row's median (or mean) is subtracted independently so that neighbouring rows align.  Does not affect within-row variation or surface slope.

| Parameter | Default | Description |
|---|---|---|
| `method` | `"median"` | `"median"` (robust) or `"mean"` |

### plane_bg / subtract_background

Fits and subtracts a polynomial background plane (order 1 = flat tilt, order 2 = parabolic bow).  An optional ROI geometry (`fit_geometry`) restricts the fit to a user-drawn region, so that islands or step edges are excluded from the fit.

| Parameter | Default | Description |
|---|---|---|
| `order` | `1` | Polynomial order (1–3) |
| `step_tolerance` | `False` | Segment rows at large steps before fitting |
| `fit_geometry` | `None` | ROI dict limiting pixels used for the fit |

### stm_line_bg

Line-by-line background subtraction tuned for STM images with step edges.  Each row is fitted and subtracted separately in a step-tolerant way, preventing step-edge height from biasing the background fit.  Equivalent to the ImageJ `STM_Background` plugin approach.

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"step_tolerant"` | `"step_tolerant"` or `"linear"` |

### facet_level

Detects atomically flat terraces using an angular-threshold criterion (pixels within `threshold_deg` degrees of horizontal are considered terrace pixels) and levels each terrace to a common height.  Useful for multi-terrace STM images where a single plane fit would tilt the result.

| Parameter | Default | Description |
|---|---|---|
| `threshold_deg` | `3.0` | Angular tolerance for terrace classification |

### smooth

Gaussian low-pass filter applied in real space.  Reduces high-frequency noise while preserving large-scale features.  `sigma_px` controls the blur radius in pixels.

| Parameter | Default | Description |
|---|---|---|
| `sigma_px` | `1.0` | Gaussian σ in pixels |

### gaussian_high_pass

Subtracts a Gaussian-blurred version of the image from the original (equivalent to a high-pass filter in frequency space).  Enhances short-range features such as atomic corrugation while removing long-range background.

| Parameter | Default | Description |
|---|---|---|
| `sigma_px` | `8.0` | σ of the subtracted Gaussian |

### fft_soft_border

Fourier-domain low-pass or high-pass filter with a soft (cosine-tapered) border applied before the FFT to suppress ringing.  The border taper blends the image edges to their mean, so that periodic boundary conditions imposed by the FFT do not introduce streak artefacts.  Matches the ImageJ `FFT_Soft_Border` plugin.

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"low_pass"` | `"low_pass"` or `"high_pass"` |
| `cutoff` | `0.10` | Normalised frequency cutoff (0–0.5) |
| `border_frac` | `0.12` | Fraction of image width used for the taper |

### fourier_filter

General Fourier-domain filter without the soft-border taper.  Suitable when edge artefacts are not a concern (e.g. when the image has already been windowed or the scan area is large relative to the feature of interest).

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"low_pass"` | `"low_pass"` or `"high_pass"` |
| `cutoff` | `0.10` | Normalised frequency cutoff |
| `window` | `"hanning"` | Window function applied before FFT |

### periodic_notch_filter

Suppresses periodic noise by zeroing a disc of radius `radius_px` around each specified peak in the FFT magnitude spectrum.  The `peaks` list contains `[qx, qy]` pairs in normalised frequency coordinates.  Conjugate peaks are automatically included.

| Parameter | Default | Description |
|---|---|---|
| `peaks` | `[]` | List of `[qx, qy]` frequency-space peak positions |
| `radius_px` | `3.0` | Notch radius in FFT pixels |

### patch_interpolate

Fills a masked region (e.g. a defect or scan artefact) by interpolating from the surrounding data.

| Parameter | Default | Description |
|---|---|---|
| `method` | `"line_fit"` | `"line_fit"` (recommended) or `"laplace"` |
| `rim_px` | `20` | Width of rim used for slope estimation in `"line_fit"` |
| `iterations` | `200` | Iterations for `"laplace"` relaxation |

**`"line_fit"` (default)** — for each masked row, fits a straight line through the unmasked rim pixels immediately to the left and right of the masked segment and extrapolates across it.  This preserves the local surface slope across the patch, which is essential for STM terraces: a flat-terrace repair that introduces a tilt artefact would create a phantom step.  Rows that are entirely masked are filled by interpolation from neighbouring rows.  This matches the ImageJ `Patch_Interpolation` algorithm.

**`"laplace"`** — iterative harmonic (Laplace) relaxation inside the mask.  Isotropic and smooth, but does *not* preserve scan-line slope.  Retained for cases where the slope-preserving fit is unsuitable (e.g. non-STM data with genuinely isotropic structure).

### linear_undistort

Corrects linear geometric distortions introduced by piezo creep or cross-coupling: a shear along x and a scale along y.

| Parameter | Default | Description |
|---|---|---|
| `shear_x` | `0.0` | Shear coefficient (pixels of x shift per row) |
| `scale_y` | `1.0` | Vertical scale factor |

### edge_detect

Applies a derivative-based edge-detection kernel.  Useful for visualising atomic step edges or grain boundaries.

| Parameter | Default | Description |
|---|---|---|
| `method` | `"laplacian"` | `"laplacian"`, `"sobel"`, or `"dog"` (difference of Gaussians) |
| `sigma` | `1.0` | Primary Gaussian σ |
| `sigma2` | `2.0` | Secondary σ for `"dog"` |

---

## Notes

* The `.sxm` timestamp parser expects Createc filenames of the form `AyyMMdd.HHmmss.dat`.
* Failed batch conversions are logged to `errors.json` in the output directory.
* Sine-mode Createc scans are currently rejected with a clear error (not silently corrupted).

---

## License

ProbeFlow is released under the [MIT License](LICENSE). You are free to use,
modify, and redistribute it for any purpose, including commercially, as long
as the copyright notice is preserved.

## Acknowledgements

**ProbeFlow** is developed at **[SPMQT-Lab](https://github.com/SPMQT-Lab)**, under the supervision of **Dr. Peter Jacobson** at **The University of Queensland**.

The core Createc-decoding algorithms were originally written by **[Rohan Platts](https://github.com/rohanplatts)**. ProbeFlow is a refactored, extended, and GUI-enabled evolution of that work — his contributions are the foundation of the conversion pipeline.

> *"Standing on the shoulders of giants."*
