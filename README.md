<p align="center">
  <img src="assets/logo.gif" alt="ProbeFlow logo" width="100%"/>
</p>

<h1 align="center">ProbeFlow</h1>
<p align="center"><em>An imaging-lab workflow tool for scanning-probe microscopy.</em></p>

---

## Why It Exists

A scanning-probe lab generates data faster than it can look at it.  A week of STM imaging leaves you with hundreds of scans scattered across four vendor formats, a stack of spectroscopy curves tied to unknown tip positions, a handful of notebooks half-analysed in Gwyddion, and no routine way to go from raw file to a paper figure without manual clicking through five tools.  The interesting science — *did the molecules organise?  did the lattice shift?  how clean is the step?* — gets buried in file-format juggling.

**ProbeFlow collapses that pipeline.**  Point it at a folder; it reads every scan and spectrum file it recognises, renders thumbnails, lets you flatten / smooth / filter / measure in a GUI *or* from bash, runs feature detection (particles, lattices, unit cells, line profiles, grain counts), and exports to whatever format the next step of your workflow wants — publication PDF, Gwyddion `.gwy`, TIFF for ImageJ, or CSV for a plot.  Every operation preserves physical units.  Every corrective operation is a one-liner you can script across a dataset.

It's not another viewer — it's the glue between your microscope, your analysis habits, and the figures you put in a paper.

## What's In It

* **Browse** — a thumbnail grid over an entire imaging session.  Recognises **Createc `.dat`**, **Nanonis `.sxm`**, **Gwyddion `.gwy`**, **RHK `.sm4`**, **Omicron Matrix `.mtrx`**, and **Createc `.VERT`** spectroscopy in a single folder.  Live clip sliders, per-card colormaps, per-scan undo, full-size viewer with interactive histogram, and side-by-side metadata — designed to make "did anything happen in this scan?" a one-second answer.
* **Process** — plane / facet flattening, row-offset alignment, bad-line interpolation, Gaussian / FFT / edge / TV denoising, grain detection, periodicity measurement.  All usable from the GUI, chainable as bash pipelines, and importable as plain Python functions (no Qt needed).
* **Analyse features** — particle / molecule segmentation, template-match counting, few-shot classification, SIFT lattice extraction, unit-cell averaging, line profiles.  Built for *discrete-object* STM work: molecular adsorbates, defect sites, coverage statistics.  Exports per-object JSON so your counts live alongside your raw scans in git or a lab notebook.
* **Read any, write what's useful** — ProbeFlow reads the five main STM vendor formats natively (with the optional `access2thematrix` / `spym` / `gwyfile` extras installed); anything else falls through to the system `gwyddion` binary if it's on `PATH`, which adds Bruker, Park, NTEGRA, Nanoscope, JPK and ~30 more.  On the output side it writes **`.sxm`**, **PNG**, **PDF**, **TIFF** (float or uint16), **GWY**, **CSV**, and **JSON** — driven purely by the suffix you pick.
* **Point-measurement spectroscopy** — `.VERT` bias sweeps, time traces and Z spectroscopy read into a unit-aware model you can smooth, differentiate, overlay, waterfall, and map back onto a topography image to show *where* each spectrum came from.

## Why It Matters in the Lab

* **Fewer clicks per figure.**  "Open dat, flatten, apply colormap, add scale bar, save PNG" becomes `probeflow pipeline scan.dat --steps plane-bg:1 align-rows:median --png`.  When you're doing this for 200 scans, the difference is *hours*.
* **Consistent corrections across a dataset.**  Processing is parameterised in a way you can commit to git.  Two people analysing the same growth series apply the same flattening the same way.
* **Units aren't lost.**  Scale bars, colour-bar ticks, JSON exports — all carry metres / amperes / volts throughout, so a grain area in nm² today can be re-verified in a year.
* **Format is no longer a gatekeeper.**  A `.dat` straight off Createc is as analysable as a polished `.sxm`.  An Omicron file from a collaborator can go straight into a pipeline.  Conversion happens only when another tool (or a journal) asks for it — and it's one CLI call away.
* **Reproducibility by default.**  Every CLI invocation is a command you can paste into a methods section.  Every Python function has no Qt dependency, so lab notebooks run without PySide6 installed.

> **Status: beta.** The on-disk formats, the CLI surface, and the Python API are still subject to change between commits. Pin a commit hash if you depend on the current shape.

---

## Installation

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e .
```

Python 3.11+ is required. The install pulls in `numpy`, `scipy`, `pillow`, `matplotlib`, and `PySide6`.

**Optional extras** for vendor-specific formats and feature-detection:

```bash
pip install probeflow[omicron]    # Omicron Matrix (.mtrx)   via access2thematrix
pip install probeflow[rhk]        # RHK             (.sm4)   via spym
pip install probeflow[gwyddion]   # Gwyddion        (.gwy)   via gwyfile
pip install probeflow[features]   # particles / lattice / count / classify  (cv2 + sklearn)
pip install probeflow[all]        # everything above
```

Without any extras, ProbeFlow still fully supports `.sxm` and `.dat` on both the read and write sides.

---

## Quick Start

### Launch the GUI

```bash
probeflow gui
```

### One-shot from the shell

```bash
# Look up what's in a scan (works for .dat, .sxm, .gwy, .sm4, .mtrx)
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

# Suggest a contrast window automatically
probeflow autoclip some_scan.sxm --json

# Convert between vendor formats when another tool asks for it
probeflow convert some_scan.dat some_scan.sxm      # Createc → Nanonis
probeflow convert some_scan.sm4 some_scan.gwy      # RHK → Gwyddion
probeflow convert some_scan.sxm some_scan.tif --tiff-mode float
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

Legacy shortcuts `dat-sxm` and `dat-png` remain available for backward compatibility.

#### `probeflow convert`

Reads any supported scan format and writes any supported output, picking both ends from file suffixes.

**Read:** `.sxm` · `.dat` · `.gwy` · `.sm4` · `.mtrx` (or `.Z_mtrx` / `.I_mtrx`).  Any other suffix is automatically routed through the system `gwyddion` binary (if installed), which adds Bruker, Park, NTEGRA, Nanoscope, JPK and the rest of Gwyddion's ~40 vendor formats.
**Write:** `.sxm` · `.png` · `.pdf` · `.tif` / `.tiff` · `.gwy` · `.csv`

```bash
probeflow convert scan.dat scan.pdf --colormap inferno            # Createc .dat → publication-ready PDF
probeflow convert scan.sxm scan.gwy                               # Nanonis .sxm → Gwyddion .gwy
probeflow convert scan.sxm scan.tif --tiff-mode float             # full-precision TIFF
probeflow convert scan.sxm scan.tif --tiff-mode uint16 --clip-low 2 --clip-high 98
probeflow convert scan.sm4 scan.png --colormap viridis            # RHK → PNG
probeflow convert default_0001.Z_mtrx scan.sxm                    # Omicron Matrix → Nanonis
probeflow convert scan.dat line0.csv --plane 0                    # single plane → CSV grid
```

### Processing (scan in → `.sxm` or `.png` out)

Each of these reads a scan (`.sxm`, `.dat`, `.gwy`, `.sm4`, `.mtrx` — auto-detected), applies a single operation to the selected plane (0 = Z forward by default), and writes a new `.sxm` — or a PNG with `--png`.

| Command            | Operation                                                         |
|--------------------|-------------------------------------------------------------------|
| `plane-bg`         | Subtract polynomial plane background (`--order 1` or `2`)         |
| `align-rows`       | Per-row offset / slope removal (`--method median|mean|linear`)    |
| `remove-bad-lines` | MAD-based outlier-row interpolation                               |
| `facet-level`      | Plane fit using only flat-terrace pixels — good for stepped surfaces |
| `smooth`           | Isotropic Gaussian smoothing (`--sigma` in pixels)                |
| `edge`             | Laplacian / LoG / DoG edge detection                              |
| `fft`              | 2-D FFT low-pass or high-pass filter                              |

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

| Command       | Purpose                                                               |
|---------------|-----------------------------------------------------------------------|
| `grains`      | Detect islands or depressions and print per-grain area / centroid     |
| `autoclip`    | Suggest GMM-based clip percentiles for display                        |
| `periodicity` | Find dominant spatial periods via the power spectrum                  |
| `info`        | Print header metadata (`--json` for machine-readable output)          |
| `profile`     | Sample z along a line (CSV / JSON / PNG; supports nm and px endpoints, swath averaging) |

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

### Spectroscopy (Createc `.VERT` files)

ProbeFlow reads Createc vertical-spectroscopy files and auto-detects the sweep type from the data:

| Sweep type   | X-axis         | Typical use                                    |
|--------------|----------------|------------------------------------------------|
| Bias sweep   | Bias (V)       | I(V) / Z(V) tunnelling spectroscopy            |
| Time trace   | Time (s)       | I(t) / Z(t) at fixed bias — telegraph noise    |

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
from probeflow.spec_io import read_spec_file
from probeflow.spec_processing import smooth_spectrum, numeric_derivative, crop
from probeflow.spec_plot import plot_spectrum, plot_spectra

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

Three tabs:

* **Browse** — point at a folder; the grid auto-detects every supported scan and spectrum format (`.sxm` / `.dat` / `.gwy` / `.sm4` / `.mtrx` / `.VERT`) and renders thumbnails for each. An *All / Images / Spectra* toggle filters the visible cards. Per-card colormap gallery, live clip sliders, per-scan undo, full-size viewer with interactive histogram, processing panel, and PNG export dialog.
* **Convert** — folder-in / folder-out batch dat→sxm and dat→png with PNG / SXM checkboxes and clip-percentile controls.
* **Features** — load the currently-selected Browse scan, choose a mode (*Particles* / *Template* / *Lattice*), tune parameters, hit *Run*. Results overlay on the canvas (contours, detection markers, primitive vectors + unit cell) and populate a sortable table. *Export JSON…* writes results with full scan provenance via `probeflow.writers.json`. Heavy analyses run on a background thread so the UI stays responsive.

Preferences (folders, theme, clip values) are saved to `~/.probeflow_config.json`.

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

Works on raw `.dat`, `.sxm`, or any other supported format — no pre-conversion step required:

```bash
for s in data/session/*.{dat,sxm,gwy}; do
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

# Works for any supported format — same API whether the input is
# Createc, Nanonis, Gwyddion, RHK, or Omicron.
scan = load_scan("raw_scan.dat")

scan.planes[0] = processing.align_rows(scan.planes[0], method="median")
scan.planes[0] = processing.subtract_background(scan.planes[0], order=1)

# Export by file suffix — sxm / png / pdf / tiff / gwy / csv.
scan.save("figure.pdf", colormap="inferno")
scan.save("archive.sxm")
```

Lower-level primitives for when you need the full vendor header or raw byte layout:

```python
from probeflow.sxm_io import parse_sxm_header, read_all_sxm_planes

hdr, planes = read_all_sxm_planes("scan.sxm")
```

Spectroscopy is a different shape of data, so it has its own module:

```python
from probeflow.spec_io import read_spec_file
from probeflow.spec_processing import smooth_spectrum, numeric_derivative

spec = read_spec_file("spectrum.VERT")
z_smooth = smooth_spectrum(spec.channels["Z"], method="savgol")
dzdv = numeric_derivative(spec.x_array, z_smooth)
```

---

## Repository layout

```
probeflow/              # installable package
├── __init__.py
├── scan.py             # Scan dataclass + load_scan dispatcher (main entry point)
├── readers/            # vendor format readers (sxm, dat, gwy, sm4, mtrx, gwy_bridge)
├── writers/            # output formats (sxm, png, pdf, tiff, gwy, csv, json)
├── processing.py       # image-processing pipeline (GUI-free) — incl. tv_denoise, line_profile
├── features.py         # particle segmentation / template counting / few-shot classify (GUI-free)
├── lattice.py          # SIFT lattice extraction + unit-cell averaging (GUI-free)
├── spec_io.py          # Createc .VERT reader → SpecData (GUI-free)
├── spec_processing.py  # spectroscopy processing functions (GUI-free)
├── spec_plot.py        # spectroscopy matplotlib plots (GUI-free)
├── sxm_io.py           # low-level .sxm byte layout (used by readers/writers)
├── common.py           # DAC / header utilities shared by Createc / Nanonis paths
├── dat_sxm.py          # Createc .dat → Nanonis .sxm (legacy CLI: `dat-sxm`)
├── dat_png.py          # Createc .dat → PNG previews (legacy CLI: `dat-png`)
├── gui.py              # PySide6 desktop interface (Browse / Convert / Features tabs)
└── cli.py              # unified "probeflow" command

src/file_cushions/      # binary layout captured from a reference .sxm file
data/                   # sample input / output for manual runs + tests
tests/                  # pytest suite (conversion, processing, .sxm round-trip,
                        #               spectroscopy reader + processing)
assets/                 # logo artwork
```

The `src/file_cushions/` directory holds the byte-level layout used to reconstruct `.sxm` files (header padding, `:SCANIT_END:` marker position, tail bytes, fixed data offset). These were reverse-engineered once from a reference Nanonis file and should be regenerated only if a future Nanonis version shifts the binary layout.

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
* `.VERT` header parsing, unit conversion, sweep-type detection, and error handling.
* Every public function in `probeflow.spec_processing`.
* Phase-2 readers / writers (`gwy`, `sm4`, `mtrx`, `pdf`, `tiff`, `csv`).
* Feature-detection: `segment_particles`, `count_features`, `classify_particles`.
* SIFT lattice extraction and `average_unit_cell`.
* Line-profile sampling with sub-pixel interpolation and swath averaging.
* Gwyddion-bridge availability and graceful-failure semantics.

---

## Notes

* The `.sxm` timestamp parser expects Createc filenames of the form `AyyMMdd.HHmmss.dat`.
* Failed batch conversions are logged to `errors.json` in the output directory.
* Sine-mode Createc scans are currently rejected with a clear error (not silently corrupted).

---

## Acknowledgements

**ProbeFlow** is developed at **[SPMQT-Lab](https://github.com/SPMQT-Lab)**, under the supervision of **Dr. Peter Jacobson** at **The University of Queensland**.

The core Createc-decoding algorithms were originally written by **[Rohan Platts](https://github.com/rohanplatts)**. ProbeFlow is a refactored, extended, and GUI-enabled evolution of that work — his contributions are the foundation of the conversion pipeline.

> *"Standing on the shoulders of giants."*
