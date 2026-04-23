<p align="center">
  <img src="assets/logo.gif" alt="ProbeFlow logo" width="100%"/>
</p>

<h1 align="center">ProbeFlow</h1>
<p align="center"><em>A scanning-tunneling-microscopy toolkit — browse, process, convert.</em></p>

---

## What It Is

**ProbeFlow** is a scanning-tunneling-microscopy (STM) toolkit built around three everyday lab needs:

1. **Getting scans out of Createc and into Nanonis.**
   Createc `.dat` files are decoded, scaled to SI units, and repackaged as Nanonis-compatible `.sxm` files or PNG previews.
2. **Looking at and cleaning up the scans.**
   Browse a folder of `.sxm` files in a thumbnail grid, switch colormaps, flatten with plane / facet levelling, fix bad scan lines, smooth, FFT-filter, detect grains, measure lattice periodicities, and export publication-ready PNGs with scale bars — **from a GUI *or* from the shell.**
3. **Reading and processing point-measurement spectroscopy.**
   Createc `.VERT` files (bias sweeps, time traces, Z spectroscopy) are read into a unit-aware data model, smoothed, differentiated, overlaid, and plotted — with tip positions optionally shown as markers on a topography image.

Everything is available as a CLI subcommand, so corrections and conversions can be scripted across hundreds of files or wired into a processing pipeline.

> **Status: beta.** The on-disk formats, the CLI surface, and the Python API are still subject to change between commits. Pin a commit hash if you depend on the current shape.

All topography commands (`plane-bg`, `align-rows`, `smooth`, `sxm2png`, `pipeline`, `info`, `grains`, `autoclip`, `periodicity`, `tv-denoise`, `profile` …) accept **either** `.sxm` **or** `.dat` inputs transparently — the format is auto-detected, so conversion is not a mandatory first step before analysis.

ProbeFlow reads **Createc `.dat`**, **Nanonis `.sxm`**, **Gwyddion `.gwy`**, **RHK `.sm4`**, and **Omicron Matrix `.mtrx`** (via optional extras), and writes **`.sxm`**, **PNG**, **PDF**, **TIFF**, **GWY**, **CSV**, and **JSON**. The `probeflow convert` subcommand is a one-shot any-in/any-out converter driven purely by file suffixes. When the `gwyddion` system binary is on `PATH`, an automatic **Gwyddion bridge** falls back to its converter for any other vendor format Gwyddion knows (Bruker, Park, NTEGRA, Nanoscope, JPK, …) — no extra Python dependency required.

A dedicated **Features tab** (mirrored as CLI subcommands) covers discrete-object STM analysis: particle / molecule **segmentation**, **template-match counting**, few-shot **classification**, **SIFT lattice extraction**, **unit-cell averaging**, **line profiles**, and edge-preserving **TV denoising** — all preserving SI units throughout.

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
# Convert a folder of Createc .dat scans into .sxm
probeflow dat2sxm -- --input-dir data/scans --output-dir out/sxm

# Export a single .sxm plane as a colour PNG with an auto-sized nm scale bar
probeflow sxm2png scan.sxm --colormap inferno --clip-low 2 --clip-high 98

# Flatten with a linear plane background, write a new .sxm
probeflow plane-bg scan.sxm --order 1 -o scan_flat.sxm

# Chain three corrections and write a PNG straight out
probeflow pipeline scan.sxm \
    --steps align-rows:median plane-bg:1 smooth:1.5 \
    --png --colormap viridis -o scan_clean.png
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

### Processing (`.sxm` in → `.sxm` or `.png` out)

Each of these reads an `.sxm`, applies a single operation to the selected plane (0 = Z forward by default), and writes a new `.sxm` — or a PNG with `--png`.

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

### Convert → flatten → export for publication

```bash
probeflow dat2sxm -- --input-dir raw --output-dir sxm
for s in sxm/*.sxm; do
    probeflow pipeline "$s" \
        --steps remove-bad-lines align-rows:median plane-bg:1 smooth:1.0 \
        --png --colormap inferno --scalebar-unit nm \
        -o "figures/${s##*/}.png"
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

The package is importable without pulling in the GUI:

```python
# Topographic image processing
from probeflow import processing
from probeflow.sxm_io import (
    read_sxm_plane, write_sxm_with_planes, read_all_sxm_planes,
    parse_sxm_header, sxm_dims, sxm_scan_range,
)

arr = read_sxm_plane("scan.sxm", plane_idx=0)
arr = processing.align_rows(arr, method="median")
arr = processing.subtract_background(arr, order=1)

# Spectroscopy
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
├── common.py           # DAC / header utilities used by both conversion paths
├── dat_sxm.py          # Createc .dat → Nanonis .sxm
├── dat_png.py          # Createc .dat → PNG previews
├── sxm_io.py           # .sxm read / write (GUI-free)
├── processing.py       # image-processing pipeline (GUI-free) — incl. tv_denoise, line_profile
├── features.py         # particle segmentation / template counting / few-shot classify (GUI-free)
├── lattice.py          # SIFT lattice extraction + unit-cell averaging (GUI-free)
├── spec_io.py          # Createc .VERT reader → SpecData (GUI-free)
├── spec_processing.py  # spectroscopy processing functions (GUI-free)
├── spec_plot.py        # spectroscopy matplotlib plots (GUI-free)
├── readers/            # vendor format readers (sxm, dat, gwy, sm4, mtrx, gwy_bridge)
├── writers/            # output writers (sxm, png, pdf, tiff, gwy, csv, json)
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
