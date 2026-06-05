<p align="center">
  <img src="probeflow/assets/logo.gif" alt="ProbeFlow logo" width="100%"/>
</p>

# ProbeFlow

ProbeFlow is a lab workflow tool for scanning tunnelling microscopy and related
SPM data. It helps you browse folders of scans and spectra, apply routine image
corrections, draw ROIs, make common measurements, and export figures or data
with enough context to understand how they were produced. It reads Createc,
Nanonis, and RHK files and runs from a desktop GUI (with a command-line
interface for scripting and batch work).

> **Beta software.** ProbeFlow is `0.0.0b0`: the GUI, CLI, Python API, and JSON
> sidecar formats may still change. Raw microscope files are treated as
> read-only — processing and exports always write new files, with provenance
> sidecars where supported.

## Quick start

Install from a checkout (Python 3.11+):

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e .
```

Launch the GUI:

```bash
probeflow gui
```

A typical first session:

1. Open a folder of `.dat`, `.sxm`, `.sm4`, or spectroscopy files.
2. Pick a scan or spectrum from the thumbnail browser.
3. Choose a channel, colormap, and display range.
4. Draw an ROI or line profile if you need one.
5. Apply a correction, open the FFT viewer, or take a measurement.
6. Export the image, table, profile, spectrum, or processed scan.

Prefer the command line for inspection, conversion, and batch pipelines? See the
[command-line guide](docs/cli.md).

## Main features

ProbeFlow is honest about being a focused toolkit rather than a do-everything
suite. What it does today:

- **Browse** folders of scans and spectra in a thumbnail grid; switch channels
  (Z / current, forward / backward), colormaps, and display contrast.
- **Process images** — row alignment, bad-line detection and repair, background
  subtraction (plane fit, STM line-by-line, facet levelling), Gaussian
  smoothing and high-pass, edge detection, Fourier low/high-pass filters,
  periodic-spot notch filters, TV denoising, point/plane zeroing, lossless and
  arbitrary geometry transforms, and derived arithmetic channels. Steps are
  recorded as a processing state so an export can be reproduced.
- **Advanced edge detection** (Process tab) — **Canny** and **Sobel/Scharr**
  detectors with a live, non-destructive overlay preview and STM-tuned presets.
  Results become reusable analysis objects: an overlay, a new image, an **active
  mask**, or ROI(s). The active-mask layer (Masks tab) supports morphological
  cleanup (remove small objects, fill holes, dilate/erode/open/close,
  skeletonize) and restricts statistics directly; convert it to ROI(s) to
  exclude regions from a plane fit. Masks are saved to a `<scan>.masks.json`
  sidecar.
- **FFT tools** (the FFT viewer) — inspect the magnitude and radial profile with
  q in nm⁻¹; overlay a draggable reciprocal-lattice grid and apply an affine
  lattice correction; show Bragg-shell rings for a known structure; predict and
  notch out **mains pickup** (50/60 Hz); and use the **inverse-FFT /
  Fourier-reconstruction** tool to select circle/ellipse features, *remove* or
  *keep* them, preview the reconstructed image and the residual, then apply.
- **ROIs and measurements** — rectangle, ellipse, polygon, freehand, line, and
  point ROIs; ROI-scoped processing; line profiles, periodicity, ROI
  statistics, step heights, distances and angles, feature points, point-mask
  FFTs, pair correlation, and lattice / grid / unit-cell measurements. ROIs save
  to a `<scan>.rois.json` sidecar.
- **Feature analysis** *(optional)* — a Feature Counting tool for particle /
  molecule segmentation, counting, few-shot classification, template-match
  counting, lattice extraction, and reproducible step-edge exclusion. Requires
  the `features` extra (OpenCV + scikit-learn).
- **Spectroscopy** — inspect single traces or overlays / waterfalls. Smoothing,
  derivative, normalization, outlier masking, and offsets operate on derived
  display data; the raw loaded arrays are left intact.
- **Convert** Createc `.dat` scans to Nanonis-compatible `.sxm` (and PNG).
- **Export with context** — PNG, PDF, CSV, JSON, SXM, and optional Gwyddion
  `.gwy`. Many exports also write a JSON provenance sidecar (source file and
  channel, display settings, processing state, ROIs, and warnings). It is not a
  full electronic lab notebook, but it makes exported figures and data easier to
  interpret later. Exports never overwrite an existing file unless you ask
  (CLI `--force` / writer `overwrite=True`).

## Supported files

| Direction | File type | Use |
|---|---|---|
| Input | Createc `.dat` | STM/SPM image scan |
| Input | Createc `.VERT` | Point spectroscopy |
| Input | Nanonis `.sxm` | STM/SPM image scan |
| Input | Nanonis `.dat` | Point spectroscopy |
| Input | RHK `.sm4` | STM/SPM image scan |
| Output | `.sxm` | Converted or processed scan data |
| Output | `.png`, `.pdf` | Figure / image export |
| Output | `.csv`, `.json` | Numerical data, metadata, or provenance |
| Output | `.gwy` | Optional Gwyddion export when `gwyfile` is installed |

Createc `.dat` reader details (interrupted-scan handling, payload conventions)
are in [docs/createc_dat_reader.md](docs/createc_dat_reader.md).

## Installation notes

Python 3.11 or newer is required. The core install pulls in numpy, scipy,
Pillow, PySide6, matplotlib, and shapely.

Optional extras:

```bash
python -m pip install -e ".[features]"   # particle/feature + lattice tools (OpenCV, scikit-learn)
python -m pip install -e ".[gwyddion]"   # Gwyddion .gwy writer (gwyfile)
python -m pip install -e ".[dev]"        # test + lint tooling
```

The Feature Counting and lattice-extraction tools are inactive until the
`features` extra is installed; everything else works with the core install.

## Using ProbeFlow from Python

ProbeFlow can also be driven as a library:

```python
from probeflow import load_scan, processing

scan = load_scan("scan.dat")
scan.planes[0] = processing.align_rows(scan.planes[0], method="median")
scan.planes[0] = processing.subtract_background(scan.planes[0], order=1)
scan.save("processed.sxm")
scan.save("processed.png", colormap="gray")
```

```python
from probeflow.io.spectroscopy import read_spec_file
from probeflow.processing.spectroscopy import smooth_spectrum, numeric_derivative

spec = read_spec_file("spectrum.VERT")
z_smooth = smooth_spectrum(spec.channels["Z"], method="savgol")
dzdv = numeric_derivative(spec.x_array, z_smooth)
```

## Documentation

- [Command-line guide](docs/cli.md)
- [Createc `.dat` reader notes](docs/createc_dat_reader.md)
- [ROI manual workflow checklist](docs/roi_manual_test_checklist.md)
- [Review and cleanup status](docs/review_status.md)
- [Contributor notes](CONTRIBUTING.md)

## Development

```bash
python -m pip install -e ".[dev,features]"
pytest                          # run the test suite
ruff check probeflow tests      # lint
```

Repository layout:

```text
probeflow/
|-- core/        # Scan model, loading dispatch, metadata, ROI, validation
|-- io/          # File sniffing, readers, writers, converters, sidecars
|-- processing/  # Numerical processing and ProcessingState
|-- analysis/    # Image, lattice, feature, and spectroscopy analysis helpers
|-- provenance/  # Export provenance and graph data structures
|-- gui/         # PySide6 GUI package
|-- cli/         # Command-line interface
`-- plugins/     # Plugin API and registry groundwork

tests/           # pytest suite
test_data/       # sample input data
docs/            # additional documentation
```

## Acknowledgements

ProbeFlow is developed at [SPMQT-Lab](https://github.com/SPMQT-Lab) at The
University of Queensland.

The original Createc-decoding work was written by
[Rohan Platts](https://github.com/rohanplatts). ProbeFlow builds on that
foundation with browsing, conversion, processing, ROI workflows, spectroscopy
handling, FFT tools, and export provenance.
