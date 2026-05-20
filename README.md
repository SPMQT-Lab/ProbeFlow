<p align="center">
  <img src="probeflow/assets/logo.gif" alt="ProbeFlow logo" width="100%"/>
</p>

# ProbeFlow

ProbeFlow is a lab workflow tool for scanning tunnelling microscopy and related
SPM data. It helps you browse folders of scans and spectra, apply routine image
corrections, draw ROIs, make common measurements, and export figures or data
with enough context to understand how they were produced.

It currently focuses on Createc and Nanonis workflows:

- Open Createc `.dat` and Nanonis `.sxm` scan images.
- Open Createc `.VERT` and Nanonis spectroscopy `.dat` traces.
- Convert Createc `.dat` scans to Nanonis-compatible `.sxm`.
- Process scans with row alignment, bad-line correction, background
  subtraction, smoothing, FFT filters, notch filters, denoising, zeroing, and
  simple geometry transforms.
- Measure line profiles, periodicity, ROI statistics, FFTs, feature points,
  pair correlations, lattice grids, unit cells, and spectroscopy traces.
- Export PNG, PDF, CSV, JSON, SXM, and optional Gwyddion `.gwy` outputs.

ProbeFlow is beta software. The GUI, CLI, Python API, and JSON sidecar formats
may still change. Raw microscope files are treated as read-only; processing and
export steps write separate output files and provenance sidecars where
supported.

## Quick Start

Install from a checkout:

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e .
```

Launch the GUI:

```bash
probeflow gui
```

Typical first GUI workflow:

1. Open a folder containing `.dat`, `.sxm`, or spectroscopy files.
2. Select a scan or spectrum from the browser.
3. Adjust display contrast and colormap.
4. Draw an ROI or line profile if needed.
5. Apply a correction or measurement.
6. Export the image, table, profile, spectrum, or processed scan.

Inspect or convert files from the command line:

```bash
probeflow info scan.dat
probeflow info scan.sxm --json
probeflow convert scan.dat scan.sxm
```

Apply a small processing pipeline and export a PNG:

```bash
probeflow pipeline scan.dat \
    --steps align-rows:median plane-bg:1 \
    --png --colormap gray \
    -o scan_processed.png
```

More CLI examples are in [docs/cli.md](docs/cli.md).

## Installation Notes

Python 3.11 or newer is required.

Optional feature and export dependencies:

```bash
python -m pip install -e ".[features]"  # OpenCV / scikit-learn feature tools
python -m pip install -e ".[gwyddion]"  # optional .gwy writer dependency
```

Development install:

```bash
python -m pip install -e ".[dev,features]"
pytest
```

The experimental ScanFlow survey/PPTX integration is optional and is not
installed with ProbeFlow by default.

## Supported Files

| Direction | File type | Use |
|---|---|---|
| Input | Createc `.dat` | STM/SPM image scan |
| Input | Createc `.VERT` | Point spectroscopy |
| Input | Nanonis `.sxm` | STM/SPM image scan |
| Input | Nanonis `.dat` | Point spectroscopy |
| Output | `.sxm` | Converted or processed scan data |
| Output | `.png`, `.pdf` | Figure/image export |
| Output | `.csv`, `.json` | Numerical data, metadata, or provenance |
| Output | `.gwy` | Optional Gwyddion export when `gwyfile` is installed |

Createc `.dat` reader details, including interrupted-scan handling and stored
payload conventions, live in [docs/createc_dat_reader.md](docs/createc_dat_reader.md).

## Key Workflows

### Browse And Process Scans

Use the GUI to browse scan folders, inspect channels, adjust display ranges,
draw ROIs, and apply common corrections. Processing operations are recorded as
processing state where supported so exported files can be audited later.

### Measure Images

ProbeFlow supports rectangle, ellipse, polygon, freehand, line, and point ROIs.
Common image measurements include line profiles, periodicity estimates, ROI
statistics, feature points, point-mask FFTs, pair correlation, feature-to-lattice
comparison, and lattice/grid measurements.

ROIs are stored in pixel coordinates with `(x, y) = (column, row)` and origin at
the top-left of the displayed image. The default ROI sidecar is:

```text
<scan-stem>.rois.json
```

### Inspect Spectroscopy

The spectroscopy viewer can inspect individual traces or overlays/waterfalls.
Smoothing, derivative, normalization, outlier masking, and offsets operate on
derived display data rather than overwriting the raw loaded arrays.

### Export With Context

ProbeFlow writes JSON sidecars for many exports. These records can include
source file, source channel, display settings, processing state, warnings, and
ROI data where relevant. They are not a full electronic lab notebook, but they
make exported figures and data easier to interpret later.

Export paths are conservative by default. Existing output artifacts and
provenance sidecars are not overwritten unless overwrite options are used
explicitly, such as CLI `--force` or writer API `overwrite=True`.

## Python Use

ProbeFlow can also be used from Python:

```python
from probeflow import load_scan, processing

scan = load_scan("scan.dat")
scan.planes[0] = processing.align_rows(scan.planes[0], method="median")
scan.planes[0] = processing.subtract_background(scan.planes[0], order=1)
scan.save("processed.sxm")
scan.save("processed.png", colormap="gray")
```

Spectroscopy:

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

Run the test suite:

```bash
pytest
```

Run the current lint check:

```bash
ruff check probeflow tests
```

Useful development scripts:

```bash
vulture probeflow/ tests/ whitelist.py --min-confidence 80
python scripts/find_orphan_modules.py
```

Repository layout:

```text
probeflow/
|-- core/        # Scan model, loading dispatch, metadata, ROI, validation
|-- assets/      # Packaged logo and GUI assets
|-- data/        # Packaged runtime resources
|-- io/          # File sniffing, readers, writers, converters, sidecars
|-- processing/  # Numerical processing and ProcessingState
|-- analysis/    # Image, lattice, feature, and spectroscopy analysis helpers
|-- provenance/  # Export provenance and graph data structures
|-- gui/         # PySide6 GUI package
|-- cli/         # Command-line interface
`-- plugins/     # Plugin API and registry groundwork

tests/           # pytest suite
test_data/       # sample/manual input data
docs/            # additional documentation
```

## Acknowledgements

ProbeFlow is developed at [SPMQT-Lab](https://github.com/SPMQT-Lab) at The
University of Queensland.

The original Createc-decoding work was written by
[Rohan Platts](https://github.com/rohanplatts). ProbeFlow builds on that
foundation with browsing, conversion, processing, ROI workflows, spectroscopy
handling, and export provenance.
