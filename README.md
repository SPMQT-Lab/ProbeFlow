<p align="center">
  <img src="probeflow/assets/logo.gif" alt="ProbeFlow logo" width="100%"/>
</p>

# ProbeFlow

ProbeFlow is a Python program for browsing scanning probe microscopy (SPM)
data, applying standard image-processing and analysis operations, and exporting
images or data with processing details alongside them.

It currently focuses on Createc and Nanonis files used in STM/SPM workflows.
The aim is practical: open a folder of data, inspect scans and spectra, apply
routine corrections, make simple measurements, and save outputs that record how
they were produced.

ProbeFlow is not a replacement for Gwyddion, Fiji/ImageJ, WSXM, or other
specialist tools. It is a lab workflow tool for common browsing, processing,
conversion, measurement, and export tasks.

## Status

ProbeFlow is beta software. The GUI, CLI, Python API, and JSON sidecar formats
may change between versions.

Raw microscope input files are treated as read-only. Processing and export
operations write to separate output paths and sidecar files.

## What It Does

- Browse folders containing supported scan and spectroscopy files.
- Load Createc `.dat` scans and `.VERT` spectroscopy files.
- Load Nanonis `.sxm` scans and Nanonis spectroscopy `.dat` files.
- Convert Createc `.dat` scans to Nanonis-compatible `.sxm` files.
- Apply standard image-processing operations such as row alignment, bad-line
  correction, background subtraction, smoothing, FFT filtering, notch filtering,
  denoising, zeroing, and simple geometric transforms.
- Draw ROIs and use them for selected processing and analysis operations.
- Make line profiles, histograms, FFT views, particle/feature summaries,
  lattice estimates, unit-cell averages, and spectroscopy plots.
- Inspect spectroscopy traces individually or as overlays/waterfalls, with basic
  smoothing, approximate numerical `dI/dV`, outlier masking, simple
  normalization, cursor readout, and CSV/JSON/TXT copy/export of displayed
  values.
- Export `.png`, `.pdf`, `.csv`, `.json`, `.sxm`, and optionally `.gwy` files.
- Write provenance sidecars for exported files where supported, including
  source information, channel information, display settings, processing state,
  warnings, and ROI data when available.

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev,features]"
pytest
```

Optional extras:

```bash
python -m pip install -e ".[features]"  # OpenCV / scikit-learn feature tools
python -m pip install -e ".[gwyddion]"  # optional Gwyddion writer dependency
```

## Quick Start

Launch the GUI:

```bash
probeflow gui
```

Inspect a scan from the command line:

```bash
probeflow info scan.dat
probeflow info scan.sxm --json
```

Convert a Createc `.dat` file to `.sxm`:

```bash
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

## Supported Files

Input:

| File type | Use |
|---|---|
| Createc `.dat` | STM/SPM image scan |
| Createc `.VERT` | Point spectroscopy |
| Nanonis `.sxm` | STM/SPM image scan |
| Nanonis `.dat` | Point spectroscopy |

Output:

| File type | Use |
|---|---|
| `.sxm` | Converted or processed scan data |
| `.png` | Image export |
| `.pdf` | Figure-style export |
| `.csv` | Numerical data export |
| `.json` | Metadata, provenance, or analysis output |
| `.gwy` | Optional Gwyddion export, when `gwyfile` is installed |

## GUI

The GUI is started with:

```bash
probeflow gui
```

It includes tools for folder browsing, viewing images, basic spectroscopy
inspection, selected-spectrum overlays, ROI drawing, image processing,
feature/lattice workflows, TV denoising, conversion, and export.

Spectroscopy display transforms are non-destructive. Smoothing, derivative,
normalization, outlier masking, and offsets are applied to derived display data,
not to the raw arrays loaded from the source file.

Preferences such as theme, recent folders, clip values, and font size are saved
in `~/.probeflow_config.json`.

## ROIs

ProbeFlow supports rectangle, ellipse, polygon, freehand, line, and point ROIs.
ROIs are stored in pixel coordinates with `(x, y) = (column, row)` and origin at
the top-left of the displayed image.

The default ROI sidecar for a scan is:

```text
<scan-stem>.rois.json
```

ROI sidecars are used by the GUI and by CLI commands that accept named ROIs.
Current limitations are simple: ROIs are stored in pixel coordinates, not
physical coordinates, and session-level ROI project files are not implemented.

## Provenance And Export Safety

ProbeFlow writes JSON sidecars for many exports. These sidecars are intended to
answer practical questions such as:

- what source file was used;
- which channel was exported;
- what display settings were used;
- what processing state was recorded;
- whether the output is processed rather than raw data;
- which ROIs were included, when relevant.

These sidecars are not a full electronic lab notebook. They are export records
for helping users understand and audit saved files.

Export paths are conservative by default. Existing output artifacts and
provenance sidecars are not overwritten unless overwrite options are used
explicitly, such as CLI `--force` or writer API `overwrite=True` /
`overwrite_sidecars=True`.

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

## Repository Layout

```text
probeflow/
|-- core/        # Scan model, loading dispatch, metadata, ROI, validation
|-- assets/      # Packaged logo artwork
|-- data/        # Packaged runtime resources
|-- io/          # File sniffing, readers, writers, converters, sidecar helpers
|-- processing/  # Numerical processing and ProcessingState
|-- analysis/    # Particles, lattice, spectroscopy plotting, feature tools
|-- provenance/  # Export provenance and graph data structures
|-- gui/         # PySide6 GUI package
|-- cli/         # Command-line interface
`-- plugins/     # Plugin API and registry groundwork

tests/           # pytest suite
test_data/       # sample/manual input data
docs/            # additional documentation
```

Architecture notes for contributors are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Testing

```bash
python -m pip install -e ".[dev,features]"
pytest
ruff check probeflow tests
```

Dead-code checks:

```bash
vulture probeflow/ tests/ whitelist.py --min-confidence 80
python scripts/find_orphan_modules.py
```

## Notes

- Failed batch conversions are logged to `errors.json`.
- Converted `.sxm` files should be checked in the target software when exact
  interoperability matters.
- Sidecar formats are JSON and may change while the project is beta.

## Acknowledgements

ProbeFlow is developed at [SPMQT-Lab](https://github.com/SPMQT-Lab) at The
University of Queensland.

The original Createc-decoding work was written by
[Rohan Platts](https://github.com/rohanplatts). ProbeFlow builds on that
foundation with browsing, conversion, processing, ROI workflows, spectroscopy
handling, and export provenance.
