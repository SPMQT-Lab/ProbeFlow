# ProbeFlow CLI

The command-line entry point is:

```bash
probeflow
```

Run `probeflow <command> --help` for exact options.

## Common Commands

| Command | Purpose |
|---|---|
| `gui` | Launch the PySide6 GUI |
| `info` | Print basic scan metadata |
| `convert` | Convert or export based on input/output suffixes |
| `dat2sxm` | Convert Createc `.dat` to `.sxm` |
| `dat2png` | Render Createc `.dat` to PNG |
| `dat2npy` | Export Createc `.dat` to NumPy bundles |
| `sxm2png` | Render `.sxm` or supported scans to PNG |
| `pipeline` | Apply ordered processing steps |
| `prepare-png` | Write a PNG handoff with provenance sidecar |
| `plane-bg` | Polynomial background subtraction |
| `facet-level` | Plane-level using only flat-terrace pixels (stepped surfaces) |
| `align-rows` | Per-row offset correction |
| `remove-bad-lines` | Detect and interpolate bad scan lines |
| `smooth` | Gaussian smoothing |
| `tv-denoise` | Total-variation denoising (Huber-ROF / TV-L1) |
| `edge` | Edge detection (Laplacian / LoG / DoG) |
| `rotate`, `rotate-90`, `rotate-180`, `rotate-270` | Rotate a scan (arbitrary angle or lossless quarter turns) |
| `flip-h`, `flip-v` | Mirror a scan about the vertical / horizontal axis |
| `fft` | Fourier-domain filtering |
| `histogram` | Pixel-value histogram |
| `fft-spectrum` | FFT magnitude spectrum |
| `profile` | Line profile from endpoints or a named line ROI |
| `periodicity` | Find dominant spatial periodicities via power spectrum |
| `autoclip` | Suggest display clip percentiles |
| `particles`, `count`, `classify` | Feature analysis workflows |
| `grains` | Detect grains / islands by threshold and print statistics |
| `lattice`, `unit-cell` | Lattice extraction and unit-cell averaging |
| `diag-z` | Diagnose Z-scale candidates for a Createc `.dat` file |
| `spec-info`, `spec-plot`, `spec-overlay`, `spec-positions` | Spectroscopy utilities |

Some feature/lattice commands require optional dependencies from the
`features` extra.

## Examples

Inspect a file:

```bash
probeflow info scan.sxm
probeflow info scan.dat --json
```

Convert Createc `.dat` to Nanonis-compatible `.sxm`:

```bash
probeflow convert scan.dat scan.sxm
```

Export Createc `.dat` to NumPy bundles:

```bash
probeflow dat2npy -- --input-dir data/scans --output-dir out/npy
```

The NumPy exporter writes a bundle directory per input file, plus
`*.probeflow.json` and `*.provenance.json` sidecars that record source path,
basis, channel mapping, warnings, and other provenance metadata.

Render a scan to PNG:

```bash
probeflow sxm2png scan.sxm -o scan.png --colormap gray
```

Apply a processing pipeline:

```bash
probeflow pipeline scan.sxm \
    --steps remove-bad-lines align-rows:median plane-bg:1 smooth:1.2 \
    -o scan_processed.sxm
```

Export a processed PNG:

```bash
probeflow pipeline scan.dat \
    --steps align-rows:median plane-bg:1 \
    --png --colormap gray \
    -o scan_processed.png
```

Prepare a PNG for downstream analysis:

```bash
probeflow prepare-png scan.dat aisurf_input.png \
    --steps align-rows:median plane-bg:1 \
    --colormap gray
```

Use a named ROI saved by the GUI:

```bash
probeflow histogram scan.sxm --roi terrace
probeflow profile scan.sxm --roi line_1 -o profile.csv
```

Plot spectroscopy:

```bash
probeflow spec-info spectrum.VERT
probeflow spec-plot spectrum.VERT -o spectrum.png
```

## Pipeline Step Syntax

Pipeline step syntax is:

```text
name[:param1,param2,...]
```

Examples:

```bash
align-rows:median
plane-bg:1
smooth:1.2
```

For full accepted parameter forms, use:

```bash
probeflow pipeline --help
```

## Output Safety

CLI exports refuse to overwrite existing output artifacts or provenance
sidecars unless `--force` is provided.

Processing commands that derive an output path use command-specific suffixes.
For example, `smooth` writes `<input-stem>_smooth.sxm` by default and
`pipeline` writes `<input-stem>_pipeline.sxm` by default.
