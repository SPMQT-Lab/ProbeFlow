# ProbeFlow GUI

Launch the graphical interface with:

```bash
probeflow gui
```

This guide walks through the most common workflows: loading images,
subtracting a background, exploring the FFT, and finding features.
The screenshots are generated from the real widgets by
`scripts/generate_gui_screenshots.py` — rerun it after UI changes to
refresh them.

## Loading images

Use **File → Open folder...** (or the **Open folder** button in the
sidebar) and pick any folder containing scans. ProbeFlow indexes the
folder and shows a thumbnail for every supported file — Createc `.dat`,
Nanonis `.sxm`, RHK `.sm4`, plus `.VERT` and Nanonis spectroscopy files.

![Browse mode with a folder of scans loaded](images/gui_browse.png)

Each card shows the scan size, setpoint, and channel info. The sidebar
controls the thumbnail colormap, channel, row alignment, and size; the
filter buttons (All / Images / Spectra) narrow the grid, cards can be
sorted by name or scan size, and the **Bias** picker lists the bias
values present in the folder so one setpoint can be shown alone.
**Double-click a thumbnail** to open it in the image viewer.

![The image viewer showing a terraced surface](images/gui_viewer.png)

The viewer opens on the raw topography with a histogram and contrast
controls in the right-hand sidebar (View tab). The toolbar above the image
switches channel and colormap; **← Prev / Next →** at the bottom steps
through the other scans in the folder. Every tool in the viewer is also
reachable from the **Search** box (or `Ctrl+K`) — type a few letters of
what you want ("background", "profile", "fft") and pick the command.

Raw microscope files are treated as read-only: everything below operates
on an in-memory copy, and saving always writes a new file.

## Subtracting a background

Scans usually come with a tilted plane or scan-line artifacts. Two tools
remove them:

* **Processing → Plane/background subtraction...** (`Ctrl+Shift+B`) —
  polynomial plane fits.
* **Processing → STM scan-line background...** (`Ctrl+Alt+B`) — per-line
  background estimation designed for terraced STM topographs.

![STM scan-line background dialog with a linear fit previewed](images/gui_stm_background.png)

In the STM background dialog:

1. Pick the **Fit region** — the whole image, or the active ROI if you
   have drawn one around a flat terrace.
2. Pick the **Line statistic** (median is robust to steps and tip
   changes) and the **Background model**. Models range from *Linear*
   through *2nd/3rd order polynomial*, *Low-pass* and *Line by line* to
   the *Piezo creep* family — switch between them with the dropdown and
   compare the fits.
3. Click **Preview corrected image**. The right-hand plots show the
   per-line statistic with the fitted background and the residual per
   scan line, plus residual RMS — switch models until the residuals stop
   shrinking.
4. Click **Apply**. The subtraction is recorded in the processing
   history (undo with `Ctrl+Z`), and exports carry the full provenance.

## Cleaning up defects

Three everyday repairs, all recorded (and undoable) processing steps:

* **Median filter (despeckle)** — Process tab → Smooth → *Median*.
  Removes salt-and-pepper noise and single-pixel tip glitches without
  blurring step edges (unlike Gaussian smoothing).
* **Remove spots** — mark a tip change, dirt speck, or glitch with an
  area ROI (right-click → *Remove spots*) or a mask from Advanced Edge
  Detection (Masks section → *Remove spots*). The region is replaced by
  a smooth surface interpolated from its surroundings.
* **Crop** — draw a rectangle selection (or right-click an area ROI →
  *Crop image to this region*), then **Image → Transform → Crop to
  selection**. The scale bar, FFT axes, ROIs, and masks all follow the
  new extent; pixel size is unchanged.

## Performing an FFT

Open **Measurements → FFT viewer...** (`Ctrl+Shift+F`, or the **FFT**
button in the quick toolbar). The viewer computes the FFT of the current
processed image — subtract the background first, or the spectrum is
dominated by the surface tilt.

The left pane shows the real-space source with its pixel and q-space
resolution; the main pane shows log-magnitude FFT with reciprocal-space
axes. The tabs below cover the common reciprocal-space tasks:

* **Inspect** — intensity histogram with min/max/brightness/contrast
  sliders, and a radial profile of the spectrum.
* **Grid** — fit a reciprocal lattice to the Bragg peaks.
* **Correction** — preview lattice undistortion from the fitted grid.
* **Mains** — detect and suppress mains-frequency pickup streaks.
* **Inverse FFT** — mask regions of the spectrum and reconstruct the
  filtered image.
* **Symmetrize** — enforce an n-fold (optionally mirrored) symmetry by
  averaging the image with its rotated copies. Rotated copies are
  registered back onto the original automatically, so the symmetry axis
  need not sit at the image centre. Always check the **Residual**
  preview: it holds everything symmetrization removed — noise, but also
  real defects and domain boundaries.

**Focus FFT** and the zoom buttons home in on the spectral content near
the origin, and the **Export** menu saves the spectrum or filtered image.
For a quick periodicity measurement without the full viewer, use
**Measurements → Find spacing from line profile...** on a line ROI.

## Detecting features and point-pattern measurements

Under **Measurements → Features** you can locate point-like features —
atoms, molecules, defects, moiré sites — and analyse their spatial pattern:

- **Feature maxima** detects local protrusions (or, with an area ROI
  selected, only within that region), marking them as points in physical
  (nm) coordinates.
- **Pair correlation** computes the pair-correlation function g(r) from the
  detected points or from point ROIs, reporting the density, the
  nearest-neighbour median spacing, and the first-peak position. With a
  calibrated area ROI, g(r) is density-normalised.
- **Point mask / FFT** builds a mask from detected features and inspects its
  FFT.

SIFT-based lattice-vector extraction is an optional tool that needs the
`lattice` extra:

```bash
pip install "probeflow[lattice]"
```

## Beyond the basics

* **ROIs** — draw rectangles, ellipses, polygons, freehand outlines,
  lines, and points from the ROI tab; ROIs restrict background fits,
  FFTs, and statistics, and are saved as sidecar files next to the scan.
* **Measurements** — distance and angle measurements, line profiles, ROI
  statistics, step heights, pair correlation.
* **Spectroscopy** — `.VERT` and Nanonis spectroscopy files open in a
  dedicated spectrum viewer; positions can be overlaid on the topograph.
* **Export** — PNG/PDF/CSV/`.sxm`/`.gwy` export with the full processing
  history embedded, so any image can be reproduced from the raw file.

See [cli.md](cli.md) for the command-line equivalents of these
workflows.
