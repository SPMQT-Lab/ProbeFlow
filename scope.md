# ProbeFlow Scope

## JOSS Publication Scope

The paper should cover only:

- folder-based browsing of large scan collections through thumbnails, channel selection, sorting, and filtering
- organisation through existing folder structure and metadata filtering
- reliable reading of supported Createc `.dat`, Nanonis `.sxm`, and RHK `.sm4` image structures, preserving calibration
- calibrated conversion and export where implemented: Createc-to-`.sxm`/`.npy`, processed `.sxm`, figures, tables, and provenance sidecars
- a small routine suite: scan-line/plane background correction, bad-line repair, smoothing, cropping, ROIs, line profiles, ROI statistics, distances, angles, and step-height measurements
- basic calibrated FFT inspection and periodicity measurement
- optional export to Gwyddion `.gwy`.
- Gwyddion and WSxM interoperability hooks.


## Out-of-Joss-Scope Functionality

Total-variation decomposition, advanced Fourier reconstruction/symmetrisation, lattice correction, SIFT extraction, feature/point statistics, masks, and broad CLI coverage are outside the paper unless separately reviewed and validated. Instrument control, feature counting, ML, dataset building, and reproduction of Gwyddion or WSxM functionality are out of scope.


