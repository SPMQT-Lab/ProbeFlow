# ProbeFlow third-party notices

ProbeFlow is released under the MIT License. Copyright © 2026 SPMQT-Lab and
contributors. The complete ProbeFlow license is in the repository `LICENSE`
file and is included with the packaged application.

## Adapted and attributed work

- The original Createc decoding work was written by
  [Rohan Platts](https://github.com/rohanplatts).
- The experimental Total Variation decomposition feature is adapted from
  [AiSurf: Automated Identification of Surface Images](https://github.com/QuantumMaterialsModelling/AiSurf-Automated-Identification-of-Surface-images).
  This ProbeFlow adaptation has not been rigorously validated and is included
  for testing purposes. Scientific users should consult and cite the
  [original AiSurf Total Variation work](https://arxiv.org/abs/2505.08843).

## Direct runtime libraries

The first desktop build is based on these direct libraries. Their complete
license texts and the licenses of resolved transitive libraries are included
under `THIRD_PARTY_LICENSES` in the application resources.

Qt and Qt for Python are used under their LGPLv3 option. The accompanying
`QT_LGPL_COMPLIANCE.md` explains the corresponding-source release assets and
how recipients can run ProbeFlow with compatible modified Qt libraries. The
unused, GPL-only Qt Virtual Keyboard component is excluded from the release.

| Library | Release baseline | License | Project |
|---|---:|---|---|
| CPython | 3.13.14 | Python Software Foundation License Version 2 | [python.org](https://www.python.org/) |
| NumPy | 2.1.3 | BSD-3-Clause | [numpy.org](https://numpy.org/) |
| SciPy | 1.15.3 | BSD-3-Clause | [scipy.org](https://scipy.org/) |
| Pillow | 12.2.0 | MIT-CMU | [python-pillow.org](https://python-pillow.org/) |
| PySide6, PySide6 Essentials, PySide6 Addons and Shiboken6 | 6.11.0 | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only | [Qt for Python](https://doc.qt.io/qtforpython-6/) |
| Matplotlib | 3.10.0 | Matplotlib License | [matplotlib.org](https://matplotlib.org/) |
| Shapely | 2.1.2 | BSD-3-Clause | [shapely.readthedocs.io](https://shapely.readthedocs.io/) |
| scikit-image | 0.25.0 | BSD-3-Clause and bundled component licenses | [scikit-image.org](https://scikit-image.org/) |

## Desktop feature libraries

The full desktop build includes the optional lattice and Gwyddion export
features. Their exact versions are pinned in the clean macOS build environment.

| Library | Current baseline | License | Project |
|---|---:|---|---|
| OpenCV Python | 4.13.0.92 | Apache-2.0 and bundled component licenses | [opencv.org](https://opencv.org/) |
| scikit-learn | 1.6.1 | BSD-3-Clause | [scikit-learn.org](https://scikit-learn.org/) |
| gwyfile | 0.3.0 | MIT | [gwyfile on PyPI](https://pypi.org/project/gwyfile/) |

This notice is an inventory and attribution index, not a replacement for the
complete license texts shipped by the libraries.
