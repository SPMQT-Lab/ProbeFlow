# Qt and Qt for Python redistribution information

ProbeFlow uses the open-source Qt and Qt for Python libraries under the GNU
Lesser General Public License version 3 (LGPLv3) option. ProbeFlow itself is
distributed under the MIT License. The release build does not modify Qt or Qt
for Python and dynamically loads their Windows libraries.

The complete applicable license texts and upstream attribution records are in
`_internal/THIRD_PARTY_LICENSES` below the ProbeFlow installation directory.
Each binary GitHub Release also publishes the checksum-pinned, unmodified Qt
corresponding-source archives named in `QT_CORRESPONDING_SOURCE.txt`.

## Using a modified Qt library

Recipients may study, modify, replace and reverse engineer the LGPL-covered
libraries for debugging those modifications. To run ProbeFlow with a compatible
modified Qt or Qt for Python build:

1. Make a working copy of the ProbeFlow installation directory.
2. Replace the corresponding DLL or Python extension below
   `_internal/PySide6/Qt`, `_internal/PySide6`, or `_internal/shiboken6`, while
   retaining compatible filenames and the Qt 6.11 ABI.
3. Launch `ProbeFlow.exe` from the copied directory.

No signature-removal step is required for this unsigned release. ProbeFlow's
complete build recipe is available in the public source repository.
