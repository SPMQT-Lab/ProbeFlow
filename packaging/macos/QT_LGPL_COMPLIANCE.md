# Qt and Qt for Python redistribution information

ProbeFlow uses the open-source Qt and Qt for Python libraries under the GNU
Lesser General Public License version 3 (LGPLv3) option. ProbeFlow itself is
distributed under the MIT License. The ProbeFlow release build does not modify
Qt or Qt for Python and dynamically links their libraries inside the macOS
application bundle.

The complete applicable license texts and upstream attribution records are in
`Contents/Resources/THIRD_PARTY_LICENSES` inside `ProbeFlow.app`. Each binary
GitHub Release must also publish the checksum-pinned, unmodified corresponding
source archives named in `QT_CORRESPONDING_SOURCE.txt` beside the DMG.

## Using a modified Qt library

Recipients may study, modify, replace and reverse engineer the LGPL-covered
libraries for debugging those modifications. To run ProbeFlow with a compatible
modified Qt or Qt for Python build:

1. Make a working copy of `ProbeFlow.app`.
2. Replace the corresponding dynamic frameworks or libraries below
   `Contents/Frameworks/PySide6/Qt/lib`, `Contents/Frameworks/PySide6`, or
   `Contents/Frameworks/shiboken6`, retaining compatible filenames, install
   names and the Qt 6.11 ABI.
3. Re-sign the modified copy locally with an ad-hoc signature:

   ```bash
   codesign --force --deep --sign - /path/to/ProbeFlow.app
   ```

4. Launch the executable at `ProbeFlow.app/Contents/MacOS/ProbeFlow`.

Apple's original notarization ticket no longer covers a user-modified bundle;
that does not restrict the recipient from running a locally re-signed copy.
ProbeFlow's complete build recipe is available in the public source repository.
