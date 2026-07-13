# ProbeFlow macOS release metadata

The first standalone ProbeFlow release targets Apple Silicon (`arm64`) and
macOS 15 Sequoia or newer. The minimum is based on the native deployment target
reported by the verified PySide6 6.11 binding binaries. The clean build in the
next release step must repeat the native-binary audit before the value is copied
into the application bundle.

`app_metadata.toml` is the source for the PyInstaller bundle settings in
`ProbeFlow.spec`. The packaged application derives its release version from
`probeflow.__version__`; the numeric bundle version fields exist for macOS
Finder, signing and update metadata.

## Reproducible build environment

The application is built with the official Python.org CPython 3.13.14 macOS
framework in a clean virtual environment. The build script downloads the
official installer, verifies its published SHA-256 checksum, and extracts the
framework under the ignored `build/macos/` tree without installing anything
system-wide. The Python.org runtime supports macOS 10.13 and later; PySide6
sets ProbeFlow's effective minimum to macOS 15.

To create and verify that environment without building the app:

```bash
scripts/build_macos_app.sh --prepare-only
```

To create the unsigned `build/macos/dist/ProbeFlow.app` used by the next
release step:

```bash
scripts/build_macos_app.sh
```

To rebuild the application and package it as a drag-to-install disk image:

```bash
scripts/build_macos_dmg.sh
```

This creates `ProbeFlow-1.0.0-rc1-macOS-arm64.dmg` and its `.sha256` checksum
under `build/macos/dist/`. The mounted image contains `ProbeFlow.app` and an
alias to `/Applications`. The script verifies the disk image, mounts it
read-only, repeats the application-bundle audit, and launches the packaged
smoke test from the mounted volume. Use `--skip-app-build` only when the
existing application was built from the exact source revision being packaged.

The application inside the resulting DMG is ad-hoc signed and suitable for
local testing or free distribution. The DMG itself is unsigned. A GitHub
Release can provide it as a direct download, but downloaded copies display the
usual macOS unidentified-developer warning because they have no Developer ID
signature or Apple notarization.

## Unsigned GitHub release

ProbeFlow currently uses the free, unnotarized distribution path. Before
publishing, `scripts/publish_github_release.sh --unsigned` requires a clean
checkout equal to `origin/main`, a valid DMG checksum, a complete mounted-bundle
audit and smoke test, and all five Qt corresponding-source archives. The
explicit flag prevents accidentally describing an unsigned build as notarized.

The release notes tell users to attempt the first launch and then approve
ProbeFlow under **System Settings → Privacy & Security → Open Anyway**. This is
an unavoidable extra installation step for an app distributed without a paid
Apple Developer ID. The release script creates the `v1.0.0-rc1` GitHub
prerelease and will not overwrite an existing release.

## Optional Developer ID release

The release build supports Developer ID signing without storing credentials in
the repository. Install a `Developer ID Application` certificate in the login
Keychain, then store notarization credentials under a named Keychain profile:

```bash
xcrun notarytool store-credentials "ProbeFlow-notary"
```

Build, sign, submit, staple and verify the release with:

```bash
PROBEFLOW_CODESIGN_IDENTITY="Developer ID Application: …" \
PROBEFLOW_NOTARY_PROFILE="ProbeFlow-notary" \
  scripts/notarize_macos_dmg.sh
```

No Apple password or API key is written to the repository or build logs. The
script uses PyInstaller's hardened-runtime signing for every collected native
binary, signs the disk image, waits for Apple's result, staples the ticket,
checks Gatekeeper assessment, and regenerates the checksum after stapling.

Every binary GitHub Release must publish the five checksum-pinned Qt 6.11
corresponding-source archives listed in `runtime_licenses.toml` beside the DMG.
The build caches those official archives under
`build/macos/downloads/qt-source/`; their full license texts and upstream
attribution records are also embedded in the application.

The notarization script remains available if the project adopts a paid Apple
Developer ID later. The unsigned publication workflow and release notes must be
changed before publishing a notarized build.

Set `PROBEFLOW_BUILD_PYTHON` to override the extracted Python 3.13.14 runtime, or
`PROBEFLOW_BUILD_ROOT` to place the disposable virtual environment and build
artifacts outside the repository. The script rejects non-arm64 interpreters,
recreates the virtual environment on every invocation, installs the complete
end-user `desktop` extra through the complete, verified resolution in
`constraints-arm64.txt`, and uses the pinned build tools in
`requirements-build.txt`. The top-level `constraints.txt` remains the lighter
known-good set for ordinary development and CI environments.

`ProbeFlow.spec` creates a windowed, one-folder app. It explicitly bundles the
GUI assets, Createc file cushions, ProbeFlow license and third-party notices;
collects GUI modules loaded through compatibility shims; excludes development
and alternate-Qt modules; and writes the release identity into `Info.plist`.
It deliberately does not sign the app yet.

After building, the hidden `--smoke-test` launcher option verifies that the
packaged main window can be created, required resources are readable, and the
OpenCV, scikit-learn and gwyfile features import successfully:

```bash
QT_QPA_PLATFORM=offscreen \
  build/macos/dist/ProbeFlow.app/Contents/MacOS/ProbeFlow --smoke-test
```

The build script runs that smoke test automatically after
`scripts/validate_macos_app.py` checks the bundle metadata and resources, every
Mach-O architecture and deployment target, build-machine library references,
and the complete ad-hoc signature.

`ProbeFlow.icns` is generated from the committed ProbeFlow logo by running:

```bash
python scripts/build_macos_icon.py
```

The initial release is intentionally native `arm64`. An Intel or `universal2`
release requires a separate build whose Python interpreter and every bundled
native dependency support that target.
