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
