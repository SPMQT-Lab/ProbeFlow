# ProbeFlow Windows release

The initial Windows release targets Windows 10 version 1809 or newer and
Windows 11 on x64 processors. It uses the same ProbeFlow 1.0.0 RC 1 source and
dependency baselines as the macOS release.

PyInstaller output is platform-specific, so this package is built on a clean
GitHub-hosted Windows runner rather than cross-compiled from macOS. The
`Build Windows installer` workflow can be started manually from the Actions
page. It performs the following release gates:

1. Installs CPython 3.13.14 x64 and the pinned Windows dependency resolution.
2. Runs the complete ProbeFlow test suite on Windows.
3. Creates a clean PyInstaller one-folder application.
4. Collects runtime license texts and Qt corresponding-source records.
5. Audits every EXE, DLL and PYD as x64 and rejects unused Qt Virtual Keyboard,
   Qt PDF, QML and Quick components.
6. Runs the packaged GUI and PDF-export smoke test.
7. Builds a per-user installer with NSIS 3.12.0.
8. Silently installs, validates, smoke-tests and uninstalls that installer.

Successful runs provide these workflow artifacts:

- `ProbeFlow-1.0.0-rc1-Windows-x64-Setup.exe`
- `ProbeFlow-1.0.0-rc1-Windows-x64-Setup.exe.sha256`
- `ProbeFlow-1.0.0-rc1-Windows-x64-portable.zip`
- `ProbeFlow-1.0.0-rc1-Windows-x64-portable.zip.sha256`

The installer uses `%LOCALAPPDATA%\Programs\ProbeFlow`, writes only current-user
registry entries, adds a Start Menu shortcut and registers an uninstaller. It
does not request administrator privileges and leaves user configuration and
data untouched during uninstall.

The first Windows package is intentionally unsigned. Microsoft Defender
SmartScreen can warn about an application without an established reputation,
and managed Windows computers may block it completely. The GitHub Release must
state this clearly and publish the SHA-256 checksum beside the installer.

After manual testing, the verified installer and checksum can be attached to
the existing `v1.0.0-rc1` GitHub prerelease. The five Qt corresponding-source
archives required by the LGPL are already release assets shared with the macOS
binary.

To reproduce the build on a Windows x64 machine with CPython 3.13.14 and NSIS
3.12.0 installed:

```powershell
pwsh -File scripts/build_windows_app.ps1
pwsh -File scripts/build_windows_installer.ps1
```
