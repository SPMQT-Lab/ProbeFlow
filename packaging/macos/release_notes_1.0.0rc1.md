# ProbeFlow 1.0.0 RC 1

This is the first standalone ProbeFlow desktop release candidate.

## System requirements

- Apple Silicon Mac (`arm64`)
- macOS 15 Sequoia or newer

## Installation

Open the DMG, drag `ProbeFlow.app` to the Applications shortcut, eject the disk
image, and launch ProbeFlow from Applications.

### macOS security notice

This release is **not Developer ID-signed or notarized by Apple**. macOS
will therefore prevent the downloaded app from opening normally the first time.
If you trust this ProbeFlow release:

1. Try to open ProbeFlow from Applications once and dismiss the warning.
2. Open **System Settings → Privacy & Security**.
3. Scroll down to Security, click **Open Anyway**, then confirm **Open**.

macOS saves that choice for ProbeFlow, so subsequent launches work normally.
The SHA-256 checksum attached to this release lets you verify that your DMG
matches the file published here. Apple documents the same exception process in
[Open apps safely on your Mac](https://support.apple.com/en-us/102445).

This candidate packages ProbeFlow's image and spectroscopy viewers, processing,
ROI and measurement tools, FFT workflow, and PNG, PDF, SXM and GWY export
support. Total Variation decomposition remains experimental, is adapted from
AiSurf, and has not been rigorously validated.

Exact corresponding source archives for the bundled LGPL-covered Qt 6.11
libraries are attached to this release alongside the application.

Please report release-candidate problems through the ProbeFlow GitHub issue
tracker.
