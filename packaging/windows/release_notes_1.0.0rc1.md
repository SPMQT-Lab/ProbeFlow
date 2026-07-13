# ProbeFlow 1.0.0 RC 1 for Windows

## System requirements

- Windows 10 version 1809 or newer, or Windows 11
- x64 processor

## Installation

Run `ProbeFlow-1.0.0-rc1-Windows-x64-Setup.exe`. ProbeFlow installs for the
current user, creates a Start Menu shortcut, and does not require administrator
access. It can be removed from **Settings → Apps → Installed apps**.

### Windows security notice

This release is not code-signed. Microsoft Defender SmartScreen may therefore
warn that ProbeFlow is an unrecognized application. Only continue if the file
came from the official SPMQT-Lab GitHub Release and its SHA-256 checksum matches
the attached checksum file. Managed computers may prevent unsigned software
from running altogether.

This candidate packages ProbeFlow's image and spectroscopy viewers, processing,
ROI and measurement tools, FFT workflow, and PNG, PDF, SXM and GWY export
support. Total Variation decomposition remains experimental, is adapted from
AiSurf, and has not been rigorously validated.
