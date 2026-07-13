param(
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildRoot = if ($env:PROBEFLOW_BUILD_ROOT) {
    $env:PROBEFLOW_BUILD_ROOT
} else {
    Join-Path $Root "build\windows"
}
$Venv = Join-Path $BuildRoot "venv"
$VenvPython = Join-Path $Venv "Scripts\python.exe"
$Python = if ($env:PROBEFLOW_BUILD_PYTHON) {
    $env:PROBEFLOW_BUILD_PYTHON
} else {
    (Get-Command python -ErrorAction Stop).Source
}
$Constraints = Join-Path $Root "packaging\windows\constraints-x64.txt"
$BuildRequirements = Join-Path $Root "packaging\windows\requirements-build.txt"
$LicenseManifest = Join-Path $Root "packaging\runtime_licenses.toml"
$LicenseDir = Join-Path $BuildRoot "licenses"

if (-not $IsWindows) {
    throw "ProbeFlow's Windows bundle must be built on Windows."
}

& $Python -c @'
import platform
import struct
import sys

if sys.version_info[:3] != (3, 13, 14):
    raise SystemExit(f"Expected Python 3.13.14, found {sys.version.split()[0]}")
if platform.machine().upper() not in {"AMD64", "X86_64"} or struct.calcsize("P") != 8:
    raise SystemExit(f"Expected 64-bit x86 Python, found {platform.machine()}")
'@
if ($LASTEXITCODE -ne 0) { throw "The Windows build interpreter is incompatible." }

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
Write-Host "Creating clean build environment: $Venv"
& $Python -m venv --clear $Venv
if ($LASTEXITCODE -ne 0) { throw "Could not create the Windows build environment." }

& $VenvPython -m pip install --constraint $Constraints --requirement $BuildRequirements
if ($LASTEXITCODE -ne 0) { throw "Could not install Windows build requirements." }
& $VenvPython -m pip install --constraint $Constraints "$Root[desktop]"
if ($LASTEXITCODE -ne 0) { throw "Could not install ProbeFlow's desktop dependencies." }

& $VenvPython -c @'
import platform

import PyInstaller
import PySide6
import cv2
import gwyfile
import probeflow
import sklearn

print(f"ProbeFlow {probeflow.__version__}")
print(f"Python {platform.python_version()} ({platform.machine()})")
print(f"PyInstaller {PyInstaller.__version__}; PySide6 {PySide6.__version__}")
print(f"OpenCV {cv2.__version__}; scikit-learn {sklearn.__version__}")
print(f"gwyfile {getattr(gwyfile, '__version__', 'installed')}")
'@
if ($LASTEXITCODE -ne 0) { throw "The Windows build environment import check failed." }

Write-Host "Collecting third-party licenses and corresponding-source metadata"
& $VenvPython (Join-Path $Root "scripts\collect_python_licenses.py") `
    --manifest $LicenseManifest `
    --output (Join-Path $LicenseDir "python")
if ($LASTEXITCODE -ne 0) { throw "Could not collect Python distribution licenses." }

$PythonBase = (& $VenvPython -c "import sys; print(sys.base_prefix)").Trim()
$PythonLicense = Join-Path $PythonBase "LICENSE.txt"
if (-not (Test-Path $PythonLicense -PathType Leaf)) {
    throw "The exact CPython runtime license is missing: $PythonLicense"
}
$RuntimeLicenseDir = Join-Path $LicenseDir "runtime\CPython-3.13.14"
New-Item -ItemType Directory -Force -Path $RuntimeLicenseDir | Out-Null
Copy-Item -Force $PythonLicense (Join-Path $RuntimeLicenseDir "LICENSE.txt")

& $VenvPython (Join-Path $Root "scripts\prepare_qt_release_materials.py") `
    --manifest $LicenseManifest `
    --download-dir (Join-Path $BuildRoot "downloads\qt-source") `
    --license-output (Join-Path $LicenseDir "qt")
if ($LASTEXITCODE -ne 0) { throw "Could not prepare Qt release materials." }

if ($PrepareOnly) {
    Write-Host "Windows build environment is ready; application build skipped."
    exit 0
}

Write-Host "Building ProbeFlow.exe"
$env:PYINSTALLER_CONFIG_DIR = Join-Path $BuildRoot "pyinstaller-cache"
$env:MPLCONFIGDIR = Join-Path $BuildRoot "matplotlib-cache"
$env:PROBEFLOW_LICENSE_DIR = $LicenseDir
& (Join-Path $Venv "Scripts\pyinstaller.exe") `
    --clean `
    --noconfirm `
    --distpath (Join-Path $BuildRoot "dist") `
    --workpath (Join-Path $BuildRoot "work") `
    (Join-Path $Root "packaging\windows\ProbeFlow.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed to build ProbeFlow." }

$App = Join-Path $BuildRoot "dist\ProbeFlow"
& $VenvPython (Join-Path $Root "scripts\validate_windows_app.py") $App
if ($LASTEXITCODE -ne 0) { throw "The Windows application bundle failed validation." }

$SmokeHome = Join-Path $BuildRoot "smoke-test-home"
New-Item -ItemType Directory -Force -Path $SmokeHome | Out-Null
$env:HOME = $SmokeHome
$env:USERPROFILE = $SmokeHome
$env:MPLCONFIGDIR = Join-Path $BuildRoot "smoke-test-matplotlib"
$env:QT_QPA_PLATFORM = "offscreen"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
$Smoke = Start-Process `
    -FilePath (Join-Path $App "ProbeFlow.exe") `
    -ArgumentList "--smoke-test" `
    -Wait `
    -PassThru
if ($Smoke.ExitCode -ne 0) {
    throw "The packaged Windows GUI smoke test failed with exit code $($Smoke.ExitCode)."
}

Write-Host "Built and validated $App"
