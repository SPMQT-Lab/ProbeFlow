#!/usr/bin/env bash
# Create an isolated Apple Silicon build environment and build ProbeFlow.app.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PROBEFLOW_BUILD_ROOT:-${ROOT}/build/macos}"
VENV="${BUILD_ROOT}/venv"
PREPARE_ONLY=0
PYTHON_VERSION="3.13.14"
PYTHON_PACKAGE="python-${PYTHON_VERSION}-macos11.pkg"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_PACKAGE}"
PYTHON_SHA256="8e58affb218c155a1dfdc27b291f817129669f8760e7a297adb2e4439ba5d2e8"
PYTHON_RUNTIME="${BUILD_ROOT}/python-runtime-${PYTHON_VERSION}"
PYTHON_FRAMEWORK="${PYTHON_RUNTIME}/Python_Framework.pkg/Payload/Versions/3.13"
DEFAULT_PYTHON="${PYTHON_FRAMEWORK}/bin/python3.13"

provision_python() {
    local download_dir="${BUILD_ROOT}/downloads"
    local installer="${download_dir}/${PYTHON_PACKAGE}"
    local actual_sha256

    if [[ -x "${DEFAULT_PYTHON}" ]]; then
        return
    fi

    mkdir -p "${download_dir}"
    if [[ ! -f "${installer}" ]]; then
        echo "Downloading official Python.org ${PYTHON_VERSION} runtime"
        curl --location --fail --show-error --output "${installer}" "${PYTHON_URL}"
    fi

    actual_sha256="$(shasum -a 256 "${installer}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${PYTHON_SHA256}" ]]; then
        echo "Python installer checksum mismatch: ${actual_sha256}" >&2
        exit 1
    fi
    if [[ -e "${PYTHON_RUNTIME}" ]]; then
        echo "Incomplete Python runtime exists at ${PYTHON_RUNTIME}; remove it and retry." >&2
        exit 1
    fi

    echo "Extracting Python.org runtime without installing it system-wide"
    pkgutil --expand-full "${installer}" "${PYTHON_RUNTIME}"

    # Python.org's framework components normally reference /Library/Frameworks.
    # Rewrite all internal references to the extracted framework itself.
    /usr/bin/python3 "${ROOT}/scripts/relocate_python_framework.py" \
        "${PYTHON_FRAMEWORK}"
}

if [[ "${1:-}" == "--prepare-only" ]]; then
    PREPARE_ONLY=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--prepare-only]" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "ProbeFlow's first macOS build requires an arm64 Mac." >&2
    exit 1
fi

if [[ -n "${PROBEFLOW_BUILD_PYTHON:-}" ]]; then
    PYTHON="${PROBEFLOW_BUILD_PYTHON}"
else
    provision_python
    PYTHON="${DEFAULT_PYTHON}"
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python ${PYTHON_VERSION} was not found at ${PYTHON}." >&2
    echo "Set PROBEFLOW_BUILD_PYTHON to an arm64 Python ${PYTHON_VERSION} executable." >&2
    exit 1
fi

"${PYTHON}" - <<'PY'
import platform
import sys

if sys.version_info[:3] != (3, 13, 14):
    raise SystemExit(f"Expected Python 3.13.14, found {sys.version.split()[0]}")
if platform.machine() != "arm64":
    raise SystemExit(f"Expected an arm64 interpreter, found {platform.machine()}")
PY

echo "Creating clean build environment: ${VENV}"
"${PYTHON}" -m venv --clear "${VENV}"
"${VENV}/bin/python" -m pip install \
    --constraint "${ROOT}/packaging/macos/constraints-arm64.txt" \
    --requirement "${ROOT}/packaging/macos/requirements-build.txt"
"${VENV}/bin/python" -m pip install \
    --constraint "${ROOT}/packaging/macos/constraints-arm64.txt" \
    "${ROOT}[desktop]"

"${VENV}/bin/python" - <<'PY'
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
PY

if [[ ${PREPARE_ONLY} -eq 1 ]]; then
    echo "Build environment is ready; application build skipped."
    exit 0
fi

echo "Building ProbeFlow.app"
export PYINSTALLER_CONFIG_DIR="${BUILD_ROOT}/pyinstaller-cache"
export MPLCONFIGDIR="${BUILD_ROOT}/matplotlib-cache"
export XDG_CACHE_HOME="${BUILD_ROOT}/cache"
"${VENV}/bin/pyinstaller" \
    --clean \
    --noconfirm \
    --distpath "${BUILD_ROOT}/dist" \
    --workpath "${BUILD_ROOT}/work" \
    "${ROOT}/packaging/macos/ProbeFlow.spec"

APP="${BUILD_ROOT}/dist/ProbeFlow.app"
"${VENV}/bin/python" "${ROOT}/scripts/validate_macos_app.py" "${APP}"

SMOKE_HOME="${BUILD_ROOT}/smoke-test-home"
mkdir -p "${SMOKE_HOME}" "${BUILD_ROOT}/smoke-test-matplotlib"
HOME="${SMOKE_HOME}" \
MPLCONFIGDIR="${BUILD_ROOT}/smoke-test-matplotlib" \
QT_QPA_PLATFORM=offscreen \
    "${APP}/Contents/MacOS/ProbeFlow" --smoke-test

echo "Built and validated ${APP}"
