#!/usr/bin/env bash
# Mount and validate a ProbeFlow DMG, including the app inside it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PROBEFLOW_BUILD_ROOT:-${ROOT}/build/macos}"
DMG="${1:-${BUILD_ROOT}/dist/ProbeFlow-1.0.0-rc1-macOS-arm64.dmg}"
MOUNT_POINT="${BUILD_ROOT}/dmg-verify-mount"
VALIDATION_PYTHON="${PROBEFLOW_VALIDATION_PYTHON:-python3}"
DEVICE=""

if [[ ! -f "${DMG}" ]]; then
    echo "DMG not found: ${DMG}" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${DEVICE}" ]]; then
        /usr/bin/hdiutil detach "${DEVICE}" >/dev/null 2>&1 || \
            /usr/bin/hdiutil detach -force "${DEVICE}" >/dev/null 2>&1 || true
    fi
    rm -rf "${MOUNT_POINT}"
}
trap cleanup EXIT

rm -rf "${MOUNT_POINT}"
mkdir -p "${MOUNT_POINT}"

/usr/bin/hdiutil verify "${DMG}"
ATTACH_OUTPUT="$(
    /usr/bin/hdiutil attach \
        -readonly \
        -nobrowse \
        -noautoopen \
        -mountpoint "${MOUNT_POINT}" \
        "${DMG}"
)"
DEVICE="$(printf '%s\n' "${ATTACH_OUTPUT}" | awk 'END {print $1}')"
if [[ -z "${DEVICE}" ]]; then
    echo "Could not determine mounted DMG device." >&2
    exit 1
fi

MOUNTED_APP="${MOUNT_POINT}/ProbeFlow.app"
if [[ ! -d "${MOUNTED_APP}" ]]; then
    echo "Mounted DMG does not contain ProbeFlow.app." >&2
    exit 1
fi
if [[ ! -L "${MOUNT_POINT}/Applications" ]]; then
    echo "Mounted DMG does not contain the Applications shortcut." >&2
    exit 1
fi
if [[ "$(readlink "${MOUNT_POINT}/Applications")" != "/Applications" ]]; then
    echo "Applications shortcut has the wrong target." >&2
    exit 1
fi

"${VALIDATION_PYTHON}" "${ROOT}/scripts/validate_macos_app.py" "${MOUNTED_APP}"

SMOKE_HOME="${BUILD_ROOT}/dmg-smoke-test-home"
SMOKE_MPL="${BUILD_ROOT}/dmg-smoke-test-matplotlib"
mkdir -p "${SMOKE_HOME}" "${SMOKE_MPL}"
HOME="${SMOKE_HOME}" \
MPLCONFIGDIR="${SMOKE_MPL}" \
QT_QPA_PLATFORM=offscreen \
    "${MOUNTED_APP}/Contents/MacOS/ProbeFlow" --smoke-test

echo "Verified $(basename "${DMG}"): readable disk image, drag-install layout, and packaged app smoke test passed"
