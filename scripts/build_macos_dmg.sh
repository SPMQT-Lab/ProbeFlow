#!/usr/bin/env bash
# Build and verify the Apple Silicon ProbeFlow drag-install DMG.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PROBEFLOW_BUILD_ROOT:-${ROOT}/build/macos}"
DIST_DIR="${BUILD_ROOT}/dist"
APP="${DIST_DIR}/ProbeFlow.app"
STAGING_DIR="${BUILD_ROOT}/dmg-staging"
SKIP_APP_BUILD=0

if [[ "${1:-}" == "--skip-app-build" ]]; then
    SKIP_APP_BUILD=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--skip-app-build]" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "ProbeFlow's first DMG build requires an arm64 Mac." >&2
    exit 1
fi

VERSION="$(
    awk -F'"' '/^__version__ = / {print $2; exit}' \
        "${ROOT}/probeflow/__init__.py"
)"
if [[ -z "${VERSION}" ]]; then
    echo "Could not read ProbeFlow version." >&2
    exit 1
fi

ARTIFACT_VERSION="${VERSION/rc/-rc}"
DISPLAY_VERSION="${VERSION/rc/ RC }"
DMG_NAME="ProbeFlow-${ARTIFACT_VERSION}-macOS-arm64.dmg"
DMG="${DIST_DIR}/${DMG_NAME}"
CHECKSUM="${DMG}.sha256"

if [[ ${SKIP_APP_BUILD} -eq 0 ]]; then
    "${ROOT}/scripts/build_macos_app.sh"
elif [[ ! -d "${APP}" ]]; then
    echo "No application bundle found at ${APP}." >&2
    echo "Run without --skip-app-build to create it." >&2
    exit 1
fi

echo "Preparing drag-install layout"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}" "${DIST_DIR}"
/usr/bin/ditto "${APP}" "${STAGING_DIR}/ProbeFlow.app"
ln -s /Applications "${STAGING_DIR}/Applications"

rm -f "${DMG}" "${CHECKSUM}"
echo "Creating ${DMG_NAME}"
/usr/bin/hdiutil create \
    -volname "ProbeFlow ${DISPLAY_VERSION}" \
    -srcfolder "${STAGING_DIR}" \
    -fs HFS+ \
    -format UDZO \
    -imagekey zlib-level=9 \
    -ov \
    "${DMG}"

if [[ -n "${PROBEFLOW_CODESIGN_IDENTITY:-}" ]]; then
    echo "Signing ${DMG_NAME} with ${PROBEFLOW_CODESIGN_IDENTITY}"
    /usr/bin/codesign \
        --force \
        --timestamp \
        --sign "${PROBEFLOW_CODESIGN_IDENTITY}" \
        "${DMG}"
    /usr/bin/codesign --verify --strict --verbose=2 "${DMG}"
fi

PROBEFLOW_VALIDATION_PYTHON="${BUILD_ROOT}/venv/bin/python" \
    "${ROOT}/scripts/validate_macos_dmg.sh" "${DMG}"

(
    cd "${DIST_DIR}"
    /usr/bin/shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256"
)

echo "Built and verified ${DMG}"
echo "SHA-256: $(awk '{print $1}' "${CHECKSUM}")"
