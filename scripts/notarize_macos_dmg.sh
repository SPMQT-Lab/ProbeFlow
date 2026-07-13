#!/usr/bin/env bash
# Build, Developer ID-sign, notarize and staple the ProbeFlow macOS DMG.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PROBEFLOW_BUILD_ROOT:-${ROOT}/build/macos}"
DIST_DIR="${BUILD_ROOT}/dist"
SKIP_BUILD=0

if [[ "${1:-}" == "--skip-build" ]]; then
    SKIP_BUILD=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--skip-build]" >&2
    exit 2
fi

if [[ -z "${PROBEFLOW_CODESIGN_IDENTITY:-}" ]]; then
    echo "Set PROBEFLOW_CODESIGN_IDENTITY to a Developer ID Application identity." >&2
    exit 1
fi
if [[ -z "${PROBEFLOW_NOTARY_PROFILE:-}" ]]; then
    echo "Set PROBEFLOW_NOTARY_PROFILE to a notarytool Keychain profile." >&2
    exit 1
fi

VERSION="$(
    awk -F'"' '/^__version__ = / {print $2; exit}' \
        "${ROOT}/probeflow/__init__.py"
)"
ARTIFACT_VERSION="${VERSION/rc/-rc}"
DMG_NAME="ProbeFlow-${ARTIFACT_VERSION}-macOS-arm64.dmg"
DMG="${DIST_DIR}/${DMG_NAME}"
CHECKSUM="${DMG}.sha256"

if [[ ${SKIP_BUILD} -eq 0 ]]; then
    "${ROOT}/scripts/build_macos_dmg.sh"
elif [[ ! -f "${DMG}" ]]; then
    echo "DMG not found: ${DMG}" >&2
    exit 1
fi

/usr/bin/codesign --verify --strict --verbose=2 "${DMG}"
echo "Submitting ${DMG_NAME} to Apple's notary service"
/usr/bin/xcrun notarytool submit \
    "${DMG}" \
    --keychain-profile "${PROBEFLOW_NOTARY_PROFILE}" \
    --wait

echo "Stapling and validating notarization ticket"
/usr/bin/xcrun stapler staple "${DMG}"
/usr/bin/xcrun stapler validate "${DMG}"
/usr/sbin/spctl \
    --assess \
    --type open \
    --context context:primary-signature \
    --verbose=2 \
    "${DMG}"
/usr/bin/hdiutil verify "${DMG}"

(
    cd "${DIST_DIR}"
    /usr/bin/shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256"
)

echo "Notarized ${DMG}"
echo "SHA-256: $(awk '{print $1}' "${CHECKSUM}")"
