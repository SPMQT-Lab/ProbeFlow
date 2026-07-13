#!/usr/bin/env bash
# Publish the unnotarized ProbeFlow DMG and required source assets as a prerelease.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PROBEFLOW_BUILD_ROOT:-${ROOT}/build/macos}"
DIST_DIR="${BUILD_ROOT}/dist"
QT_SOURCE_DIR="${BUILD_ROOT}/downloads/qt-source"
LICENSE_MANIFEST="${ROOT}/packaging/macos/runtime_licenses.toml"
NOTES="${ROOT}/packaging/macos/release_notes_1.0.0rc1.md"

VERSION="$(
    awk -F'"' '/^__version__ = / {print $2; exit}' \
        "${ROOT}/probeflow/__init__.py"
)"
ARTIFACT_VERSION="${VERSION/rc/-rc}"
DISPLAY_VERSION="${VERSION/rc/ RC }"
TAG="v${ARTIFACT_VERSION}"
DMG_NAME="ProbeFlow-${ARTIFACT_VERSION}-macOS-arm64.dmg"
DMG="${DIST_DIR}/${DMG_NAME}"

if [[ "${1:-}" != "--unsigned" || $# -ne 1 ]]; then
    echo "Usage: $0 --unsigned" >&2
    echo "The flag confirms that this release is not Developer ID-signed or notarized." >&2
    exit 2
fi

if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    echo "The repository must be clean before publishing a release." >&2
    exit 1
fi

git -C "${ROOT}" fetch origin main
LOCAL_HEAD="$(git -C "${ROOT}" rev-parse HEAD)"
REMOTE_HEAD="$(git -C "${ROOT}" rev-parse origin/main)"
if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
    echo "Local HEAD does not match origin/main." >&2
    exit 1
fi

gh auth status --hostname github.com >/dev/null
if gh release view "${TAG}" >/dev/null 2>&1; then
    echo "GitHub Release already exists: ${TAG}" >&2
    exit 1
fi

if [[ ! -f "${DMG}" || ! -f "${DMG}.sha256" ]]; then
    echo "DMG or checksum is missing under ${DIST_DIR}." >&2
    exit 1
fi

if /usr/bin/xcrun stapler validate "${DMG}" >/dev/null 2>&1; then
    echo "The DMG has a notarization ticket; review the unsigned release notes." >&2
    exit 1
fi

(
    cd "${DIST_DIR}"
    /usr/bin/shasum -a 256 -c "${DMG_NAME}.sha256"
)

PROBEFLOW_VALIDATION_PYTHON="${BUILD_ROOT}/venv/bin/python" \
    "${ROOT}/scripts/validate_macos_dmg.sh" "${DMG}"

ASSETS=("${DMG}" "${DMG}.sha256")
MANIFEST_PYTHON="${BUILD_ROOT}/venv/bin/python"
if [[ ! -x "${MANIFEST_PYTHON}" ]]; then
    echo "Release build environment is missing: ${MANIFEST_PYTHON}" >&2
    exit 1
fi
while IFS=$'\t' read -r expected_sha archive_name; do
    [[ -n "${archive_name}" ]] || continue
    archive="${QT_SOURCE_DIR}/${archive_name}"
    if [[ ! -f "${archive}" ]]; then
        echo "Required Qt corresponding-source archive is missing: ${archive}" >&2
        exit 1
    fi
    actual_sha="$(/usr/bin/shasum -a 256 "${archive}" | awk '{print $1}')"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
        echo "Qt source archive checksum mismatch: ${archive}" >&2
        exit 1
    fi
    ASSETS+=("${archive}")
done < <(
    "${MANIFEST_PYTHON}" - "${LICENSE_MANIFEST}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
for archive in config["qt_source_archives"]:
    print(f'{archive["sha256"]}\t{archive["filename"]}')
PY
)

ASSETS+=("${BUILD_ROOT}/licenses/qt/QT_CORRESPONDING_SOURCE.txt")
for asset in "${ASSETS[@]}"; do
    if [[ ! -f "${asset}" ]]; then
        echo "Release asset is missing: ${asset}" >&2
        exit 1
    fi
done

echo "Publishing unnotarized ${TAG} from ${LOCAL_HEAD}"
gh release create "${TAG}" \
    --target "${LOCAL_HEAD}" \
    --title "ProbeFlow ${DISPLAY_VERSION}" \
    --notes-file "${NOTES}" \
    --prerelease \
    "${ASSETS[@]}"

echo "Published GitHub prerelease ${TAG}"
