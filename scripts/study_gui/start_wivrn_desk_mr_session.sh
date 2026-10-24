#!/usr/bin/env bash
# WiVRn session helper for the SEPARATE Desk MR passthrough trial (ADF 16).
# Does not configure or launch the OpenXR eyepiece path.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Reuse the same WiVRn bring-up as eyepiece (dashboard + optional adb reverse).
bash "${ROOT}/scripts/study_gui/start_wivrn_eyepiece_session.sh"

echo
echo "=== Desk MR trial ==="
echo "In the study GUI, check ONLY:  Desk MR passthrough (trial)"
echo "Leave OpenXR eyepiece and OpenVR unchecked."
echo "Docs: ${ROOT}/docs/QUEST_DESK_MR.md"
echo
echo "If libambf_openxr_desk_mr.so is missing:"
echo "  bash ${ROOT}/scripts/build_openxr_desk_mr.sh"
