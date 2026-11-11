#!/usr/bin/env bash
# Launcher for the atlas placement tool (scripts/soft_tissue/atlas_placement.py).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/middle_ear_newton/.venv/bin/python"
LOG_DIR="${HOME}/.cache/fivrs-atlas-placement"
mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
unset PYTHONPATH PYTHONHOME 2>/dev/null || true
cd "${REPO_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/atlas_placement.py" "$@" >"${LOG_DIR}/atlas.log" 2>&1
