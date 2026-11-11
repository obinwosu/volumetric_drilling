#!/usr/bin/env bash
# Menu launcher for the ear placement viewer (scripts/soft_tissue/placement_viewer.py).
# Extra arguments pass through, e.g. an ADF or --ear.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/middle_ear_newton/.venv/bin/python"
LOG_DIR="${HOME}/.cache/fivrs-placement-viewer"
LOG_FILE="${LOG_DIR}/viewer.log"

export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
unset __NV_PRIME_RENDER_OFFLOAD __VK_LAYER_NV_optimus 2>/dev/null || true
unset PYTHONPATH PYTHONHOME 2>/dev/null || true

if [[ ! -x "${PYTHON}" ]]; then
    echo "Newton environment is missing: ${PYTHON}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/placement_viewer.py" "$@" >"${LOG_FILE}" 2>&1
