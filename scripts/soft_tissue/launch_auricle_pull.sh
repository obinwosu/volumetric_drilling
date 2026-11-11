#!/usr/bin/env bash
# Launcher for the auricle pull viewer (scripts/soft_tissue/auricle_pull_viewer.py).
# Ctrl + left-drag on the ear to pinch and pull it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/middle_ear_newton/.venv/bin/python"
LOG_DIR="${HOME}/.cache/fivrs-auricle-pull"
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
exec "${PYTHON}" "${SCRIPT_DIR}/auricle_pull_viewer.py" --pick "$@" >"${LOG_FILE}" 2>&1
