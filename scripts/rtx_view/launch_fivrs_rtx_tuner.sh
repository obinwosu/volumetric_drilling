#!/usr/bin/env bash
# Menu launcher for the FIVRS RTX Tuner. The simulator is a separate app.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/middle_ear_newton/.venv/bin/python"
LOG_DIR="${HOME}/.cache/fivrs-rtx-tuner"
LOG_FILE="${LOG_DIR}/tuner.log"

export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
unset __NV_PRIME_RENDER_OFFLOAD __VK_LAYER_NV_optimus 2>/dev/null || true

if [[ ! -x "${PYTHON}" ]]; then
    echo "Newton environment is missing: ${PYTHON}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
mapfile -t running < <(ps -eo pid,args | awk '$2 ~ /(^|\/)python[0-9.]*$/ && index($0, "fivrs_rtx_tuner.py") { print $1 }')
if ((${#running[@]})); then
    echo "Tuner already open ${running[*]}" >>"${LOG_FILE}"
    exit 0
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/fivrs_rtx_tuner.py" >>"${LOG_FILE}" 2>&1
