#!/usr/bin/env bash
# Menu launcher for the FIVRS RTX Tuner. The simulator is a separate app.
# With no --anatomy-adf, a picker asks for anatomy and view first.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/middle_ear_newton/.venv/bin/python"
DIALOG_PYTHON="${DIALOG_PYTHON:-python3}"
LOG_DIR="${HOME}/.cache/fivrs-rtx-tuner"
LOG_FILE="${LOG_DIR}/tuner.log"

export PYTHONUNBUFFERED=1
export DISPLAY="${DISPLAY:-:1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
unset __NV_PRIME_RENDER_OFFLOAD __VK_LAYER_NV_optimus 2>/dev/null || true
unset PYTHONPATH PYTHONHOME 2>/dev/null || true
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH QT_QPA_PLATFORMTHEME QTDIR 2>/dev/null || true

if [[ ! -x "${PYTHON}" ]]; then
    echo "Newton environment is missing: ${PYTHON}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

has_anatomy=0
for arg in "$@"; do
    if [[ "${arg}" == "--anatomy-adf" ]]; then
        has_anatomy=1
        break
    fi
done

cd "${REPO_ROOT}"
export FIVRS_SPLASH_ACTIVE=1
export PYTHONUNBUFFERED=1

if [[ "${has_anatomy}" -eq 0 ]]; then
    exec "${DIALOG_PYTHON}" "${SCRIPT_DIR}/tuner_launch_dialog.py" --launch
fi

mapfile -t running < <(ps -eo pid,args | awk '$2 ~ /(^|\/)python[0-9.]*$/ && index($0, "fivrs_rtx_tuner.py") { print $1 }')
if ((${#running[@]})); then
    kill "${running[@]}" >>"${LOG_FILE}" 2>&1 || true
    sleep 0.4
fi

if [[ -n "${FIVRS_TUNER_FOREGROUND:-}" ]]; then
    exec "${PYTHON}" "${SCRIPT_DIR}/fivrs_rtx_tuner.py" "$@"
fi
exec "${DIALOG_PYTHON}" "${SCRIPT_DIR}/rtx_splash.py" \
    --heading "Opening RTX Tuner" \
    --ready "Tuner ready" \
    --log "${LOG_FILE}" \
    -- "${PYTHON}" "${SCRIPT_DIR}/fivrs_rtx_tuner.py" "$@"
