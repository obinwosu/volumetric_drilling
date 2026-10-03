#!/usr/bin/env bash
# Launch the study GUI on Ubuntu 24.04 / ROS 2 Jazzy / AMBF 3.
# Working directory must be scripts/study_gui (gui_setup.yaml paths are relative).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WS_ROOT="${ROS_AMBF_WS:-${HOME}/ros_ambf_ws}"

set +u
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
elif [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi
if [[ -f "${WS_ROOT}/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "${WS_ROOT}/install/setup.bash"
fi
set -u

cd "${SCRIPT_DIR}"

# ROS 2 does not use roscore. Ensure ambf_simulator is on PATH when installed via colcon.
if ! command -v ambf_simulator >/dev/null 2>&1; then
    if [[ -x "${WS_ROOT}/install/ambf_simulator/lib/ambf_simulator/ambf_simulator" ]]; then
        export PATH="${WS_ROOT}/install/ambf_simulator/lib/ambf_simulator:${PATH}"
    elif [[ -x "${HOME}/ros_ambf_ws/src/ambf/bin/lin-x86_64/ambf_simulator" ]]; then
        export PATH="${HOME}/ros_ambf_ws/src/ambf/bin/lin-x86_64:${PATH}"
    fi
fi

exec python3 "${SCRIPT_DIR}/study_gui.py"
