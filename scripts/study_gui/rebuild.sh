#!/usr/bin/env bash
# Rebuild volumetric_drilling msgs + plugin in the ROS 2 / AMBF3 workspace.
# Use:  source ./rebuild.sh   OR   ./rebuild.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WS_ROOT="${ROS_AMBF_WS:-${HOME}/ros_ambf_ws}"

set +u
if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
fi
set -u

cd "${WS_ROOT}"
colcon build --packages-select volumetric_drilling_msgs volumetric_plugin \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"
cd "${SCRIPT_DIR}"
echo "Rebuild complete. cwd=${SCRIPT_DIR}"
