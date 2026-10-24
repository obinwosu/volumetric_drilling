#!/usr/bin/env bash
# Build AMBF-ORG/ambf_openvr_plugin against the local AMBF3 colcon install.
set -euo pipefail

WS_ROOT="${ROS_AMBF_WS:-${HOME}/ros_ambf_ws}"
PLUGIN_SRC="${WS_ROOT}/src/ambf_openvr_plugin"

if [[ ! -d "${PLUGIN_SRC}/.git" && ! -f "${PLUGIN_SRC}/CMakeLists.txt" ]]; then
  echo "Cloning ambf_openvr_plugin..."
  git clone https://github.com/AMBF-ORG/ambf_openvr_plugin.git "${PLUGIN_SRC}"
fi

if ! pkg-config --exists openvr; then
  echo "OpenVR SDK missing. Install with:"
  echo "  sudo apt-get install -y libopenvr-dev libopenvr-api1t64"
  exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"
set -u

export CMAKE_PREFIX_PATH="${WS_ROOT}/install/AMBF:${CMAKE_PREFIX_PATH:-}"

mkdir -p "${PLUGIN_SRC}/build"
cd "${PLUGIN_SRC}/build"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

# Keep colcon from treating this standalone tree as a workspace package.
touch "${PLUGIN_SRC}/COLCON_IGNORE"

echo
echo "Built: ${PLUGIN_SRC}/build/libambf_openvr_camera.so"
echo "Enable OpenVR HMD in the study GUI, or launch with -l 0,7,14"
echo "See docs/QUEST_OPENVR_SETUP.md for ALVR + SteamVR."
