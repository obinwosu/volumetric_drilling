#!/usr/bin/env bash
# Build ambf_openxr_eyepiece against local AMBF3 + system OpenXR loader.
set -euo pipefail

WS_ROOT="${ROS_AMBF_WS:-${HOME}/ros_ambf_ws}"
PLUGIN_SRC="${WS_ROOT}/src/ambf_openxr_eyepiece"

if [[ ! -f "${PLUGIN_SRC}/CMakeLists.txt" ]]; then
  echo "ERROR: missing ${PLUGIN_SRC}"
  exit 1
fi

if ! pkg-config --exists openxr; then
  echo "OpenXR SDK missing. Install with:"
  echo "  sudo apt-get install -y libopenxr-dev libopenxr-loader1"
  echo "Or: bash $(dirname "$0")/setup_wivrn_openxr.sh"
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

touch "${PLUGIN_SRC}/COLCON_IGNORE"

echo
echo "Built: ${PLUGIN_SRC}/build/libambf_openxr_eyepiece.so"
echo "See docs/QUEST_OPENXR_EYEPIECE.md"
