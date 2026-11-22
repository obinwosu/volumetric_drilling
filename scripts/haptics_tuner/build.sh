#!/usr/bin/env bash
# Builds this branch's plugin into .haptics/, so the main simulator's install is not touched.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS="${ROS_AMBF_WS:-$HOME/ros_ambf_ws}"
set +u
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
cd "$REPO"
export COLCON_LOG_PATH="$REPO/.haptics/log"
colcon build --base-paths "$REPO" \
  --build-base "$REPO/.haptics/build" --install-base "$REPO/.haptics/install" \
  --packages-select volumetric_drilling_msgs volumetric_plugin \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
