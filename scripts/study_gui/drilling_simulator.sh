#!/usr/bin/env bash
# AMBF3 / ROS 2 launcher for the study GUI (no roscore).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/launch_study_gui.sh"
