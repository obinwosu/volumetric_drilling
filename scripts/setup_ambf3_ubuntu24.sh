#!/usr/bin/env bash
# Idempotent A51 setup: NVIDIA check, ROS 2 Jazzy, AMBF 3.0, volumetric_drilling plugin.
# Usage: bash scripts/setup_ambf3_ubuntu24.sh
set -euo pipefail

WS_ROOT="${ROS_AMBF_WS:-${HOME}/ros_ambf_ws}"
AMBF_BRANCH="${AMBF_BRANCH:-ambf-3.0}"
VD_DIR="${WS_ROOT}/src/volumetric_drilling"
AMBF_DIR="${WS_ROOT}/src/ambf"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need_sudo() {
  if [[ "$(id -u)" -eq 0 ]]; then
    return 0
  fi
  if sudo -n true 2>/dev/null; then
    return 0
  fi
  die "sudo is required (passwordless or an interactive terminal). Run: sudo -v && re-run this script."
}

log "Checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. Install Ubuntu 24.04 proprietary NVIDIA drivers for the RTX 5090, reboot, then re-run."
fi
nvidia-smi | head -n 15

need_sudo

log "Installing base packages (git, build tools, GUI deps)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  git gh curl ca-certificates build-essential cmake pkg-config \
  libasound2-dev libgl1-mesa-dev xorg-dev libusb-1.0-0-dev \
  libx11-dev libxi-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  zlib1g-dev python3-pip python3-yaml python3-pyqt5 python3-numpy python3-zmq \
  python3-natsort xdotool software-properties-common locales

log "Ensuring ROS 2 Jazzy"
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  sudo apt-get install -y -qq locales
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8
  sudo apt-get install -y -qq curl gnupg lsb-release
  if [[ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
  fi
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init || true
  fi
  rosdep update || true
else
  log "ROS 2 Jazzy already present"
fi

# ROS setup scripts reference optional unset vars; disable nounset while sourcing.
set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u

mkdir -p "${WS_ROOT}/src"

log "Cloning / updating AMBF ${AMBF_BRANCH}"
if [[ ! -d "${AMBF_DIR}/.git" ]]; then
  git clone --branch "${AMBF_BRANCH}" --single-branch https://github.com/WPI-AIM/ambf.git "${AMBF_DIR}"
else
  git -C "${AMBF_DIR}" fetch origin "${AMBF_BRANCH}" || true
  git -C "${AMBF_DIR}" checkout "${AMBF_BRANCH}" || true
fi

if [[ ! -d "${VD_DIR}" ]]; then
  die "volumetric_drilling not found at ${VD_DIR}. Place the ported tree there first."
fi

# Keep only ambf + one volumetric_drilling tree under src/ (no zip extracts / forks).
find "${WS_ROOT}/src" -maxdepth 1 -type d \( -name 'volumetric_drilling-*' -o -name 'volumetric_drilling_VRE*' \) \
  ! -path "${VD_DIR}" -exec rm -rf {} + 2>/dev/null || true
find "${WS_ROOT}/src" -maxdepth 1 -type f -name '*.zip' -delete 2>/dev/null || true

log "rosdep install (best effort)"
cd "${WS_ROOT}"
set +e
rosdep install --from-paths src --ignore-src -r -y
set -e

log "colcon build (Release)"
cd "${WS_ROOT}"
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

set +u
# shellcheck disable=SC1091
source "${WS_ROOT}/install/setup.bash"
set -u

log "Patching gui_setup.yaml for this machine"
GUI_YAML="${VD_DIR}/scripts/study_gui/gui_setup.yaml"
if [[ -f "${GUI_YAML}" ]]; then
  python3 - <<PY
from pathlib import Path
import yaml, os, glob
p = Path("${GUI_YAML}")
data = yaml.safe_load(p.read_text())
# Prefer installed ambf_simulator on PATH after sourcing install/setup.bash
data["ambf_executable"] = "ambf_simulator"
data["launch_file"] = str((Path("${VD_DIR}") / "launch.yaml").resolve())
data["recording_base_path"] = str(Path.home() / "Volumetric_Drilling_Userstudy")
data["recording_script_executable"] = str((Path("${VD_DIR}") / "scripts" / "data_record.py").resolve())
# Footpedal: VOXEL-MAN joystick, else PCsensor FootSwitch, else js0/js1
candidates = (
    sorted(glob.glob("/dev/input/by-id/*Pedal*joystick"))
    + sorted(glob.glob("/dev/input/by-id/*FootSwitch*event-kbd"))
    + ["/dev/input/js0", "/dev/input/js1"]
)
for c in candidates:
    if Path(c).exists():
        data["footpedal_device"] = c
        break
p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
print("Updated", p)
print("  ambf_executable:", data["ambf_executable"])
print("  footpedal_device:", data.get("footpedal_device"))
print("  launch_file:", data["launch_file"])
PY
fi

log "Footpedal /dev/input permissions"
UDEV_SRC="${VD_DIR}/scripts/udev/99-volumetric-drilling-input.rules"
if [[ -f "${UDEV_SRC}" ]]; then
  cat <<EOF
Install footpedal udev rules (one-time, requires sudo + re-login):

  sudo cp ${UDEV_SRC} /etc/udev/rules.d/
  sudo usermod -aG input "\$USER"
  sudo udevadm control --reload-rules && sudo udevadm trigger
  # then log out and back in
EOF
fi

log "Smoke hints"
cat <<EOF
Source the workspace, then launch the study GUI:

  source /opt/ros/jazzy/setup.bash
  source ${WS_ROOT}/install/setup.bash
  cd ${VD_DIR}/scripts/study_gui
  python3 study_gui.py

Select Anatomy A (RT143), start simulation, then check:
  - camera nav / pause menu (Ctrl+P)
  - DVR / beam lighting
  - footpedal hold-to-run drill (middle pedal) — HUD shows DRILL ON
  - camera clutch via hold key 1 (SayoDevice)
  - video record checkbox
  - volume list
EOF

log "Setup finished"
