#!/usr/bin/env bash
# Start ALVR streamer (and optionally SteamVR), then the volumetric study GUI.
# Meta Quest connects through ALVR → SteamVR → ambf_openvr_plugin.
set -euo pipefail

ALVR_ROOT="${ALVR_ROOT:-${HOME}/Downloads/alvr_streamer_linux}"
ALVR_BIN="${ALVR_ROOT}/bin/alvr_dashboard"
WS_ROOT="${ROS_AMBF_WS:-${HOME}/ros_ambf_ws}"
STUDY_GUI="${WS_ROOT}/src/volumetric_drilling/scripts/study_gui/launch_study_gui.sh"
STEAMVR_DIR="${HOME}/.local/share/Steam/steamapps/common/SteamVR"

if [[ ! -x "${ALVR_BIN}" ]]; then
  echo "ALVR dashboard not found at: ${ALVR_BIN}"
  echo "Set ALVR_ROOT to your alvr_streamer_linux extract, e.g.:"
  echo "  export ALVR_ROOT=${HOME}/Downloads/alvr_streamer_linux"
  exit 1
fi

# ALVR needs its lib64 on LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${ALVR_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Hybrid GPU (Intel iGPU + NVIDIA dGPU): force SteamVR / ALVR / AMBF onto NVIDIA.
# Without this, tracking often works while the Quest stays black (wrong encoder / compositor GPU).
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
export __VK_LAYER_NV_optimus=NVIDIA_only
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
# Prefer NVIDIA VA-API if nvidia-vaapi-driver is installed; else ALVR uses NVENC via Vulkan.
export LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME:-nvidia}"
export NVD_BACKEND="${NVD_BACKEND:-direct}"

# Start Steam if needed (SteamVR lives in the Steam library)
if ! pgrep -x steam >/dev/null 2>&1; then
  echo "Starting Steam..."
  nohup env \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __NV_PRIME_RENDER_OFFLOAD=1 \
    VK_DRIVER_FILES="${VK_DRIVER_FILES}" \
    steam -silent >/tmp/steam_vr_session.log 2>&1 &
  sleep 5
fi

# Persist SteamVR NVIDIA launch options (ALVR requirement on hybrid NVIDIA laptops/desktops)
LOCAL_VDF="${HOME}/.local/share/Steam/userdata"
if [[ -d "${LOCAL_VDF}" ]]; then
  python3 - <<'PY' || true
from pathlib import Path
import os
home = Path.home()
roots = list((home/".local/share/Steam/userdata").glob("*/config/localconfig.vdf"))
# ALVR Linux mandatory: wrap with vrmonitor.sh or the HMD stays black even
# when poses/presents look fine. Keep NVIDIA env for hybrid GPUs.
launch = (
    '__GLX_VENDOR_LIBRARY_NAME=nvidia __NV_PRIME_RENDER_OFFLOAD=1 '
    '__VK_LAYER_NV_optimus=NVIDIA_only '
    'VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json '
    f'{home}/.local/share/Steam/steamapps/common/SteamVR/bin/vrmonitor.sh %command%'
)
import re
for p in roots:
    text = p.read_text(errors="replace")
    idx = text.find('"250820"')
    if idx < 0:
        continue
    sub = text[idx:idx+800]
    m = re.search(r'"LaunchOptions"\s*"([^"]*)"', sub)
    if m:
        new = f'"LaunchOptions"\t\t"{launch}"'
        if m.group(0) == new or (m.group(1) == launch):
            print(f"SteamVR launch options OK in {p}")
            continue
        text = text[:idx] + sub.replace(m.group(0), new, 1) + text[idx+800:]
        p.write_text(text)
        print(f"Updated SteamVR LaunchOptions in {p}")
        continue
    needle = '"250820"\n\t\t\t\t\t{\n\t\t\t\t\t\t"LastPlayed"'
    insert = (
        '"250820"\n\t\t\t\t\t{\n'
        f'\t\t\t\t\t\t"LaunchOptions"\t\t"{launch}"\n'
        '\t\t\t\t\t\t"LastPlayed"'
    )
    if needle in text:
        p.write_text(text.replace(needle, insert, 1))
        print(f"Set SteamVR LaunchOptions in {p}")
    else:
        print(f"Could not auto-set LaunchOptions in {p}; set manually in Steam → SteamVR → Properties")
PY
fi

# Keep SteamVR Home from stealing the scene (causes menu overlay in the headset)
VRSETTINGS="${HOME}/.local/share/Steam/config/steamvr.vrsettings"
if [[ -f "${VRSETTINGS}" ]]; then
  python3 - <<'PY' || true
import json
from pathlib import Path
p = Path.home()/".local/share/Steam/config/steamvr.vrsettings"
d = json.loads(p.read_text())
sv = d.setdefault("steamvr", {})
changed = False
# Home overlays the sim; keep it off. Do NOT disable the SteamVR dashboard —
# that triggers SteamVR Fail (-203 / Compositor_OverlayInitFailed) and a black HMD.
if sv.get("enableHomeApp", True) is not False:
    sv["enableHomeApp"] = False
    changed = True
dash = d.setdefault("dashboard", {})
if dash.get("enableDashboard") is False:
    dash["enableDashboard"] = True
    changed = True
# Locked microscope view: async reprojection late-warps the framebuffer when the
# head moves (delayed "app slide"). Prefer it off for this sim.
if sv.get("disableAsync") is not True:
    sv["disableAsync"] = True
    changed = True
if sv.get("enableLinuxVulkanAsync") is not False:
    sv["enableLinuxVulkanAsync"] = False
    changed = True
if changed:
    p.write_text(json.dumps(d, indent=3) + "\n")
    print("Updated steamvr.vrsettings (Home off, dashboard on, async reprojection off)")
PY
fi

# Stub SteamVR Home launcher if present (enableHomeApp=false is not always enough)
TOURS_SH="${STEAMVR_DIR}/tools/steamvr_environments/game/steamtours.sh"
TOURS_BAK="${TOURS_SH}.disabled_by_ambf"
if [[ -x "${TOURS_SH}" && ! -f "${TOURS_BAK}" ]]; then
  if ! grep -q 'disabled for volumetric drilling' "${TOURS_SH}" 2>/dev/null; then
    mv "${TOURS_SH}" "${TOURS_BAK}"
    printf '%s\n' '#!/bin/sh' 'echo "SteamVR Home disabled for volumetric drilling" >&2' 'exit 0' > "${TOURS_SH}"
    chmod +x "${TOURS_SH}"
    echo "Disabled SteamVR Home launcher (backup: ${TOURS_BAK})"
  fi
fi

# Start SteamVR if installed
if [[ -d "${STEAMVR_DIR}" ]]; then
  if ! pgrep -af 'vrserver|vrmonitor|steamvr' | grep -vi grep >/dev/null 2>&1; then
    echo "Starting SteamVR (NVIDIA)..."
    nohup env \
      __GLX_VENDOR_LIBRARY_NAME=nvidia \
      __NV_PRIME_RENDER_OFFLOAD=1 \
      __VK_LAYER_NV_optimus=NVIDIA_only \
      VK_DRIVER_FILES="${VK_DRIVER_FILES}" \
      steam -applaunch 250820 >/tmp/steamvr_session.log 2>&1 &
    sleep 4
  fi
else
  echo "WARNING: SteamVR not found under ${STEAMVR_DIR}"
  echo "Install it from Steam → Library → Tools → SteamVR (app 250820)."
  echo "Opening SteamVR install page..."
  steam steam://install/250820 >/dev/null 2>&1 || true
fi

# Start ALVR dashboard if not already running
if ! pgrep -af 'alvr_dashboard' | grep -vi grep >/dev/null 2>&1; then
  echo "Starting ALVR dashboard..."
  nohup env \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __NV_PRIME_RENDER_OFFLOAD=1 \
    __VK_LAYER_NV_optimus=NVIDIA_only \
    VK_DRIVER_FILES="${VK_DRIVER_FILES}" \
    LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME}" \
    NVD_BACKEND="${NVD_BACKEND}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    "${ALVR_BIN}" >/tmp/alvr_dashboard.log 2>&1 &
  sleep 2
else
  echo "ALVR dashboard already running."
fi

echo
echo "On the Quest: open ALVR and connect to this PC."
echo "When SteamVR shows the headset Ready, start the study GUI with OpenVR HMD checked."
echo "If the headset is black but the PC view tracks your head: SteamVR must use the NVIDIA launch options"
echo "  (already applied if this script ran), then fully restart SteamVR + ALVR and reconnect."
echo

if [[ "${1:-}" == "--gui" || "${1:-}" == "-g" ]]; then
  exec "${STUDY_GUI}"
fi

echo "ALVR is up. Launch study GUI when ready:"
echo "  ${STUDY_GUI}"
echo "Or re-run: $0 --gui"
