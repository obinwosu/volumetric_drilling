#!/usr/bin/env bash
# Start WiVRn (and optional wired Quest reverse) for the OpenXR eyepiece pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if ! flatpak info io.github.wivrn.wivrn >/dev/null 2>&1; then
  echo "WiVRn Flatpak not installed. Run:"
  echo "  bash ${ROOT}/scripts/setup_wivrn_openxr.sh"
  exit 1
fi

# Flatpak only exposes NVIDIA when this extension matches the host driver.
# Otherwise WiVRn starts on the Intel GPU and the eyepiece sim crashes.
if [[ -r /proc/driver/nvidia/version ]]; then
  DRIVER_VER="$(awk 'match($0, /[0-9]+\.[0-9]+\.[0-9]+/) { print substr($0, RSTART, RLENGTH); exit }' /proc/driver/nvidia/version)"
  if [[ -n "${DRIVER_VER}" ]]; then
    EXT_ID="org.freedesktop.Platform.GL.nvidia-${DRIVER_VER//./-}"
    if ! flatpak info "${EXT_ID}" >/dev/null 2>&1; then
      echo "ERROR! Host NVIDIA driver is ${DRIVER_VER}, but ${EXT_ID} is not installed."
      echo "WiVRn will start on the Intel GPU, the sim will crash, and the Quest stays blank."
      echo "Install the matching runtime, then run this script again:"
      echo "  flatpak install -y flathub ${EXT_ID}"
      exit 1
    fi
  fi
fi

# WiVRn writes a runtime manifest whose library_path is relative to
# ~/.config/openxr/1/. That resolves to ~/lib/wivrn on the host.
WIVRN_LIB="/var/lib/flatpak/app/io.github.wivrn.wivrn/current/active/files/lib/wivrn"
if [[ -e "${WIVRN_LIB}/libopenxr_wivrn.so" ]]; then
  mkdir -p "${HOME}/lib"
  ln -sfn "${WIVRN_LIB}" "${HOME}/lib/wivrn"
fi

echo "Starting WiVRn dashboard..."
flatpak run io.github.wivrn.wivrn >/tmp/wivrn_dashboard.log 2>&1 &
sleep 2

if command -v adb >/dev/null 2>&1; then
  if adb devices 2>/dev/null | grep -qE 'device$'; then
    echo "USB device present — setting adb reverse for wired WiVRn..."
    adb reverse tcp:9757 tcp:9757 || true
    # Meta Store package; ignore failure if sideloaded under another id.
    adb shell am start -a android.intent.action.VIEW -d "wivrn+tcp://localhost" org.meumeu.wivrn 2>/dev/null || true
  else
    echo "No adb device — connect Quest over Wi‑Fi in the WiVRn app, or plug USB + enable Dev Mode."
  fi
fi

# Prefer WiVRn active runtime if the dashboard wrote one.
if [[ -f "${HOME}/.config/openxr/1/active_runtime.json" ]]; then
  export XR_RUNTIME_JSON="${HOME}/.config/openxr/1/active_runtime.json"
  echo "XR_RUNTIME_JSON=${XR_RUNTIME_JSON}"
  cat "${XR_RUNTIME_JSON}" || true
fi

echo
echo "On Quest: WiVRn → Connection ready."
echo "Then launch study GUI with OpenXR eyepiece checked (not OpenVR/SteamVR)."
echo "  bash ${ROOT}/scripts/study_gui/launch_study_gui.sh"
echo
echo "Docs: ${ROOT}/docs/QUEST_OPENXR_EYEPIECE.md"
