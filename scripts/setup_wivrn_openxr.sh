#!/usr/bin/env bash
# Install OpenXR build deps + WiVRn (Flatpak) for the AMBF eyepiece pipeline.
set -euo pipefail

echo "==> OpenXR SDK (headers + loader)"
sudo apt-get update
sudo apt-get install -y libopenxr-dev libopenxr-loader1 libglx-dev libx11-dev adb flatpak

echo "==> Flathub + WiVRn"
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install -y flathub io.github.wivrn.wivrn

echo
echo "WiVRn installed. Next on this PC:"
echo "  flatpak run io.github.wivrn.wivrn"
echo "  (first-run wizard → install Quest client from Meta Store or APK)"
echo
echo "Build the AMBF plugin:"
echo "  bash $(dirname "$0")/build_openxr_eyepiece.sh"
echo
echo "Full steps: docs/QUEST_OPENXR_EYEPIECE.md"
