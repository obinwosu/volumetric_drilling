#!/usr/bin/env bash
# Environment for 3D Systems / Geomagic Touch HID with OpenHaptics + chai3d (AMBF).
# Sourced by launch_study_gui.sh; safe to source in any shell before ambf_simulator.
#
# The volumetric_drilling plugin opens the Touch via chai3d (device index 0).
# Keep ambf_simulator -n/--ndevs at 0 so AMBF does not also claim the device.

export GTDD_HOME="${GTDD_HOME:-/usr/share/3DSystems}"
export OH_SDK_BASE="${OH_SDK_BASE:-/opt/OpenHaptics/Developer/3.4-0}"

# chai3d dlopens "libhdPhantom.so" by name; Ubuntu's ld cache indexes the
# odd SONAME "libhdPhantom.so." — put /usr/local/lib on LD_LIBRARY_PATH.
case ":${LD_LIBRARY_PATH:-}:" in
  *":/usr/local/lib:"*) ;;
  *) export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

# OpenHaptics libs live under /usr/lib after install; keep explicit for safety.
case ":${LD_LIBRARY_PATH:-}:" in
  *":/usr/lib:"*) ;;
  *) export LD_LIBRARY_PATH="/usr/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

# Profile snippets (if present) for interactive shells / future packages.
# shellcheck disable=SC1091
[[ -f /etc/profile.d/3ds-touch-drivers.sh ]] && source /etc/profile.d/3ds-touch-drivers.sh
# shellcheck disable=SC1091
[[ -f /etc/profile.d/openhaptics.sh ]] && source /etc/profile.d/openhaptics.sh

# Rebuilt libhdPhantom opens the second Touch as "Phantom2". It has to be
# found before the one-device copy in /usr/local/lib. Keep this last so
# profile snippets cannot put the old library ahead of it.
HAPTIC_LIB_DIR="${HOME}/.local/lib"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${HAPTIC_LIB_DIR}:"*) ;;
  *) export LD_LIBRARY_PATH="${HAPTIC_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
