#!/usr/bin/env bash
# Builds the standalone Touch reader used by the haptic monitor when the simulator is closed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/bin"
g++ -std=c++17 -O2 -Wall -o "${HERE}/bin/haptic_probe" "${HERE}/haptic_probe.cpp" -ldl -lpthread
echo "built ${HERE}/bin/haptic_probe"
