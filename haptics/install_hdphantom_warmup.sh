#!/usr/bin/env bash
# Apply, build and install the HID Touch warm-up fix for libhdPhantom.
# See haptics/README.md. Run as your normal user (not sudo).
#
#   haptics/install_hdphantom_warmup.sh [path to ambf repo]
#
# 1. applies ambf_hdPhantom_hid_warmup.patch to ambf (skipped if already in)
# 2. builds chai3d/extras/hdPhantom with make
# 3. backs up and installs ~/.local/lib/libhdPhantom.so (the copy the
#    launcher loads first, see scripts/setup_haptic_env.sh)
# 4. prints the sudo command for the /usr/local/lib fallback copy
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMBF="${1:-$(cd "$HERE/../../ambf" 2>/dev/null && pwd || true)}"
PATCH="$HERE/ambf_hdPhantom_hid_warmup.patch"
SRC_DIR="$AMBF/core/external/chai3d/extras/hdPhantom"
STAMP="$(date +%F)"

if [[ -z "$AMBF" || ! -f "$SRC_DIR/hdPhantom.cpp" ]]; then
    echo "ambf repo not found. Pass its path: $0 /path/to/ros_ambf_ws/src/ambf" >&2
    exit 1
fi

echo "== 1/4 patch ($AMBF)"
if grep -q FIVRS_HAPTIC_WARMUP "$SRC_DIR/hdPhantom.cpp"; then
    echo "   already applied"
elif git -C "$AMBF" apply --check "$PATCH" 2>/dev/null; then
    git -C "$AMBF" apply "$PATCH"
    echo "   applied (uncommitted in ambf; commit it there if you want: git -C $AMBF am $PATCH)"
else
    echo "   patch does not apply cleanly to $SRC_DIR/hdPhantom.cpp; apply it by hand" >&2
    exit 1
fi

echo "== 2/4 build"
make -C "$SRC_DIR" >/dev/null
LIB="$(ls -t "$SRC_DIR"/lib/release/lin-x86_64-*/libhdPhantom.so... | head -1)"
strings "$LIB" | grep -q FIVRS_HAPTIC_WARMUP || { echo "   built library has no warm-up: $LIB" >&2; exit 1; }
echo "   built $LIB"

echo "== 3/4 install ~/.local/lib/libhdPhantom.so"
mkdir -p "$HOME/.local/lib"
if [[ -f "$HOME/.local/lib/libhdPhantom.so" ]] && ! strings "$HOME/.local/lib/libhdPhantom.so" | grep -q FIVRS_HAPTIC_WARMUP; then
    cp -a "$HOME/.local/lib/libhdPhantom.so" "$HOME/.local/lib/libhdPhantom.so.bak-$STAMP"
    echo "   old copy backed up to ~/.local/lib/libhdPhantom.so.bak-$STAMP"
fi
install -m 0775 "$LIB" "$HOME/.local/lib/libhdPhantom.so"
echo "   installed"

echo "== 4/4 /usr/local/lib fallback (needs sudo; run it yourself in a terminal)"
if strings /usr/local/lib/libhdPhantom.so 2>/dev/null | grep -q FIVRS_HAPTIC_WARMUP; then
    echo "   already has the warm-up"
else
    echo "   sudo cp -a /usr/local/lib/libhdPhantom.so /usr/local/lib/libhdPhantom.so.bak-$STAMP && sudo install -m 0755 \"$LIB\" /usr/local/lib/libhdPhantom.so && sudo ldconfig"
fi

echo
echo "Check while the simulator runs (must be a file with the warm-up):"
echo "   grep libhdPhantom /proc/\$(pgrep -x ambf_simulator)/maps"
