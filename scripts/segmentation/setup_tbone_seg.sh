#!/usr/bin/env bash
# Install the temporal bone segmentation model used by the launcher's "Add a volume" tab.
#
#   scripts/segmentation/setup_tbone_seg.sh [target_dir]
#
# target_dir defaults to ../tbone-seg-nnunetv2 next to this repo (where the tab looks;
# gui_setup.yaml key tbone_seg_dir overrides it). Safe to run again: each step is skipped
# when it is already done. Needs ~10 GB of disk (torch + 1.1 GB of weights) and internet.
#
# What it does (same as the 5090 desk machine, 2026-11-13):
#   1. clone andy-s-ding/tbone-seg-nnunetv2 at the tested commit, with its nnUNet fork
#   2. .venv with torch cu128 (RTX 50xx cards need CUDA 12.8 builds), nnUNet, SimpleITK, ...
#   3. download the trained weights (Git LFS zip) and check their sha256
#   4. check that torch sees the GPU and that the model folder is complete
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${1:-$REPO/../tbone-seg-nnunetv2}"
TARGET="$(realpath -m "$TARGET")"
URL="https://github.com/andy-s-ding/tbone-seg-nnunetv2.git"
COMMIT="ac0290d8f24aad978c0f3c808d7474566bb27dd1"
WEIGHTS_URL="https://media.githubusercontent.com/media/andy-s-ding/tbone-seg-nnunetv2/main/tbone_nnUNet_Vanilla.zip"
WEIGHTS_SHA256="b669bb17ff7e56ac34c8519a3cabd32b040c60f13432e9d83802cb7d66c30c94"
MODEL="$TARGET/nnUNet_results/Dataset101_TemporalBone/nnUNetTrainer__nnUNetPlans__3d_fullres"

say() { printf '\n== %s\n' "$*"; }

# 1. Source --------------------------------------------------------------------
if [ ! -d "$TARGET/.git" ]; then
    say "Cloning $URL into $TARGET"
    git clone "$URL" "$TARGET"
fi
cd "$TARGET"
if [ "$(git rev-parse HEAD)" != "$COMMIT" ] && [ -z "$(git status --porcelain --untracked-files=no)" ]; then
    git fetch -q origin || true
    git -c advice.detachedHead=false checkout -q "$COMMIT"
fi
# Only the nnUNet fork is needed (not scripts/surface-distance).
git submodule update --init nnUNet
say "Source: $(git rev-parse --short HEAD), nnUNet $(git -C nnUNet describe --tags --always)"

# 2. Python environment ----------------------------------------------------------
PY="$TARGET/.venv/bin/python"
if [ ! -x "$PY" ]; then
    say "Creating .venv"
    if ! python3 -m venv .venv >/dev/null 2>&1; then
        # Ubuntu without python3-venv: no ensurepip. Make it bare and borrow pip.
        rm -rf .venv
        python3 -m venv --without-pip .venv
    fi
fi
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    HELPER=""
    for candidate in "$REPO/middle_ear_newton/.venv/bin/python" python3; do
        if "$candidate" -m pip --version >/dev/null 2>&1; then HELPER="$candidate"; break; fi
    done
    if [ -z "$HELPER" ]; then
        echo "No pip anywhere to bootstrap from. Install python3-venv (sudo apt install python3-venv) and run again." >&2
        exit 1
    fi
    say "Installing pip into .venv with $HELPER"
    "$HELPER" -m pip --python "$PY" install -q pip
fi
if ! "$PY" -c "import torch" 2>/dev/null; then
    say "Installing torch (CUDA 12.8 build, ~3 GB)"
    "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
fi
say "Installing nnUNet and the worker's packages"
"$PY" -m pip install -q -e ./nnUNet SimpleITK pydicom pillow pyyaml edt

# 3. Weights ---------------------------------------------------------------------
if [ ! -f "$MODEL/fold_0/checkpoint_final.pth" ]; then
    say "Downloading the trained weights (1.1 GB)"
    mkdir -p model_download nnUNet_results
    curl -L --fail --retry 3 -o model_download/tbone_nnUNet_Vanilla.zip "$WEIGHTS_URL"
    echo "$WEIGHTS_SHA256  model_download/tbone_nnUNet_Vanilla.zip" | sha256sum -c -
    if command -v unzip >/dev/null; then
        unzip -q -o model_download/tbone_nnUNet_Vanilla.zip -d nnUNet_results
    else
        "$PY" -m zipfile -e model_download/tbone_nnUNet_Vanilla.zip nnUNet_results
    fi
    rm -rf model_download
fi

# 4. Check -----------------------------------------------------------------------
say "Checking"
for f in plans.json dataset.json fold_0/checkpoint_final.pth; do
    [ -f "$MODEL/$f" ] || { echo "Missing $MODEL/$f" >&2; exit 1; }
done
"$PY" - <<'EOF'
import torch, nnunetv2, SimpleITK, pydicom, edt  # noqa: F401
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}, {total / 2**30:.1f} GiB ({free / 2**30:.1f} free)")
    x = torch.ones(8, device="cuda")  # fails here if this torch build does not support the card
    assert float(x.sum()) == 8.0
else:
    print("WARNING: torch sees no GPU; segmentation would run on the CPU (very slow).")
EOF
say "Ready. Open the launcher's \"Add a volume\" tab."
if [ "$TARGET" != "$(realpath -m "$REPO/../tbone-seg-nnunetv2")" ]; then
    echo "Not the default place: add   tbone_seg_dir: $TARGET   to scripts/study_gui/gui_setup.yaml"
fi
