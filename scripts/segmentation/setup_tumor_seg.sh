#!/usr/bin/env bash
# Install the tumor tools of the launcher's "Register MRI" tab (Tumor box).
#
#   scripts/segmentation/setup_tumor_seg.sh
#
# Needs scripts/segmentation/setup_tbone_seg.sh first: the new venv borrows that venv's
# torch (cu128) instead of downloading another 3 GB. Safe to run again: finished steps are
# skipped. Needs ~10 GB of disk and internet. Same as the 5090 desk machine, 2026-11-15.
#
#   1. ../nninteractive/.venv: nnInteractive 2.7.0 + nnU-Net v2, torch from ../tbone-seg-nnunetv2
#      (a .pth file), torchvision built for that torch from the PyTorch index (the PyPI
#      build fails with "operator torchvision::nms does not exist").
#   2. ../vs-seg-kujawa: the Kujawa et al. 2024 vestibular schwannoma nnU-Net models,
#      Multi-Centre Routine Clinical set (Zenodo 10.5281/zenodo.10363647, CC BY 4.0, 8.7 GB,
#      md5 checked); only the contrast T1 (Dataset910) and T2 (Dataset911) models are unpacked.
#   3. nnInteractive v1.0 weights (Hugging Face MIC-DKFZ/nnInteractive, CC BY-NC-SA 4.0:
#      non-commercial) into ~/.nninteractive.
#   4. Check: imports, GPU, both model folders.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$(realpath -m "$REPO/..")"
TBONE="$SRC/tbone-seg-nnunetv2/.venv"
VENV="$SRC/nninteractive/.venv"
MODELS="$SRC/vs-seg-kujawa"
ZIP_URL="https://zenodo.org/api/records/10363647/files/Multi-Centre-Routine-Clinical-(MC-RC)-models.zip/content"
ZIP_MD5="59d04c51dd5055d8985fdc03d653d7a0"
UNPACKED="$MODELS/Multi-Centre-Routine-Clinical-(MC-RC)-models"

say() { printf '\n== %s\n' "$*"; }

[ -x "$TBONE/bin/python" ] || { echo "Run scripts/segmentation/setup_tbone_seg.sh first ($TBONE missing)."; exit 1; }

# 1. venv ------------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    say "Creating $VENV"
    mkdir -p "$(dirname "$VENV")"
    python3 -m venv --without-pip "$VENV"   # no python3-venv package on these machines
fi
SITE="$(ls -d "$VENV"/lib/python3*/site-packages)"
TBONE_SITE="$(ls -d "$TBONE"/lib/python3*/site-packages)"
echo "$TBONE_SITE" > "$SITE/zz_tbone_torch.pth"
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    "$TBONE/bin/python" -m pip --python "$VENV/bin/python" install -q pip
fi
if ! "$VENV/bin/python" -c "import nnInteractive" 2>/dev/null; then
    say "Installing nnInteractive"
    "$VENV/bin/python" -m pip install -q "nnInteractive==2.7.0"
fi
TORCH="$("$VENV/bin/python" -c 'import torch; print(torch.__version__)')"
MINOR="$(echo "$TORCH" | cut -d. -f2)"
CUDA_TAG="$(echo "$TORCH" | sed -n 's/.*+\(cu[0-9]*\).*/\1/p')"
WANT="0.$((MINOR + 15)).0"  # torchvision 0.(minor+15) goes with torch 2.minor
if ! "$VENV/bin/python" -c "import torchvision" 2>/dev/null; then
    say "torchvision $WANT+$CUDA_TAG for torch $TORCH"
    "$VENV/bin/python" -m pip install -q --no-deps --force-reinstall "torchvision==$WANT" \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG:-cu128}"
fi

# 2. vestibular schwannoma models -------------------------------------------------
if [ ! -d "$UNPACKED/Dataset910_VSMCRCT1" ] || [ ! -d "$UNPACKED/Dataset911_VSMCRCT2" ]; then
    mkdir -p "$MODELS"
    cd "$MODELS"
    if ! echo "$ZIP_MD5  MC-RC-models.zip" | md5sum -c --status 2>/dev/null; then
        say "Downloading the VS models (8.7 GB; Zenodo serves ~5 MB/s, so about 30 min)"
        curl -L --retry 5 -C - -o MC-RC-models.zip "$ZIP_URL"
        echo "$ZIP_MD5  MC-RC-models.zip" | md5sum -c
    fi
    say "Unpacking the contrast T1 and T2 models"
    unzip -q -n MC-RC-models.zip -x "*Dataset912*"
fi

# 3. nnInteractive weights ---------------------------------------------------------
"$VENV/bin/python" - <<'EOF'
from nnInteractive.model_management import ensure_model_available, get_default_model_id
print("nnInteractive weights:", ensure_model_available(get_default_model_id()))
EOF

# 4. check -----------------------------------------------------------------------------
"$VENV/bin/python" - <<EOF
import torch, torchvision, nnunetv2, nnInteractive, SimpleITK
print("torch", torch.__version__, "torchvision", torchvision.__version__, "GPU:", torch.cuda.is_available())
import pathlib
for d in ("Dataset910_VSMCRCT1", "Dataset911_VSMCRCT2"):
    folds = sorted(pathlib.Path("$UNPACKED", d).glob("*3d_fullres/fold_*/checkpoint_final.pth"))
    print(d, len(folds), "folds")
EOF
say "Done. The launcher finds $VENV by default (gui_setup.yaml key nninteractive_python overrides it)."
