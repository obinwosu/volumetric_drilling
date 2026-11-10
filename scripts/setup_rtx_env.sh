#!/usr/bin/env bash
# Build the Python environment the RTX view runs in: middle_ear_newton/.venv.
# Packages are pinned in scripts/rtx_view/requirements.txt (all from PyPI, ~5 GB).
#
#   bash scripts/setup_rtx_env.sh            # create or update the environment
#   bash scripts/setup_rtx_env.sh --check    # only test the imports
#
# Needs Python 3.12, an NVIDIA driver with RTX, and either python3.12-venv or
# python3-pip from apt (sudo apt install python3.12-venv).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${FIVRS_RTX_VENV:-${REPO}/middle_ear_newton/.venv}"
REQS="${REPO}/scripts/rtx_view/requirements.txt"
PY="${VENV}/bin/python"

check() {
  echo "Checking the RTX view imports..."
  "${PY}" - <<'PY'
import importlib, sys
names = ["numpy", "warp", "newton", "pxr.Usd", "ovrtx", "ovstage", "pyglet", "imgui_bundle", "glfw", "trimesh", "scipy", "yaml"]
bad = []
for name in names:
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append(f"{name}: {exc}")
if bad:
    print("Missing or broken:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("RTX view environment OK:", sys.executable)
PY
}

if [[ "${1:-}" == "--check" ]]; then
  check
  exit
fi

if ! command -v python3.12 >/dev/null; then
  echo "ERROR: python3.12 not found. Ubuntu 24.04 ships it as python3."
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "WARNING: nvidia-smi not found. The RTX view needs the NVIDIA driver."
fi

if [[ ! -x "${PY}" ]]; then
  echo "Creating ${VENV}"
  mkdir -p "$(dirname "${VENV}")"
  if python3.12 -c "import ensurepip" 2>/dev/null; then
    python3.12 -m venv "${VENV}"
  elif python3.12 -m pip --version >/dev/null 2>&1; then
    # No python3.12-venv: make the environment bare and install pip into it
    # with the system pip.
    python3.12 -m venv --without-pip "${VENV}"
    python3.12 -m pip --python "${PY}" install --quiet pip
  else
    echo "ERROR: install one of these first, then run this again:"
    echo "  sudo apt install python3.12-venv"
    echo "  sudo apt install python3-pip"
    exit 1
  fi
fi

echo "Installing pinned packages (first run downloads ~5 GB)..."
"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install -r "${REQS}"
check
echo
echo "Done. Start the study GUI again; its RTX renderer check should now pass."
