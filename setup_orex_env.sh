#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment setup for OReX
#  (Object Reconstruction from Planar Cross-sections Using Neural Fields,
#   https://github.com/haimsaw/OReX)
#
#  Works from ANY clone path:
#    * script inside the clone -> just run:  bash setup_orex_env.sh
#    * elsewhere              -> bash setup_orex_env.sh /path/to/OReX
#    * parent dir containing the clone -> auto-detected one level down
#
#  Why this recipe:
#    * requirements.txt pins numpy~=1.24.2 / scipy~=1.9.1 / scikit-learn~=1.0.2
#      (no python-3.7 wheels) and torch==1.13.1 (no python-3.11 wheels), so
#      python 3.10 is the intersection that installs everything from wheels.
#    * "torch==1.13.1" on PyPI is the CPU build; OReX runs on GPU
#      (--cuda_device), so we install torch 1.13.1+cu117 from the official
#      PyTorch index instead (supports Ampere/RTX-30 sm_86). CPU-only machines:
#      OREX_TORCH_CUDA=cpu.
#    * requirements.txt omits numpy-stl, but Slicer.py does "from stl import
#      mesh" -> installed as an extra so Slicer.py works out of the box.
#
#  Usage:
#      bash setup_orex_env.sh [repo_dir] [--recreate] [--dry-run]
#  Overrides (env vars):
#      OREX_ENV_NAME=orex  OREX_PYTHON=3.10
#      OREX_TORCH_VERSION=1.13.1  OREX_TORCH_CUDA=117  (or 113/116/102/cpu)
# =============================================================================
set -eo pipefail  # NOTE: no -u: conda activation hooks reference unset vars

ENV_NAME="${OREX_ENV_NAME:-orex}"
PYTHON_VERSION="${OREX_PYTHON:-3.10}"
TORCH_VERSION="${OREX_TORCH_VERSION:-1.13.1}"
TORCH_CUDA="${OREX_TORCH_CUDA:-117}"
RECREATE=0
DRY_RUN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=""

usage() {
    local last; last="$(awk '/^[^#[:space:]]/{print NR; exit}' "${BASH_SOURCE[0]}")"
    sed -n "2,$((last-1))p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)     usage ;;
        --recreate)    RECREATE=1 ;;
        --dry-run)     DRY_RUN=1 ;;
        --repo)        shift; REPO_DIR="$1" ;;
        -*)            echo "unknown option: $1" >&2; usage ;;
        *)             REPO_DIR="$1" ;;
    esac
    shift
done

# ------------------------------------------------- find the OReX repo root
is_orex_repo() {
    [ -f "$1/Main.py" ] && [ -f "$1/Globals.py" ] && [ -f "$1/requirements.txt" ] \
        && grep -q "meshcut" "$1/requirements.txt" \
        && grep -q "torch==1.13.1" "$1/requirements.txt"
}
find_repo_root() {
    local base="$1"
    if is_orex_repo "$base"; then echo "$base"; return 0; fi
    local f
    while IFS= read -r f; do
        if is_orex_repo "$(dirname "$f")"; then dirname "$f"; return 0; fi
    done < <(find "$base" -mindepth 2 -maxdepth 2 -name requirements.txt 2>/dev/null)
    return 1
}

if [ -n "$REPO_DIR" ]; then REPO_DIR="$(cd "$REPO_DIR" && pwd)"; else REPO_DIR="$SCRIPT_DIR"; fi
if ! REPO_DIR="$(find_repo_root "$REPO_DIR")"; then
    echo "ERROR: cannot find the OReX repo under '$REPO_DIR'." >&2
    echo "       Run: bash setup_orex_env.sh /path/to/OReX" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "==> OReX repo: $REPO_DIR"

# ------------------------------------------------------------- preflight
command -v conda >/dev/null || { echo "ERROR: conda not found on PATH"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then ENV_EXISTS=1; else ENV_EXISTS=0; fi

EXISTS_STR=$([ "$ENV_EXISTS" -eq 1 ] && echo yes || echo no)
RECREATE_STR=$([ "$RECREATE" -eq 1 ] && echo yes || echo no)
DRYRUN_STR=$([ "$DRY_RUN" -eq 1 ] && echo yes || echo no)
echo "==> plan:"
echo "    env          : $ENV_NAME (python $PYTHON_VERSION; exists: $EXISTS_STR)"
echo "    torch        : $TORCH_VERSION (+cu$TORCH_CUDA)"
echo "    recreate     : $RECREATE_STR"
echo "    dry-run      : $DRYRUN_STR"
echo "    steps: create env -> pip -r requirements.txt (minus torch) ->"
echo "           install torch -> numpy-stl extra -> verify (+ tiny csl smoke)"

run() {
    if [ "$DRY_RUN" -eq 1 ]; then echo "    [dry-run] $*"; else echo "    \$ $*"; "$@"; fi
}
do_activate() {
    if [ "$DRY_RUN" -eq 1 ]; then echo "    [dry-run] conda activate $ENV_NAME"; else conda activate "$ENV_NAME"; fi
}

# ------------------------------------------------------------ create env
if [ "$RECREATE" -eq 1 ] && [ "$ENV_EXISTS" -eq 1 ]; then
    echo "==> removing existing env '$ENV_NAME' (--recreate)"
    run conda env remove -n "$ENV_NAME" -y
    run conda create -y -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION"
    do_activate
elif [ "$ENV_EXISTS" -eq 1 ]; then
    echo "==> env '$ENV_NAME' already exists, reusing it (use --recreate to rebuild)"
    do_activate
else
    echo "==> creating env '$ENV_NAME' (python $PYTHON_VERSION)"
    run conda create -y -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION"
    do_activate
fi

# ---------------------------------------------------- install core deps
run python -m pip install -U pip
echo "==> installing requirements.txt (torch line skipped, installed separately)"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "    [dry-run] pip install -r <requirements minus torch>"
else
    grep -vE '^\s*torch\s*==' requirements.txt > /tmp/orex_requirements_notorch.txt
    python -m pip install -r /tmp/orex_requirements_notorch.txt
fi

# ----------------------------------------------------------------- torch
echo "==> installing torch $TORCH_VERSION (+cu$TORCH_CUDA, python$PYTHON_VERSION-compatible)"
if [ "$TORCH_CUDA" = "cpu" ]; then
    run python -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
else
    run python -m pip install "torch==$TORCH_VERSION+cu$TORCH_CUDA" \
        --index-url "https://download.pytorch.org/whl/cu$TORCH_CUDA"
fi

# ---------------------------------------------------- extra: numpy-stl
# Slicer.py does `from stl import mesh`; not listed in requirements.txt
echo "==> installing numpy-stl (needed by Slicer.py)"
run python -m pip install numpy-stl

# --------------------------------------------------------------- verify
if [ "$DRY_RUN" -eq 1 ]; then echo "==> dry-run finished (nothing installed)."; exit 0; fi
echo "==> verifying installation"
python - <<'PY'
import sys
print("python :", sys.version.split()[0])
import numpy, scipy, matplotlib, sklearn, shapely, meshcut, trimesh, parse
print("numpy   :", numpy.__version__)
print("scipy   :", scipy.__version__)
print("sklearn :", sklearn.__version__)
print("shapely :", shapely.__version__)
print("trimesh :", trimesh.__version__)
import stl
print("numpy-stl OK")
import torch
print("torch   :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      ("| gpu: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    x = torch.rand(1000, device="cuda")
    print("cuda op :", float((x @ x) ** 0.5).__round__(3))
# tiny smoke: parse the bundled example .csl (no training)
from Dataset.CSL import CSL
import glob
cands = sorted(glob.glob("Data/csl_with_ref/*.csl"))
if cands:
    csl = CSL.from_csl_file(cands[0])
    print("csl smoke OK:", csl.model_name, "| planes:", len([p for p in csl.planes if not p.is_empty]))
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:  conda activate $ENV_NAME"
echo "  Repo: $REPO_DIR"
echo "  Run e.g.: python Main.py ./Artifacts ./Data/csl_with_ref/eight_15.csl --cuda_device 0"
echo "================================================================"
