#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment setup for MR-Net
#  "Shape Registration with Learned Deformations for 3D Shape Reconstruction
#   from Sparse and Incomplete Point Clouds" (Medical Image Analysis 2021)
#  https://github.com/XiangChen1994/MR-Net
#
#  Works from ANY clone path:
#    * script inside the clone -> just run:  bash setup_mrnet_env.sh
#    * elsewhere              -> bash setup_mrnet_env.sh /path/to/MR-Net
#    * parent dir containing the clone -> auto-detected one level down
#
#  IMPORTANT: the upstream repo is Python 2.7 + TensorFlow 1.7 + TensorLayer and
#  CANNOT be installed in a modern conda env (no py2.7/TF1.7 wheels, no GPU
#  support for RTX 30/40). The previously validated path is the PyTorch
#  re-implementation in `pytorch_mrnet/` (synced from the reference checkout),
#  which runs the full train/val/test loop on the DeepSDF-format heart data.
#
#  Creates / reuses a conda env named `mrnet`
#  (python 3.9 + torch 2.2.0 + torchvision 0.17.0 + pytorch-cuda 11.8 +
#   numpy/scipy/trimesh/Pillow/matplotlib — all the PyTorch MR-Net needs).
#
#  Syncs from the reference checkout (default /home/nay/github/MR-Net):
#      pytorch_mrnet/            (PyTorch MR-Net + baselines)
#      MR_Net_Analysis.md        (paper analysis / migration notes)
#      Data/heart/               (template meshes + demo contour, if missing)
#
#  Usage:
#      bash setup_mrnet_env.sh [repo_dir] [--recreate] [--dry-run] [--no-sync]
#  Overrides (env vars):
#      MRNET_ENV_NAME=mrnet  MRNET_PYTHON=3.9
#      MRNET_TORCH_VERSION=2.2.0  MRNET_TORCHVISION_VERSION=0.17.0  MRNET_CUDA=118
#      MRNET_SOURCE=/home/nay/github/MR-Net
# =============================================================================
set -eo pipefail  # NOTE: no -u: conda activation hooks reference unset vars

ENV_NAME="${MRNET_ENV_NAME:-mrnet}"
PYTHON_VERSION="${MRNET_PYTHON:-3.9}"
TORCH_VERSION="${MRNET_TORCH_VERSION:-2.2.0}"
TORCHVISION_VERSION="${MRNET_TORCHVISION_VERSION:-0.17.0}"
CUDA_TAG="${MRNET_CUDA:-118}"
CONDA_CUDA_VERSION="${MRNET_CUDA_VERSION:-$(echo "$CUDA_TAG" | sed 's/^\([0-9][0-9]*\)\([0-9]\)$/\1.\2/')}"
MRNET_SOURCE="${MRNET_SOURCE:-/home/nay/github/MR-Net}"
RECREATE=0; DRY_RUN=0; DO_SYNC=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=""

usage() {
    local last; last="$(awk '/^[^#[:space:]]/{print NR; exit}' "${BASH_SOURCE[0]}")"
    sed -n "2,$((last-1))p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)      usage ;;
        --recreate)     RECREATE=1 ;;
        --dry-run)      DRY_RUN=1 ;;
        --no-sync)      DO_SYNC=0 ;;
        --repo)         shift; REPO_DIR="$1" ;;
        -*)             echo "unknown option: $1" >&2; usage ;;
        *)              REPO_DIR="$1" ;;
    esac
    shift
done

# ---------------------------------------- find the MR-Net repo root
is_mrnet_repo() {
    { [ -f "$1/README.md" ] && [ -f "$1/train.py" ] && [ -f "$1/test.py" ] && [ -d "$1/mrnet" ]; } \
        || [ -f "$1/pytorch_mrnet/model.py" ]
}
find_repo_root() {
    local base="$1"
    if is_mrnet_repo "$base"; then echo "$base"; return 0; fi
    local d
    while IFS= read -r d; do
        if is_mrnet_repo "$d"; then echo "$d"; return 0; fi
    done < <(find "$base" -mindepth 1 -maxdepth 2 -type d \( -name 'MR-Net' -o -name 'MRNet' \) 2>/dev/null)
    return 1
}

if [ -n "$REPO_DIR" ]; then REPO_DIR="$(cd "$REPO_DIR" && pwd)"; else REPO_DIR="$SCRIPT_DIR"; fi
if ! REPO_DIR="$(find_repo_root "$REPO_DIR")"; then
    echo "ERROR: cannot find the MR-Net repo under '$REPO_DIR'." >&2
    echo "       Run: bash setup_mrnet_env.sh /path/to/MR-Net" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "==> MR-Net repo: $REPO_DIR"

command -v conda >/dev/null || { echo "ERROR: conda not found on PATH"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then ENV_EXISTS=1; else ENV_EXISTS=0; fi

EXISTS_STR=$([ "$ENV_EXISTS" -eq 1 ] && echo yes || echo no)
RECREATE_STR=$([ "$RECREATE" -eq 1 ] && echo yes || echo no)
DRYRUN_STR=$([ "$DRY_RUN" -eq 1 ] && echo yes || echo no)
SYNC_STR=$([ "$DO_SYNC" -eq 1 ] && echo "yes ($MRNET_SOURCE)" || echo no)
echo "==> plan:"
echo "    env        : $ENV_NAME (python $PYTHON_VERSION; exists: $EXISTS_STR)"
echo "    torch      : $TORCH_VERSION / torchvision $TORCHVISION_VERSION / cuda $CONDA_CUDA_VERSION"
echo "    recreate   : $RECREATE_STR"
echo "    dry-run    : $DRYRUN_STR"
echo "    sync       : $SYNC_STR"
echo "    steps: env -> conda torch -> pip deps -> sync pytorch_mrnet -> verify"

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

# --------------------------------------------------- torch (conda or skip)
if [ "$DRY_RUN" -eq 1 ]; then
    echo "    [dry-run] conda install pytorch=$TORCH_VERSION torchvision=$TORCHVISION_VERSION pytorch-cuda=$CONDA_CUDA_VERSION -c pytorch -c nvidia"
elif python -c "import torch, torchvision" >/dev/null 2>&1; then
    echo "==> torch already present in '$ENV_NAME', skipping conda install"
else
    echo "==> installing pytorch $TORCH_VERSION + torchvision $TORCHVISION_VERSION (cuda $CONDA_CUDA_VERSION)"
    if [ "$CUDA_TAG" = "cpu" ]; then
        conda install -y -n "$ENV_NAME" "pytorch=$TORCH_VERSION" "torchvision=$TORCHVISION_VERSION" cpuonly -c pytorch
    else
        conda install -y -n "$ENV_NAME" "pytorch=$TORCH_VERSION" "torchvision=$TORCHVISION_VERSION" "pytorch-cuda=$CONDA_CUDA_VERSION" -c pytorch -c nvidia
    fi
fi

# ----------------------------------------------------------- pip deps
echo "==> installing python deps (PyTorch MR-Net)"
run python -m pip install -U pip
run python -m pip install \
    "numpy==1.26.4" "scipy==1.13.1" "trimesh==4.12.2" \
    "Pillow==11.3.0" "matplotlib==3.9.4" "tqdm==4.68.4" \
    "psutil==7.2.2" "PyYAML==6.0.2"

# ------------------------------------- sync validated PyTorch MR-Net
if [ "$DO_SYNC" -eq 1 ]; then
    if [ ! -d "$MRNET_SOURCE" ]; then
        echo "WARNING: reference checkout '$MRNET_SOURCE' not found; skipping sync."
    elif [ "$DRY_RUN" -eq 1 ]; then
        echo "    [dry-run] sync pytorch_mrnet/ + MR_Net_Analysis.md + Data/heart from $MRNET_SOURCE"
    else
        echo "==> syncing PyTorch MR-Net from $MRNET_SOURCE"
        mkdir -p pytorch_mrnet
        cp -rf "$MRNET_SOURCE/pytorch_mrnet/." pytorch_mrnet/
        [ -f "$MRNET_SOURCE/MR_Net_Analysis.md" ] && cp -f "$MRNET_SOURCE/MR_Net_Analysis.md" .
        if [ ! -d Data/heart ]; then
            mkdir -p Data
            cp -rf "$MRNET_SOURCE/Data/heart" Data/heart
        fi
        echo "PyTorch MR-Net synced."
    fi
fi

# --------------------------------------------------------------- verify
if [ "$DRY_RUN" -eq 1 ]; then echo "==> dry-run finished (nothing installed)."; exit 0; fi
echo "==> verifying installation"
python - <<'PY'
import importlib
for m in ["torch", "torchvision", "numpy", "scipy", "trimesh", "PIL", "matplotlib", "tqdm"]:
    importlib.import_module(m)
import numpy as np, torch, trimesh
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      ("| gpu: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")

import sys
sys.path.insert(0, ".")
from pytorch_mrnet.model import MinimalMRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MinimalMRNet().to(device).eval()
# synthetic fixed-topology template (icosphere) + 3000 random contour points
sphere = trimesh.creation.icosphere(subdivisions=2)
verts = torch.as_tensor(sphere.vertices, dtype=torch.float32, device=device)  # (V,3); model adds batch dim
faces = np.asarray(sphere.faces)
edges = np.unique(np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]], axis=0), axis=1), axis=0)
edges = torch.as_tensor(edges, dtype=torch.long, device=device)
points = (torch.rand(1, 3000, 3, device=device) * 2 - 1)
with torch.no_grad():
    out = model(points, verts, edges)
print("MR-Net synthetic forward OK | output mesh:", tuple(out.shape))
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:  conda activate $ENV_NAME"
echo "  Repo: $REPO_DIR"
echo "  PyTorch MR-Net: python -m pytorch_mrnet.train_feasibility --data-root <3d_data> ..."
echo "================================================================"
