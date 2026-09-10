#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment setup for NeuralDeformableModels (NDM)
#  ICCV 2023: Neural Deformable Models for 3D Bi-Ventricular Heart Shape
#  Reconstruction and Modeling from 2D Sparse Cardiac MRI
#  https://github.com/DeepTag/NeuralDeformableModels
#
#  Works from ANY clone path:
#    * script inside the clone -> just run:  bash setup_ndm_env.sh
#    * elsewhere              -> bash setup_ndm_env.sh /path/to/NeuralDeformableModels
#    * parent dir containing the clone -> auto-detected one level down
#
#  Creates / reuses a conda env named `ndm` (python 3.9 + torch 2.2.0 +
#  pytorch-cuda 11.8 + torchdiffeq/trimesh/scipy/... needed by the validated
#  two-surface heart pipeline).
#
#  It also syncs the previously validated heart pipeline from the reference
#  checkout (default /home/nay/github/NeuralDeformableModels) into this repo:
#    root:      train_simple.py, evaluate_two_surface.py, chamfer_loss.py,
#               test_input_gt_alignment.py
#    package:   NeuralDeformableModel/dataset/heart_dataset.py,
#               acquisition_geometry.py
#               NeuralDeformableModel/model/model.py, pointnet_util.py
#               (latter two contain torch>=2.0 compat + memory fixes)
#    tools/:    convert_deepsdf_3d_data_to_ndm.py, smoke_test_ndm.py, ...
#  Use --no-sync to skip, or NDM_HEART_SOURCE=<dir> to point elsewhere.
#
#  Usage:
#      bash setup_ndm_env.sh [repo_dir] [--recreate] [--dry-run]
#                            [--no-sync] [--with-official-deps]
#  Overrides (env vars):
#      NDM_ENV_NAME=ndm  NDM_PYTHON=3.9
#      NDM_TORCH_VERSION=2.2.0  NDM_TORCHVISION_VERSION=0.17.0  NDM_CUDA=118
# =============================================================================
set -eo pipefail  # NOTE: no -u: conda activation hooks reference unset vars

ENV_NAME="${NDM_ENV_NAME:-ndm}"
PYTHON_VERSION="${NDM_PYTHON:-3.9}"
TORCH_VERSION="${NDM_TORCH_VERSION:-2.2.0}"
TORCHVISION_VERSION="${NDM_TORCHVISION_VERSION:-0.17.0}"
CUDA_TAG="${NDM_CUDA:-118}"
CONDA_CUDA_VERSION="${NDM_CUDA_VERSION:-$(echo "$CUDA_TAG" | sed 's/^\([0-9][0-9]*\)\([0-9]\)$/\1.\2/')}"
HEART_SOURCE="${NDM_HEART_SOURCE:-/home/nay/github/NeuralDeformableModels}"
RECREATE=0; DRY_RUN=0; DO_SYNC=1; OFFICIAL_DEPS=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=""

usage() {
    local last; last="$(awk '/^[^#[:space:]]/{print NR; exit}' "${BASH_SOURCE[0]}")"
    sed -n "2,$((last-1))p" "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)              usage ;;
        --recreate)             RECREATE=1 ;;
        --dry-run)              DRY_RUN=1 ;;
        --no-sync)              DO_SYNC=0 ;;
        --with-official-deps)   OFFICIAL_DEPS=1 ;;
        --repo)                 shift; REPO_DIR="$1" ;;
        -*)                     echo "unknown option: $1" >&2; usage ;;
        *)                      REPO_DIR="$1" ;;
    esac
    shift
done

# ---------------------------------------- find the NDM repo root
is_ndm_repo() {
    [ -f "$1/README.md" ] && [ -f "$1/NeuralDeformableModel/model/model.py" ] \
        && [ -f "$1/NeuralDeformableModel/train_ndm.py" ]
}
find_repo_root() {
    local base="$1"
    if is_ndm_repo "$base"; then echo "$base"; return 0; fi
    local d
    while IFS= read -r d; do
        if is_ndm_repo "$d"; then echo "$d"; return 0; fi
    done < <(find "$base" -mindepth 1 -maxdepth 2 -type d -name NeuralDeformableModels 2>/dev/null)
    return 1
}

if [ -n "$REPO_DIR" ]; then REPO_DIR="$(cd "$REPO_DIR" && pwd)"; else REPO_DIR="$SCRIPT_DIR"; fi
if ! REPO_DIR="$(find_repo_root "$REPO_DIR")"; then
    echo "ERROR: cannot find the NeuralDeformableModels repo under '$REPO_DIR'." >&2
    echo "       Run: bash setup_ndm_env.sh /path/to/NeuralDeformableModels" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "==> NDM repo: $REPO_DIR"

command -v conda >/dev/null || { echo "ERROR: conda not found on PATH"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then ENV_EXISTS=1; else ENV_EXISTS=0; fi

EXISTS_STR=$([ "$ENV_EXISTS" -eq 1 ] && echo yes || echo no)
RECREATE_STR=$([ "$RECREATE" -eq 1 ] && echo yes || echo no)
DRYRUN_STR=$([ "$DRY_RUN" -eq 1 ] && echo yes || echo no)
SYNC_STR=$([ "$DO_SYNC" -eq 1 ] && echo "yes ($HEART_SOURCE)" || echo no)
echo "==> plan:"
echo "    env           : $ENV_NAME (python $PYTHON_VERSION; exists: $EXISTS_STR)"
echo "    torch         : $TORCH_VERSION / torchvision $TORCHVISION_VERSION / cuda $CONDA_CUDA_VERSION"
echo "    recreate      : $RECREATE_STR"
echo "    dry-run       : $DRYRUN_STR"
echo "    sync heart    : $SYNC_STR"
echo "    official deps : $([ "$OFFICIAL_DEPS" -eq 1 ] && echo yes || echo no)"
echo "    steps: env -> conda torch -> pip deps -> sync pipeline -> verify"

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
    echo "==> installing pytorch $TORCH_VERSION + torchvision $TORCHVISION_VERSION (cuda $CUDA_TAG)"
    if [ "$CUDA_TAG" = "cpu" ]; then
        conda install -y -n "$ENV_NAME" "pytorch=$TORCH_VERSION" "torchvision=$TORCHVISION_VERSION" cpuonly -c pytorch
    else
        conda install -y -n "$ENV_NAME" "pytorch=$TORCH_VERSION" "torchvision=$TORCHVISION_VERSION" "pytorch-cuda=$CONDA_CUDA_VERSION" -c pytorch -c nvidia
    fi
fi

# ----------------------------------------------------------- pip deps
echo "==> installing python deps (two-surface heart pipeline)"
run python -m pip install -U pip
run python -m pip install \
    "numpy==1.26.4" "scipy==1.13.1" "torchdiffeq==0.2.5" \
    "trimesh==4.12.2" "tqdm==4.68.4" "psutil==7.2.2" "PyYAML==6.0.2" \
    "scikit-image==0.24.0" "matplotlib==3.9.4"

if [ "$OFFICIAL_DEPS" -eq 1 ]; then
    echo "==> installing optional deps for the official pipeline / metrics"
    run python -m pip install "open3d==0.19.0" "SimpleITK" "pykeops" "imageio" "comet_ml" || true
    if [ "$DRY_RUN" -ne 1 ]; then
        python -m pip install --no-build-isolation "pytorch3d" \
            || echo "WARNING: pytorch3d install failed (only needed by official test/eval scripts)"
        echo "NOTE: 'mesh_intersection' (torch-mesh-isect) is not on PyPI; build it manually if you need non-manifold metrics."
    fi
fi

# ------------------------------------- sync validated heart pipeline
if [ "$DO_SYNC" -eq 1 ]; then
    if [ ! -d "$HEART_SOURCE" ]; then
        echo "WARNING: heart pipeline source '$HEART_SOURCE' not found; skipping sync."
    elif [ "$DRY_RUN" -eq 1 ]; then
        echo "    [dry-run] rsync heart pipeline from $HEART_SOURCE -> $REPO_DIR"
    else
        echo "==> syncing validated heart pipeline from $HEART_SOURCE"
        mkdir -p NeuralDeformableModel/dataset NeuralDeformableModel/model tools
        for f in train_simple.py evaluate_two_surface.py chamfer_loss.py \
                 test_input_gt_alignment.py test_dataset.py quick_test.py; do
            [ -f "$HEART_SOURCE/$f" ] && cp -f "$HEART_SOURCE/$f" "$f"
        done
        for f in dataset/heart_dataset.py dataset/acquisition_geometry.py \
                 model/model.py model/pointnet_util.py; do
            [ -f "$HEART_SOURCE/NeuralDeformableModel/$f" ] && \
                cp -f "$HEART_SOURCE/NeuralDeformableModel/$f" "NeuralDeformableModel/$f"
        done
        for f in convert_deepsdf_3d_data_to_ndm.py realign_deepsdf_points.py smoke_test_ndm.py; do
            [ -f "$HEART_SOURCE/tools/$f" ] && cp -f "$HEART_SOURCE/tools/$f" "tools/$f"
        done
        [ -f "$HEART_SOURCE/alignment.json" ] && cp -f "$HEART_SOURCE/alignment.json" .
        echo "heart pipeline synced."
    fi
fi

# -------------------------------- torch>=2.0 compat patches (idempotent)
if [ "$DRY_RUN" -ne 1 ]; then
    echo "==> applying torch>=2.0 compatibility patches (idempotent)"
    python - <<'PY'
import pathlib
fixes = []
p = pathlib.Path("NeuralDeformableModel/model/model.py")
s = p.read_text()
if "torch.meshgrid(theta, phi)" in s:
    s = s.replace("torch.meshgrid(theta, phi)", "torch.meshgrid(theta, phi, indexing='ij')")
    p.write_text(s)
    fixes.append("model.py: torch.meshgrid(..., indexing='ij')")
q = pathlib.Path("NeuralDeformableModel/model/pointnet_util.py")
s = q.read_text()
if "return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)" in s:
    s = s.replace(
        "return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)",
        "# memory-efficient squared distance (5600-point inputs OOM otherwise)\n"
        "    dist = -2 * torch.matmul(src, dst.transpose(1, 2))\n"
        "    dist = dist + torch.sum(src.square(), dim=-1, keepdim=True)\n"
        "    dist = dist + torch.sum(dst.square(), dim=-1).unsqueeze(1)\n"
        "    return dist.clamp_min_(0)")
    q.write_text(s)
    fixes.append("pointnet_util.py: matmul-based square_distance")
print("patches applied:", fixes if fixes else "none needed (already patched)")
PY
fi

# --------------------------------------------------------------- verify
if [ "$DRY_RUN" -eq 1 ]; then echo "==> dry-run finished (nothing installed)."; exit 0; fi
echo "==> verifying installation"
python - <<'PY'
import importlib
for m in ["torch", "torchvision", "torchdiffeq", "trimesh", "scipy", "numpy", "tqdm"]:
    importlib.import_module(m)
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      ("| gpu: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")

import sys
sys.path.insert(0, ".")
from NeuralDeformableModel.model.model import NeuralDeformableModel
model = NeuralDeformableModel(zdim=512).eval()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(dev).float()
x = torch.randn(1, 5600, 3, device=dev)
with torch.no_grad():
    out = model(x)
print("NDM model forward OK | outputs:", len(out), "| primitive:", tuple(out[-1].shape))
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:  conda activate $ENV_NAME"
echo "  Repo: $REPO_DIR"
echo "  Train: python train_simple.py --data-root <3d_data> ..."
echo "================================================================"
