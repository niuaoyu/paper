#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment setup for 3DShape2VecSet
#  (A 3D Shape Representation for Neural Fields and Generative Diffusion Models,
#   SIGGRAPH 2023, https://github.com/1zb/3DShape2VecSet)
#
#  Works from ANY clone path:
#    * script inside the clone -> just run:  bash setup_shape2vecset_env.sh
#    * elsewhere              -> bash setup_shape2vecset_env.sh /path/to/3DShape2VecSet
#    * parent dir containing the clone -> auto-detected one level down
#
#  Creates / reuses a conda env named `shape` (python 3.9 + torch 2.2.0 +
#  pytorch-cuda 11.8 + the PyG wheel for torch_cluster + the project deps).
#
#  Also applies two small torch>=2.0 compatibility patches to the source
#  (idempotent; no-op when already applied):
#    1. util/misc.py  : `from torch._six import inf` was removed in torch 2.x
#                       -> replaced with `from math import inf`
#    2. util/shapenet.py: also accept point data stored under
#                       <cat>/4_pointcloud/ (besides the official <cat>/ layout)
#
#  Usage:
#      bash setup_shape2vecset_env.sh [repo_dir] [--recreate] [--dry-run]
#  Overrides (env vars):
#      S2VS_ENV_NAME=shape   S2VS_PYTHON=3.9
#      S2VS_TORCH_VERSION=2.2.0  S2VS_TORCHVISION_VERSION=0.17.0
#      S2VS_CUDA=118   (e.g. 117 / 121 / cpu)
# =============================================================================
set -eo pipefail  # NOTE: no -u: conda activation hooks reference unset vars

ENV_NAME="${S2VS_ENV_NAME:-shape}"
PYTHON_VERSION="${S2VS_PYTHON:-3.9}"
TORCH_VERSION="${S2VS_TORCH_VERSION:-2.2.0}"
TORCHVISION_VERSION="${S2VS_TORCHVISION_VERSION:-0.17.0}"
CUDA_TAG="${S2VS_CUDA:-118}"
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

# ---------------------------------------- find the 3DShape2VecSet repo root
is_s2vs_repo() {
    [ -f "$1/main_ae.py" ] && [ -f "$1/models_ae.py" ] \
        && [ -f "$1/util/datasets.py" ] && [ -f "$1/util/shapenet.py" ]
}
find_repo_root() {
    local base="$1"
    if is_s2vs_repo "$base"; then echo "$base"; return 0; fi
    local f
    while IFS= read -r f; do
        if is_s2vs_repo "$(dirname "$f")"; then dirname "$f"; return 0; fi
    done < <(find "$base" -mindepth 2 -maxdepth 2 -name main_ae.py 2>/dev/null)
    return 1
}

if [ -n "$REPO_DIR" ]; then REPO_DIR="$(cd "$REPO_DIR" && pwd)"; else REPO_DIR="$SCRIPT_DIR"; fi
if ! REPO_DIR="$(find_repo_root "$REPO_DIR")"; then
    echo "ERROR: cannot find the 3DShape2VecSet repo under '$REPO_DIR'." >&2
    echo "       Run: bash setup_shape2vecset_env.sh /path/to/3DShape2VecSet" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "==> 3DShape2VecSet repo: $REPO_DIR"

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
echo "    torch        : $TORCH_VERSION / torchvision $TORCHVISION_VERSION / cu$CUDA_TAG"
echo "    recreate     : $RECREATE_STR"
echo "    dry-run      : $DRYRUN_STR"
echo "    steps: create env -> conda torch -> pip deps (+torch_cluster) ->"
echo "           auto-patch torch2 compat -> verify"

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
    echo "    [dry-run] conda install pytorch=$TORCH_VERSION torchvision=$TORCHVISION_VERSION pytorch-cuda=$CUDA_TAG -c pytorch -c nvidia"
elif python -c "import torch, torchvision" >/dev/null 2>&1; then
    echo "==> torch already present in '$ENV_NAME', skipping conda install"
else
    echo "==> installing pytorch $TORCH_VERSION + torchvision $TORCHVISION_VERSION (cuda $CUDA_TAG)"
    if [ "$CUDA_TAG" = "cpu" ]; then
        conda install -y -n "$ENV_NAME" "pytorch=$TORCH_VERSION" "torchvision=$TORCHVISION_VERSION" cpuonly -c pytorch
    else
        conda install -y -n "$ENV_NAME" "pytorch=$TORCH_VERSION" "torchvision=$TORCHVISION_VERSION" "pytorch-cuda=$CUDA_TAG" -c pytorch -c nvidia
    fi
fi

# ----------------------------------------------------------- pip deps
echo "==> installing python deps"
run python -m pip install -U pip
run python -m pip install \
    "numpy==1.26.4" "scipy==1.13.1" "Pillow==11.3.0" "h5py==3.14.0" \
    "PyYAML==6.0.2" "PyMCubes==0.1.6" "scikit-image==0.24.0" \
    "tensorboard==2.21.0" "tqdm==4.68.4" "psutil==7.2.2" \
    "einops==0.8.2" "timm==1.0.27" "trimesh==4.12.2"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "    [dry-run] pip install torch-cluster from data.pyg.org (torch ${TORCH_VERSION}+cu${CUDA_TAG})"
elif python -c "import torch_cluster" >/dev/null 2>&1; then
    echo "==> torch_cluster already present, skipping"
else
    echo "==> installing torch_cluster (PyG wheel for torch $TORCH_VERSION + cu$CUDA_TAG)"
    python -m pip install "torch-cluster" \
        -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_TAG}.html"
fi

# ------------------------------------ torch>=2.0 compatibility auto-patch
if [ "$DRY_RUN" -ne 1 ]; then
    echo "==> applying torch>=2.0 compatibility patches (idempotent)"
    python - <<'PY'
import pathlib
fixes = []
p = pathlib.Path("util/misc.py")
s = p.read_text()
if "from torch._six import inf" in s:
    p.write_text(s.replace("from torch._six import inf",
                           "from math import inf  # torch>=2.0 removed torch._six", 1))
    fixes.append("util/misc.py: torch._six -> math.inf")
q = pathlib.Path("util/shapenet.py")
s = q.read_text()
marker = "if not os.path.exists(point_path):"
if "4_pointcloud" not in s.split("try:")[0]:
    old = """        point_path = os.path.join(self.point_folder, category, model+'.npz')
        try:"""
    new = """        point_path = os.path.join(self.point_folder, category, model+'.npz')
        # compatibility: some converters store point data under <cat>/4_pointcloud/
        if not os.path.exists(point_path):
            point_path = os.path.join(self.point_folder, category, '4_pointcloud', model+'.npz')
        try:"""
    if old in s:
        q.write_text(s.replace(old, new, 1))
        fixes.append("util/shapenet.py: accept <cat>/4_pointcloud/ layout")
print("patches applied:", fixes if fixes else "none needed (already patched)")
PY
fi

# --------------------------------------------------------------- verify
if [ "$DRY_RUN" -eq 1 ]; then echo "==> dry-run finished (nothing installed)."; exit 0; fi
echo "==> verifying installation"
python - <<'PY'
import importlib
for m in ["torch", "torchvision", "timm", "einops", "torch_cluster", "mcubes",
          "trimesh", "scipy", "numpy", "yaml", "h5py", "tensorboard", "tqdm"]:
    importlib.import_module(m)
import torch, torch_cluster
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      ("| gpu: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    idx = torch_cluster.fps(torch.randn(2048, 3), ratio=0.5)
    print("torch_cluster.fps OK", tuple(idx.shape))
# repo modules import (needs the compat patches above)
import util.misc, util.datasets, models_ae, engine_ae
print("repo modules import OK")
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:  conda activate $ENV_NAME"
echo "  Repo: $REPO_DIR"
echo "================================================================"
