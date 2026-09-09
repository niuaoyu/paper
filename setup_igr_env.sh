#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment setup for IGR
#  (Implicit Geometric Regularization for Learning Shapes, ICML 2020,
#   https://github.com/amosgropp/IGR)
#
#  Works from ANY clone path:
#    * script inside the clone -> just run:  bash setup_igr_env.sh
#    * elsewhere              -> bash setup_igr_env.sh /path/to/IGR
#    * parent dir containing the clone -> auto-detected one level down
#
#  Why this recipe:
#    * README says python 3.7 + torch 1.2, but torch 1.2 has NO kernels for
#      Ampere/RTX-30 (sm_86). The newest torch with python-3.7 wheels is
#      1.13.1; +cu117 supports sm_86, and IGR's code (plain torch.nn +
#      torch.autograd.grad) runs fine on it.
#    * Dependencies actually used by IGR code (verified by import scan):
#      numpy, scipy, pyhocon, plotly, scikit-image, trimesh, GPUtil.
#      Versions are pinned to the python-3.7 wheel set (same as the digs env).
#    * scikit-image must be 0.18.x: IGR calls measure.marching_cubes_lewiner,
#      which was renamed/removed in later versions.
#
#  Usage:
#      bash setup_igr_env.sh [repo_dir] [--recreate] [--dry-run]
#  Overrides (env vars):
#      IGR_ENV_NAME=igr  IGR_PYTHON=3.7.9
#      IGR_TORCH_VERSION=1.13.1  IGR_TORCH_CUDA=117  (or 113/116/102/cpu)
# =============================================================================
set -eo pipefail  # NOTE: no -u: conda activation hooks reference unset vars

ENV_NAME="${IGR_ENV_NAME:-igr}"
PYTHON_VERSION="${IGR_PYTHON:-3.7.9}"
TORCH_VERSION="${IGR_TORCH_VERSION:-1.13.1}"
TORCH_CUDA="${IGR_TORCH_CUDA:-117}"
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

# ------------------------------------------------- find the IGR repo root
is_igr_repo() {
    [ -f "$1/README.md" ] && [ -d "$1/code" ] && [ -f "$1/code/reconstruction/run.py" ] \
        && [ -f "$1/code/model/network.py" ]
}
find_repo_root() {
    local base="$1"
    if is_igr_repo "$base"; then echo "$base"; return 0; fi
    local d
    while IFS= read -r d; do
        if is_igr_repo "$d"; then echo "$d"; return 0; fi
    done < <(find "$base" -mindepth 1 -maxdepth 2 -type d -name code 2>/dev/null | sed 's#/code$##')
    return 1
}

if [ -n "$REPO_DIR" ]; then REPO_DIR="$(cd "$REPO_DIR" && pwd)"; else REPO_DIR="$SCRIPT_DIR"; fi
if ! REPO_DIR="$(find_repo_root "$REPO_DIR")"; then
    echo "ERROR: cannot find the IGR repo under '$REPO_DIR'." >&2
    echo "       Run: bash setup_igr_env.sh /path/to/IGR" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "==> IGR repo: $REPO_DIR"

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
echo "    steps: create env -> pip deps (numpy/scipy/pyhocon/plotly/"
echo "           scikit-image/trimesh/GPUtil) -> install torch -> verify"

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

# ------------------------------------------------ python-3.7 compatible deps
run python -m pip install -U "pip<24"          # last pip supporting python 3.7
echo "==> installing IGR runtime deps (python3.7 wheel set)"
run python -m pip install \
    "numpy==1.19.2" \
    "scipy==1.6.2" \
    "pyhocon" \
    "plotly==5.9.0" \
    "scikit-image==0.18.1" \
    "trimesh==3.9.14" \
    "GPUtil"

# ----------------------------------------------------------------- torch
echo "==> installing torch $TORCH_VERSION (+cu$TORCH_CUDA, python3.7-compatible)"
if [ "$TORCH_CUDA" = "cpu" ]; then
    run python -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
else
    run python -m pip install "torch==$TORCH_VERSION+cu$TORCH_CUDA" \
        --index-url "https://download.pytorch.org/whl/cu$TORCH_CUDA"
fi

# --------------------------------------------------------------- verify
if [ "$DRY_RUN" -eq 1 ]; then echo "==> dry-run finished (nothing installed)."; exit 0; fi
echo "==> verifying installation"
python - <<'PY'
import sys
print("python :", sys.version.split()[0])
import numpy, scipy, pyhocon, plotly, skimage, trimesh, GPUtil
print("numpy   :", numpy.__version__)
print("scipy   :", scipy.__version__)
print("skimage :", skimage.__version__)
print("plotly  :", plotly.__version__)
print("trimesh :", trimesh.__version__)
import torch
print("torch   :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      ("| gpu: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    x = torch.rand(1000, device="cuda")
    print("cuda op :", float((x @ x) ** 0.5).__round__(3))
PY
echo "==> smoke-testing IGR model code (ImplicitNet + eikonal-style gradient)"
cd "$REPO_DIR/code"
python - <<'PY'
import sys
sys.path.insert(0, ".")
import numpy as np
import torch
from model.network import ImplicitNet, gradient
torch.manual_seed(0)
net = ImplicitNet(d_in=3, dims=[128, 128, 128, 128], skip_in=(4,))
if torch.cuda.is_available():
    net = net.cuda()
    pts = torch.randn(200, 3, device="cuda", requires_grad=True)
else:
    pts = torch.randn(200, 3, requires_grad=True)
out = net(pts)
grad = gradient(pts, out)
loss = ((grad.norm(dim=-1) - 1.0) ** 2).mean()  # eikonal regularizer, as in IGR
loss.backward()
print("model smoke OK | output:", out.shape, "| eikonal loss:", float(loss))
import utils.plots   # plotly + skimage import path used by IGR
print("utils.plots import OK")
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:  conda activate $ENV_NAME"
echo "  Repo: $REPO_DIR"
echo "  Run e.g.: cd code && python reconstruction/run.py"
echo "================================================================"
