#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment setup for DiGS
#  (Divergence guided shape implicit neural representation, ICCV 2021)
#
#  Works from ANY clone path:
#    * script inside the clone -> just run:  bash setup_digs_env.sh
#    * elsewhere              -> bash setup_digs_env.sh /path/to/DiGS
#    * parent dir containing the clone -> auto-detected (looks for the DiGS
#      requirements.txt marker one level down)
#
#  Why this recipe (problems solved):
#    * open3d==0.11.2 (required by surface_reconstruction/recon_dataset.py)
#      declares a dependency on the DEPRECATED PyPI placeholder 'sklearn',
#      which now refuses to install -> pip fails with "The 'sklearn' PyPI
#      package is deprecated ...". Nothing in DiGS imports sklearn; the env var
#      SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True lets pip install
#      that empty placeholder so open3d resolves. (scikit-image, i.e. the real
#      'skimage', is installed normally from requirements.txt.)
#    * Official README uses python 3.7.9 + torch 1.8/CUDA 10.2, but torch 1.8
#      has NO kernels for Ampere/RTX 30-series (sm_86). The newest torch that
#      still ships python-3.7 wheels is 1.13.1; the +cu117 build supports
#      sm_86, so we install torch 1.13.1+cu117 by default. Override below.
#    * plotly is REQUIRED at import time (DiGS utils/visualizations.py does
#      "import plotly"), so the script installs it by default. plotly-orca is
#      optional (--with-viz) and only needed for static-image export.
#
#  Usage:
#      bash setup_digs_env.sh [repo_dir] [--recreate] [--dry-run] [--with-viz]
#  Overrides (env vars):
#      DIGS_ENV_NAME=digs        DIGS_PYTHON=3.7.9
#      DIGS_TORCH_VERSION=1.13.1 DIGS_TORCH_CUDA=117   (or e.g. 113 / 102 / cpu)
# =============================================================================
set -eo pipefail  # NOTE: no -u: conda activation hooks reference unset vars

ENV_NAME="${DIGS_ENV_NAME:-digs}"
PYTHON_VERSION="${DIGS_PYTHON:-3.7.9}"
TORCH_VERSION="${DIGS_TORCH_VERSION:-1.13.1}"
TORCH_CUDA="${DIGS_TORCH_CUDA:-117}"     # torch wheel tag: 117/116/113/102/cpu
RECREATE=0
DRY_RUN=0
WITH_VIZ=0
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
        --with-viz)    WITH_VIZ=1 ;;
        --repo)        shift; REPO_DIR="$1" ;;
        -*)            echo "unknown option: $1" >&2; usage ;;
        *)             REPO_DIR="$1" ;;
    esac
    shift
done

# ----------------------------------------------- find the DiGS repo root
is_digs_repo() {  # $1 = dir ; true if it is a DiGS checkout
    [ -f "$1/requirements.txt" ] \
        && grep -q "open3d==0.11.2" "$1/requirements.txt" \
        && grep -q "scikit_image==0.18.1" "$1/requirements.txt" \
        && [ -d "$1/surface_reconstruction" ]
}
find_repo_root() {
    local base="$1"
    if is_digs_repo "$base"; then echo "$base"; return 0; fi
    local f
    while IFS= read -r f; do
        if is_digs_repo "$(dirname "$f")"; then
            dirname "$f"; return 0
        fi
    done < <(find "$base" -mindepth 2 -maxdepth 2 -name requirements.txt 2>/dev/null)
    return 1
}

if [ -n "$REPO_DIR" ]; then REPO_DIR="$(cd "$REPO_DIR" && pwd)"; else REPO_DIR="$SCRIPT_DIR"; fi
if ! REPO_DIR="$(find_repo_root "$REPO_DIR")"; then
    echo "ERROR: cannot find the DiGS repo under '$REPO_DIR'." >&2
    echo "       Run: bash setup_digs_env.sh /path/to/DiGS" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "==> DiGS repo: $REPO_DIR"

# ------------------------------------------------------------- preflight
command -v conda >/dev/null || { echo "ERROR: conda not found on PATH"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then ENV_EXISTS=1; else ENV_EXISTS=0; fi

EXISTS_STR=$([ "$ENV_EXISTS" -eq 1 ] && echo yes || echo no)
RECREATE_STR=$([ "$RECREATE" -eq 1 ] && echo yes || echo no)
DRYRUN_STR=$([ "$DRY_RUN" -eq 1 ] && echo yes || echo no)
VIZ_STR=$([ "$WITH_VIZ" -eq 1 ] && echo "yes (plotly/orca)" || echo no)
echo "==> plan:"
echo "    env          : $ENV_NAME (python $PYTHON_VERSION; exists: $EXISTS_STR)"
echo "    torch        : $TORCH_VERSION (+cu$TORCH_CUDA)"
echo "    recreate     : $RECREATE_STR"
echo "    dry-run      : $DRYRUN_STR"
echo "    plotly-orca  : $VIZ_STR   (optional static-image export)"
echo "    steps: create env -> pip<24 -> SKLEARN_ALLOW_DEPRECATED... -> "
echo "           pip install -r requirements.txt -> install torch -> verify"

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

# ------------------------------------------------------------ pip + deps
run python -m pip install -U "pip<24"          # last pip supporting python 3.7
echo "==> installing DiGS requirements (sklearn-placeholder fix enabled)"
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
run python -m pip install -r requirements.txt

# ----------------------- real scikit-learn (open3d.ml needs it at import)
# open3d==0.11.2 imports "from sklearn.neighbors import KDTree" when the
# package is imported, and its pip dependency is only the DEPRECATED empty
# 'sklearn' placeholder. Install the real scikit-learn (py3.7-compatible,
# matching numpy 1.19 / scipy 1.6) so open3d can be imported.
echo "==> installing scikit-learn 0.24.2 (real one; open3d.ml imports sklearn)"
run python -m pip install "scikit-learn==0.24.2"
# tensorboardX 2.3 ships old protobuf-generated files; needs protobuf < 4
echo "==> pinning protobuf 3.20.3 (tensorboardX 2.3 compatibility)"
run python -m pip install "protobuf==3.20.3"
# DiGS utils/visualizations.py does "import plotly" at module load, and the
# dataset / train scripts import it -> plotly is required even to train.
# (plotly-orca / kaleido static-image export stays optional, see --with-viz.)
echo "==> installing plotly (required: utils/visualizations.py imports it)"
run python -m pip install "plotly==5.9.0"

# ----------------------------------------------------------------- torch
echo "==> installing torch $TORCH_VERSION (+cu$TORCH_CUDA, python3.7-compatible)"
if [ "$TORCH_CUDA" = "cpu" ]; then
    run python -m pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
else
    run python -m pip install "torch==$TORCH_VERSION+cu$TORCH_CUDA" \
        --index-url "https://download.pytorch.org/whl/cu$TORCH_CUDA"
fi

# ------------------------------------------- optional visualisation deps
if [ "$WITH_VIZ" -eq 1 ]; then
    echo "==> installing optional visualisation deps (plotly / plotly-orca)"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    [dry-run] conda install -y -n $ENV_NAME -c plotly plotly plotly-orca"
    else
        conda install -y -n "$ENV_NAME" -c plotly plotly plotly-orca \
            || echo "WARNING: plotly/plotly-orca install failed (visualisation only, skipping)"
    fi
fi

# --------------------------------------------------------------- verify
if [ "$DRY_RUN" -eq 1 ]; then echo "==> dry-run finished (nothing installed)."; exit 0; fi
echo "==> verifying installation"
python - <<'PY'
import sys
print("python :", sys.version.split()[0])
import numpy, scipy, trimesh, matplotlib, skimage, open3d, plyfile, tensorboardX, PIL
print("numpy   :", numpy.__version__)
print("scipy   :", scipy.__version__)
print("trimesh :", trimesh.__version__)
print("skimage :", skimage.__version__)
print("open3d  :", open3d.__version__)
import torch
print("torch   :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      ("| gpu: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "")
if torch.cuda.is_available():   # sanity: real kernel runs on this GPU
    x = torch.rand(1000, device="cuda")
    print("cuda op :", float((x @ x) ** 0.5).__round__(3))
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:  conda activate $ENV_NAME"
echo "  Repo: $REPO_DIR"
echo "  (visualisation: conda install -c plotly plotly plotly-orca  -- optional)"
echo "================================================================"
