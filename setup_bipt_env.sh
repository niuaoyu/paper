#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment setup for Bi-PT
#  (https://github.com/Chenchuhui/Bi-PT  — 4-chamber heart reconstruction)
#
#  Works from ANY git clone path:
#    * if this script sits inside the clone (recommended): just run
#        bash setup_bipt_env.sh
#      it auto-detects the repo root (the directory containing environment.yml)
#      from its own location.
#    * if the script lives elsewhere, pass the repo path explicitly:
#        bash /path/to/setup_bipt_env.sh /path/to/Bi-PT
#      or place it in a parent folder that contains a Bi-PT clone
#      (it searches one level down for an environment.yml with "name: bipt").
#
#  It performs exactly the official install steps:
#      conda env create -f environment.yml
#      conda activate bipt
#      pip install -e .
#      pip install -r requirements-preprocess.txt
#      pip install -r requirements-eval.txt
#  plus one auto-fix: requirements-preprocess.txt needs vtk>=9.3 while the
#  conda recipe can resolve vtk 9.2.x; pip cannot replace a conda-installed
#  (distutils) vtk, so if vtk < 9.3 is present the script removes it first and
#  lets pip install vtk>=9.3. Skipped automatically when vtk is already >= 9.3.
#
#  Usage:
#      bash setup_bipt_env.sh [repo_dir] [--recreate] [--dry-run] [--help]
#
#  Notes:
#    * environment.yml pins pytorch-cuda=12.1; on machines with a different
#      CUDA toolkit / driver, edit environment.yml before running (see its
#      header comments). BIPT_ENV_NAME overrides the env name if you changed it.
#    * requirements-eval.txt mentions torch-mesh-isect (CUDA, not on PyPI);
#      it is OPTIONAL, only used for the SI metric -- the script leaves it to
#      you (see https://github.com/vchoutas/torch-mesh-isect).
# =============================================================================
set -eo pipefail  # NOTE: no -u: conda activation hooks reference unset vars

ENV_NAME="${BIPT_ENV_NAME:-bipt}"
RECREATE=0
DRY_RUN=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=""

usage() {
    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# ---------------------------------------------------------------- arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)    usage ;;
        --recreate)   RECREATE=1 ;;
        --dry-run)    DRY_RUN=1 ;;
        --repo)       shift; REPO_DIR="$1" ;;
        -*)           echo "unknown option: $1" >&2; usage ;;
        *)            REPO_DIR="$1" ;;
    esac
    shift
done

# ------------------------------------------------- find the repo root
# 1) environment.yml right here -> this is the repo root
# 2) otherwise look one level down for an environment.yml whose name is bipt
find_repo_root() {
    local base="$1"
    if [ -f "$base/environment.yml" ] && grep -q "name: bipt" "$base/environment.yml"; then
        echo "$base"; return 0
    fi
    local f
    while IFS= read -r f; do
        if grep -q "name: bipt" "$f"; then
            dirname "$f"; return 0
        fi
    done < <(find "$base" -mindepth 2 -maxdepth 2 -name environment.yml 2>/dev/null)
    return 1
}

if [ -n "$REPO_DIR" ]; then
    REPO_DIR="$(cd "$REPO_DIR" && pwd)"
else
    REPO_DIR="$SCRIPT_DIR"
fi
if ! REPO_DIR="$(find_repo_root "$REPO_DIR")"; then
    echo "ERROR: cannot find the Bi-PT repo (no environment.yml with 'name: bipt'" >&2
    echo "       under '$REPO_DIR'). Run: bash setup_bipt_env.sh /path/to/Bi-PT" >&2
    exit 1
fi
cd "$REPO_DIR"
echo "==> Bi-PT repo: $REPO_DIR"

# ---------------------------------------------------------------- preflight
command -v conda >/dev/null || { echo "ERROR: conda not found on PATH"; exit 1; }
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    ENV_EXISTS=1
else
    ENV_EXISTS=0
fi

echo "==> plan:"
echo "    env name   : $ENV_NAME (exists: $([ $ENV_EXISTS -eq 1 ] && echo yes || echo no))"
echo "    recreate   : $([ $RECREATE -eq 1 ] && echo yes || echo no)"
echo "    dry-run    : $([ $DRY_RUN -eq 1 ] && echo yes || echo no)"
echo "    steps      : conda env create -> conda activate -> pip install -e ."
echo "                 -> (vtk>=9.3 auto-fix) -> pip install -r requirements-preprocess.txt"
echo "                 -> pip install -r requirements-eval.txt"

# run a real command, or print it when in dry-run mode
run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    [dry-run] $*"
    else
        echo "    \$ $*"
        "$@"
    fi
}
# conda activate is a shell function -> must be executed in this shell
do_activate() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    [dry-run] conda activate $ENV_NAME"
    else
        conda activate "$ENV_NAME"
    fi
}

# ------------------------------------------------------------------ create
if [ "$RECREATE" -eq 1 ] && [ "$ENV_EXISTS" -eq 1 ]; then
    echo "==> removing existing env '$ENV_NAME' (--recreate)"
    run conda env remove -n "$ENV_NAME" -y
    run conda env create -f environment.yml
    do_activate
elif [ "$ENV_EXISTS" -eq 1 ]; then
    echo "==> env '$ENV_NAME' already exists, reusing it (use --recreate to rebuild)"
    do_activate
else
    echo "==> creating env '$ENV_NAME' from environment.yml"
    run conda env create -f environment.yml
    do_activate
fi

# ------------------------------------------ vtk>=9.3 fix (preprocess needs it)
# requirements-preprocess.txt needs vtk>=9.3, but the conda recipe can resolve
# vtk 9.2.x; pip cannot replace a conda-installed (distutils) vtk, so remove it
# first and let pip install vtk>=9.3. Skipped when vtk is already >=9.3 or
# already pip-managed (source column == "pypi").
if [ "$DRY_RUN" -eq 1 ]; then
    echo "    [dry-run] ensure vtk>=9.3 (remove conda-managed vtk<9.3 if present)"
else
    vtk_row="$(conda list -n "$ENV_NAME" vtk 2>/dev/null | awk '$1=="vtk"{print; exit}')"
    if [ -n "$vtk_row" ]; then
        v="$(echo "$vtk_row" | awk '{print $2}')"
        src="$(echo "$vtk_row" | awk '{print $NF}')"
        if [ "$src" != "pypi" ] && ! python -c "import sys; from packaging.version import Version; sys.exit(0 if Version('$v') >= Version('9.3') else 1)"; then
            echo "==> conda-managed vtk $v is < 9.3; removing it so pip can install vtk>=9.3"
            conda remove -n "$ENV_NAME" vtk -y
        fi
    fi
fi

# ------------------------------------------------------------------ install
run python -m pip install -e .
run python -m pip install -r requirements-preprocess.txt
run python -m pip install -r requirements-eval.txt

# ------------------------------------------------------------------ verify
if [ "$DRY_RUN" -eq 1 ]; then
    echo "==> dry-run finished (nothing was installed)."
    exit 0
fi

echo "==> verifying installation"
python - <<'PY'
import sys
import torch
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "| cuda available:", torch.cuda.is_available())
try:
    import vtk
    print("vtk    :", vtk.vtkVersion.GetVTKVersion())
except Exception as e:
    print("vtk import warning:", repr(e))
try:
    import bipt
    print("bipt   : import OK ->", bipt.__file__)
except Exception as e:  # package import may need torch; report but don't fail
    print("bipt import warning:", repr(e))
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:  conda activate $ENV_NAME"
echo "  Repo: $REPO_DIR"
echo "================================================================"
