#!/usr/bin/env bash
# =============================================================================
#  One-click conda environment for MeshDiffusion + its bundled nvdiffrec
#  (https://github.com/nv-tlabs/MeshDiffusion)
#
#  Validated on:
#    - Ubuntu 22.04+ x86_64, NVIDIA GPU (RTX 3090 Ti / sm_86), conda
#    - CUDA host driver >= 525 (any newer driver is forward compatible)
#
#  What it installs (exact versions, all pitfalls pre-resolved):
#    python 3.9 | CUDA toolkit 12.1 (in-env, provides nvcc)
#    torch==2.1.0+cu121  torchvision==0.16.0+cu121  numpy==1.26.4  (<2 !)
#    nvdiffrast (git, pinned)  tiny-cuda-nn (git, pinned)
#    pytorch3d 0.7.7 (built from source for the detected GPU arch)
#    kaolin 0.15.0 (built from source: NVIDIA wheels are numpy2-only)
#    pymeshlab, xatlas, opencv-python-headless 4.10 (numpy1-compatible,
#        imageio needs it to read .hdr envmaps), imageio, ml_collections,
#        absl-py, tensorboard, scipy, tqdm, PyOpenGL, glfw
#
#  Why these exact choices (troubles we hit):
#    * This repo's code + prebuilt nvdiffrast/tiny-cuda-nn target torch 2.1;
#      installing a newer/CPU torch silently breaks everything.
#    * numpy must stay < 2: torch 2.1 / kaolin / pytorch3d are built against 1.x.
#      Never let opencv>=5 or pymeshlab upgrades pull in numpy 2.
#    * NVIDIA kaolin wheels on the S3 index are compiled against numpy 2, so we
#      build kaolin from source (needs cython 0.29.36 + numpy 1.x headers).
#    * A conda "activate" hook sets LD_LIBRARY_PATH so `import torch` + CUDA work.
#
#  Usage:
#    bash setup_meshdiff_env.sh [env_name]        # default env: meshdiff
#  Optional overrides (env vars):
#    MESHDIFF_CUDA_ARCH=86          # tiny-cuda-nn arch (default: auto from GPU)
#    MAX_JOBS=8                    # parallel compile jobs
# =============================================================================
set -euo pipefail

ENV_NAME="${1:-meshdiff}"
MAX_JOBS="${MAX_JOBS:-8}"

echo "==> MeshDiffusion env setup: env='$ENV_NAME'"

# ------------------------------------------------------------------ checks
for c in conda git nvidia-smi gcc g++; do
    command -v "$c" >/dev/null || { echo "ERROR: '$c' not found on PATH"; exit 1; }
done
command -v conda >/dev/null
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ------------------------------------------------------------ create env
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> conda env '$ENV_NAME' exists, reusing it"
    conda activate "$ENV_NAME"
else
    conda create -y -n "$ENV_NAME" python=3.9
    conda activate "$ENV_NAME"
fi
python -m pip install -U pip wheel

export CONDA_PREFIX="$(conda info --base)/envs/$ENV_NAME"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export PYOPENGL_PLATFORM=egl
export MAX_JOBS="$MAX_JOBS"

# ------------------------------------------------- CUDA toolkit (nvcc)
if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then
    echo "==> installing CUDA toolkit 12.1 into the env (nvcc for JIT builds)"
    conda install -y -n "$ENV_NAME" -c nvidia/label/cuda-12.1.0 cuda
fi

# --------------------------------------------------- PyTorch + numpy
echo "==> installing torch 2.1.0+cu121 / torchvision 0.16.0+cu121 (big download)"
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install "numpy==1.26.4"   # keep <2: never let anything upgrade this

# torch needs torch/lib on LD_LIBRARY_PATH to import correctly in this env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# -------------------------------------------------- GPU arch detection
ARCH="$(python -c "import torch;c=torch.cuda.get_device_capability();print('%d%d'%c)")"
TORCH_ARCH="$(python -c "import torch;c=torch.cuda.get_device_capability();print('%d.%d'%c)")"
export TCNN_CUDA_ARCHITECTURES="${MESHDIFF_CUDA_ARCH:-$ARCH}"
export TORCH_CUDA_ARCH_LIST="${MESHDIFF_TORCH_ARCH:-$TORCH_ARCH}"
echo "==> detected GPU compute capability $TORCH_ARCH (TCNN arch=$TCNN_CUDA_ARCHITECTURES)"

# ------------------------------------------------------- build tools
pip install ninja "setuptools==67.8.0" "cython==0.29.36"

# ------------------------------------------- pip deps (pure wheels)
pip install ml_collections absl-py tensorboard tqdm scipy imageio \
    "opencv-python-headless==4.10.0.84" pymeshlab xatlas PyOpenGL glfw

# -------------------------------------------- compiled GPU extensions
echo "==> building nvdiffrast (pinned commit, matches torch 2.1)"
pip install --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast@253ac4fcea7de5f396371124af597e6cc957bfae"

echo "==> building tiny-cuda-nn (pinned commit, no networks)"
pip install --no-build-isolation \
    --config-settings="--global-option=--no-networks" \
    "git+https://github.com/NVlabs/tiny-cuda-nn@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"

echo "==> building pytorch3d 0.7.7 from source (for torch 2.1)"
pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

echo "==> building kaolin 0.15.0 from source (numpy-1.x compatible)"
pip install --no-build-isolation \
    "git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.15.0"

# ----------------------------- activation hook (runtime environment)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/meshdiff_env.sh" <<EOF
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:\$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
export CPATH="$CONDA_PREFIX/include:\$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:\$LIBRARY_PATH"
export PYOPENGL_PLATFORM=egl
export TCNN_CUDA_ARCHITECTURES=$TCNN_CUDA_ARCHITECTURES
EOF

# --------------------------------------------------------- verify
echo "==> verifying installation"
conda activate "$ENV_NAME"
python - <<'PY'
import numpy as np, torch
assert torch.__version__.startswith("2.1.0"), torch.__version__
assert np.__version__.startswith("1.26"), np.__version__
assert torch.cuda.is_available(), "CUDA not available - check driver"
print("gpu:", torch.cuda.get_device_name(0))
import torchvision
import nvdiffrast.torch as dr
import tinycudann, xatlas, pymeshlab, cv2, imageio, scipy, ml_collections
import kaolin, pytorch3d
print("torch", torch.__version__, "| numpy", np.__version__, "| kaolin", kaolin.__version__,
      "| pytorch3d", pytorch3d.__version__)
PY

echo
echo "================================================================"
echo "  DONE. Activate anytime with:"
echo "      conda activate $ENV_NAME"
echo "  Then, from the repo:"
echo "      cd MeshDiffusion/nvdiffrec"
echo "      python eval.py --config ./configs/res64.json \\"
echo "          --out-dir ./outputs/meshes \\"
echo "          --sample-path ../outputs/my_chair_gen/0.npy --deform-scale 3.0"
echo "================================================================"
