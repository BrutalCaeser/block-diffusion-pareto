#!/bin/bash
# =============================================================================
# build_blockpareto_env.sbatch
# Builds the dedicated conda env for the Block-Size Pareto project on Explorer.
#
# WHY a SLURM job (not login node): Explorer login nodes cap interactive
# memory/CPU and KILL heavy processes (observed: conda create -> "Killed").
# Env build is CPU/IO/network heavy. We run it on the `gpu-short` partition
# with 1 GPU: (a) the CPU `short` partition was jammed (0 idle nodes) so jobs
# sat in queue, while `gpu-short` had idle nodes -> immediate start; (b) bonus
# -> running on a GPU node lets us verify torch.cuda.is_available()==True as a
# real Gate-G0 check. This is a one-time expedient for a ~20-min build; future
# heavy CPU-only work should use `short`. Env lives in /scratch (no home quota),
# matching the user's existing pattern (/scratch/gupta.yashv/flm/envs/flm).
#
# Submit:  sbatch env/build_blockpareto_env.sbatch
# Log:     /scratch/gupta.yashv/block-pareto/logs/env_build_<jobid>.log
# =============================================================================
#SBATCH --job-name=bp-env-build
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/env_build_%j.log

set -euo pipefail

PROJECT=/scratch/gupta.yashv/block-pareto
REPO=$PROJECT/bd3lms
ENV_PREFIX=$PROJECT/envs/blockpareto

echo "=========================================================="
echo "node      : $(hostname)"
echo "date      : $(date)"
echo "project   : $PROJECT"
echo "env prefix: $ENV_PREFIX"
echo "=========================================================="

# --- 0. Sanity: compute node internet (pip/conda need PyPI/anaconda) ---------
echo "### [0] connectivity check"
if ! curl -sI --max-time 20 https://pypi.org/simple/ | head -n 1 ; then
  echo "FATAL: no outbound internet on this compute node — cannot pip install."
  echo "Fallback: build env on a node/partition that has internet."
  exit 2
fi

# --- 1. Conda available -------------------------------------------------------
echo "### [1] load conda"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda --version

# Isolate this project's conda PACKAGE cache. The shared cache (set in ~/.condarc
# pkgs_dirs -> /scratch/.../matrix-game/conda-pkgs) had a CORRUPTED wheel-0.45.1
# package (CondaVerificationError) that failed env creation on jobs 7380571/7380683.
# A fresh project-local pkgs dir makes conda DOWNLOAD clean packages here...
export CONDA_PKGS_DIRS="$PROJECT/conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"
echo "CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"

# ...BUT CONDA_PKGS_DIRS only controls where conda WRITES. Conda still scans every
# pkgs_dir in ~/.condarc for an already-extracted package to hard-link, so it kept
# reusing the corrupted wheel-0.45.1 extraction. Purge that one broken package from
# ALL known pkgs_dirs (dir + .conda/.tar.bz2 tarball) so conda must re-download it
# clean. We delete ONLY the corrupted package, nothing else in the shared cache.
echo "### [1b] purge corrupted wheel-0.45.1 from all pkgs_dirs"
for d in "$CONDA_PKGS_DIRS" $(conda config --show pkgs_dirs 2>/dev/null | grep -E '^\s+-\s' | sed 's/^\s*-\s*//'); do
  for victim in "$d"/wheel-0.45.1-py39h06a4308_0 "$d"/wheel-0.45.1-py39h06a4308_0.conda "$d"/wheel-0.45.1-py39h06a4308_0.tar.bz2; do
    if [ -e "$victim" ]; then echo "  rm -rf $victim"; rm -rf "$victim"; fi
  done
done

# --- 2. Clean any partial/failed env -----------------------------------------
echo "### [2] clean prior attempts"
conda env remove -n blockpareto -y 2>/dev/null || true   # home-default failed attempt
conda env remove -p "$ENV_PREFIX" -y 2>/dev/null || true
rm -rf "$ENV_PREFIX"

# --- 3. Create env (python 3.9, faithful to bd3lms repo) ---------------------
echo "### [3] create env (python 3.9)"
conda create -y -p "$ENV_PREFIX" python=3.9
conda activate "$ENV_PREFIX"
echo "python: $(which python)  ->  $(python --version)"

# --- 4. Install dependencies from the REPO's own requirements.txt -------------
# (install from the real file, not a transcription, so it stays grounded.)
# NOTE: requirements.txt pins torch==2.7.1 (PyPI default = CUDA 12.6 wheel,
# self-contained runtime). flash-attn is intentionally NOT installed: bd3lm
# trains with FlexAttention (native to torch>=2.5) and samples with sdpa;
# flash-attn 2.5.6 (README "optional, for MDLM baselines") will not build
# against torch 2.7.1 and is deferred until/if baselines need it.
echo "### [4] pip install -r requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r "$REPO/requirements.txt"

# --- 5. Verify the install (imports + CUDA build tag) ------------------------
echo "### [5] verify"
python - <<'PY'
import sys
print("python      :", sys.version.split()[0])
import torch
print("torch       :", torch.__version__)
print("torch.cuda  :", torch.version.cuda, "(CUDA build tag)")
print("cuda avail  :", torch.cuda.is_available(), "(expect True on gpu-short node)")
if torch.cuda.is_available():
    print("gpu         :", torch.cuda.get_device_name(0))
import transformers, lightning, hydra, omegaconf, datasets, triton, einops, timm, wandb
print("transformers:", transformers.__version__)
print("lightning   :", lightning.__version__)
print("datasets    :", datasets.__version__)
print("triton      :", triton.__version__)
print("hydra-core  :", hydra.__version__)
print("ALL CORE IMPORTS OK")
PY

# --- 6. Freeze the exact resolved versions (provenance) ----------------------
echo "### [6] freeze -> env/requirements.lock.txt"
python -m pip freeze > "$PROJECT/repo/env/requirements.lock.txt" 2>/dev/null || \
  python -m pip freeze > "$PROJECT/requirements.lock.txt"

echo "=========================================================="
echo "DONE env build: $(date)"
echo "activate with:  conda activate $ENV_PREFIX"
echo "=========================================================="
