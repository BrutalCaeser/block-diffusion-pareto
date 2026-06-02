#!/bin/bash
# =============================================================================
# build_blockpareto_env.sbatch
# Builds the dedicated conda env for the Block-Size Pareto project on Explorer.
#
# WHY a SLURM job (not login node): Explorer login nodes cap interactive
# memory/CPU and KILL heavy processes (observed: conda create -> "Killed").
# Env build is CPU/IO/network heavy (no GPU needed), so we run it on a compute
# node with generous CPU/mem. Env lives in /scratch (huge, no home quota),
# matching the user's existing pattern (/scratch/gupta.yashv/flm/envs/flm).
#
# Submit:  sbatch env/build_blockpareto_env.sbatch
# Log:     /scratch/gupta.yashv/block-pareto/logs/env_build_<jobid>.log
# =============================================================================
#SBATCH --job-name=bp-env-build
#SBATCH --partition=short
#SBATCH --time=01:30:00
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
print("torch.cuda  :", torch.version.cuda, "(build tag; is_available False on CPU node is EXPECTED)")
print("cuda avail  :", torch.cuda.is_available())
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
