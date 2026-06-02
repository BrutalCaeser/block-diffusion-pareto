#!/bin/bash
# =============================================================================
# p1_prep_owt.sbatch  —  Phase 1, step A: warm the OpenWebText download cache.
#
# WHY a separate CPU job: bd3lms ppl_eval needs the OWT *valid* split
# (openwebtext train[-100000:]). HF's `openwebtext` is a single-split dataset, so
# even the 100k-doc valid slice forces a download of ALL shards (~40 GB) + extract.
# This is CPU/network/IO heavy and unbounded in time -> run it on `short` (CPU,
# 2-day limit), NOT on a GPU partition (2h cap + wasted idle GPU during download).
# The follow-up GPU job (p1_ppl_owt.sbatch) reuses this cache (same cache_dir) so
# its load_dataset() is a cache HIT and it only tokenizes valid (~minutes) + evals.
#
# We do NOT tokenize here on purpose: tokenization params (eos/special tags) must
# match main.py exactly; rather than hand-replicate them, we let main.py tokenize
# on the GPU job. Here we only populate the raw HF arrow cache -> zero param risk.
#
# Submit:  sbatch exp/p1_prep_owt.sbatch
# =============================================================================
#SBATCH --job-name=bp-p1-prep
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/p1_prep_%j.log

set -euo pipefail

PROJECT=/scratch/gupta.yashv/block-pareto
ENV_PREFIX=$PROJECT/envs/blockpareto
DATA_DIR=$PROJECT/data           # == data.cache_dir used by the GPU eval job
export HF_HOME=$PROJECT/hf
mkdir -p "$DATA_DIR" "$HF_HOME"

echo "=========================================================="
echo "node     : $(hostname)"
echo "date     : $(date)"
echo "data dir : $DATA_DIR  (shared with GPU eval job -> cache hit)"
echo "=========================================================="

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
echo "python: $(which python) -> $(python --version)"

echo "### connectivity"
curl -sI --max-time 20 https://huggingface.co | head -n 1 || { echo "FATAL: no net"; exit 2; }

echo "### download + cache OpenWebText valid slice (train[-100000:])"
python - <<'PY'
import time, datasets
print("datasets:", datasets.__version__)
t0 = time.time()
ds = datasets.load_dataset(
    'openwebtext',
    split='train[-100000:]',
    cache_dir='/scratch/gupta.yashv/block-pareto/data',
    trust_remote_code=True,
)
dt = time.time() - t0
print(f"OK: loaded {len(ds)} valid docs in {dt:.0f}s")
print("sample[0] head:", repr(ds[0]['text'][:160]))
PY

echo "=========================================================="
echo "DONE prep: $(date)"
echo "=========================================================="
