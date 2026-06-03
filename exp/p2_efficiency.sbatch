#!/bin/bash
# =============================================================================
# p2_efficiency.sbatch  —  Phase 2: generation-efficiency Pareto sweep.
#
# For each block size in {4,8,16,32,64,128}, run bench/bench_gen.py once and
# collect throughput / NFE / peak-memory. Uses the NATIVE random-init `dit`
# backbone -> works for the UNRELEASED 32/64/128 (efficiency is weight-independent).
# Each block size runs in a FRESH python process so peak-memory is clean.
#
# Output: results/phase2_efficiency.jsonl (one BENCH_JSON record per block size).
#
# Knobs:  BATCH=1 (latency-oriented, matches README var-len eval_batch_size=1)
#         T=5000  (algo.T; first_hitting breaks early per block)
#         LEN=1024
# Submit: sbatch exp/p2_efficiency.sbatch
# =============================================================================
#SBATCH --job-name=bp-p2-eff
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/p2_eff_%j.log

set -euo pipefail

PROJECT=/scratch/gupta.yashv/block-pareto
ENV_PREFIX=$PROJECT/envs/blockpareto
REPO=$PROJECT/repo
export BD3LMS_DIR=$PROJECT/bd3lms
export HF_HOME=$PROJECT/hf
BATCH=${BATCH:-1}
T=${T:-5000}
LEN=${LEN:-1024}
BLOCKS=${BLOCKS:-"4 8 16 32 64 128"}

# Write results OUTSIDE the git clone (avoids `git pull` conflicts); copied back
# into the repo from the local machine and committed there.
RESULTS=$PROJECT/results/phase2_efficiency.jsonl
mkdir -p "$(dirname "$RESULTS")" "$PROJECT/bench_runs/$SLURM_JOB_ID"

echo "node:$(hostname) date:$(date) batch=$BATCH T=$T len=$LEN blocks=[$BLOCKS]"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
python -c "import torch;print('gpu:',torch.cuda.get_device_name(0))"

for BS in $BLOCKS; do
  echo "==================== block_size=$BS ===================="
  RUNDIR=$PROJECT/bench_runs/$SLURM_JOB_ID/bs$BS
  # capture stdout so we can extract the BENCH_JSON line into the results file
  set +e
  OUT=$(python -u "$REPO/bench/bench_gen.py" \
      model=small \
      algo=bd3lm \
      algo.backbone=dit \
      algo.T=$T \
      data=openwebtext-split \
      data.cache_dir=$PROJECT/data \
      model.length=$LEN \
      model.attn_backend=sdpa \
      block_size=$BS \
      loader.eval_batch_size=$BATCH \
      sampling.kv_cache=true \
      sampling.first_hitting=true \
      wandb=null \
      hydra.run.dir="$RUNDIR" \
      hydra.output_subdir=null \
      2>&1)
  rc=$?
  set -e
  echo "$OUT" | tail -n 25
  if [ $rc -ne 0 ]; then echo "WARN: block_size=$BS failed (rc=$rc); continuing"; continue; fi
  echo "$OUT" | grep -a '^BENCH_JSON ' | sed 's/^BENCH_JSON //' >> "$RESULTS" || echo "WARN: no BENCH_JSON for $BS"
done

echo "==================== results ===================="
cat "$RESULTS"
echo "DONE p2 efficiency: $(date)"
