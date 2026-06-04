#!/bin/bash
# =============================================================================
# nfe_throughput.sbatch  —  NfePareto Phase 3: throughput at the matched
# (block, NFE) operating points, to fuse with the gen-PPL curves into the
# unified quality-vs-throughput Pareto.
#
# Uses bench/bench_gen.py (the BlockPareto efficiency harness) at EXACTLY the
# (block, T, first_hitting) points measured by exp/nfe_genppl.sbatch, with REAL
# released weights (algo.backbone=hf_dit, BENCH_BACKEND=hf_dit) — blocks 4/8/16
# are all released, so we don't need the random-init proxy here, and real weights
# guarantee the measured NFE + tok/s join cleanly to the gen-PPL run (same sampler).
#
# Throughput is weight-independent in compute/forward; we measure it on the
# production HF path (the deployed impl, per BlockPareto). One fresh python process
# per point (clean peak-mem). Output: results/nfe_throughput.jsonl (BENCH_JSON/line).
#
# Submit: sbatch exp/nfe_throughput.sbatch
# =============================================================================
#SBATCH --job-name=nfe-thru
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/nfe_thru_%j.log

set -uo pipefail

PROJECT=/scratch/gupta.yashv/block-pareto
ENV_PREFIX=$PROJECT/envs/blockpareto
REPO=$PROJECT/repo
export BD3LMS_DIR=$PROJECT/bd3lms
export HF_HOME=$PROJECT/hf
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BENCH_BACKEND=hf_dit          # real released weights -> NFE/tok-s match sample_eval

LEN=${LEN:-1024}
BATCH=${BATCH:-1}
RESULTS=$PROJECT/results/nfe_throughput.jsonl
mkdir -p "$(dirname "$RESULTS")" "$PROJECT/logs" "$PROJECT/nfe_runs/$SLURM_JOB_ID"
: > "$RESULTS"   # fresh

# Grid = the exact samplable gen-PPL points (block:T:first_hitting).
GRID="\
4:1:false 4:3:false 4:4:false 4:6:false 4:5000:true \
8:1:false 8:3:false 8:4:false 8:6:false 8:8:false 8:5000:true \
16:4:false 16:6:false 16:8:false 16:12:false 16:16:false 16:24:false 16:32:false 16:48:false 16:64:false 16:5000:true"

echo "node:$(hostname) date:$(date) len=$LEN batch=$BATCH"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$BD3LMS_DIR"
python -c "import torch;print('gpu:',torch.cuda.get_device_name(0))"

for triple in $GRID; do
  IFS=: read -r BS T FH <<< "$triple"
  CKPT=kuleshov-group/bd3lm-owt-block_size${BS}
  RUNDIR=$PROJECT/nfe_runs/$SLURM_JOB_ID/b${BS}_T${T}_fh${FH}
  echo "==================== block=$BS T=$T fh=$FH ===================="
  OUT=$(python -u "$REPO/bench/bench_gen.py" \
      model=small \
      algo=bd3lm \
      algo.backbone=hf_dit \
      algo.T=$T \
      data=openwebtext-split \
      data.cache_dir=$PROJECT/data \
      model.length=$LEN \
      model.attn_backend=sdpa \
      block_size=$BS \
      loader.eval_batch_size=$BATCH \
      sampling.kv_cache=true \
      sampling.first_hitting=$FH \
      eval.checkpoint_path=$CKPT \
      wandb=null \
      hydra.run.dir="$RUNDIR" \
      hydra.output_subdir=null \
      2>&1)
  rc=$?
  echo "$OUT" | tail -n 6
  if [ $rc -ne 0 ]; then echo "WARN: block=$BS T=$T fh=$FH failed (rc=$rc); continuing"; continue; fi
  # tag the record with the knob settings so it joins to the gen-PPL table
  echo "$OUT" | grep -a '^BENCH_JSON ' | sed "s/^BENCH_JSON //" \
    | python -c "import sys,json;
d=json.load(sys.stdin); d['req_block']=$BS; d['req_T']=$T; d['req_fh']='$FH'; print(json.dumps(d))" \
    >> "$RESULTS" || echo "WARN: no BENCH_JSON for block=$BS T=$T"
done

echo "==================== results ===================="
cat "$RESULTS"
echo "DONE nfe_throughput: $(date)"
