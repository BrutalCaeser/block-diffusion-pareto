#!/bin/bash
# =============================================================================
# nfe_genppl.sbatch  —  NfePareto: generative-quality vs NFE harness.
#
# Drives the bd3lms repo's OWN validated pipeline: `main.py mode=sample_eval`
# (generate_samples -> restore_model_and_sample -> _sample -> semi_ar sampler ->
#  metrics.record_generative_perplexity). We do NOT re-implement gen-PPL scoring
# (grounding rule #1) — we use the paper's exact metric (gpt2-large gen-PPL +
# unigram-entropy diversity guard) and parse its CSV.
#
# THE NFE DIAL (verified in diffusion.py::_semi_ar_sampler):
#   first_hitting=true  -> ~1 token/step  -> NFE ~= seqlen, INDEPENDENT of T
#                          (paper's high-quality mode; used with T=5000 >> block)
#   first_hitting=false -> fixed T-step schedule -> NFE ~= (seqlen/block)*T,
#                          CONTROLLABLE via algo.T  <-- the dial we sweep.
# NFE is recorded as the sampler's actual returned `sampling_steps` (measured).
#
# Loops over TS (list of algo.T) x SEEDS. One unique CSV + .log per run, written
# OUTSIDE the git clone ($PROJECT/results/nfe/), copied back & committed locally.
#
# DEFAULTS reproduce scripts/gen_ppl/genppl_bd3lm.sh (the paper anchor):
#   BLOCK=16 TS="5000" FH=true SEEDS="1" NSB=25 EBS=1 LEN=1024 NUCLEUS=0.9
# Phase-0 NFE-dial sanity:   FH=false TS="2 32"
#
# Submit (validation): sbatch exp/nfe_genppl.sbatch
# Submit (sanity):     FH=false TS="2 32" sbatch exp/nfe_genppl.sbatch
# =============================================================================
#SBATCH --job-name=nfe-genppl
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/nfe_genppl_%j.log

set -euo pipefail

PROJECT=/scratch/gupta.yashv/block-pareto
ENV_PREFIX=$PROJECT/envs/blockpareto
export BD3LMS_DIR=$PROJECT/bd3lms
export HF_HOME=$PROJECT/hf
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- knobs (env-overridable) ----
BLOCK=${BLOCK:-16}
TS=${TS:-"5000"}            # space-separated list of algo.T values
FH=${FH:-true}             # sampling.first_hitting (true=paper quality mode; false=NFE dial)
SEEDS=${SEEDS:-"1"}
NSB=${NSB:-25}             # sampling.num_sample_batches  (N = NSB * EBS per seed)
EBS=${EBS:-1}              # loader.eval_batch_size
LEN=${LEN:-1024}
NUCLEUS=${NUCLEUS:-0.9}
CKPT=${CKPT:-kuleshov-group/bd3lm-owt-block_size${BLOCK}}

OUT=$PROJECT/results/nfe
mkdir -p "$OUT" "$PROJECT/logs" "$PROJECT/nfe_runs/$SLURM_JOB_ID"

echo "node:$(hostname) date:$(date)"
echo "BLOCK=$BLOCK TS=[$TS] FH=$FH SEEDS=[$SEEDS] NSB=$NSB EBS=$EBS LEN=$LEN NUCLEUS=$NUCLEUS CKPT=$CKPT"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
cd "$BD3LMS_DIR"   # main.py uses @hydra.main(config_path='configs') -> run from repo root
python -c "import torch;print('gpu:',torch.cuda.get_device_name(0),'torch',torch.__version__)"

for SEED in $SEEDS; do
  for T in $TS; do
    TAG=b${BLOCK}_T${T}_fh${FH}_s${SEED}_len${LEN}
    CSV=$OUT/genppl_${TAG}.csv
    LOG=$OUT/genppl_${TAG}.log
    RUNDIR=$PROJECT/nfe_runs/$SLURM_JOB_ID/$TAG
    rm -f "$CSV"   # fresh file (update_and_save_csv APPENDS) so reruns don't accumulate
    echo "==================== $TAG ===================="
    set +e
    python -u "$BD3LMS_DIR/main.py" \
        loader.eval_batch_size=$EBS \
        model=small \
        algo=bd3lm \
        algo.T=$T \
        algo.backbone=hf_dit \
        data=openwebtext-split \
        data.cache_dir=$PROJECT/data \
        model.length=$LEN \
        block_size=$BLOCK \
        wandb=null \
        mode=sample_eval \
        eval.checkpoint_path=$CKPT \
        model.attn_backend=sdpa \
        seed=$SEED \
        sampling.num_sample_batches=$NSB \
        sampling.nucleus_p=$NUCLEUS \
        sampling.kv_cache=true \
        sampling.first_hitting=$FH \
        sampling.logdir="$CSV" \
        hydra.run.dir="$RUNDIR" \
        hydra.output_subdir=null \
        2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then echo "WARN: $TAG failed (rc=$rc); continuing"; continue; fi
    # echo the headline lines for the slurm log
    grep -aE 'Generative perplexity:|Entropy:' "$LOG" | tail -2 || true
    echo "wrote: $CSV"
  done
done

echo "==================== done ===================="
ls -l "$OUT" | tail -20
echo "DONE nfe_genppl: $(date)"
