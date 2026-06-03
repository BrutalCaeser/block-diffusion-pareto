#!/bin/bash
# =============================================================================
# p3_quality_sweep.sbatch  —  Phase 3 controlled quality sweep (one block size).
#
# Trains ONE BD3-LM block size from scratch on wikitext103 for a FIXED budget,
# then validates. Run once per block in {4,16,32,64,128}; compare final val NLL/PPL
# -> the quality axis to pair with the Phase 2 throughput curve.
#
# HONEST SCOPE: from-scratch + fixed-budget => this is "achieved PPL under a matched
# cheap training budget", which blends optimization speed with model capacity. It is
# NOT converged quality (the paper trains ~1M steps from a pretrain init). We report
# the TREND under matched compute, with this caveat stated.
#
# Config rationale: LEN=256 (dense sdpa block mask = (2*LEN)^2 fits V100/A100);
# sdpa (flex needs Ampere+ and we want portability); from_pretrained=null (scratch).
#
# Submit one per block:  sbatch --export=ALL,BS=32 exp/p3_quality_sweep.sbatch
# =============================================================================
#SBATCH --job-name=bp-p3-qual
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/p3_qual_bs%x_%j.log

set -euo pipefail
PROJECT=/scratch/gupta.yashv/block-pareto
ENV_PREFIX=$PROJECT/envs/blockpareto
REPO=$PROJECT/bd3lms
export HF_HOME=$PROJECT/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BS=${BS:-32}
STEPS=${STEPS:-2500}
GBATCH=${GBATCH:-16}
LEN=${LEN:-256}
RUNDIR=$PROJECT/train_runs/p3_bs${BS}_${SLURM_JOB_ID}

echo "node:$(hostname) date:$(date) block_size=$BS steps=$STEPS gbatch=$GBATCH len=$LEN"
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
python -c "import torch;print('gpu:',torch.cuda.get_device_name(0))"

cd "$REPO"
set -x
python -u main.py \
  mode=train \
  loader.global_batch_size=$GBATCH \
  loader.eval_global_batch_size=$GBATCH \
  loader.batch_size=$GBATCH \
  loader.eval_batch_size=$GBATCH \
  model=small \
  algo=bd3lm \
  algo.backbone=dit \
  data=wikitext103 \
  data.cache_dir=$PROJECT/data \
  model.length=$LEN \
  model.attn_backend=sdpa \
  block_size=$BS \
  training.from_pretrained=null \
  trainer.max_steps=$STEPS \
  trainer.val_check_interval=$STEPS \
  trainer.limit_val_batches=50 \
  trainer.log_every_n_steps=50 \
  hydra.run.dir=$RUNDIR \
  wandb=null
set +x
echo "DONE p3 quality (block_size=$BS, steps=$STEPS): $(date)"
echo "RUNDIR=$RUNDIR"
