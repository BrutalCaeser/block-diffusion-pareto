#!/bin/bash
# =============================================================================
# p3_train_probe.sbatch  —  Phase 3 step-timing probe (SPEC gate G2 input).
#
# Goal: confirm the TRAINING pipeline runs for BD3-LM on a small dataset, and
# measure training it/s + peak memory, so we can size the fixed-budget controlled
# quality sweep (train block sizes {4,16,32,64,128} for the SAME #steps on the
# same data, then compare val PPL -> the quality axis of the Pareto).
#
# Choices (grounded):
#  - data=wikitext103: small standard LM benchmark; train split tokenizes in
#    minutes (OWT train = 8M docs, deliberately avoided -> too expensive for a
#    "cheap" sweep).
#  - attn_backend=sdpa: the paper trains with flex, but flex wants Ampere+ and
#    gpu-short gives V100. dit.py applies the SAME block_diff_mask in the sdpa
#    training path -> equivalent gradients, just denser. Faithful + portable.
#  - from_pretrained=null: train from scratch (controlled, comparable across
#    block sizes; absolute PPL will be high/undertrained — we want the TREND).
#  - tiny batch + 30 steps: just measuring step time + memory here.
#
# Submit: sbatch exp/p3_train_probe.sbatch
# =============================================================================
#SBATCH --job-name=bp-p3-probe
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/p3_probe_%j.log

set -euo pipefail
PROJECT=/scratch/gupta.yashv/block-pareto
ENV_PREFIX=$PROJECT/envs/blockpareto
REPO=$PROJECT/bd3lms
export HF_HOME=$PROJECT/hf
BS=${BS:-32}
STEPS=${STEPS:-30}
GBATCH=${GBATCH:-16}
LEN=${LEN:-256}
# bd3lm cross-attn doubles the seq (x ++ x0); the SDPA path materializes a DENSE
# (2*LEN)^2 block mask per layer -> at LEN=1024 (2048^2) it OOMs a 32GB V100.
# LEN=256 (512^2 mask) fits comfortably and still gives a valid block-size quality
# TREND. expandable_segments reduces allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
  trainer.limit_val_batches=2 \
  trainer.log_every_n_steps=1 \
  hydra.run.dir=$PROJECT/train_runs/probe_bs${BS}_${SLURM_JOB_ID} \
  wandb=null
set +x
echo "DONE p3 probe: $(date)"
