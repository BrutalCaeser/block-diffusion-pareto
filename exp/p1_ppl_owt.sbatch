#!/bin/bash
# =============================================================================
# p1_ppl_owt.sbatch  —  Phase 1, step B: reproduce a released-checkpoint PPL.
#
# Smoke test of the whole eval pipeline against a KNOWN target: load the released
# HF checkpoint, tokenize OWT valid (cache hit from p1_prep_owt.sbatch), compute
# test perplexity, compare to the paper.
#
# Paper (BD3-LMs, arXiv:2503.09573, Table 1, OWT, model.length=1024) PPL targets:
#   block_size 4  -> ~20.73 ;  block_size 8 -> ~21.x ;  block_size 16 -> ~22.27
# Released checkpoints exist ONLY for block sizes 4, 8, 16. Default smoke = 16.
#
# attn_backend=sdpa (NOT flex): dit.py applies the SAME block_diff_mask in the
# sdpa path (F.scaled_dot_product_attention attn_mask=mask), so PPL is faithful,
# and sdpa runs on any GPU (gpu-short gives V100; flex wants Ampere+).
#
# Env knobs:
#   BS=16      block size / which released checkpoint (4|8|16)
#   LIMIT=8    quick sanity on N val batches; unset/blank = full valid set
#
# Submit (quick sanity): sbatch --export=ALL,BS=16,LIMIT=8 exp/p1_ppl_owt.sbatch
# Submit (full PPL):     sbatch --export=ALL,BS=16          exp/p1_ppl_owt.sbatch
# =============================================================================
#SBATCH --job-name=bp-p1-ppl
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/scratch/gupta.yashv/block-pareto/logs/p1_ppl_%j.log

set -euo pipefail

PROJECT=/scratch/gupta.yashv/block-pareto
ENV_PREFIX=$PROJECT/envs/blockpareto
REPO=$PROJECT/bd3lms
DATA_DIR=$PROJECT/data
export HF_HOME=$PROJECT/hf            # checkpoint + dataset downloads -> /scratch
BS=${BS:-16}
LIMIT=${LIMIT:-}

echo "=========================================================="
echo "node : $(hostname)   date: $(date)"
echo "block_size=$BS   limit_val_batches=${LIMIT:-<full>}   backend=sdpa"
echo "=========================================================="

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
python -c "import torch; print('cuda avail:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

cd "$REPO"
EXTRA=""
[ -n "$LIMIT" ] && EXTRA="+trainer.limit_val_batches=$LIMIT"

set -x
python -u main.py \
  loader.eval_batch_size=16 \
  model=small \
  algo=bd3lm \
  algo.backbone=hf_dit \
  data=openwebtext-split \
  data.cache_dir="$DATA_DIR" \
  data.insert_valid_special=False \
  model.length=1024 \
  model.attn_backend=sdpa \
  block_size=$BS \
  eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BS} \
  wandb=null \
  mode=ppl_eval $EXTRA
set +x

echo "=========================================================="
echo "DONE ppl_eval (block_size=$BS): $(date)"
echo "=========================================================="
