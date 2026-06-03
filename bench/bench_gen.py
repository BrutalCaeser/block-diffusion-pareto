#!/usr/bin/env python
# =============================================================================
# bench_gen.py  —  Phase 2 efficiency harness for BD3-LMs.
#
# Measures, for ONE block size, the weight-independent generation efficiency:
#   - wall-clock latency to generate a full sequence
#   - throughput (generated tokens / second)
#   - NFE (number of function evals = denoising forward passes), as reported by
#     the bd3lm semi-AR sampler itself
#   - peak GPU memory during generation
#
# WHY weight-independent / random init: throughput, NFE and memory depend only on
# the architecture + block size + sampler config, NOT on the trained weights. So we
# use the NATIVE `dit` backbone (random init), which exists for ANY block size —
# this is what lets us measure the UNRELEASED block sizes 32/64/128 (the whole
# point of the study). We separately validate this claim by comparing block_size=16
# random-init throughput against the real hf_dit checkpoint (see exp/p2_validate).
#
# Faithful to the repo's own sampler: calls diffusion.Diffusion._semi_ar_sampler
# with the canonical inference config (first_hitting + kv_cache + sdpa, algo.T),
# under a bf16 autocast context (matches trainer.precision='bf16').
#
# Run (via hydra, like main.py); emits one machine-readable line:
#   BENCH_JSON {...}
# =============================================================================
import os
import sys
import json
import time

# bd3lms lives outside this repo (pinned upstream clone); make its modules importable.
BD3LMS = os.environ.get('BD3LMS_DIR', '/scratch/gupta.yashv/block-pareto/bd3lms')
sys.path.insert(0, BD3LMS)

import hydra  # noqa: E402
import torch  # noqa: E402


@hydra.main(version_base=None,
            config_path=os.path.join(BD3LMS, 'configs'),
            config_name='config')
def main(cfg):
    import dataloader   # noqa: E402  (import here so hydra/cwd are set)
    import diffusion     # noqa: E402

    assert torch.cuda.is_available(), 'need a GPU'
    assert cfg.model.attn_backend == 'sdpa', 'inference + kv_cache requires sdpa (flex unsupported)'
    torch.manual_seed(0)

    tokenizer = dataloader.get_tokenizer(cfg)
    model = diffusion.Diffusion(cfg, tokenizer=tokenizer).to('cuda').eval()

    # --- adapt the NATIVE `dit` backbone to the sampling path -----------------
    # diffusion.py's sampler was maintained for the hf_dit backbone. Two gaps for
    # the native dit backbone, both fixed here WITHOUT touching pinned upstream:
    #  (1) reset_kv_cache(): native takes no args, but the sampler calls it with
    #      eval_batch_size=...  (native reads config.loader.eval_batch_size, which
    #      EQUALS the passed value -> safe to accept-and-ignore the kwarg).
    #  (2) gen_mask(): Diffusion.__init__ only calls it for hf_dit. The native
    #      forward needs self.block_diff_mask to exist for the kv-cache sampling
    #      branch (`cross_attn = hasattr(self, 'block_diff_mask')`). Build it.
    if cfg.algo.backbone == 'dit':
        _orig_reset = model.backbone.reset_kv_cache
        model.backbone.reset_kv_cache = lambda *a, **k: _orig_reset()
        if not hasattr(model.backbone, 'block_diff_mask'):
            model.backbone.gen_mask(cfg.model.length, model.block_size,
                                    attn_backend='sdpa')

    seqlen = cfg.model.length
    bsz = cfg.loader.eval_batch_size
    T = cfg.algo.T
    block = model.block_size
    assert seqlen % block == 0, f'seqlen {seqlen} not divisible by block {block}'
    num_strides = seqlen // block

    def run_once():
        torch.cuda.synchronize()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            x, nfe = model._semi_ar_sampler(
                n_samples=bsz, num_steps=T, num_strides=num_strides, seqlen=seqlen)
        torch.cuda.synchronize()
        return x, nfe

    # warmup (compile kernels / allocator) — not timed
    _ = run_once()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    x, nfe = run_once()
    dt = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    gen_tokens = seqlen * bsz
    rec = {
        'block_size': int(block),
        'seqlen': int(seqlen),
        'batch': int(bsz),
        'algo_T': int(T),
        'backbone': str(cfg.algo.backbone),
        'attn_backend': str(cfg.model.attn_backend),
        'kv_cache': bool(cfg.sampling.kv_cache),
        'first_hitting': bool(cfg.sampling.first_hitting),
        'precision': 'bf16',
        'wall_s': round(dt, 4),
        'tok_per_s': round(gen_tokens / dt, 2),
        'nfe': int(nfe),
        'sec_per_nfe': round(dt / max(int(nfe), 1), 5),
        'peak_mem_gb': round(peak_gb, 4),
        'gpu': torch.cuda.get_device_name(0),
    }
    print('BENCH_JSON ' + json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
