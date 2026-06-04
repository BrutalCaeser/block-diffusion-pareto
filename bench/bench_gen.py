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

    # Backbone selection via BENCH_BACKEND env var:
    #   dit            : native random-init DIT (reference implementation)
    #   hf_pretrained  : HF modeling_bd3lm with REAL released weights (only bs 4/8/16)
    #   hf_random      : HF modeling_bd3lm built from_config (random) — the PRODUCTION
    #                    code path, available for ANY block size (incl. unreleased
    #                    32/64/128). This is the faithful efficiency measurement.
    # NOTE: native dit and HF modeling_bd3lm are DIFFERENT implementations with
    # different speed/memory (verified: bs16 dit 49 tok/s vs hf 62). So the headline
    # sweep uses hf_random; weight-independence is validated by hf_random vs
    # hf_pretrained at bs16 (same impl, random vs trained weights).
    bench_backend = os.environ.get('BENCH_BACKEND', str(cfg.algo.backbone))
    ref_ckpt = os.environ.get('BENCH_REF_CKPT',
                              'kuleshov-group/bd3lm-owt-block_size16')

    model = diffusion.Diffusion(cfg, tokenizer=tokenizer)

    if bench_backend == 'hf_random':
        # cfg.algo.backbone was passed as 'dit' (so Diffusion didn't from_pretrained);
        # replace that native backbone with a random-init HF modeling_bd3lm.
        import transformers
        hfcfg = transformers.AutoConfig.from_pretrained(ref_ckpt, trust_remote_code=True)
        hfcfg.block_size = int(model.block_size)
        hfcfg.model_length = int(cfg.model.length)
        hfcfg.attn_backend = 'sdpa'   # inference (flex unsupported w/ kv-cache)
        model.backbone = transformers.AutoModelForMaskedLM.from_config(
            hfcfg, trust_remote_code=True)
        # HF DITBackbone.__init__ already calls gen_mask + provides reset_kv_cache(eval_batch_size=)

    model = model.to('cuda').eval()

    if bench_backend == 'dit':
        # adapt native dit to the sampler (gaps vs hf path), without touching upstream:
        #  (1) reset_kv_cache(): native takes no args (reads config.loader.eval_batch_size,
        #      == the value the sampler passes) -> accept-and-ignore the kwarg.
        #  (2) gen_mask(): Diffusion.__init__ only calls it for hf_dit; native forward
        #      needs self.block_diff_mask for the kv-cache sampling branch.
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

    def run_attempt():
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            x, nfe = model._semi_ar_sampler(
                n_samples=bsz, num_steps=T, num_strides=num_strides, seqlen=seqlen)
        torch.cuda.synchronize()
        return x, nfe, time.time() - t0

    def run_until_success(max_tries=15):
        # _semi_ar_sampler returns (None, None) when a sample trips the entropy<4
        # stop-condition (degenerate, common at very low NFE). diffusion._sample
        # retries the same way; we mirror it so low-NFE operating points still yield
        # a clean TIMED generation. Only the successful attempt's wall-time is kept
        # (failed attempts discarded) -> tok/s = time for one good sample, consistent
        # with the gen-PPL runs (which also discard rejected samples).
        for _ in range(max_tries):
            x, nfe, dt = run_attempt()
            if x is not None:
                return x, nfe, dt
        raise RuntimeError(f'sampler returned None after {max_tries} tries '
                           f'(operating point too few-NFE to sample: block={block} T={T})')

    # warmup (compile kernels / allocator) — not timed
    _ = run_until_success()

    torch.cuda.reset_peak_memory_stats()
    x, nfe, dt = run_until_success()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    gen_tokens = seqlen * bsz
    rec = {
        'block_size': int(block),
        'seqlen': int(seqlen),
        'batch': int(bsz),
        'algo_T': int(T),
        'backbone': str(cfg.algo.backbone),
        'bench_backend': bench_backend,
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
