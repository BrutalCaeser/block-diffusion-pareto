# Upstream pin

This project runs on top of **BD3-LMs** (Block Diffusion) — we do **not** vendor or
fork it; we keep a clean clone on the cluster and pin the exact commit here for
reproducibility. If we ever need to patch BD3-LM internals, we'll fork it and
update this pin.

- **Repo:** https://github.com/kuleshov-group/bd3lms
- **Paper:** Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models," ICLR 2025 (Oral). arXiv:2503.09573
- **Pinned commit:** `1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7`
- **Commit date:** 2025-07-10
- **Cluster clone path:** `/scratch/gupta.yashv/block-pareto/bd3lms`
- **Released checkpoints used (HuggingFace `kuleshov-group/`):**
  `bd3lm-owt-block_size{4,8,16}`, `bd3lm-owt-block_size1024-pretrain`,
  `mdlm-owt`, `{mdlm,sedd,ar}-noeos-owt`. (No 32/64/128 — those are ours to train.)

To re-clone the exact version on the cluster:
```bash
git clone https://github.com/kuleshov-group/bd3lms.git
cd bd3lms && git checkout 1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7
```
