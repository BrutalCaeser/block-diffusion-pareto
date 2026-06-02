# Block-Size Pareto Frontier for Block-Diffusion LMs

**What:** A controlled study of how **block size** trades off quality vs. inference efficiency in Block-Diffusion language models (BD3-LMs). The public BD3-LMs paper reports block sizes **4 / 8 / 16 / 128**; Inception Labs' Mercury reportedly runs **block size 32** (per Stefano Ermon, in person, Apr 2026) — a point **nobody has published**. This repo measures the full frontier — perplexity, generative quality, **throughput (tok/s)**, and **KV-cache memory** — across block sizes {4, 8, 16, 32, 64, 128} and asks: **why is 32 the production sweet spot?**

**Why it matters:** It's the exact design choice a commercial diffusion-LLM (Mercury) makes, framed as a clean, reproducible scientific question. Companion analysis: `~/Documents/wiki/wiki/concepts/diffusion-llms.md`.

## Status
See `LOG.md` (engineering log) and `SPEC.md` (the blueprint with phases, methodology, compute budget, decision gates).

## Layout
```
SPEC.md      blueprint: phases, hypotheses, methodology, compute budget, risks
LOG.md       running engineering/devops log (newest on top)
env/         conda env build (SLURM) + locked requirements
slurm/       our batch scripts (smoke, ppl_eval, throughput, train)   [added per phase]
src/         our measurement/analysis code (throughput, KV-mem, plots) [added per phase]
results/     small metrics (CSV/JSON) — large dumps are gitignored
```

## Infra
- **Cluster:** Northeastern Explorer (`ssh explorer`), H200/A100 (`gpu`, 8h), `short` (CPU, 2d).
- **Workspace (cluster):** `/scratch/gupta.yashv/block-pareto/` — `bd3lms/` (upstream), `envs/blockpareto` (conda), `data/`, `ckpts/`, `logs/`.
- **Upstream:** [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) (ICLR 2025). Env: torch 2.7.1, py3.9; trains with FlexAttention, samples with sdpa + KV cache.

## Reproduce the environment
```bash
ssh explorer
cd /scratch/gupta.yashv/block-pareto/repo
sbatch env/build_blockpareto_env.sbatch     # builds /scratch/.../envs/blockpareto
```
