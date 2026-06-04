# Block-Size Pareto Frontier for Block-Diffusion LMs

> **Why does Inception Labs' Mercury run block size 32?**
> Block-Diffusion LMs (BD3-LMs) interpolate between autoregressive and diffusion
> models via a **block size** knob. Public checkpoints cover only 4/8/16; Stefano
> Ermon said in person that Mercury runs **32**. This repo measures the full
> **quality ↔ efficiency frontier** on commodity GPUs to explain why — every
> number from a real run.

**TL;DR:** On the production code path, generation throughput **peaks at block size 32**.
Quality degrades monotonically as blocks grow. On the quality-vs-throughput Pareto, the
non-dominated set is **{4, 16, 32}**, and **block 32 is the throughput-optimal endpoint** —
blocks 64/128 are strictly dominated. Picking 32 = choosing the fastest point on the
frontier at a small, bounded quality cost. Full writeup: **[FINDINGS.md](FINDINGS.md)**.

![Block-size Pareto: quality vs throughput](results/pareto_quality_speed.png)

---

## Headline results

| | |
|---|---|
| **Pipeline validated** | Reproduced released BD3-LM block-16 OWT perplexity to ~0.1% (**22.30** vs paper **≤22.27**) |
| **Throughput peaks at block 32** | 62.5 tok/s on the production HF path; non-monotonic (see below) |
| **Quality** | Monotonically worsens with block size (4→32), plateaus at 64/128 |
| **Pareto frontier** | **{4, 16, 32}** — 64/128 strictly dominated by 32 |
| **Weight-independence** | random-init vs real-checkpoint throughput match to **0.03%** ⇒ unreleased 32/64/128 points are faithful |

### Generation throughput vs block size (production HF path, V100, seqlen 1024, batch 1)
![Throughput vs block size](results/phase2_throughput.png)

Small blocks pay per-stride loop + KV-store overhead; large blocks pay expensive
per-step attention. NFE is constant (first_hitting ⇒ ~1 token/step). Memory rises
monotonically with block size (`results/phase2_memory.png`).

### Quality vs block size (wikitext103, matched 2500-step from-scratch budget)
![Quality vs block size](results/phase3_quality.png)

Smaller blocks are closer to autoregressive ⇒ better likelihood. _(Scope: this is
quality under a matched **cheap** budget — the block-size **ordering** is the claim,
not converged absolute PPL. See caveats in [FINDINGS.md](FINDINGS.md).)_

---

## Sister study — NfePareto: the *denoising-step (NFE)* axis

BlockPareto maps the **block-size** knob; **[FINDINGS_NFE.md](FINDINGS_NFE.md)** maps the **NFE knob**
(how many denoising steps you spend — the literal "fast vs quality" dial) on the released checkpoints,
then fuses both into one operating map. Plan: **[SPEC_NFE.md](SPEC_NFE.md)**.

![Unified gen-PPL vs throughput Pareto](results/pareto_genppl_throughput.png)

- **Validated:** released-checkpoint gen-PPL reproduces the BD3-LM paper Table 7 within ~6% (block 4/8/16 = 24.2/31.0/31.2 vs 25.7/30.4/33.4).
- **The NFE dial** (block 16): gen-PPL is monotone-decreasing in NFE with a **soft knee NFE\*≈550–650**; no hard plateau. **Few-step sampling is genuinely lossy on the base model** (knee gen-PPL 81 = 2.6× the full-NFE 31) — the exact gap consistency-distillation (CDLM / Mercury's "1–3 step" claim) exists to close. `results/nfe_quality_curve.png`.
- **Block × NFE interact — the curves cross:** at matched mid-NFE the *larger* block (16) wins (115 vs 327 @ NFE~557); at full NFE the *smaller* block (4) wins (24.2). **The optimal block depends on the NFE budget.**
- **Unified Pareto:** the frontier is a clean **turbo → quality** arc (block 16 owns the middle; block 4 + full NFE = the quality endpoint) — the latency-tiered surface a commercial dLLM exposes.

---

## Repository layout
```
README.md       this file — front door + results
FINDINGS.md     BlockPareto writeup: question, setup, results, Pareto, honest caveats
SPEC.md         BlockPareto blueprint: phases, hypotheses, methodology, gates
FINDINGS_NFE.md NfePareto writeup: the NFE↔quality dial + unified Pareto (sister study)
SPEC_NFE.md     NfePareto blueprint (NFE axis)
LOG.md          running engineering/devops log (newest on top)
CONTRIBUTING.md commit-message convention
UPSTREAM.md     pinned bd3lms commit SHA (upstream is NOT vendored)

env/            conda env build (SLURM) + locked requirements (requirements.lock.txt)
exp/            SLURM batch jobs:
                  p1_prep_owt.sbatch       OWT download/cache (CPU)
                  p1_ppl_owt.sbatch        PPL reproduction (Phase 1)
                  p2_efficiency.sbatch     throughput/memory/NFE sweep (Phase 2)
                  p3_train_probe.sbatch    training step-timing probe
                  p3_quality_sweep.sbatch  quality sweep, one block size/job (Phase 3)
                  nfe_genppl.sbatch        [NFE] drive sample_eval: gen-PPL vs NFE
                  nfe_throughput.sbatch    [NFE] throughput at matched (block,NFE) points
bench/          bench_gen.py       generation-efficiency harness (retries low-NFE stop)
analysis/       analyze_phase2.py  efficiency CSV + plots
                analyze_pareto.py  combined quality↔throughput Pareto
                parse_genppl.py    [NFE] aggregate sample_eval CSVs → table
                analyze_nfe.py     [NFE] gen-PPL/entropy vs NFE curves
                analyze_pareto_nfe.py [NFE] unified gen-PPL vs throughput Pareto
results/        committed metrics (CSV/JSON) + figures (PNG)
```

## Method at a glance
- **Upstream** [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) (ICLR 2025), pinned by SHA in [UPSTREAM.md](UPSTREAM.md) — cloned on the cluster, never vendored.
- **Efficiency** measured on the production HF `modeling_bd3lm` path (random init, validated equivalent to real checkpoints) ⇒ the unreleased 32/64/128 are measurable.
- **Quality** from controlled matched-budget training on wikitext103.
- **Env:** torch 2.7.1+cu126, py3.9; `sdpa` + KV-cache at inference (flex needs Ampere+).

## Infra & workflow (git is the spine)
- **Cluster:** Northeastern Explorer — `gpu` (H200/A100, 8h), `gpu-short` (2h), `sharing`/`short` (CPU).
- **GitHub** `BrutalCaeser/block-diffusion-pareto` (public) = source of truth. Author + commit
  locally → `git push` → cluster `git pull` to run. Large artifacts stay on `/scratch`
  (gitignored); small results (CSV/JSON/PNG) are committed.

## Reproduce
```bash
ssh explorer
cd /scratch/gupta.yashv/block-pareto/repo && git pull
sbatch env/build_blockpareto_env.sbatch                              # 1. env (Gate G0)
sbatch exp/p1_prep_owt.sbatch                                        # 2. cache OWT valid
sbatch --partition=gpu --time=04:00:00 --export=ALL,BS=16 exp/p1_ppl_owt.sbatch   # 3. PPL (Gate G1)
sbatch --export=ALL,BACKEND=hf_random,BLOCKS="4 8 16 32 64 128" exp/p2_efficiency.sbatch  # 4. throughput
for BS in 4 16 32 64 128; do sbatch --export=ALL,BS=$BS exp/p3_quality_sweep.sbatch; done # 5. quality
python analysis/analyze_phase2.py && python analysis/analyze_pareto.py             # 6. figures
```

## Status: complete (both studies)
**BlockPareto** (block-size axis): env ✓ · PPL reproduction ✓ · efficiency sweep ✓ · quality sweep ✓ · Pareto + writeup ✓.
**NfePareto** (NFE axis, [FINDINGS_NFE.md](FINDINGS_NFE.md)): harness validated (G0-N) ✓ · NFE dial (G1-N) ✓ · block×NFE crossover (G2-N) ✓ · unified Pareto + writeup ✓.
See [LOG.md](LOG.md) for the full run-by-run history. Next: the consistency-distillation extension —
close the few-step quality gap NfePareto quantifies (the CDLM / Mercury recipe). Other extensions:
finetune-from-pretrain converged quality, batch/seqlen throughput sweeps, blog post.
