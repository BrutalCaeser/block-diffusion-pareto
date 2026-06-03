# Block-Size Pareto Frontier for Block-Diffusion LMs — Findings

_Last updated: 2026-06-03 · repo: https://github.com/BrutalCaeser/block-diffusion-pareto_

**Question.** BD3-LMs (Kuleshov group, ICLR 2025) interpolate between autoregressive
and diffusion LMs via a **block size** knob. Public checkpoints exist only for block
sizes 4/8/16. Stefano Ermon (Inception Labs) said in person that Mercury runs **block
size 32**. Why 32? This study measures the **quality ↔ efficiency frontier** across
block sizes to explain it — on commodity hardware, every number grounded in a run.

## TL;DR
- **Reproduced** the released BD3-LM block-16 OWT perplexity to within ~0.1%
  (22.30 vs paper ≤22.27) — pipeline is trustworthy.
- **On the production HF code path, generation throughput PEAKS at block size 32**
  (62.5 tok/s; non-monotonic — falls off on both sides). This matches Inception's
  reported value.
- **Quality** (matched-budget val NLL) **monotonically worsens** as block size grows
  from 4→32, then plateaus at 64/128.
- **Pareto frontier = {4, 16, 32}.** Blocks 64/128 are *strictly dominated* by 32
  (slower AND worse quality). **Block 32 is the throughput-optimal endpoint of the
  frontier** — buying maximum speed at a small, bounded quality cost vs 16.

## Setup (grounded)
- Hardware: Northeastern Explorer cluster; V100-32GB (efficiency) + A100/H200 (training).
- Code: upstream `kuleshov-group/bd3lms` pinned @ `1c3e8f4`; env torch 2.7.1+cu126.
- Efficiency: production HF `modeling_bd3lm`, `sdpa` + KV-cache + first_hitting, bf16,
  seqlen 1024, batch 1. Measured on **random-init** weights — validated equivalent to
  the real checkpoint (hf_random@16 = 61.72 tok/s vs real ckpt 61.74; 0.03% diff), so
  the unreleased 32/64/128 points are faithful.
- Quality: BD3-LM trained from scratch on wikitext103 for a **matched 2500-step budget**
  (len 256, sdpa). This is "achieved PPL under matched cheap compute" — the block-size
  *ordering* is the claim, not converged absolute quality (caveat below).

## Results

### Efficiency (production HF path, V100, seqlen 1024, batch 1)
| block | throughput (tok/s) | peak mem (GB) | NFE |
|------:|------:|-----:|----:|
| 4   | 53.57 | 0.77 | 1023 |
| 8   | 60.06 | 0.77 | 1023 |
| 16  | 61.72 | 0.78 | 1023 |
| **32**  | **62.52 (peak)** | 0.79 | 1023 |
| 64  | 60.64 | 0.82 | 1023 |
| 128 | 57.78 | 0.90 | 1023 |
| 256 | 52.26 | 1.06 | 1023 |
| 512 | 40.83 | 1.36 | 1023 |
| 1024| 23.98 | 1.98 | 1023 |

Throughput is **non-monotonic** with a knee at block 32: small blocks pay per-stride
loop + KV-store overhead (256 strides at block 4); large blocks pay expensive per-step
attention (bigger query each denoising step). NFE is constant (first_hitting unmasks
~1 token/step ⇒ NFE ≈ seqlen), so the curve is pure per-step cost. Peak memory rises
monotonically with block size. _Figure: `results/phase2_throughput.png`._

### Quality (wikitext103, matched 2500-step from-scratch budget, len 256)
| block | val NLL | val PPL |
|------:|------:|------:|
| 4   | 5.965 | 389 |
| 16  | 6.035 | 418 |
| 32  | 6.157 | 472 |
| 64  | 6.214 | 500 |
| 128 | 6.198 | 492 |

Quality worsens monotonically 4→32, then plateaus (64≈128 within noise). Smaller blocks
are closer to autoregressive ⇒ better likelihood. _Figure: `results/phase3_quality.png`._

### The Pareto (quality vs throughput)
Non-dominated set = **{4, 16, 32}**. Block 32 is the **fastest** point on the frontier;
64 and 128 are dominated by 32 on both axes. Moving 4→16→32 trades a little quality for
throughput; beyond 32 you lose throughput AND quality. _Figure: `results/pareto_quality_speed.png`._

**Why 32 is a defensible production sweet spot:** it maximizes generation throughput on
the production code path while sitting on the quality/speed frontier — the quality cost
vs block 16 is small (NLL 6.04→6.16) and it is never dominated. A latency-sensitive
commercial dLLM (Mercury) picking 32 is choosing the speed-optimal frontier endpoint.

## Honest caveats
- **Quality is fixed-budget-from-scratch, not converged.** It blends optimization speed
  with capacity; the paper trains ~1M steps from a pretrain init. The *ordering* is the
  result; absolute PPL is high by construction. A finetune-from-pretrain sweep (compute
  permitting) would sharpen the absolute quality axis.
- **Throughput is one operating point** (V100, batch 1, seqlen 1024). The knee can shift
  with GPU / batch / sequence length; the qualitative non-monotonicity is the robust claim.
- **Two backbones differ:** a native reference DIT impl peaks at block 16; the production
  HF impl peaks at 32. We report the production path (what's deployed) and keep the
  reference as a labeled comparison. Lesson: benchmark the deployed implementation.

## Reproduce
See `SPEC.md` (blueprint) and `LOG.md` (full run log). Scripts: `exp/p1_*` (PPL),
`exp/p2_efficiency.sbatch` + `bench/bench_gen.py` (throughput), `exp/p3_quality_sweep.sbatch`
(quality), `analysis/analyze_*.py` (figures).
