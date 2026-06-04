# Findings — Quality ↔ NFE "Speed Dial" for Block-Diffusion LMs (NfePareto)

**Sister study to [BlockPareto](FINDINGS.md).** Where BlockPareto mapped the *block-size* axis,
this maps the *denoising-step (NFE) axis* and fuses both into one inference operating map.
Spec: `SPEC_NFE.md` · run log: `LOG.md` · all numbers reproducible from committed scripts + logged commands.

> **The question.** A diffusion LM's headline knob is *how many denoising steps (NFE) you spend*.
> Fewer = faster but lower quality; more = slower but better. This is the literal "fast vs quality"
> dial a commercial dLLM (Mercury) exposes. We measure **generative quality (gen-PPL under GPT-2-large)
> vs NFE**, per block size, on the **released BD3-LM checkpoints** (real weights → faithful quality;
> no training → cheap), and build the unified **gen-PPL-vs-throughput Pareto**.

## How (grounded, no re-implementation)
- We drive the bd3lms repo's **own** `main.py mode=sample_eval` — its validated pipeline
  (gpt2-large gen-PPL + unigram-entropy diversity guard). We did **not** re-implement scoring.
- **The NFE lever (verified in `diffusion.py::_semi_ar_sampler`):** `sampling.first_hitting=false`
  + `algo.T` ⇒ measured NFE ≈ (seqlen/block)·T (capped); `first_hitting=true` ⇒ NFE ≈ seqlen
  (1 token/step — the paper's quality mode). **NFE is measured** (the sampler's actual `sampling_steps`), not assumed.
- Throughput from `bench/bench_gen.py` (real released weights, production HF path) at the **exact same**
  (block, T, first_hitting) points, so NFE and tok/s join cleanly to the gen-PPL.
- Fixed: `model=small`, len 1024, nucleus 0.9, kv_cache, sdpa, N=25 samples/point, blocks {4,8,16}, single GPU.

## Validation — Gate G0-N (we reproduce the paper)
Released-checkpoint gen-PPL at full NFE (`first_hitting=true`) vs **BD3-LM paper Table 7** (OWT, L=1024):

| block | ours (gen-PPL) | paper | Δ |
|------:|---:|---:|---:|
| 4  | 24.2 | 25.7 | −5.9% |
| 8  | 31.0 | 30.4 | +2.1% |
| 16 | 31.2 | 33.4 | −6.6% |

All within ~6% (gen-PPL is sampler/seed-sensitive; N=25). Pipeline trusted.

## Result 1 — the NFE dial (block 16) — Gate G1-N
`first_hitting=false`, sweep `algo.T`; N=25/point:

| T | NFE (measured) | gen-PPL | entropy |
|--:|--:|--:|--:|
| 4 | 247 | 442 | 5.24 |
| 6 | 343 | 379 | 5.57 |
| 8 | 425 | 226 | 5.49 |
| 12 | 557 | 115 | 5.41 |
| 16 | 643 | 81 | 5.44 |
| 24 | 748 | 57 | 5.41 |
| 32 | 809 | 44 | 5.36 |
| 48 | 874 | 42 | 5.39 |
| 64 | 910 | 37 | 5.32 |
| fh=true | 1023 | 31 | 5.31 |

- **gen-PPL is monotone-decreasing in NFE** with a **soft knee at NFE\* ≈ 550–650** (T≈12–16): steep below,
  gentle above. **No hard plateau** — quality keeps improving to full NFE (31 @ 1023 is 30% better than 44 @ 809).
- **Few-step sampling is genuinely lossy on the *base* (undistilled) model:** even at the knee, gen-PPL 81 is
  2.6× the full-NFE 31; at a "fast" NFE 247 it's 442 (incoherent). **This is the exact quality gap that
  consistency-distillation (CDLM / the Mercury "1–3 step" story) exists to close** — quantified end-to-end.
- **The entropy guard is healthy (5.24–5.57) at every point, *including* the gen-PPL-442 garbage** ⇒ those
  samples are *incoherent-but-diverse*, not repetition-degenerate. gen-PPL catches incoherence; entropy catches
  repetition — both are needed (the FlowLM lesson, confirmed empirically). `results/nfe_quality_curve.png`.

## Result 2 — block × NFE: the curves CROSS — Gate G2-N (H2 flipped)
gen-PPL at matched NFE across blocks:

| matched NFE | block 4 | block 8 | block 16 |
|--:|--:|--:|--:|
| ~130–260 | 808 | 803 | 442 |
| ~557 | 327 | 357 | **115** |
| ~650 | 320 | 190 | **81** |
| 1023 (full) | **24.2** | 31.0 | 31.2 |

- **Pre-registered H2 ("larger blocks need more steps → NFE\* grows with block size") is REJECTED — the
  opposite holds.** **Smaller blocks need *more total* NFE to converge** (block-4 is still gen-PPL 225 at NFE 770;
  block-16 is already 57 at NFE 748). At *equal steps-per-token*, the larger block wins — more tokens denoised
  jointly per step, and a shorter AR chain (64 vs 256 blocks) → less cross-block error compounding.
- **The deeper, confirmed finding: block size and NFE interact — the optimal block depends on the step budget.**
  The full-NFE ordering (small block best: 24.2 < 31) and the reduced-NFE ordering (large block best: 115 < 327)
  are *opposite*; the curves cross. There is no single best block size — only a best (block, NFE) operating point.

## Result 3 — the unified operating map (gen-PPL vs throughput Pareto)
Fusing gen-PPL × measured throughput over all samplable (block, NFE) points (`results/pareto_genppl_throughput.png`):

- The non-dominated frontier runs from a **turbo corner** — block 8, NFE 128, ~277 tok/s but gen-PPL 803
  (fast garbage) — down to a **quality corner**: **block 4 at full NFE = best gen-PPL (24.2), slowest (~42 tok/s)**.
- **Block 16 owns the speed–quality middle of the frontier** (e.g. NFE 425 → ~144 tok/s @ gen-PPL 226;
  NFE 643 → ~98 tok/s @ 81; NFE 809 → ~75 tok/s @ 44): for any given throughput in the usable range, block 16
  gives the best quality. 11 of 16 measured points are non-dominated.
- *(Throughput is single-stream, batch 1, gpu-short-class GPUs; absolute tok/s carries node-to-node timing
  noise — the frontier shape, not the exact tok/s, is the claim. 5 very-low-NFE points are omitted: they trip
  the entropy<4 stop too often to time at batch 1, and are dominated garbage-quality points regardless.)*
- **The two product "modes" fall straight out of the map:** *turbo* = larger block + low NFE; *quality* =
  smaller block + high NFE. This is exactly the latency-tiered surface a commercial dLLM exposes.

## Headline takeaways
1. The "fast vs quality" dial is real and measurable; on the base model it is a **smooth, lossy trade**
   (soft knee NFE\*≈550–650), not a free lunch — few-step quality must be *bought back* (consistency distillation).
2. **Block size and NFE interact** — the best block depends on your NFE budget; the curves cross. (Novel: the
   paper reports only one NFE point per block; we trace the whole sub-NFE surface.)
3. The unified Pareto **is** the enterprise turbo/quality operating map, with block 16 dominating the middle.

## Honest caveats
- **N=25 samples/point** → gen-PPL noisy (std large at low NFE); we report the corpus aggregate + co-report entropy.
  Trends are robust; absolute low-NFE gen-PPLs are coarse. (Playbook target N≥256 for publication-grade numbers.)
- Throughput measured **single-stream, batch 1, len 1024** on one GPU (gpu-short class) — one operating point;
  absolute tok/s is hardware-specific, the *shape* of the Pareto is the claim.
- **Block 32** (Inception's value) has no released checkpoint, so it is absent from the quality axis here;
  BlockPareto covers its (weight-independent) throughput separately.
- gen-PPL under GPT-2-large is the paper's own metric (comparable), not ground-truth quality; a MAUVE
  cross-check is left as future work. Low-NFE points have high generation-time rejection rates (entropy<4 stop).

## Reproduce
```bash
# gen-PPL curve (drives the repo's sample_eval):
FH=false TS="4 6 8 12 16 24 32 48 64" NSB=25 BLOCK=16 sbatch --partition=gpu --time=04:00:00 exp/nfe_genppl.sbatch
# blocks 4/8 + anchors; then throughput at matched points:
sbatch exp/nfe_throughput.sbatch
# parse + plot:
python analysis/parse_genppl.py && python analysis/analyze_nfe.py && python analysis/analyze_pareto_nfe.py
```

## Status
Phases 0–3 complete (Gates G0-N, G1-N, G2-N passed). Next: the consistency-distillation extension — close the
few-step gap this study quantifies (the Mercury / CDLM recipe) — tracked as menu idea #4 / a Phase-5 candidate.
