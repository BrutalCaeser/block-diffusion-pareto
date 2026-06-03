# SPEC_NFE — Quality ↔ NFE "Speed Dial" for Block-Diffusion LMs

**Owner:** Yashvardhan Gupta · **Started:** 2026-06-03 · **Cluster:** Northeastern Explorer
**Predecessor:** [BlockPareto](./SPEC.md) (block-size axis — COMPLETE). This study adds the **denoising-step (NFE) axis** and fuses the two into one inference-cost map.
**Upstream:** [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) @ `1c3e8f4` (see `UPSTREAM.md`).
**Wiki playbook:** `~/Documents/wiki/wiki/projects/NfePareto.md` (this file is its repo-side execution spec).

> Blueprint, grounded in the **verified** bd3lms source (read 2026-06-03: `main.py`,
> `metrics.py`, `diffusion.py` sampler, `configs/`). Every override below is traced to a
> real config key or code path. Estimates tagged `[ESTIMATE]`.

---

## 1. Objective (one sentence)
Map how **generative quality (gen-PPL under GPT-2-large)** trades against the **number of
denoising function evaluations (NFE)** per generation, across block sizes, on the **released
BD3-LM checkpoints** (real weights → faithful quality; **no training → cheap**), and fuse it
with BlockPareto's throughput data into a single 2-D operating map: *given a latency budget,
which (block size, NFE) should you run?* — i.e. the dial Mercury exposes as latency tiers.

## 2. The NFE lever — verified mechanism (no assumptions)
From `diffusion.py::_semi_ar_sampler` (lines ~979–1052) and `_ddpm_caching_update` (~559–603):

- The sampler loops `for i in range(num_steps)` per block, where `num_steps = algo.T`.
- **`sampling.first_hitting=true` (config default):** `t *= u**(1/num_masked)` — a stochastic
  first-hitting schedule that unmasks ~**1 token per productive step**. To fill a block it needs
  ≈ `block_size` productive steps regardless of T, so **NFE ≈ seqlen** (independent of T).
  *This is why BlockPareto measured a flat NFE=1023.* **Not a controllable dial.** This is the
  paper's high-quality sampling mode (used with `T=5000 ≫ block`).
- **`sampling.first_hitting=false`:** `t = timesteps[i]` over a fixed linear schedule of `T`
  points; each DDPM update can unmask **many** tokens at once. The loop runs up to `T` steps
  (early-break `if mask_index not in x_accum: break`). So **NFE ≈ (seqlen/block) × T**, directly
  controllable via `algo.T`. **← this is the dial we sweep.**
- **NFE is measured, not assumed.** The sampler returns `sampling_steps`, incremented only when a
  denoising update *changes* the sequence (`_ddpm_caching_update` returns `p_x0=None` ⇒ KV/p_x0
  cache invalidated ⇒ a real forward next step). We record the **actual** returned NFE every run.
- **Built-in degeneracy guard:** `_check_stop_conds` rejects a sample if unigram entropy of the
  last 256 tokens `< 4` (non-var-length ⇒ resample, up to 10 tries). Degenerate/repetitive
  samples are already filtered at generation time; we *additionally* report entropy (below).

## 3. Faithful measurement design (grounding rule #1: use the repo's validated metric)
We do **not** re-implement gen-PPL scoring (risk of diverging from the paper). Instead:

- **Quality / entropy / NFE ← drive the repo's own `main.py mode=sample_eval`.** That path
  (`generate_samples` → `restore_model_and_sample` → `_sample` → `record_generative_perplexity`)
  already: generates N = `num_sample_batches × eval_batch_size` unconditional samples, scores
  **gen-PPL under `eval.gen_ppl_eval_model_name_or_path=gpt2-large`** via the sliding-window NLL,
  computes **unigram token entropy** (the diversity guard — `metrics.py::record_generative_perplexity`),
  records **NFE** and **length**, and appends one CSV row per sample-batch
  (`sampling.logdir` = the CSV path; columns `gen_ppl, gen_nfes, gen_entropy, gen_lengths, samples, seed`).
  It also prints the corpus-level aggregate `Generative perplexity: …` and `Entropy: …` to stdout.
- **Throughput ← `bench/bench_gen.py`** (weight-independent; built in BlockPareto) at the
  **matched** `(block, T, first_hitting=false)` point → gives `tok/s` + actual NFE for the same
  operating point, with no need for trained weights.
- We join the two on `(block, T, NFE)`.

**Unconditional sampling ⇒ no dataset needed** for `sample_eval` (verified: `main.py` builds only
the tokenizer for this mode). Phase 0 is therefore cheap — no OWT prep.

## 4. Metrics & protocol (controlled)
Fixed unless varied: `model=small`, `algo=bd3lm`, `algo.backbone=hf_dit` (the **deployed** HF path —
BlockPareto lesson: benchmark the impl that ships), `model.attn_backend=sdpa`, `model.length=1024`,
`sampling.nucleus_p=0.9`, `sampling.kv_cache=true`, released ckpt `kuleshov-group/bd3lm-owt-block_size{B}`.
Per `(block, T, seed)` record: **gen-PPL (corpus aggregate + per-batch mean±std)**, **unigram entropy**,
**actual NFE**, **gen length**, **throughput (tok/s)** (from bench_gen), seed, GPU.
**N ≥ 256** generated sequences per point (gen-PPL is noisy) via `num_sample_batches × eval_batch_size`
across ≥3 seeds; report mean ± std.

**Paper anchors — BD3-LM Table 7 (OWT, L=1024, 300 samples, GPT-2-large judge):**
| Model | Gen PPL | NFEs | | Model | Gen PPL | NFEs |
|---|---|---|---|---|---|---|
| AR | 14.1 | 1K | | BD3-LM L'=16 | **33.4** | 1K |
| MDLM | 46.8 | 1K | | BD3-LM L'=8 | 30.4 | 1K |
| SEDD | 52.0 | 1K | | BD3-LM L'=4 | 25.7 | 1K |

⇒ **G0-N target: block-16 gen-PPL ≈ 33.4 at NFE≈1024** (within ~5–10%). NFE=1K confirms
first_hitting unmasks ~1 tok/step (≈ seqlen). **Note the paper reports only this single
full-NFE point per block — our `first_hitting=false` sweep traces the whole gen-PPL-vs-NFE curve
*below* NFE=seqlen, which the paper never does. That under-NFE curve is the novel contribution.**

## 5. Phases & gates
### Phase 0 — Harness + validation (cheap, ~1–2 GPU-hr) ✅ DONE
- [x] Read `main.py` / `metrics.py` / `diffusion.py` sampler / `configs` (done 2026-06-03).
- [x] `exp/nfe_genppl.sbatch` (drives `main.py mode=sample_eval`; knobs BLOCK, TS, FH, SEEDS, NSB, EBS, LEN).
- [x] `analysis/parse_genppl.py` (per-run CSV + stdout `.log` → `results/nfe_genppl.csv`).
- [x] **Validation run** (jobs 7414624/7414625): block-16 gen-PPL **31.2** vs paper Table 7 **33.4**
      at NFE≈1024 → **6.6% within ballpark**. fh=false sanity: NFE {424,807} at T {8,32}, gen-PPL
      {246,43} → dial controllable + gen-PPL monotone-decreasing with NFE. T=2 unsamplable (entropy floor).
- **Gate G0-N ✅ PASSED:** harness reproduces paper gen-PPL, NFE is controllable+measured via
  `first_hitting=false`+`algo.T`, entropy guard healthy & complementary. CSV is headerless (repo util
  quirk) → parsed positionally.

### Phase 1 — NFE sweep at fixed block (cheap) ✅ DONE
- Block 16, `first_hitting=false`, `T∈{4,6,8,12,16,24,32,48,64}` (+ fh=true anchor), N=25 (job 7415004).
- **Gate G1-N ✅:** H1 confirmed — gen-PPL monotone-decreasing in NFE; **soft knee NFE*≈550–650** (T≈12–16);
  no hard plateau (31@1023 vs 44@809). Few-step sampling lossy on the base model (knee gen-PPL 81 = 2.6×
  full-NFE). Entropy healthy 5.24–5.57 throughout. Curve: `results/nfe_quality_curve.png`.

### Phase 2 — 2-D sweep block × NFE (moderate)
- Block ∈ {4,8,16} (released) × NFE grid; build gen-PPL & throughput surfaces.
- **Gate G2-N:** H2 verdict (does NFE* grow with block size?).

### Phase 3 — Unified Pareto
- Merge with BlockPareto throughput; plot gen-PPL vs throughput over all (block, NFE); mark frontier;
  overlay a latency-budget line. Figures: `nfe_quality_curve.png`, `nfe_surface.png`, `pareto_genppl_throughput.png`.

### Phase 4 — Writeup
- `FINDINGS_NFE.md` + a sister-result paragraph in `README.md`. Outreach hook for Inception.

## 6. Hypotheses (pre-registered)
- **H1:** gen-PPL decreases with NFE then saturates at a knee NFE*.
- **H2:** NFE* grows with block size (more tokens to jointly denoise).
- **H3:** the (block, NFE) frontier explains tiered modes — "turbo" = low NFE + mid block, "quality" = high NFE.

## 7. Compute budget
- `[ESTIMATE]` Validation: block16, len1024, first_hitting=true (NFE≈1024) ≈ 25 samples × ~25 s/sample
  ≈ 10–15 min generation + gpt2-large scoring → fits `gpu-short` (2h). Refine after G0-N.
- No training ⇒ no 8h-limit gymnastics. Full study ≈ a handful of GPU-hours–days.

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| gen-PPL gameable by repetition (FlowLM trap) | Always co-report unigram entropy; flag low-entropy points as degenerate; built-in entropy<4 reject already active |
| gen-PPL noisy | N≥256 samples, ≥3 seeds, mean±std |
| `first_hitting=false` path under-tested upstream | Validate NFE behaviour in P0 before sweeping |
| Block-32 unreleased | Faithful curves on {4,8,16}; mark 32 as extrapolated or plug our trained ckpt with a caveat |
| gpt2-large ≠ ground-truth quality | It's the paper's own metric ⇒ comparable; optional MAUVE cross-check |
| Compute-node download of a ckpt | gpt2-large + bd3lm-b16 already cached; {4,8} fetched at runtime (nodes have internet) |

## 9. Grounding rules (inherited, non-negotiable)
1. No code against an unread API (done: source read 2026-06-03). 2. Every number reproducible from a
committed script + logged command. 3. `[ESTIMATE]` on guesses. 4. Negative results reported honestly.
5. `LOG.md` every session; small Conventional Commits. 6. gen-PPL **never** reported without the entropy guard.

## 10. Definition of done
Validated gen-PPL harness (G0-N) + NFE curves for ≥3 block sizes + the unified gen-PPL-vs-throughput
Pareto with the enterprise-mode reading + honest caveats. Then → DiffuGRPO.
