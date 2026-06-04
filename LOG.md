# Engineering Log — Block-Size Pareto

Newest entries at top. This is the running devops/lab log: what was run, where, the result, and the decision. Wiki mirror: `~/Documents/wiki/wiki/projects/BlockPareto.md`.

---

## 2026-06-03 — ✅ NfePareto Phase 3+4: unified Pareto + writeup (NfePareto COMPLETE)

Throughput at matched (block,NFE) points via `bench_gen.py` (real released weights, hf_dit) — jobs
7418923 + 7419597. Fused with the gen-PPL curves → `results/pareto_genppl_throughput.png`, `nfe_pareto.csv`.

- **bench_gen fix:** it called `_semi_ar_sampler` single-shot; at very low NFE the entropy<4 stop returns
  (None,None) → crash. Mirrored `_sample`'s resample loop (retry-to-success, time only the success). Recovered
  most points; **5 very-low-NFE points (b4 T3/4, b8 T3/4, b16 T4) still untimeable at batch-1/bf16** (>15
  consecutive rejects) — all dominated garbage-quality points, omitted with a caveat.
- **Unified Pareto (16 pts, 11 non-dominated):** frontier is a clean **turbo → quality** arc —
  turbo corner b8 NFE128 ~277 tok/s @ gen-PPL 803; **block 16 owns the middle** (NFE425 ~144 tok/s @226;
  NFE643 ~98 @81; NFE809 ~75 @44); **quality corner b4 full-NFE ~42 tok/s @ 24.2**. = the latency-tiered
  product surface. Throughput is single-stream/batch-1/gpu-short-class (node timing noise; shape is the claim).
- **Phase 4:** `FINDINGS_NFE.md` (full writeup), README sister-result section + figure, layout/status updated.
- **NfePareto COMPLETE (Phases 0–4; Gates G0-N/G1-N/G2-N passed).** Next: consistency-distillation extension
  (close the few-step gap this study quantifies — the CDLM/Mercury recipe), tracked as menu idea #4.

## 2026-06-03 — ✅ NfePareto Phase 2: block×NFE — H2 FLIPPED (curves cross) (Gate G2-N)

Jobs 7417509 (b4), 7417510 (b8) fh=false sweeps + 7417511/12 fh=true anchors. All COMPLETED.
Blocks {4,8,16} × NFE. `results/nfe_quality_curve.png` (now 3 series + paper lines).

**Anchors reproduce paper (fh=true, NFE 1023):** b4 24.2 (paper 25.7), b8 31.0 (30.4), b16 31.2 (33.4) — all within ~6%.

**gen-PPL at matched NFE (the headline — curves CROSS):**
| NFE | block4 | block8 | block16 |
|--:|--:|--:|--:|
| ~130–260 | 808 | 803 | 442 |
| ~557 | 327 | 357 | **115** |
| ~650 | 320 | 190 | **81** |
| 1023 | **24.2** | 31.0 | 31.2 |

- **Gate G2-N — H2 REJECTED & FLIPPED.** Original H2 (NFE* grows with block size) is wrong: **smaller blocks
  need MORE total NFE to converge** (b4 still 225 @ NFE 770; b16 57 @ 748). At *equal steps-per-token* the
  larger block wins (more tokens denoised jointly/step; shorter AR chain → less cross-block error compounding).
- **Richer confirmed finding: block size × NFE interact — optimal block depends on the NFE budget.**
  Low NFE (turbo) → bigger block (16) degrades gracefully; full NFE (quality) → smaller block (4) best.
  The two orderings cross ⇒ a concrete turbo/quality operating recipe (the 2-D map this project targets).
- T=2 fails for ALL blocks (entropy<4 auto-reject) = consistent low-step degeneracy floor. Entropy 4.9–5.6 healthy.
- **Next (Phase 3):** weight-independent throughput (`bench_gen`, hf_random, fh=false) at matched (block,NFE)
  points → fuse into the unified gen-PPL-vs-throughput Pareto + the enterprise "turbo/quality" reading.

## 2026-06-03 — ✅ NfePareto Phase 1: NFE→quality curve at block 16 (Gate G1-N)

Job 7415004 (gpu/V100, 57:51, COMPLETED clean). block-16, fh=false, T∈{4,6,8,12,16,24,32,48,64}
+ fh=true anchor. N=25/point. `results/nfe_genppl.csv`, `results/nfe_quality_curve.png`, `nfe_vs_t.png`.

| T | NFE | gen-PPL (corpus) | mean±std | entropy |
|--:|--:|--:|--:|--:|
| 4  | 247 | 442.3 | 452±103 | 5.24 |
| 6  | 343 | 379.4 | 399±135 | 5.57 |
| 8  | 425 | 225.9 | 240±78  | 5.49 |
| 12 | 557 | 114.5 | 126±65  | 5.41 |
| 16 | 643 | 80.8  | 85±27   | 5.44 |
| 24 | 748 | 56.9  | 60±22   | 5.41 |
| 32 | 809 | 44.1  | 46±14   | 5.36 |
| 48 | 874 | 42.3  | 43±8    | 5.39 |
| 64 | 910 | 36.6  | 38±11   | 5.32 |
| fh=true | 1023 | 31.2 | 32±8 | 5.31 |

- **Gate G1-N ✅ — H1 confirmed (nuanced):** gen-PPL **monotone-decreasing** in NFE; **soft knee NFE*≈550–650**
  (T≈12–16). Steep below (442→81), gentle above (81→31). **No hard plateau** — quality keeps improving to
  full NFE (31@1023 is 30% better than 44@809). On the *base* model, more steps = better all the way down,
  with diminishing returns. Even at the knee, gen-PPL 81 is 2.6× the full-NFE 31 ⇒ **few-step sampling is
  genuinely lossy on the undistilled model** (this is the gap consistency-distillation/CDLM exists to close).
- **Entropy healthy 5.24–5.57 at every point** incl. the gen-PPL-442 garbage ⇒ incoherent-but-diverse, not
  repetition-degenerate. gen-PPL and entropy catch different failures across the whole sweep (FlowLM lesson).
- **Mechanistic note:** fh=false asymptotes at NFE≈910 (T=64, ~14 steps/block); fh=true reaches 1023
  (1 tok/step = 16 steps/block) — the true max-NFE/quality endpoint.
- **Next (Phase 2):** repeat for blocks {4,8} (released ckpts) → test H2 (does NFE* grow with block size?).

## 2026-06-03 — ▶ NfePareto Phase 0 START (gen-PPL ↔ NFE harness)

New study (sister to BlockPareto, reuses this repo/env): quality (gen-PPL) vs denoising
steps (NFE). Spec: `SPEC_NFE.md`. Wiki: `projects/NfePareto.md`.

- **Source read (grounding rule #1):** `main.py` (`generate_samples`), `metrics.py`
  (`record_generative_perplexity` = gpt2-large gen-PPL + **unigram-entropy diversity guard**),
  `diffusion.py::_semi_ar_sampler`/`_ddpm_caching_update`, `configs/`. Pulled read-only to
  `/Volumes/Crucial_X9/Projects/_refs/bd3lms-ro` (NOT vendored into repo).
- **NFE dial verified in code:** `first_hitting=true` ⇒ ~1 tok/step ⇒ NFE≈seqlen, independent
  of T (explains BlockPareto's flat 1023; it's the paper's quality mode w/ T=5000). 
  `first_hitting=false` ⇒ fixed T-step schedule ⇒ **NFE≈(seqlen/block)·T, controllable via algo.T**.
  NFE recorded as the sampler's actual `sampling_steps` (measured, not assumed).
- **Design:** drive the repo's OWN `main.py mode=sample_eval` (validated metric — no re-scoring)
  + parse CSV; throughput from weight-independent `bench_gen.py` at matched (block,T,fh=false).
  `sample_eval` is unconditional ⇒ no dataset needed ⇒ Phase 0 cheap.
- **Built:** `exp/nfe_genppl.sbatch` (drives sample_eval; knobs BLOCK/TS/FH/SEEDS/NSB/EBS/LEN;
  defaults reproduce `scripts/gen_ppl/genppl_bd3lm.sh`), `analysis/parse_genppl.py` (CSV+log → table).
- **Cluster ready:** env intact; gpt2-large + bd3lm-owt-block_size16 already cached in $HF_HOME.
- **Jobs 7414624 (validation) + 7414625 (sanity)** on gpu-short.

### ✅ Gate G0-N PASSED (2026-06-03)

| block | T | first_hitting | NFE (measured) | gen-PPL (corpus) | gen-PPL mean±std | entropy | n |
|------:|--:|:--|--:|--:|--:|--:|--:|
| 16 | 8    | false | **424**  | 246.0 | 256±70 | 5.53 | 10 |
| 16 | 32   | false | **807**  | 43.2  | 46±18  | 5.39 | 10 |
| 16 | 5000 | true  | **1023** | **31.2** | 32±8 | 5.31 | 25 |

- **Reproduction:** block-16 gen-PPL **31.2** vs paper Table 7 = **33.4** at NFE≈1K → **6.6% better,
  within the ~5–10% gate**. (24/25 samples; gpt2-large judge, nucleus 0.9, len 1024.)
- **NFE dial verified controllable + measured:** fh=false T∈{8,32} → NFE {424, 807}; fh=true → 1023
  (≈seqlen, 1 tok/step). gen-PPL **monotone-decreasing with NFE** (246→43→31) ⇒ H1 direction holds.
- **Entropy guard healthy (5.3–5.5) and complementary:** the bad NFE-424 point still has high unigram
  entropy ⇒ incoherent-but-diverse, not repetition-degenerate. gen-PPL catches incoherence; entropy
  catches repetition — both needed (FlowLM lesson, confirmed empirically).
- **Low-NFE floor found:** T=2 (~NFE 128) is UNSAMPLABLE — auto-rejected by the built-in entropy<4
  stop-condition after 10 retries. The Phase-1 sweep must start above this floor.
- **Harness quirk fixed:** repo's `utils.update_and_save_csv` writes NO header (opens append-mode
  before its exists-check) ⇒ `parse_genppl.py` reads CSV positionally.
- **Next (Phase 1):** block-16 fh=false sweep T∈{4,6,8,12,16,24,32,48,64} (NSB=25) on `gpu` → full
  gen-PPL/entropy/NFE-vs-T curve; locate knee NFE*; then bump key points to N≥256.

## 2026-06-03 — ✅ Phase 3 + 4: quality sweep + the Pareto (block 32 = frontier endpoint)

- **Quality sweep:** trained BD3-LM from scratch on wikitext103, matched 2500-step budget, len 256, sdpa, blocks {4,16,32,64,128} (5 parallel `gpu`-partition jobs, ~47 min each on A100/H200). Probe lessons: sdpa OOMs at len 1024 (dense (2L)² mask) → len 256; `enable_checkpointing=false` clashes w/ default ModelCheckpoint → dropped.
- **val NLL / PPL:** 4→5.965/389 · 16→6.035/418 · 32→6.157/472 · 64→6.214/500 · 128→6.198/492. Quality worsens monotonically 4→32, plateaus 64/128 (within noise). Smaller block ⇒ closer to AR ⇒ better likelihood.
- **Pareto (quality vs throughput):** non-dominated = **{4,16,32}**; 64/128 strictly dominated by 32. **Block 32 = throughput-optimal frontier endpoint** — max speed at small quality cost vs 16; never dominated. `results/pareto_quality_speed.png`.
- **Writeup:** `FINDINGS.md` (the Inception-facing artifact). Honest caveats logged: quality is fixed-budget-from-scratch (ordering, not converged absolutes); throughput is one operating point; production vs reference backbone differ.
- **Phases 0–4 COMPLETE.** Next (optional): finetune-from-pretrain sweep to sharpen absolute quality; blog post; author/Inception outreach.

## 2026-06-02 (cont.) — ✅✅ Phase 2 CORRECTED & STRONGER — production peak at block 32

**Why corrected:** the weight-independence check (native `dit` random vs `hf_dit` real ckpt @ bs16) did NOT match — 49.2 vs 61.7 tok/s. Root cause: those are DIFFERENT implementations (native DIT vs HF modeling_bd3lm), so the check conflated implementation + weights. Fix: benchmark the PRODUCTION HF path with random weights (`hf_random`, built via `AutoModelForMaskedLM.from_config` — works for any block size), and validate properly = same impl, random vs trained weights.

**Weight-independence PROVEN (correct test):** `hf_random@16` = 61.72 tok/s vs `hf_pretrained@16` (real checkpoint) = 61.74 tok/s → **0.03% diff**. Random init is a faithful timing proxy ⇒ the unreleased 32/64/128 production points are trustworthy.

**PRODUCTION (HF modeling_bd3lm) throughput — PEAK AT BLOCK 32:**

| block | tok/s | peak mem GB |
|------:|------:|-----:|
| 4   | 53.57 | 0.77 |
| 8   | 60.06 | 0.77 |
| 16  | 61.72 | 0.78 |
| **32**  | **62.52 ← PEAK (100%)** | 0.79 |
| 64  | 60.64 | 0.82 |
| 128 | 57.78 | 0.90 |
| 256 | 52.26 | 1.06 |
| 512 | 40.83 | 1.36 |
| 1024| 23.98 | 1.98 |

- **Headline:** on the production code path, throughput is **maximized at block_size 32** — exactly the value Ermon said Inception runs (native reference DIT, less optimized, peaks at 16; kept as labeled comparison in `results/phase2_efficiency_dit.jsonl`).
- NFE flat 1023 (first_hitting); peak mem monotonic 0.77→1.98 GB. Plot: `results/phase2_throughput.png`; data: `results/phase2_efficiency.csv`.
- **Caveat (honest):** measured on a V100 at batch 1, seqlen 1024. The exact knee can shift with GPU/batch/seqlen; throughput is one axis — quality (Phase 3) completes the Pareto. Lesson: always benchmark the production implementation, not just an architecturally-equivalent one.

## 2026-06-02 (cont.) — Phase 2 first pass (native DIT reference; superseded as headline)

Built a generation-efficiency harness (`bench/bench_gen.py`) + sweep job (`exp/p2_efficiency.sbatch`). Uses the **native random-init `dit` backbone** (weight-independent → works for unreleased 32/64/128). Adapted two native-backbone sampler gaps in the harness (reset_kv_cache signature, gen_mask) without touching pinned upstream. Canonical inference config: sdpa + KV-cache + first_hitting, algo.T=5000, bf16, seqlen 1024, batch 1, V100.

**Result — throughput is NON-MONOTONIC with a clear knee (H1 CONFIRMED):**

| block | tok/s | sec/NFE | peak mem GB |
|------:|------:|--------:|-----:|
| 4   | 42.48 | 0.0236 | 1.17 |
| 8   | 46.92 | 0.0213 | 1.17 |
| **16**  | **49.18 (peak)** | 0.0204 | 1.18 |
| **32**  | 47.55 (97% of peak) | 0.0211 | 1.18 |
| 64  | 46.88 | 0.0214 | 1.20 |
| 128 | 44.59 | 0.0225 | 1.22 |
| 256 | 38.98 | 0.0257 | 1.35 |
| 512 | 28.94 | 0.0346 | 1.66 |
| 1024| 18.46 | 0.0542 | 2.27 |

- **Throughput peaks at block 16, falls off both sides:** small blocks pay per-stride/loop + KV-store overhead (256 strides @ block 4); large blocks pay expensive per-step attention (bigger query). **NFE flat at 1023** for all (first_hitting ⇒ ~1 token/step ⇒ NFE≈seqlen), so the curve is pure per-step cost — clean.
- **Peak memory rises monotonically** (1.17→2.27 GB) with block size (larger activation working set).
- **Block 32 = ~97% of peak throughput** at near-min memory → consistent with the "32 is the production sweet spot" thesis (quality upside measured in Phase 3). Headline plot: `results/phase2_throughput.png`; data: `results/phase2_efficiency.csv`.
- **Weight-independence validation** (real hf_dit ckpt vs random dit @ block 16) running (job 7401784) to empirically confirm timing is backbone/weight agnostic.

## 2026-06-02 (cont.) — ✅✅ Phase 1 COMPLETE — Gate G1 PASSED (full PPL)

- **Full-valid ppl_eval (job 7392432, `gpu` partition V100, block_size 16, sdpa, all 6891 batches, 1h59m):** COMPLETED.
  - **`val/ppl = 22.297`** · val/nll 3.1044 · val/bpd 4.4788.
  - **Paper (BD3-LM L'=16, OWT, len 1024): ≤ 22.27 → reproduced within ~0.1%.** Faithful end-to-end reproduction on our infra/pipeline. Result logged to `results/phase1_ppl.csv`.
- **Gate G1 PASSED** → pipeline is trustworthy; proceed to Phase 2 (efficiency Pareto).
- Throughput baseline: ~0.97 it/s @ eval_batch_size 16, len 1024 on V100 (sdpa) → ~15.9 seq/s · ~16.3k tok/s aggregate for scoring (NOT generation throughput; that's a Phase 2 measurement).

## 2026-06-02 (cont.) — ✅ Phase 1 smoke test PASSED (PPL reproduces paper)

- **OWT prep (job 7389503, `sharing`):** COMPLETED in 7.5 min — 100k valid docs, 38 GB raw cache. (`short` was saturated → moved to idle `sharing`; 1h cap was ample.)
- **Hydra fix:** `+trainer.limit_val_batches` failed (key exists in config) → plain override `trainer.limit_val_batches=8`. Commit `2218b54`.
- **ppl_eval sanity (job 7390025, gpu-short V100, block_size 16, sdpa, 8 val batches):** COMPLETED 2:05.
  - **`val/ppl = 22.23`** · val/nll 3.1015 · val/bpd 4.4746.
  - **Paper target (BD3-LM L'=16, OWT): ≤ 22.27 → MATCH** (within ~0.2%, even on 8 batches). End-to-end pipeline validated: HF ckpt load, OWT tokenize+cache (`openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat`), block-diffusion **sdpa** masking, PPL compute — all correct.
  - GPU throughput observed ~0.8 it/s (eval_batch_size 16, len 1024) on V100 — informs Phase 2/3 timing.
- **Full-valid run sizing lesson:** full OWT valid = **6891 batches @ ~0.98 it/s on V100 ≈ 1h57m** → would hit the gpu-short **2h wall** (PPL prints only after the last batch ⇒ result lost). Killed job 7390111 at 27% and **resubmitted on the 8h `gpu` partition** (job 7392432, `--partition=gpu --time=04:00:00`). Script updated with this sizing note; full runs must use `gpu`, gpu-short is for LIMIT sanity only.
- Harmless: "Token indices sequence length is longer than 1024" = GPT-2 tokenizer notice before bd3lm `wrap` packs into 1024-blocks. Not an error.

## 2026-06-02 (cont.) — Phase 1 wired up (smoke-test plan, grounded in repo)

Read the bd3lms source to design Phase 1 correctly (not guess):
- **ppl_eval skips train.** `main.py:_ppl_eval` → `get_dataloaders(..., skip_train=True)`. Only the OWT **valid** split (`openwebtext` `train[-100000:]`, 100k docs) is tokenized — no 8M-doc train tokenization. Big cost saver.
- **One-time data cost = the OWT raw download (~40 GB).** HF `openwebtext` is single-split, so even the valid slice forces a full-shard download. Split off into a **CPU job** (`exp/p1_prep_owt.sbatch`, `short`, 16 CPU/64G/12h) that only warms the raw HF cache — no GPU idle, no 2h cap risk, and zero tokenizer-param matching (main.py tokenizes on the GPU job against the warm cache).
- **sdpa is faithful** for PPL. `dit.py` feeds the SAME `block_diff_mask` to both backends; the sdpa path (`cross_attn`) does `F.scaled_dot_product_attention(attn_mask=mask, is_causal=False)`. So we use `attn_backend=sdpa` and run on the V100 that `gpu-short` provides (flex wants Ampere+). Eval job: `exp/p1_ppl_owt.sbatch` (1 GPU, staged: `LIMIT=8` sanity → full valid).
- **Cache key:** with `data.insert_valid_special=False` (paper protocol + released-script default), the tokenized file is `openwebtext-valid_validation_bs1024_wrapped_specialFalse.dat`.
- **Targets** (paper Table 1, OWT, len 1024): block_size 4 → ~20.73, 16 → ~22.27. Smoke passes if within ~2%.
- Jobs: prep **7383163** submitted (PENDING on `short`). Eval queued behind it.

## 2026-06-02 (cont.) — ✅ env built, Gate G0 PASSED

- **Job 7380846 COMPLETED** (53 min, exit 0:0) on `gpu-short` node d1007 (Tesla V100-SXM2-32GB).
- **Isolation fix confirmed:** effective `pkgs_dirs` listed ONLY the project cache; corrupted `matrix-game` cache off the path. Everything downloaded clean.
- **Gate G0 verify (all green):** python 3.9.25 · **torch 2.7.1+cu126** · CUDA build tag 12.6 · `torch.cuda.is_available()=True` · GPU=Tesla V100-SXM2-32GB · transformers 4.49.0 · lightning 2.5.0.post0 · datasets 3.3.2 · triton 3.3.1 · hydra-core 1.3.2 · **ALL CORE IMPORTS OK**.
- **Lockfile frozen & committed:** `env/requirements.lock.txt` (177 pkgs) — full provenance. Versions match bd3lms `requirements.txt` exactly.
- Build was slow (~53 min) due to the large dependency tree (full torch+CUDA wheels + Jupyter/sklearn/pandas long tail) written to network /scratch. One-time cost; env is cached now.
- Activate: `conda activate /scratch/gupta.yashv/block-pareto/envs/blockpareto`.
- **Next:** Phase 1 smoke test — `mode=ppl_eval` on `kuleshov-group/bd3lm-owt-block_size16`, target PPL within ~2% of paper (≤22.27).

## 2026-06-02 (cont.) — env-build: root-caused corrupted conda cache

- **Two FAILED env builds (jobs 7380571, 7380683)** — both `CondaVerificationError`: the package `wheel-0.45.1-py39h06a4308_0` in the shared cache `/scratch/gupta.yashv/matrix-game/conda-pkgs` is corrupted (extracted dir missing files listed in its manifest).
- **First fix attempt (insufficient):** set `CONDA_PKGS_DIRS=$PROJECT/conda-pkgs`. **Did not work** — `CONDA_PKGS_DIRS` only controls where conda *writes/downloads*; conda still *scans every `pkgs_dirs` in `~/.condarc`* (which hard-codes the matrix-game cache) for an already-extracted package to hard-link, so it kept reusing the broken `wheel-0.45.1`.
- **Real fix (job 7380728):** keep the isolated `CONDA_PKGS_DIRS`, **and** add step `[1b]` that deletes ONLY the corrupted `wheel-0.45.1` package (dir + `.conda`/`.tar.bz2`) from all `pkgs_dirs` before `conda create`, forcing a clean re-download. Targeted — nothing else in the shared cache is touched. Commit `3cb8e32`.
- Lesson: `CONDA_PKGS_DIRS` ≠ isolation when `~/.condarc` lists a shared (and corruptible) `pkgs_dirs`. Cache hits are read across all of them.

## 2026-06-02 (cont.) — GitHub spine + env-build requeue to gpu-short

- **GitHub repo created (public):** https://github.com/BrutalCaeser/block-diffusion-pareto . Local `main` pushed. This is now the source of truth/backup/portfolio. Cluster becomes a `git clone` (was an rsync target); rsync now only for pulling large generated results back.
- **Upstream pinned:** bd3lms @ `1c3e8f43d88dfbcee5ff2aa6932a9e74b31ae1d7` (2025-07-10) → `UPSTREAM.md`. Not vendored.
- **Env-build requeue:** job 7380430 sat `PD` on `short` (partition jammed — 0 idle nodes). Requeued to **`gpu-short` + `--gres=gpu:1`** (idle nodes → immediate start) — and a GPU node lets us verify `torch.cuda.is_available()` for real. One-time expedient; heavy CPU-only work → `short` in future.

## 2026-06-02 — Project start: recon + env build

**Recon (verified, not assumed):**
- Cluster: Explorer (`gupta.yashv@explorer.northeastern.edu`), key-based SSH OK (`ssh explorer`), login node `explorer-01`.
- GPUs: partition `gpu` has **H200 (8/node)**, A100 (3–4/node); 8h limit. `gpu-short`/`gpu-interactive` = 2h (dev/smoke). CPU partition `short` = 2 days.
- Storage: `/scratch/gupta.yashv` (petabyte-scale, ample) for code/data/ckpts/envs; home is quota-limited.
- Modules: `anaconda3/2024.06`, `miniconda3/25.9.1`, `cuda/{12.1.1,12.3.0,12.8.0,13.2.0}`. wandb API key already on box (`~/.wandb_api_key`).
- Existing project envs live in **/scratch via `-p` prefix** (e.g. `/scratch/gupta.yashv/flm/envs/flm`) → we follow that pattern.
- BD3-LMs cloned (shallow) → `/scratch/gupta.yashv/block-pareto/bd3lms`. Real `requirements.txt`: torch==2.7.1, triton==3.3.1, transformers==4.49.0, lightning==2.5.0.post0, hydra-core, datasets==3.3.2, wandb==0.19.8. **No flash-attn pin** (uses FlexAttention/sdpa).
- **Confirmed research gap**: HF releases block sizes **4/8/16 only** (`kuleshov-group/bd3lm-owt-block_size{4,8,16}`) + a `block_size1024-pretrain` init + MDLM/SEDD/AR baselines. **No 32/64/128** → the novel points.
- Block size must divide context length (1024) → valid sweep: {4,8,16,32,64,128,...}.

**Decisions:**
- `conda create` on the **login node was Killed** (mem/cpu cap) → build env via **SLURM job on a compute node** (`env/build_blockpareto_env.sbatch`, partition `short`, 8 CPU / 48G).
- Env name/location: `-p /scratch/gupta.yashv/block-pareto/envs/blockpareto` (scratch prefix, matches FLM pattern; avoids home quota).
- Install from the **repo's own requirements.txt** (grounded, not transcribed). Skip flash-attn (would not build vs torch 2.7.1; not needed for bd3lm flex/sdpa).

**Status:** env build job submitted (see below / wiki). Next: smoke-test PPL on a released ckpt to validate the pipeline before any sweep.
