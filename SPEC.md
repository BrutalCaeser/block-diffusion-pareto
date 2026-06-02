# SPEC — Block-Size Pareto Frontier for Block-Diffusion LMs

**Owner:** Yashvardhan Gupta · **Started:** 2026-06-02 · **Cluster:** Northeastern Explorer
**Upstream:** [kuleshov-group/bd3lms](https://github.com/kuleshov-group/bd3lms) (BD3-LMs, ICLR 2025 Oral)
**Companion knowledge:** `~/Documents/wiki/wiki/concepts/diffusion-llms.md`, `projects/MicroDLM.md`, `startups/InceptionLabs.md`

> This is the blueprint. It is grounded in **verified** recon (cluster resources, the real repo, released checkpoints) — every compute/claim is tagged where it is an estimate. Both outcomes of every hypothesis are designed to be informative (pre-registered, MicroDLM-style).

---

## 1. Objective (one sentence)
Map how **block size** trades quality (perplexity, generative quality) against **inference efficiency** (throughput, KV-cache memory) in Block-Diffusion LMs, and explain **why block size 32 is a production sweet spot** — the choice Inception's Mercury reportedly makes but which is unpublished.

## 2. Why this, why now
- **The gap is real and verified.** BD3-LMs publicly release/report block sizes **4 / 8 / 16 / 128**. Ermon stated *in person* (2026-04-14) that Inception runs **block size 32**. No one has published the 32 point or a clean quality↔efficiency Pareto sweep including it.
- **It's the exact lever a commercial dLLM pulls.** Block size sets the AR-vs-diffusion interpolation: small blocks → AR-like quality, less parallelism; large blocks → more parallel speed, looser bound / more factorization error. "Why 32" is a concrete, defensible, peer-level question to bring to Inception.
- **It's tractable on our infra** and extends the MicroDLM scaling muscle directly.

## 3. Background (what BD3-LMs actually is — grounded)
- Sequence split into blocks of size `L'`; **autoregressive across blocks, diffusion (parallel masked denoising) within a block**. `L'=1` ≡ pure AR; `L'=L` ≡ pure (single-block) diffusion.
- Restores **KV caching** (clean prior blocks cached via block-causal mask) and **arbitrary length** (append blocks to EOS) that pure masked diffusion lacks.
- Backbone: DiT-style transformer (`model=small/medium/tiny`); trains with **FlexAttention** (`attn_backend=flex`), samples with **sdpa + `sampling.kv_cache=true`**.
- Signature trick: **variance reduction via clipped/data-driven noise schedules** (`algo.clip_search_widths`), re-fit per block size.
- **Released artifacts (verified on HF):** `kuleshov-group/bd3lm-owt-block_size{4,8,16}`, `bd3lm-owt-block_size1024-pretrain` (shared init), `mdlm-owt`, `{mdlm,sedd,ar}-noeos-owt`. **No 32/64/128.**
- Context length 1024 (OWT); **block size must divide length** → valid sweep {4,8,16,32,64,128,256,512}.

## 4. Research questions & pre-registered hypotheses
- **RQ1 (efficiency, weight-independent):** How do **throughput (tok/s)** and **peak KV-cache memory** vary with block size at fixed generation length + sampler on one H200?
  - **H1:** Throughput is **non-monotonic / has a knee** — small blocks pay per-block-step overhead (many sequential steps), very large blocks pay quadratic within-block attention + memory; an intermediate size (hypothesized 16–64) maximizes tok/s. *Falsifiable:* if throughput is monotonic in block size, "32 as a speed sweet spot" is wrong and the value of 32 must be a pure quality choice.
- **RQ2 (quality):** How do **validation PPL** (NLL bound) and **generative perplexity** vary with block size?
  - **H2:** Quality **degrades monotonically** as block size grows (looser NELBO + more within-block factorization error), per the paper's 4<8<16 ordering — so 32 sits on a quality–speed trade, not a quality optimum.
- **RQ3 (the synthesis):** On the **(quality, throughput, memory)** Pareto frontier, **where does 32 land** and is it Pareto-efficient (i.e., is there a regime — given a latency/memory budget — where 32 dominates 16 and 64)?
  - **H3:** 32 is **Pareto-optimal under a realistic latency-or-memory budget** even if it's neither the fastest nor the highest-quality single point — which would *explain* Inception's choice.

## 5. Metrics & measurement protocol (controlled)
Fixed unless varied: `model=small`, `model.length=1024`, OWT, single **H200**, identical sampler (`algo.T`, `sampling.nucleus_p=0.9`, `sampling.kv_cache=true`), 3 repeats, report mean ± std, GPU clocks not locked (note it).

| Metric | Type | How | Needs trained weights? |
|---|---|---|---|
| **Throughput** (tok/s) | efficiency | wall-clock generate N tokens, warmup discarded, `torch.cuda.synchronize`, median of 3 | **No** (architecture+sampler) |
| **Peak KV-cache memory** (GB) | efficiency | `torch.cuda.max_memory_allocated` during a fixed-length generation | **No** |
| **NFEs / steps** | efficiency | sampler step count to reach a fixed length | **No** |
| **Validation PPL** (NLL bound) | quality | `mode=ppl_eval` on OWT val | **Yes** |
| **Generative PPL** | quality | `scripts/gen_ppl` — GPT-2-Large judge on generated text | **Yes** |
| **Output entropy** | quality guardrail | token entropy of generations (catch the low-entropy→low-genPPL confound) | **Yes** |

> **Key design consequence:** RQ1 (throughput, memory) is **weight-independent**, so we can measure it for **all** block sizes {4..128} *immediately and cheaply* by instantiating `model=small` at each block size (released weights for 4/8/16; pretrain-init or our partial weights for 32/64/128) — quality of weights does not change the compute pattern. RQ2 quality for 32/64/128 requires **training** (Phase 3). We will never present an efficiency number as a quality number. **Entropy guardrail is mandatory** (lesson from FlowLM: GPT-2 judge rewards low-entropy text — report gen-PPL only alongside entropy vs dataset entropy).

## 6. Phases, with decision gates

### Phase 0 — Environment & repo  *(in progress)*
- [x] Recon (cluster, repo, checkpoints) — see LOG.md
- [x] Clone bd3lms → `/scratch/gupta.yashv/block-pareto/bd3lms`
- [~] Conda env `/scratch/gupta.yashv/block-pareto/envs/blockpareto` via `env/build_blockpareto_env.sbatch` (SLURM, compute node)
- **Gate G0:** env imports torch 2.7.1 (CUDA build) + transformers/lightning/hydra OK → proceed.

### Phase 1 — Pipeline validation (cheap, ~1 GPU-hr)
- Run `mode=ppl_eval` on `bd3lm-owt-block_size16` (and `4`) on a single H200; confirm our number matches the paper (OWT: `L'=16` ≤ 22.27, `L'=4` ≤ 20.73; AR 17.54, MDLM ≤ 22.98).
- Run a tiny `sample_eval` to confirm sampling + KV cache path works and to **time one step** (calibrates Phase 3 budget).
- **Gate G1:** PPL within ~2% of paper → pipeline trusted. Capture per-step wall-clock to size training.

### Phase 2 — Efficiency Pareto (cheap, high-value headline; ~few GPU-hrs)
- Build `src/throughput.py` + `src/kv_mem.py` **after reading the real `diffusion.py` sampler API** (no guessed API).
- Sweep block sizes {4,8,16,32,64,128} (+ AR `L'=1` and pure-diffusion anchors) measuring throughput, peak KV mem, NFEs at fixed length 1024 on one H200.
- Deliver the **throughput-vs-block-size** and **memory-vs-block-size** curves → first answer to "why 32" (RQ1/H1).
- **Gate G2:** if H1 holds (knee near 16–64), 32 is a *speed* story; if not, pivot the narrative to quality/quality-per-memory.

### Phase 3 — Quality axis (compute-gated; the expensive part)
Decision made **after G1 step-timing**:
- **Option A (default, tractable): controlled LM1B sweep.** Train *all* sizes {4,8,16,32,64,128} ourselves on **LM1B** (length 128 → cheaper) at one fixed budget → fully apples-to-apples quality curve. Cross-check our 4/8/16 vs released trend.
- **Option B (rigorous, expensive): OWT.** Use released 4/8/16; **train 32 (then 64,128)** finetuned from `bd3lm-owt-block_size1024-pretrain` at matched budget. Validate our pipeline by reproducing released `16` before trusting our `32`. Requires H200 + 8h-limit SIGUSR1 resubmit (already in user's toolkit).
- **Gate G3:** pick A if estimated OWT wall-clock for one block size > ~3 GPU-days; else B (or B for 32 only + A for the full curve).

### Phase 4 — Analysis, write-up, outreach
- `src/pareto.py`: assemble (quality, throughput, memory) frontier; mark where 32 lands; test H3 (Pareto-optimality under a latency/memory budget).
- Honest write-up + blog (plain language, like the wiki concept page). Repo public.
- Author email (Arriola/Sahoo/Kuleshov) sharing the block-32 frontier + one open question. Update `people/*` + wiki per CLAUDE.md.

## 7. Compute budget (grounded estimates — refine after G1)
- **GPU:** `gpu` partition H200×(1–8), **8h wall limit** → long trains use checkpoint + SIGUSR1 auto-resubmit (proven in MicroDLM/FLM).
- **Phases 0–2:** cheap — single GPU, hours total. No long jobs.
- **Phase 3 Option A (LM1B):** 6 trainings, small model, length 128 — [ESTIMATE] hours–1 day each; total a few GPU-days.
- **Phase 3 Option B (OWT, full 1M-step reproduction):** [ESTIMATE] multi-day per block size at global batch 512 — heavy; mitigated by finetuning-from-pretrain and resubmit. **Do not commit before G1 step-timing.**
- Storage: data/ckpts in `/scratch` (ample). wandb enabled (key on box) for train curves.

## 8. Risks & mitigations
- **Login-node process kills** → all heavy work via SLURM (done for env).
- **Compute-node internet** for pip → env script checks connectivity, exits clean if absent (fallback documented).
- **flash-attn** → not installed (bd3lm uses flex/sdpa); only add a torch-2.7-compatible build if MDLM/SEDD baselines are needed.
- **Weight-quality confound** → efficiency metrics declared weight-independent and never reported as quality; quality only from trained/released weights.
- **gen-PPL entropy confound** (FlowLM lesson) → always report output entropy beside gen-PPL.
- **Block-size ≠ only variable** → re-fit the clipped noise schedule per block size (don't hold the 16-tuned schedule fixed); document.
- **Training cost overrun** → Option A (LM1B) as the tractable default; Option B gated on measured step time.
- **Reproduction drift** → validate against a released checkpoint (G1) before trusting any self-trained number.

## 9. Success criteria
- **Minimum (ship-worthy):** validated pipeline (G1) + the **efficiency Pareto** (throughput + KV-mem vs block size, incl. 32) with the H1 verdict. This alone is a novel, shareable artifact.
- **Target:** + quality curve incl. our trained **block-32**, the full (quality, speed, memory) frontier, and a defensible answer to "why 32" (H3).
- **Stretch:** the frontier replicated on OWT at matched budget; co-authored note / strong blog; author engagement.

## 10. Grounding rules (non-negotiable)
1. No code written against an API I haven't read in the cloned repo. 2. Every reported number reproducible from a committed script + logged command. 3. Estimates tagged `[ESTIMATE]`; verified facts plain. 4. Negative/ null results reported honestly (MicroDLM precedent). 5. LOG.md updated every work session; commits small and message-clear.
