# Engineering Log — Block-Size Pareto

Newest entries at top. This is the running devops/lab log: what was run, where, the result, and the decision. Wiki mirror: `~/Documents/wiki/wiki/projects/BlockPareto.md`.

---

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
