# Engineering Log — Block-Size Pareto

Newest entries at top. This is the running devops/lab log: what was run, where, the result, and the decision. Wiki mirror: `~/Documents/wiki/wiki/projects/BlockPareto.md`.

---

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
