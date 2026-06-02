# Conventions

## Commit messages
Format: **`type: imperative summary`** (≤ ~72 chars). Optional body after a blank line for the *why*.

**Types**
| type | use for |
|------|---------|
| `feat`  | new code / capability (measurement, analysis, training scripts) |
| `fix`   | bug or error fix |
| `docs`  | README, SPEC, LOG, notes, this file |
| `build` | environment, SLURM, dependencies, infra |
| `chore` | housekeeping (gitignore, cleanup, file moves) |
| `exp`   | experiment runs and their results (research-specific) |

**Rules**
- Imperative mood ("add", not "added"/"adds"); one logical change per commit.
- Optional phase scope: `exp(P2): ...`, `feat(P2): ...`.

**Examples**
- `chore: scaffold project repo + SLURM conda env-build script`
- `docs: add SPEC blueprint (phases, methodology, budget, gates)`
- `build: isolate conda pkgs cache to fix corrupted shared cache`
- `feat(P2): throughput + KV-mem harness for block-size sweep`
- `exp(P1): reproduce bd3lm-owt-block_size16 PPL (within paper)`

## Engineering log
Append a dated entry to `LOG.md` (newest on top) each work session: what ran, where, the result, the decision.
