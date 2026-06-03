#!/usr/bin/env python
# =============================================================================
# analyze_phase2.py  —  Phase 2 efficiency analysis (production HF path + ref).
#
# Inputs (one BENCH_JSON record per block size):
#   results/phase2_efficiency_hf.jsonl   PRODUCTION path: HF modeling_bd3lm
#                                        (hf_random; validated weight-independent)
#   results/phase2_efficiency_dit.jsonl  REFERENCE: native random-init DIT
#
# Outputs:
#   results/phase2_efficiency.csv        merged tidy table
#   results/phase2_throughput.png        tok/s vs block size (prod + ref)
#   results/phase2_memory.png            peak GPU mem vs block size (prod + ref)
#
# Weight-independence note: hf_random@16 == hf_pretrained@16 (real checkpoint)
# to within 0.03% (61.72 vs 61.74 tok/s) -> random init is a faithful proxy, so
# the unreleased 32/64/128 production points are trustworthy.
#
# Usage:  python analysis/analyze_phase2.py
# =============================================================================
import json
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')


def load(name, label):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    df = df.sort_values('block_size').reset_index(drop=True)
    df['series'] = label
    return df


prod = load('phase2_efficiency_hf.jsonl', 'production (HF modeling_bd3lm)')
ref = load('phase2_efficiency_dit.jsonl', 'reference (native DIT)')
frames = [d for d in (prod, ref) if d is not None]
alldf = pd.concat(frames, ignore_index=True)
alldf.to_csv(os.path.join(RES, 'phase2_efficiency.csv'), index=False)

gpu = prod['gpu'].iloc[0]
peak = prod.loc[prod['tok_per_s'].idxmax()]
r32 = prod[prod.block_size == 32].iloc[0]
print('PRODUCTION (HF) path:')
print(prod[['block_size', 'tok_per_s', 'sec_per_nfe', 'nfe', 'peak_mem_gb']].to_string(index=False))
print(f"\npeak throughput: block_size={int(peak.block_size)} @ {peak.tok_per_s} tok/s")

# ---- throughput vs block size -----------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(prod['block_size'], prod['tok_per_s'], 'o-', color='#1f77b4', lw=2.2, ms=7,
        label='production (HF modeling_bd3lm)')
if ref is not None:
    ax.plot(ref['block_size'], ref['tok_per_s'], 's--', color='#7f7f7f', lw=1.5, ms=5,
            alpha=0.7, label='reference (native DIT)')
ax.set_xscale('log', base=2)
ax.set_xticks(prod['block_size'])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlabel('block size (log scale)')
ax.set_ylabel('throughput (tokens / s)')
ax.set_title('BD3-LM generation throughput vs block size\n'
             f'(seqlen 1024, batch 1, sdpa+KV-cache, bf16, {gpu})')
ax.grid(True, alpha=0.3)
ax.axvline(int(peak.block_size), color='#d62728', ls=':', alpha=0.8,
           label=f"peak = block {int(peak.block_size)} ({peak.tok_per_s:.1f} tok/s) — Inception's value")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(RES, 'phase2_throughput.png'), dpi=140)

# ---- peak memory vs block size ----------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
ax2.plot(prod['block_size'], prod['peak_mem_gb'], 'o-', color='#9467bd', lw=2.2, ms=7,
         label='production (HF modeling_bd3lm)')
if ref is not None:
    ax2.plot(ref['block_size'], ref['peak_mem_gb'], 's--', color='#7f7f7f', lw=1.5, ms=5,
             alpha=0.7, label='reference (native DIT)')
ax2.set_xscale('log', base=2)
ax2.set_xticks(prod['block_size'])
ax2.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax2.set_xlabel('block size (log scale)')
ax2.set_ylabel('peak GPU memory (GB)')
ax2.set_title(f'BD3-LM peak generation memory vs block size\n(seqlen 1024, batch 1, {gpu})')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(os.path.join(RES, 'phase2_memory.png'), dpi=140)

print(f"\nblock 32: {r32.tok_per_s} tok/s ({100*r32.tok_per_s/peak.tok_per_s:.1f}% of peak), "
      f"{r32.peak_mem_gb} GB")
print(f"wrote: phase2_efficiency.csv, phase2_throughput.png, phase2_memory.png in {RES}")
