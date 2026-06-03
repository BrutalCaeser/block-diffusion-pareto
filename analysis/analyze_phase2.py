#!/usr/bin/env python
# =============================================================================
# analyze_phase2.py  —  turn the Phase 2 efficiency JSONL into a CSV + figures.
#
# Reads results/phase2_efficiency.jsonl (one BENCH_JSON record per block size),
# writes results/phase2_efficiency.csv, and renders:
#   results/phase2_throughput.png   throughput vs block size (the non-monotonic knee)
#   results/phase2_memory.png       peak GPU memory vs block size
#
# Usage:  python analysis/analyze_phase2.py
# =============================================================================
import json
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
SRC = os.path.join(RES, 'phase2_efficiency.jsonl')

rows = [json.loads(l) for l in open(SRC) if l.strip()]
df = pd.DataFrame(rows).sort_values('block_size').reset_index(drop=True)
df.to_csv(os.path.join(RES, 'phase2_efficiency.csv'), index=False)

peak = df.loc[df['tok_per_s'].idxmax()]
gpu = df['gpu'].iloc[0]
print(df[['block_size', 'tok_per_s', 'sec_per_nfe', 'nfe', 'peak_mem_gb', 'wall_s']].to_string(index=False))
print(f"\npeak throughput: block_size={int(peak.block_size)} @ {peak.tok_per_s} tok/s")

# ---- throughput vs block size ------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df['block_size'], df['tok_per_s'], 'o-', color='#1f77b4', lw=2, ms=7)
ax.set_xscale('log', base=2)
ax.set_xticks(df['block_size'])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('block size (log scale)')
ax.set_ylabel('throughput (tokens / s)')
ax.set_title(f'BD3-LM generation throughput vs block size\n(seqlen 1024, batch 1, sdpa+KV-cache, bf16, {gpu})')
ax.grid(True, alpha=0.3)
# annotate peak + the production value 32
ax.axvline(int(peak.block_size), color='#2ca02c', ls='--', alpha=0.6,
           label=f'peak: block {int(peak.block_size)} ({peak.tok_per_s:.1f} tok/s)')
if 32 in set(df['block_size']):
    r32 = df[df.block_size == 32].iloc[0]
    ax.axvline(32, color='#d62728', ls=':', alpha=0.7,
               label=f"Inception's 32 ({r32.tok_per_s:.1f} tok/s, {100*r32.tok_per_s/peak.tok_per_s:.0f}% of peak)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RES, 'phase2_throughput.png'), dpi=140)

# ---- peak memory vs block size ----------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(df['block_size'], df['peak_mem_gb'], 's-', color='#9467bd', lw=2, ms=7)
ax2.set_xscale('log', base=2)
ax2.set_xticks(df['block_size'])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_xlabel('block size (log scale)')
ax2.set_ylabel('peak GPU memory (GB)')
ax2.set_title(f'BD3-LM peak generation memory vs block size\n(seqlen 1024, batch 1, {gpu})')
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(RES, 'phase2_memory.png'), dpi=140)

print(f"\nwrote: phase2_efficiency.csv, phase2_throughput.png, phase2_memory.png in {RES}")
