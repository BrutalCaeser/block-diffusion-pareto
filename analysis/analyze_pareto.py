#!/usr/bin/env python
# =============================================================================
# analyze_pareto.py  —  Phase 4: combine efficiency (Phase 2) + quality (Phase 3)
# into the block-size Pareto picture.
#
# Inputs:
#   results/phase2_efficiency.csv   (production HF path; has tok_per_s, peak_mem_gb)
#   results/phase3_quality.csv      (matched fixed-budget val NLL/PPL per block size)
#
# Outputs:
#   results/phase3_quality.png      val PPL vs block size (monotonic ↑ with block)
#   results/pareto_quality_speed.png  quality (val NLL, ↓ better) vs throughput,
#                                     with the non-dominated frontier highlighted.
#
# Honest scope: Phase 3 PPL is "achieved under a matched cheap from-scratch budget"
# (2500 steps, wikitext103, len 256), NOT converged quality. The block-size ORDERING
# is the claim; absolute PPL is high by construction.
#
# Usage:  python analysis/analyze_pareto.py
# =============================================================================
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

eff = pd.read_csv(os.path.join(RES, 'phase2_efficiency.csv'))
eff = eff[eff['series'].str.startswith('production')].copy()  # HF production path
qual = pd.read_csv(os.path.join(RES, 'phase3_quality.csv'))

# ---- quality vs block size ---------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(qual['block_size'], qual['val_ppl'], 'o-', color='#d62728', lw=2, ms=7)
ax.set_xscale('log', base=2)
ax.set_xticks(qual['block_size'])
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlabel('block size (log scale)')
ax.set_ylabel('validation PPL (lower = better)')
ax.set_title('BD3-LM quality vs block size\n(wikitext103, matched 2500-step from-scratch budget, len 256)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RES, 'phase3_quality.png'), dpi=140)

# ---- Pareto: quality (val NLL, lower better) vs throughput (higher better) ----
m = pd.merge(eff[['block_size', 'tok_per_s', 'peak_mem_gb']], qual[['block_size', 'val_nll', 'val_ppl']],
             on='block_size', how='inner').sort_values('block_size').reset_index(drop=True)

# non-dominated set: maximize tok_per_s, minimize val_nll
pts = m.sort_values('tok_per_s', ascending=False).reset_index(drop=True)
frontier, best_nll = [], float('inf')
for _, r in pts.iterrows():
    if r['val_nll'] < best_nll:
        frontier.append(int(r['block_size']))
        best_nll = r['val_nll']
m['pareto'] = m['block_size'].isin(frontier)

fig2, ax2 = plt.subplots(figsize=(7.5, 5))
ax2.plot(m['tok_per_s'], m['val_nll'], '-', color='#bbbbbb', lw=1, zorder=1)
ax2.scatter(m['tok_per_s'], m['val_nll'], s=70, color='#1f77b4', zorder=3)
front = m[m['pareto']]
ax2.scatter(front['tok_per_s'], front['val_nll'], s=160, facecolors='none',
            edgecolors='#2ca02c', linewidths=2.2, zorder=4, label='Pareto frontier')
for _, r in m.iterrows():
    ax2.annotate(f"  block {int(r.block_size)}", (r.tok_per_s, r.val_nll), fontsize=9,
                 va='center')
# mark 32
r32 = m[m.block_size == 32].iloc[0]
ax2.scatter([r32.tok_per_s], [r32.val_nll], s=240, marker='*', color='#d62728', zorder=5,
            label="block 32 (Inception)")
ax2.set_xlabel('throughput (tokens / s)  →  faster')
ax2.set_ylabel('validation NLL  →  worse (up)')
ax2.invert_yaxis()  # better quality (lower NLL) at top
ax2.set_title('Block-size Pareto: quality vs generation throughput\n'
              '(throughput=prod HF path, seqlen1024; quality=matched budget, len256)')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='lower left')
fig2.tight_layout()
fig2.savefig(os.path.join(RES, 'pareto_quality_speed.png'), dpi=140)

print(m.to_string(index=False))
print('\nPareto frontier (non-dominated):', frontier)
print('wrote: phase3_quality.png, pareto_quality_speed.png')
