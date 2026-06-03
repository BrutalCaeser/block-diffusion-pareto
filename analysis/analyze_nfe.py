#!/usr/bin/env python
# =============================================================================
# analyze_nfe.py  —  NfePareto: the gen-PPL <-> NFE "speed dial" curves.
#
# Input:  results/nfe_genppl.csv  (written by analysis/parse_genppl.py)
#         columns: block, T, first_hitting, seed, length, n_batches,
#                  genppl_agg, genppl_mean, genppl_std, entropy_agg,
#                  entropy_mean, nfe_mean, nfe_min, nfe_max, genlen_mean
#
# Outputs:
#   results/nfe_quality_curve.png   2 panels: gen-PPL vs NFE (log-y) + entropy vs NFE
#                                   (the diversity guard), one series per block size.
#                                   fh=true points drawn as the "full-NFE" anchor (star);
#                                   paper Table-7 gen-PPL marked as a dashed line.
#   results/nfe_vs_t.png            measured NFE vs algo.T (documents the dial mechanics:
#                                   fh=false NFE~=(seqlen/block)*T, capped; fh=true~=seqlen).
#
# gen-PPL is NEVER shown without the entropy guard (FlowLM lesson): a point is only
# "good quality" if gen-PPL is low AND entropy stays healthy (>~4, the repo's reject floor).
#
# Stdlib csv + matplotlib (Agg); no pandas. Usage: python analysis/analyze_nfe.py
# =============================================================================
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
ENTROPY_FLOOR = 4.0  # repo's _check_stop_conds rejects samples below this (degenerate)

# Paper Table 7 (OWT, L=1024, GPT-2-large judge): gen-PPL at full NFE (~1K = seqlen).
PAPER_GENPPL = {4: 25.7, 8: 30.4, 16: 33.4}
COLORS = {4: '#1f77b4', 8: '#2ca02c', 16: '#d62728', 32: '#9467bd', 64: '#8c564b', 128: '#e377c2'}


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def load():
    path = os.path.join(RES, 'nfe_genppl.csv')
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append({
                'block': int(r['block']),
                'T': int(r['T']),
                'fh': r['first_hitting'] == 'true',
                'nfe': fnum(r['nfe_mean']),
                'genppl': fnum(r['genppl_agg']) or fnum(r['genppl_mean']),
                'genppl_std': fnum(r['genppl_std']),
                'entropy': fnum(r['entropy_agg']) or fnum(r['entropy_mean']),
            })
    return [r for r in rows if r['nfe'] and r['genppl']]


def blocks(rows):
    return sorted({r['block'] for r in rows})


def quality_curve(rows):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for b in blocks(rows):
        c = COLORS.get(b, None)
        dial = sorted([r for r in rows if r['block'] == b and not r['fh']], key=lambda r: r['nfe'])
        anch = sorted([r for r in rows if r['block'] == b and r['fh']], key=lambda r: r['nfe'])
        if dial:
            xs = [r['nfe'] for r in dial]
            ax1.plot(xs, [r['genppl'] for r in dial], '-o', color=c, label=f'block {b} (first_hitting=false dial)')
            ax2.plot(xs, [r['entropy'] for r in dial], '-o', color=c, label=f'block {b}')
        for r in anch:  # full-NFE anchor (first_hitting=true)
            ax1.plot(r['nfe'], r['genppl'], '*', color=c, ms=16, mec='k',
                     label=f'block {b} (first_hitting=true, full NFE)')
            ax2.plot(r['nfe'], r['entropy'], '*', color=c, ms=16, mec='k')
        if b in PAPER_GENPPL:
            ax1.axhline(PAPER_GENPPL[b], ls='--', lw=1, color=c, alpha=0.6)
            ax1.text(ax1.get_xlim()[1], PAPER_GENPPL[b], f' paper {PAPER_GENPPL[b]}',
                     color=c, va='center', fontsize=8)

    ax1.set_yscale('log')
    ax1.set_ylabel('Generative PPL  (gpt2-large, ↓ better)')
    ax1.set_title('Quality vs denoising budget (NFE) — the "speed dial"\nBD3-LM, OWT, len 1024, nucleus 0.9')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=7, loc='upper right')

    ax2.axhline(ENTROPY_FLOOR, ls=':', color='k', alpha=0.7)
    ax2.text(ax2.get_xlim()[0], ENTROPY_FLOOR, ' degeneracy floor (auto-reject < 4)',
             fontsize=8, va='bottom')
    ax2.set_ylabel('Unigram entropy (diversity guard, ↑ healthier)')
    ax2.set_xlabel('NFE  (number of denoising function evaluations, measured)')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RES, 'nfe_quality_curve.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


def nfe_vs_t(rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    for b in blocks(rows):
        c = COLORS.get(b, None)
        dial = sorted([r for r in rows if r['block'] == b and not r['fh']], key=lambda r: r['T'])
        if dial:
            ax.plot([r['T'] for r in dial], [r['nfe'] for r in dial], '-o', color=c,
                    label=f'block {b} (first_hitting=false)')
        for r in [r for r in rows if r['block'] == b and r['fh']]:
            ax.axhline(r['nfe'], ls='--', lw=1, color=c, alpha=0.6)
            ax.text(ax.get_xlim()[1], r['nfe'], f' fh=true ≈ seqlen ({int(r["nfe"])})',
                    color=c, fontsize=8, va='center')
    ax.set_xscale('log')
    ax.set_xlabel('algo.T  (denoising steps per block)')
    ax.set_ylabel('NFE (measured sampling_steps)')
    ax.set_title('The NFE dial: first_hitting=false → NFE ≈ (seqlen/block)·T (capped),\n'
                 'first_hitting=true → NFE ≈ seqlen regardless of T')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, 'nfe_vs_t.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)


def main():
    rows = load()
    print(f'loaded {len(rows)} points across blocks {blocks(rows)}')
    quality_curve(rows)
    nfe_vs_t(rows)


if __name__ == '__main__':
    main()
