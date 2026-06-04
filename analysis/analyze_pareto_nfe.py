#!/usr/bin/env python
# =============================================================================
# analyze_pareto_nfe.py  —  NfePareto Phase 3: the unified quality-vs-throughput
# Pareto over the full (block size, NFE) operating grid.
#
# Joins:
#   results/nfe_genppl.csv       (gen-PPL, entropy, NFE per block,T,first_hitting)
#   results/nfe_throughput.jsonl (tok/s, peak-mem, NFE per req_block,req_T,req_fh)
# on (block, T, first_hitting).
#
# Outputs:
#   results/nfe_pareto.csv                merged tidy table (+ on_frontier flag)
#   results/pareto_genppl_throughput.png  gen-PPL (↓) vs throughput (↑), colored by
#                                         block, annotated with NFE; non-dominated
#                                         frontier highlighted; turbo/quality corners.
#
# Frontier = non-dominated for (minimize gen-PPL, maximize tok/s). gen-PPL is the
# repo's gpt2-large metric; every plotted point also passed the entropy>4 guard at
# generation (degenerate samples were auto-rejected), so low gen-PPL == real quality.
#
# Stdlib csv/json + matplotlib (Agg). Usage: python analysis/analyze_pareto_nfe.py
# =============================================================================
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
COLORS = {4: '#1f77b4', 8: '#2ca02c', 16: '#d62728', 32: '#9467bd'}


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def load_genppl():
    out = {}
    with open(os.path.join(RES, 'nfe_genppl.csv'), newline='') as f:
        for r in csv.DictReader(f):
            key = (int(r['block']), int(r['T']), r['first_hitting'])
            out[key] = {
                'genppl': fnum(r['genppl_agg']) or fnum(r['genppl_mean']),
                'entropy': fnum(r['entropy_agg']) or fnum(r['entropy_mean']),
                'nfe': fnum(r['nfe_mean']),
            }
    return out


def load_throughput():
    out = {}
    path = os.path.join(RES, 'nfe_throughput.jsonl')
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        key = (int(d['req_block']), int(d['req_T']), str(d['req_fh']))
        out[key] = {'tok_s': fnum(d.get('tok_per_s')), 'nfe_bench': fnum(d.get('nfe')),
                    'peak_mem_gb': fnum(d.get('peak_mem_gb'))}
    return out


def pareto_front(points):
    """points: list of dicts with 'genppl' (min) and 'tok_s' (max). Returns set of ids."""
    front = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if j == i:
                continue
            # q dominates p if q is >= on tok_s and <= on genppl, strictly better on one
            if (q['tok_s'] >= p['tok_s'] and q['genppl'] <= p['genppl']
                    and (q['tok_s'] > p['tok_s'] or q['genppl'] < p['genppl'])):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return set(front)


def main():
    gp = load_genppl()
    th = load_throughput()
    rows = []
    for key, g in gp.items():
        block, T, fh = key
        t = th.get(key, {})
        if g['genppl'] is None or t.get('tok_s') is None:
            continue
        rows.append({'block': block, 'T': T, 'fh': fh, 'nfe': g['nfe'],
                     'genppl': g['genppl'], 'entropy': g['entropy'],
                     'tok_s': t['tok_s'], 'peak_mem_gb': t.get('peak_mem_gb')})
    if not rows:
        print('no joined points yet (throughput job may still be running)')
        return

    front = pareto_front(rows)
    for i, r in enumerate(rows):
        r['on_frontier'] = i in front

    # write merged table
    rows.sort(key=lambda r: (r['block'], r['nfe'] or 0))
    out_csv = os.path.join(RES, 'nfe_pareto.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['block', 'T', 'fh', 'nfe', 'genppl', 'entropy',
                                          'tok_s', 'peak_mem_gb', 'on_frontier'])
        w.writeheader()
        w.writerows(rows)

    # plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for b in sorted({r['block'] for r in rows}):
        c = COLORS.get(b)
        pts = sorted([r for r in rows if r['block'] == b], key=lambda r: r['tok_s'])
        ax.plot([p['tok_s'] for p in pts], [p['genppl'] for p in pts], '-', color=c, alpha=0.35)
        ax.scatter([p['tok_s'] for p in pts], [p['genppl'] for p in pts], color=c, s=45,
                   label=f'block {b}', zorder=3)
        for p in pts:
            ax.annotate(f"{int(p['nfe'])}", (p['tok_s'], p['genppl']), fontsize=6,
                        xytext=(3, 3), textcoords='offset points', color=c)
    # frontier
    fr = sorted([r for r in rows if r['on_frontier']], key=lambda r: r['tok_s'])
    ax.plot([p['tok_s'] for p in fr], [p['genppl'] for p in fr], 'k--', lw=1.5,
            label='Pareto frontier', zorder=2)
    ax.scatter([p['tok_s'] for p in fr], [p['genppl'] for p in fr], s=140,
               facecolors='none', edgecolors='k', linewidths=1.5, zorder=4)

    ax.set_yscale('log')
    ax.set_xlabel('Throughput (tok/s, ↑ faster)')
    ax.set_ylabel('Generative PPL (gpt2-large, ↓ better)')
    ax.set_title('Unified inference operating map: quality vs throughput over (block size, NFE)\n'
                 'BD3-LM, OWT, len 1024 — point labels = measured NFE')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(RES, 'pareto_genppl_throughput.png')
    fig.savefig(out_png, dpi=130)

    # report
    cols = ['block', 'T', 'fh', 'nfe', 'genppl', 'tok_s', 'on_frontier']
    print('  '.join(c.rjust(10) for c in cols))
    for r in rows:
        print('  '.join(str(round(r[c], 2) if isinstance(r[c], float) else r[c]).rjust(10) for c in cols))
    print(f'\nfrontier points: {len(fr)}  | wrote {out_csv} + {out_png}')


if __name__ == '__main__':
    main()
