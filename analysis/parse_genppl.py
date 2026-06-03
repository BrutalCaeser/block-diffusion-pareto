#!/usr/bin/env python
# =============================================================================
# parse_genppl.py  —  NfePareto: aggregate the sample_eval runs into one table.
#
# Reads, from a directory of runs produced by exp/nfe_genppl.sbatch:
#   genppl_b{BLOCK}_T{T}_fh{FH}_s{SEED}_len{LEN}.csv   (per-sample-batch rows:
#        gen_ppl, gen_nfes, gen_entropy, gen_lengths, samples, seed)
#   genppl_b..._len....log                              (stdout; corpus-level
#        "Generative perplexity: tensor(..)" + "Entropy: tensor(..)")
#
# Emits results/nfe_genppl.csv with one row per run:
#   block, T, first_hitting, seed, length, n_batches,
#   genppl_agg (corpus-level, from .log — the paper-comparable number),
#   genppl_mean, genppl_std (over batches), entropy_agg, entropy_mean,
#   nfe_mean, nfe_min, nfe_max, genlen_mean
#
# Stdlib only (runs locally after results are copied back). No re-scoring — these
# are the repo's own validated gen-PPL / entropy numbers; we only aggregate.
#
# Usage:
#   python analysis/parse_genppl.py [--runs results/nfe] [--out results/nfe_genppl.csv]
# =============================================================================
import argparse
import csv
import glob
import os
import re
import statistics as st

TAG_RE = re.compile(
    r'genppl_b(?P<block>\d+)_T(?P<T>\d+)_fh(?P<fh>true|false)_s(?P<seed>\d+)_len(?P<len>\d+)')
# tolerant float grab after a label, optionally wrapped in tensor(...)
PPL_RE = re.compile(r'Generative perplexity:\s*(?:tensor\()?\s*([0-9.eE+-]+)')
ENT_RE = re.compile(r'Entropy:\s*(?:tensor\()?\s*([0-9.eE+-]+)')


def _floats(rows, key):
    out = []
    for r in rows:
        v = (r.get(key) or '').strip()
        if v == '' or v.lower() == 'nan':
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def _mean(xs):
    return round(st.mean(xs), 4) if xs else ''


def _std(xs):
    return round(st.pstdev(xs), 4) if len(xs) > 1 else 0.0 if xs else ''


def parse_log(log_path):
    agg_ppl = agg_ent = ''
    if os.path.exists(log_path):
        with open(log_path, errors='ignore') as f:
            txt = f.read()
        m = PPL_RE.search(txt)
        if m:
            agg_ppl = round(float(m.group(1)), 4)
        m = ENT_RE.search(txt)
        if m:
            agg_ent = round(float(m.group(1)), 4)
    return agg_ppl, agg_ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='results/nfe')
    ap.add_argument('--out', default='results/nfe_genppl.csv')
    args = ap.parse_args()

    rows_out = []
    for csv_path in sorted(glob.glob(os.path.join(args.runs, 'genppl_*.csv'))):
        base = os.path.basename(csv_path)
        m = TAG_RE.match(base)
        if not m:
            print(f'skip (bad tag): {base}')
            continue
        meta = m.groupdict()
        with open(csv_path, newline='', errors='ignore') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f'skip (empty): {base}')
            continue
        ppls = _floats(rows, 'gen_ppl')
        nfes = _floats(rows, 'gen_nfes')
        ents = _floats(rows, 'gen_entropy')
        lens = _floats(rows, 'gen_lengths')
        agg_ppl, agg_ent = parse_log(csv_path[:-4] + '.log')
        rows_out.append({
            'block': int(meta['block']),
            'T': int(meta['T']),
            'first_hitting': meta['fh'],
            'seed': int(meta['seed']),
            'length': int(meta['len']),
            'n_batches': len(rows),
            'genppl_agg': agg_ppl,             # corpus-level, paper-comparable
            'genppl_mean': _mean(ppls),        # over batches
            'genppl_std': _std(ppls),
            'entropy_agg': agg_ent,            # mean per-sample unigram entropy (diversity guard)
            'entropy_mean': _mean(ents),
            'nfe_mean': _mean(nfes),
            'nfe_min': int(min(nfes)) if nfes else '',
            'nfe_max': int(max(nfes)) if nfes else '',
            'genlen_mean': _mean(lens),
        })

    if not rows_out:
        print(f'no runs found under {args.runs}')
        return
    rows_out.sort(key=lambda r: (r['block'], r['first_hitting'], r['T'], r['seed']))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fields = list(rows_out[0].keys())
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    # pretty print
    cols = ['block', 'T', 'first_hitting', 'seed', 'nfe_mean', 'genppl_agg',
            'genppl_mean', 'genppl_std', 'entropy_agg', 'genlen_mean', 'n_batches']
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows_out)) for c in cols}
    line = '  '.join(c.ljust(widths[c]) for c in cols)
    print(line)
    print('-' * len(line))
    for r in rows_out:
        print('  '.join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f'\nwrote {args.out}  ({len(rows_out)} runs)')


if __name__ == '__main__':
    main()
