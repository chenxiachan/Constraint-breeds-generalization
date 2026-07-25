#!/usr/bin/env python3
"""
NeurIPS Rebuttal ExpB stats: significance tests on the re-run RL per-run data.

Addresses AC priority #3 / Reviewer 3zEC Q3: "report statistical tests for
Exp 4 and 5 to show if the overlapping bars are different with statistical
significance."

Input: experiment_outputs/eval_summary_{ts}.csv  (per run x difficulty)
       experiment_outputs/gap_analysis_{ts}.csv  (per run, gaps & retention)
Output: output/rl_stats.md  (markdown tables ready for the rebuttal)
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_OUT = os.path.join(BASE_DIR, '..', 'output')
OUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def newest(pattern):
    files = sorted(glob.glob(os.path.join(EXP_OUT, pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


eval_df = pd.read_csv(newest('eval_summary_*.csv'))
gap_df = pd.read_csv(newest('gap_analysis_*.csv'))
print(f"eval_summary: {len(eval_df)} rows | columns: {list(eval_df.columns)}")
print(f"gap_analysis: {len(gap_df)} rows | columns: {list(gap_df.columns)}")
print(f"groups: {sorted(eval_df['group'].unique())}")
print(f"difficulties: {sorted(eval_df['difficulty'].unique())}")


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def compare(a, b):
    """Welch t + Mann-Whitney + Cliff's delta."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    try:
        U, p_u = stats.mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:      # identical values
        U, p_u = np.nan, 1.0
    return {
        'mean_a': a.mean(), 'std_a': a.std(), 'n_a': len(a),
        'mean_b': b.mean(), 'std_b': b.std(), 'n_b': len(b),
        'welch_t': t, 'p_welch': p_t, 'U': U, 'p_mw': p_u,
        'cliffs': cliffs_delta(a, b),
    }


lines = ["# RL significance tests (re-run, N=10 runs per condition)", ""]

# ------------------------------------------------------------------
# 1. Transfer reward per difficulty: key pairwise comparisons
# ------------------------------------------------------------------
KEY_PAIRS = [
    ('tran_snn', 'exp_snn'),
    ('tran_snn', 'diss_snn'),
    ('tran_snn', 'ann_raw'),
    ('tran_snn', 'ann_tran'),
    ('ann_tran', 'ann_raw'),
]
# tolerate alternative group naming
groups_present = set(eval_df['group'].unique())


def resolve(name):
    if name in groups_present:
        return name
    cands = [g for g in groups_present if name.split('_')[0] in g.lower()
             and name.split('_')[1] in g.lower()]
    return cands[0] if cands else None


lines.append("## Transfer reward by difficulty (mean_reward per run)")
lines.append("")
lines.append("| comparison | difficulty | A mean±std | B mean±std | Welch p | "
             "MW p | Cliff's δ |")
lines.append("|---|---|---|---|---|---|---|")

for ga, gb in KEY_PAIRS:
    ga_r, gb_r = resolve(ga), resolve(gb)
    if not ga_r or not gb_r:
        lines.append(f"| {ga} vs {gb} | - | MISSING GROUP | | | | |")
        continue
    for diff in sorted(eval_df['difficulty'].unique()):
        a = eval_df[(eval_df.group == ga_r) & (eval_df.difficulty == diff)
                    ]['mean_reward']
        b = eval_df[(eval_df.group == gb_r) & (eval_df.difficulty == diff)
                    ]['mean_reward']
        if len(a) == 0 or len(b) == 0:
            continue
        r = compare(a, b)
        lines.append(
            f"| {ga_r} vs {gb_r} | {diff} | "
            f"{r['mean_a']:.1f}±{r['std_a']:.1f} | "
            f"{r['mean_b']:.1f}±{r['std_b']:.1f} | "
            f"{r['p_welch']:.4g} | {r['p_mw']:.4g} | {r['cliffs']:+.2f} |")

# ------------------------------------------------------------------
# 2. Generalization gap / retention per run
# ------------------------------------------------------------------
lines.append("")
lines.append("## Generalization gap & retention (per-run)")
lines.append("")

metric_cols = [c for c in ('avg_gap', 'performance_retention') if c in gap_df.columns]
lines.append("| comparison | metric | A mean±std | B mean±std | Welch p | MW p | "
             "Cliff's δ |")
lines.append("|---|---|---|---|---|---|---|")
for ga, gb in KEY_PAIRS:
    ga_r, gb_r = resolve(ga), resolve(gb)
    if not ga_r or not gb_r:
        continue
    for m in metric_cols:
        a = gap_df[gap_df.group == ga_r][m]
        b = gap_df[gap_df.group == gb_r][m]
        if len(a) == 0 or len(b) == 0:
            continue
        r = compare(a, b)
        lines.append(
            f"| {ga_r} vs {gb_r} | {m} | "
            f"{r['mean_a']:.2f}±{r['std_a']:.2f} | "
            f"{r['mean_b']:.2f}±{r['std_b']:.2f} | "
            f"{r['p_welch']:.4g} | {r['p_mw']:.4g} | {r['cliffs']:+.2f} |")

# ------------------------------------------------------------------
# 3. Kruskal-Wallis across all SNN encoding groups per difficulty
# ------------------------------------------------------------------
lines.append("")
lines.append("## Omnibus test across SNN encoding groups")
lines.append("")
snn_groups = [g for g in groups_present if 'snn' in g.lower()]
for diff in sorted(eval_df['difficulty'].unique()):
    samples = [eval_df[(eval_df.group == g) & (eval_df.difficulty == diff)
                       ]['mean_reward'].values for g in snn_groups]
    samples = [s for s in samples if len(s) > 0]
    if len(samples) >= 2:
        H, p = stats.kruskal(*samples)
        lines.append(f"- {diff}: Kruskal-Wallis H={H:.2f}, p={p:.4g} "
                     f"({', '.join(snn_groups)})")

out_path = os.path.join(OUT_DIR, 'rl_stats.md')
with open(out_path, 'w') as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved: {out_path}")
print("\n".join(lines[:40]))
