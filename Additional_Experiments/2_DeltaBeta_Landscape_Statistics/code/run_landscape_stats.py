#!/usr/bin/env python3
"""
NeurIPS Rebuttal ExpE: Statistical analysis of the delta-beta joint landscape.

Data: ICML rebuttal Rebuttal_Joint_landscape/output/delta_beta_landscape.json
      (5 delta x 5 beta x 5 runs = 125 per-run records; no re-running needed)

Two questions:

Q1 (Reviewer oBwj): "poor performance at very low beta might reflect SNN
    firing-rate or optimization artifacts."
    If the collapse at low beta were an intrinsic artifact of the SNN
    (firing rate / gradient flow depend only on beta), OOD accuracy at a
    given beta would be INDEPENDENT of the input encoding delta.
    Test: at each low beta, compare OOD across encoding deltas
    (Mann-Whitney U, expansive vs dissipative encodings) + a delta x beta
    permutation interaction test. A significant delta-dependence at fixed
    beta refutes the beta-intrinsic artifact hypothesis.

Q2 (Reviewer 3zEC): significance of the main landscape contrasts
    (transition-vs-expansive at functional beta, etc.)
"""

import json
import os
import itertools
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, '..', 'output', 'delta_beta_landscape.json')
OUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH) as f:
    records = json.load(f)

# Normalize record structure
if isinstance(records, dict):
    # possibly {'results': [...]} or keyed structure
    for key in ('results', 'records', 'runs', 'data'):
        if key in records and isinstance(records[key], list):
            records = records[key]
            break

print(f"Loaded {len(records)} records")
print(f"Fields: {sorted(records[0].keys())}")

deltas = sorted({r['delta'] for r in records})
betas = sorted({r['beta'] for r in records})
print(f"deltas: {deltas}")
print(f"betas:  {betas}")


def ood(delta, beta):
    vals = [r['mean_ood'] for r in records
            if r['delta'] == delta and r['beta'] == beta]
    return np.array(vals, dtype=float)


lines = []


def emit(s=""):
    print(s)
    lines.append(s)


emit("=" * 78)
emit("Cell means (mean OOD accuracy % +- std, n per cell shown)")
emit("=" * 78)
header = "delta \\ beta |" + "".join(f" {b:>12}" for b in betas)
emit(header)
for d in deltas:
    row = f"{d:>11} |"
    for b in betas:
        v = ood(d, b)
        row += f" {v.mean():5.1f}±{v.std():4.1f}({len(v)})"
    emit(row)

# ---------------------------------------------------------------
# Q1a: at each low beta, does OOD depend on encoding delta?
# ---------------------------------------------------------------
emit("")
emit("=" * 78)
emit("Q1a: delta-dependence at fixed low beta (refutes beta-intrinsic artifact)")
emit("     Mann-Whitney U (two-sided) + Kruskal-Wallis across all deltas")
emit("=" * 78)

expansive_d = min(deltas)               # -1.5
dissipative_ds = [d for d in deltas if d >= 2.0]

for b in betas:
    groups = [ood(d, b) for d in deltas]
    H, p_kw = stats.kruskal(*groups)
    exp_v = ood(expansive_d, b)
    diss_v = np.concatenate([ood(d, b) for d in dissipative_ds])
    U, p_mw = stats.mannwhitneyu(exp_v, diss_v, alternative='two-sided')
    emit(f"beta={b}: Kruskal-Wallis H={H:6.2f} p={p_kw:.4f} | "
         f"expansive(d={expansive_d}) {exp_v.mean():5.1f}±{exp_v.std():4.1f} vs "
         f"dissipative(d>=2) {diss_v.mean():5.1f}±{diss_v.std():4.1f} : "
         f"U={U:.0f} p={p_mw:.4f}")

# ---------------------------------------------------------------
# Q1b: permutation test for delta x beta interaction on OOD
# ---------------------------------------------------------------
emit("")
emit("=" * 78)
emit("Q1b: permutation interaction test (delta x beta)")
emit("     H0 (artifact hypothesis): OOD = f(beta) + g(delta) + noise, no interaction")
emit("=" * 78)

y = np.array([r['mean_ood'] for r in records], dtype=float)
d_idx = np.array([deltas.index(r['delta']) for r in records])
b_idx = np.array([betas.index(r['beta']) for r in records])


def interaction_stat(yv):
    """Residual-based interaction F-like statistic: variance explained by cell
    means beyond additive (row+col) model."""
    grand = yv.mean()
    row_m = np.array([yv[d_idx == i].mean() for i in range(len(deltas))])
    col_m = np.array([yv[b_idx == j].mean() for j in range(len(betas))])
    additive = row_m[d_idx] + col_m[b_idx] - grand
    cell_m = np.zeros_like(yv)
    for i in range(len(deltas)):
        for j in range(len(betas)):
            m = (d_idx == i) & (b_idx == j)
            cell_m[m] = yv[m].mean()
    ss_inter = ((cell_m - additive) ** 2).sum()
    ss_resid = ((yv - cell_m) ** 2).sum()
    return ss_inter / (ss_resid + 1e-12)


obs = interaction_stat(y)
rng = np.random.default_rng(0)
n_perm = 10000
count = 0
# Permute residuals from the additive model (Freedman-Lane style)
grand = y.mean()
row_m = np.array([y[d_idx == i].mean() for i in range(len(deltas))])
col_m = np.array([y[b_idx == j].mean() for j in range(len(betas))])
additive_fit = row_m[d_idx] + col_m[b_idx] - grand
resid = y - additive_fit
for _ in range(n_perm):
    y_perm = additive_fit + rng.permutation(resid)
    if interaction_stat(y_perm) >= obs:
        count += 1
p_inter = (count + 1) / (n_perm + 1)
emit(f"Observed interaction statistic: {obs:.3f}")
emit(f"Permutation p-value ({n_perm} perms): p={p_inter:.5f}")
emit("=> significant interaction means the low-beta collapse is CONDITIONAL on")
emit("   encoding delta, inconsistent with a beta-intrinsic firing-rate artifact.")

# ---------------------------------------------------------------
# Q2: key planned contrasts with Mann-Whitney + Cliff's delta
# ---------------------------------------------------------------
emit("")
emit("=" * 78)
emit("Q2: planned contrasts (Mann-Whitney U two-sided + Cliff's delta effect size)")
emit("=" * 78)


def cliffs_delta(a, b):
    gt = sum(x > y_ for x in a for y_ in b)
    lt = sum(x < y_ for x in a for y_ in b)
    return (gt - lt) / (len(a) * len(b))


func_betas = [b for b in betas if b >= 0.9]
low_betas = [b for b in betas if b <= 0.5]

contrasts = []
# transition vs expansive at functional beta
tran_d = [d for d in deltas if 0.0 <= d <= 2.5]
a = np.concatenate([ood(d, b) for d in tran_d for b in func_betas])
b_ = np.concatenate([ood(expansive_d, b) for b in func_betas])
contrasts.append(("transition(0<=d<=2.5) vs expansive(d=-1.5), beta>=0.9", a, b_))

# dissipative encodings: functional vs low beta
a = np.concatenate([ood(d, b) for d in dissipative_ds for b in func_betas])
b_ = np.concatenate([ood(d, b) for d in dissipative_ds for b in low_betas])
contrasts.append(("dissipative enc (d>=2): beta>=0.9 vs beta<=0.5", a, b_))

# expansive encoding: functional vs low beta
a = np.concatenate([ood(expansive_d, b) for b in func_betas])
b_ = np.concatenate([ood(expansive_d, b) for b in low_betas])
contrasts.append(("expansive enc (d=-1.5): beta>=0.9 vs beta<=0.5", a, b_))

for name, a, b_ in contrasts:
    U, p = stats.mannwhitneyu(a, b_, alternative='two-sided')
    cd = cliffs_delta(a, b_)
    emit(f"{name}")
    emit(f"    {a.mean():5.1f}±{a.std():4.1f} (n={len(a)}) vs "
         f"{b_.mean():5.1f}±{b_.std():4.1f} (n={len(b_)}) | "
         f"U={U:.0f} p={p:.2e} | Cliff's delta={cd:+.3f}")

with open(os.path.join(OUT_DIR, 'landscape_stats.txt'), 'w') as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved: {os.path.join(OUT_DIR, 'landscape_stats.txt')}")
