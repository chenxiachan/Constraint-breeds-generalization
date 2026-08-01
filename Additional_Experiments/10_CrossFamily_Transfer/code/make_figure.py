#!/usr/bin/env python3
"""Cross-family transfer matrix figure + per-cell breakdown CSV.

Panel A: 4 train conditions x 3 test families (aggregate, mean±std annotated).
Panel B: 4 train conditions x 22 test cells (per-cell means, grouped by family).
In-family cells outlined with dashed border; the fully out-of-family
Lorenz->Duffing row highlighted. Sequential single-hue colormap (magnitude).
Max side <= 2000 px (dpi=150, figsize <= 13 in).
"""

import json
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, '..', 'output')

d = json.load(open(os.path.join(OUT, 'crossfamily_detailed.json')))
recs = d['records']

CONDS = ['Duffing_transition', 'Lorenz_dissipative', 'ExpDecay', 'Gaussian']
COND_LABELS = ['Duffing transition\n(train)', 'Lorenz dissipative\n(train)',
               'Exponential decay\n(train)', 'Gaussian smoothing\n(train)']
FAMILIES = ['Duffing', 'ExpDecay', 'Lorenz']
FAM_LABELS = ['Duffing test family', 'Exp-decay test family', 'Lorenz test family']

# in-family (train condition, test family) pairs
IN_FAMILY = {('Duffing_transition', 'Duffing'), ('Lorenz_dissipative', 'Lorenz'),
             ('ExpDecay', 'ExpDecay')}

# ---- aggregate & per-cell stats ----
cell_runs = {}
for r in recs:
    for fam, sub in r['test_accs'].items():
        for p, acc in sub.items():
            cell_runs.setdefault((r['condition'], fam, p), []).append(acc)

fam_params = {f: sorted({p for (c, ff, p) in cell_runs if ff == f}, key=float)
              for f in FAMILIES}

agg_mean = np.zeros((len(CONDS), len(FAMILIES)))
agg_std = np.zeros_like(agg_mean)
for i, c in enumerate(CONDS):
    for j, f in enumerate(FAMILIES):
        vals = [np.mean(cell_runs[(c, f, p)]) for p in fam_params[f]]
        per_run = []
        for run_vals in zip(*[cell_runs[(c, f, p)] for p in fam_params[f]]):
            per_run.append(np.mean(run_vals))
        agg_mean[i, j] = np.mean(per_run)
        agg_std[i, j] = np.std(per_run)

# ---- per-cell CSV ----
csv_path = os.path.join(OUT, 'per_cell_breakdown.csv')
with open(csv_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['train_condition', 'test_family', 'test_param',
                'acc_mean', 'acc_std', 'n_runs'])
    for c in CONDS:
        for f in FAMILIES:
            for p in fam_params[f]:
                v = cell_runs[(c, f, p)]
                w.writerow([c, f, p, round(float(np.mean(v)), 2),
                            round(float(np.std(v)), 2), len(v)])
print('CSV saved:', csv_path)

# ---- figure ----
fig = plt.figure(figsize=(12.8, 6.2), dpi=150)
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.52)
CMAP = plt.get_cmap('Blues')
VMIN, VMAX = 0, 100
INK, INK_LIGHT = '#1a2733', '#ffffff'


def ink_for(v):
    return INK_LIGHT if v > 62 else INK


# Panel A: aggregate
axA = fig.add_subplot(gs[0])
axA.imshow(agg_mean, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect='auto')
for i in range(len(CONDS)):
    for j in range(len(FAMILIES)):
        axA.text(j, i, f'{agg_mean[i, j]:.1f} ± {agg_std[i, j]:.1f}',
                 ha='center', va='center', fontsize=11,
                 color=ink_for(agg_mean[i, j]))
        if (CONDS[i], FAMILIES[j]) in IN_FAMILY:
            axA.add_patch(plt.Rectangle((j - .46, i - .42), .92, .84, fill=False,
                                        ls=(0, (4, 3)), lw=1.6, ec='#5b6673',
                                        zorder=5))
# killer cell: Lorenz -> Duffing
axA.add_patch(plt.Rectangle((-0.46, 0.58), .92, .84, fill=False, lw=2.4,
                            ec='#c2410c', zorder=6))
axA.set_xticks(range(len(FAMILIES)), FAM_LABELS, fontsize=10)
axA.set_yticks(range(len(CONDS)), COND_LABELS, fontsize=9.5)
axA.set_title('A    Cross-family transfer, mean OOD accuracy % (10 runs; '
              'dashed = in-family; orange = fully out-of-family Lorenz'
              r'$\rightarrow$Duffing)', fontsize=11, loc='left', color=INK)
axA.tick_params(length=0)
for s in axA.spines.values():
    s.set_visible(False)

# white cell gaps
axA.set_xticks(np.arange(-.5, len(FAMILIES)), minor=True)
axA.set_yticks(np.arange(-.5, len(CONDS)), minor=True)
axA.grid(which='minor', color='white', linewidth=2)
axA.tick_params(which='minor', length=0)

# Panel B: per-cell
all_cols, col_labels, fam_bounds = [], [], []
for f in FAMILIES:
    fam_bounds.append(len(all_cols))
    for p in fam_params[f]:
        all_cols.append((f, p))
        col_labels.append(p.rstrip('0').rstrip('.') if '.' in p else p)
fam_bounds.append(len(all_cols))

M = np.zeros((len(CONDS), len(all_cols)))
for i, c in enumerate(CONDS):
    for j, (f, p) in enumerate(all_cols):
        M[i, j] = np.mean(cell_runs[(c, f, p)])

axB = fig.add_subplot(gs[1])
axB.imshow(M, cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect='auto')
for i in range(len(CONDS)):
    for j, (f, p) in enumerate(all_cols):
        axB.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center',
                 fontsize=8, color=ink_for(M[i, j]))
        if (CONDS[i], f) in IN_FAMILY:
            axB.add_patch(plt.Rectangle((j - .44, i - .42), .88, .84, fill=False,
                                        ls=(0, (3, 2)), lw=1.1, ec='#5b6673',
                                        zorder=5))
for b in fam_bounds[1:-1]:
    axB.axvline(b - .5, color=INK, lw=1.6)
axB.set_xticks(range(len(all_cols)), col_labels, fontsize=7.5)
axB.set_yticks(range(len(CONDS)),
               [c.replace('_', ' ') for c in CONDS], fontsize=9.5)
mids = [(fam_bounds[k] + fam_bounds[k + 1]) / 2 - .5 for k in range(3)]
for m, lab, sub in zip(mids, FAM_LABELS,
                       [r'test $\delta$', 'decay rate', r'test $\rho$']):
    axB.text(m, -0.95, f'{lab}  ({sub})', ha='center', fontsize=9, color=INK)
axB.set_title('B    Per-cell mean accuracy % (chance = 10%)',
              fontsize=11, loc='left', color=INK, pad=26)
axB.tick_params(length=0)
for s in axB.spines.values():
    s.set_visible(False)
axB.set_xticks(np.arange(-.5, len(all_cols)), minor=True)
axB.set_yticks(np.arange(-.5, len(CONDS)), minor=True)
axB.grid(which='minor', color='white', linewidth=1.4)
axB.tick_params(which='minor', length=0)

cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP,
                  norm=plt.Normalize(VMIN, VMAX)),
                  ax=[axA, axB], fraction=0.025, pad=0.015)
cb.set_label('accuracy %', fontsize=9)
cb.outline.set_visible(False)

fig_path = os.path.join(OUT, 'crossfamily_matrix.png')
fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
print('Figure saved:', fig_path)
