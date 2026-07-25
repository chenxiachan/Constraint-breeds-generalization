#!/usr/bin/env python3
"""
NeurIPS Rebuttal ExpD: Robustness to oscillator initialization.

Addresses AC priority #5 / Reviewer oBwj Q2:
  "What was the justification for the particular choice of initialization for
   the oscillator? Are the results robust to alternative initializations?"

The paper's encoder maps each (per-sample-normalized) feature value v to the
initial condition (x0, y0, z0) = (v, 0.2v, -v) of the 3D Duffing-style system.
Here we re-run the Exp1 cross-encoding paradigm (train delta = 2.0, transition)
under 5 alternative initialization schemes. The init variant applies to BOTH
train and test encodings (it is part of the encoder specification).

The ODE integrator is imported unchanged from the original codebase
(core/encoding.py: mixed_oscillator_transformer_vectorized); only the initial
condition is parameterized.
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import csv
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..', '..', '..')
EXP1_CODE_DIR = os.path.join(REPO_DIR, '0_Experiment 1', 'code')
sys.path.insert(0, EXP1_CODE_DIR)

from core.model_dup import SimpleSNN
from core.encoding import mixed_oscillator_transformer_vectorized

OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialization variants: v -> (x0, y0, z0)
INIT_VARIANTS = {
    'baseline_(v,0.2v,-v)':   lambda v: (v, 0.2 * v, -v),
    'x_only_(v,0,0)':         lambda v: (v, 0.0, 0.0),
    'sym_mix_(v,0.5v,-0.5v)': lambda v: (v, 0.5 * v, -0.5 * v),
    'sign_flip_(v,-0.2v,v)':  lambda v: (v, -0.2 * v, v),
    'scaled_(0.5v,0.1v,-0.5v)': lambda v: (0.5 * v, 0.1 * v, -0.5 * v),
}

# Encoding params identical to Exp1 (encoding_wrapper.py EncodingConfig)
ENC_PARAMS = dict(alpha=2.0, beta=0.1, gamma=0.1, omega=1.0, drive=0.0)
NUM_STEPS = 30
TMAX = 4.0


def encode_sample(sample_np, delta, init_fn):
    """Faithful re-implementation of mixed_oscillator_encode for ONE sample,
    with parameterized init. Per-sample normalization semantics match the
    original sample-by-sample encoding path (batch of 1 -> global max == sample
    max). ODE integration itself is the original numba function."""
    num_features = sample_np.shape[0]
    out = np.zeros((NUM_STEPS, num_features * 3))

    m = np.max(np.abs(sample_np))
    s = sample_np / m if m > 0 else sample_np

    for f in range(num_features):
        v = s[f]
        x0, y0, z0 = init_fn(v)
        traj = mixed_oscillator_transformer_vectorized(
            x0, y0, z0,
            alpha=ENC_PARAMS['alpha'], beta=ENC_PARAMS['beta'], delta=delta,
            gamma=ENC_PARAMS['gamma'], omega=ENC_PARAMS['omega'],
            drive=ENC_PARAMS['drive'], tmax=TMAX
        )
        for dim in range(3):
            out[:, f * 3 + dim] = np.interp(
                np.linspace(0, 1, NUM_STEPS),
                np.linspace(0, 1, traj.shape[0]),
                traj[:, dim]
            )
    return out


def encode_split(data, delta, init_fn, device):
    data_np = data.numpy()
    enc = np.stack([encode_sample(data_np[i], delta, init_fn)
                    for i in range(data_np.shape[0])])
    return torch.from_numpy(enc).float().to(device)


def train_snn(train_ds, val_ds, input_dim, n_classes, device,
              epochs=200, lr=1e-4, patience=10, batch_size=32, seed=0):
    torch.manual_seed(seed)
    model = SimpleSNN(input_dim, 32, n_classes, num_steps=NUM_STEPS, beta=0.95)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)

    best_val, wait = 0.0, 0
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                out, _ = model(bx)
                _, pred = out.max(1)
                total += by.size(0)
                correct += pred.eq(by).sum().item()
        val_acc = 100.0 * correct / total
        if val_acc > best_val:
            best_val, wait = val_acc, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model, best_val


def evaluate(model, ds, batch_size=64):
    loader = DataLoader(ds, batch_size, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in loader:
            out, _ = model(bx)
            _, pred = out.max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Device: {DEVICE}", flush=True)

    TRAIN_DELTA = 2.0
    TEST_DELTAS = [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.0, 2.5, 5.0, 7.0, 10.0]
    N_RUNS = args.runs
    EPOCHS = 200
    variants = dict(INIT_VARIANTS)

    if args.smoke:
        N_RUNS, EPOCHS = 1, 3
        TEST_DELTAS = [-1.5, 2.0, 10.0]
        variants = {k: variants[k] for k in list(variants)[:2]}
        print("SMOKE TEST MODE", flush=True)

    digits = load_digits()
    X_np = StandardScaler().fit_transform(digits.data)
    y_np = digits.target
    N_SAMPLES = X_np.shape[0]
    N_CLASSES = len(np.unique(y_np))
    INPUT_DIM = X_np.shape[1] * 3

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    results = {name: [] for name in variants}

    for run in range(1, N_RUNS + 1):
        print(f"\n===== RUN {run}/{N_RUNS} =====", flush=True)
        n_train = int(0.7 * N_SAMPLES)
        n_val = int(0.15 * N_SAMPLES)
        n_test = N_SAMPLES - n_train - n_val
        tr, va, te = random_split(
            list(zip(X, y)), [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42 + run))
        train_X = torch.stack([a for a, _ in tr]); train_y = torch.stack([b for _, b in tr])
        val_X = torch.stack([a for a, _ in va]);   val_y = torch.stack([b for _, b in va])
        test_X = torch.stack([a for a, _ in te]);  test_y = torch.stack([b for _, b in te])

        for name, init_fn in variants.items():
            print(f"  [{name}] encoding...", flush=True)
            tr_enc = encode_split(train_X, TRAIN_DELTA, init_fn, DEVICE)
            va_enc = encode_split(val_X, TRAIN_DELTA, init_fn, DEVICE)
            train_ds = TensorDataset(tr_enc, train_y.to(DEVICE))
            val_ds = TensorDataset(va_enc, val_y.to(DEVICE))

            model, best_val = train_snn(train_ds, val_ds, INPUT_DIM, N_CLASSES,
                                        DEVICE, epochs=EPOCHS, seed=1000 + run)

            cross = {}
            for d in TEST_DELTAS:
                te_enc = encode_split(test_X, d, init_fn, DEVICE)
                cross[d] = evaluate(model, TensorDataset(te_enc, test_y.to(DEVICE)))

            id_acc = cross[TRAIN_DELTA]
            mean_ood = float(np.mean(list(cross.values())))
            results[name].append({
                'run': run, 'best_val': best_val, 'id_acc': id_acc,
                'mean_ood': mean_ood,
                'cross': {str(k): v for k, v in cross.items()},
            })
            print(f"    BestVal={best_val:.1f} ID={id_acc:.1f} MeanOOD={mean_ood:.1f}",
                  flush=True)

        # checkpoint
        with open(os.path.join(OUTPUT_DIR, 'init_robustness_detailed.json'), 'w') as f:
            json.dump(results, f, indent=1, default=float)

    rows = []
    for name, runs in results.items():
        if not runs:
            continue
        oods = [r['mean_ood'] for r in runs]
        ids_ = [r['id_acc'] for r in runs]
        rows.append({
            'init_variant': name, 'n_runs': len(runs),
            'id_acc_mean': round(float(np.mean(ids_)), 2),
            'id_acc_std': round(float(np.std(ids_)), 2),
            'ood_acc_mean': round(float(np.mean(oods)), 2),
            'ood_acc_std': round(float(np.std(oods)), 2),
        })
    with open(os.path.join(OUTPUT_DIR, 'init_robustness_summary.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
