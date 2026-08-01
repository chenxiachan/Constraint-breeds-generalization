#!/usr/bin/env python3
"""
NeurIPS Phase 2 ExpH: Cross-Encoding-Family Generalization Matrix

Addresses Reviewer oBwj's "similarity" objection:
  "Exponential decay training transfers well to Duffing test encodings maybe just
   because exp-decay signals RESEMBLE Duffing dissipative trajectories (train-test
   distribution similarity), not because dissipative structure per se matters."

Design: train x test both cross-family.
  Train conditions (10 runs each):
    1. Duffing_transition  : Duffing delta=2.0 (paper's optimal regime)
    2. Lorenz_dissipative  : Lorenz rho=1.0 (stable/dissipative regime, sigma=10, beta=8/3)
    3. ExpDecay            : 3-channel exponential decay, delta=2.0
    4. Gaussian            : static input + Gaussian-smoothed noise (smoothing control)
  Test families (every trained model evaluated on ALL cells):
    - Duffing grid : delta in [-1.5,-1.0,-0.3,0.0,0.3,1.0,1.5,2.0,2.5,5.0,7.0,10.0]
    - ExpDecay grid: delta in [0.5,1.0,2.0,5.0,10.0]
    - Lorenz grid  : rho in [0.5,1.0,1.5,10.0,28.0]  (rho<1.5 stable/dissipative,
                     rho=10 spiral convergence, rho=28 chaotic; paper Appendix Fig 3
                     scanned rhos=[0.1,0.5,1.0,1.2,1.5,3.0,5.0,28.0])

Killer cell: Lorenz-trained -> Duffing test. Training distribution has no relation
to the Duffing family; if it still transfers while Gaussian fails everywhere, the
similarity explanation is dead.

Paradigm faithful to Additional_Experiments/7_Dissipation_vs_Smoothing/
run_lowpass_control.py: sklearn digits, 70/15/15 split (seed 42+run),
sample-by-sample encoding, SimpleSNN(192,32,10,num_steps=30,beta=0.95),
early stopping patience=10, lr=1e-4, batch=32.

Efficiency note: per run, each test-cell encoding of the test set is computed ONCE
and shared across all trained models (identical tensors; pure caching, no paradigm
change vs. the template which re-encoded per model).

Crash safety: detailed JSON is rewritten after EVERY (run, condition) completes.
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')  # before core.encoding imports matplotlib

import sys
import argparse
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime

# ============================================================
# Path setup: import from the Experiment 1 codebase
# (same pattern as ExpA_MatchedFiltering/code/run_matched_filtering.py)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..', '..', '..')
EXP1_CODE_DIR = os.path.join(REPO_DIR, '0_Experiment 1', 'code')
sys.path.insert(0, EXP1_CODE_DIR)

from core.model_dup import SimpleSNN
from core.encoding import mixed_oscillator_encode, lorenz_encode

OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Encoders
# Duffing / ExpDecay / Gaussian verbatim from run_lowpass_control.py (7_Dissipation_vs_Smoothing).
# Lorenz wraps the paper's own lorenz_encode (0_Experiment 1/code/core/encoding.py):
#   signature lorenz_encode(data, num_steps, tmax=2, sigma=10, beta=8/3, rho=28),
#   init (v, 0.2v, -v), RK4 h=0.01, output (B, T, F*3) -- same shape as Duffing.
#   Called sample-by-sample => per-sample normalization, matching the paradigm.
# ============================================================

class DuffingEncoder:
    """Full nonlinear Duffing encoder (original from paper)."""

    def __init__(self, num_steps=30, tmax=4.0):
        self.num_steps = num_steps
        self.tmax = tmax

    def encode(self, data, delta, device='cpu'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        params = {
            'alpha': 2.0, 'beta': 0.1, 'delta': delta,
            'gamma': 0.1, 'omega': 1.0, 'drive': 0.0
        }
        encoded = mixed_oscillator_encode(
            data.cpu(), num_steps=self.num_steps, tmax=self.tmax, params=params
        )
        return encoded.detach().clone().to(device).float()


class LorenzEncoder:
    """Lorenz attractor encoder using the paper's lorenz_encode (numba RK4).
    Regime knob is rho: rho<1.5 stable/dissipative, rho=28 chaotic."""

    def __init__(self, num_steps=30, tmax=2.0, sigma=10.0, beta=8.0 / 3.0):
        self.num_steps = num_steps
        self.tmax = tmax
        self.sigma = sigma
        self.beta = beta

    def encode(self, data, rho, device='cpu'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        encoded = lorenz_encode(
            data.cpu(), self.num_steps, tmax=self.tmax,
            sigma=self.sigma, beta=self.beta, rho=rho
        )
        return encoded.detach().clone().to(device).float()


class ExponentialDecayEncoder:
    """3-channel exponential decay from initial conditions matching Duffing."""

    def __init__(self, num_steps=30, tmax=4.0):
        self.num_steps = num_steps
        self.tmax = tmax

    def encode(self, data, delta=2.0, device='cpu'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        data_np = data.cpu().numpy()
        batch_size, num_features = data_np.shape

        encoded = np.zeros((batch_size, self.num_steps, num_features * 3))
        t = np.linspace(0, self.tmax, self.num_steps)

        for b_idx in range(batch_size):
            sample = data_np[b_idx]
            sample_max = np.max(np.abs(sample))
            if sample_max > 0:
                sample = sample / sample_max

            taus = [1.0 / max(delta, 0.1),
                    1.0 / max(delta * 0.5, 0.1),
                    1.0 / max(delta * 2.0, 0.1)]

            for f_idx in range(num_features):
                x_i = sample[f_idx]
                inits = [x_i, 0.2 * x_i, -x_i]  # Same init as Duffing
                for ch, (init_val, tau) in enumerate(zip(inits, taus)):
                    encoded[b_idx, :, f_idx * 3 + ch] = init_val * np.exp(-t / tau)

        return torch.from_numpy(encoded).float().to(device)


class GaussianSmoothedEncoder:
    """Static input + Gaussian-smoothed temporal noise (generic smoothing control)."""

    def __init__(self, num_steps=30, tmax=4.0, sigma_t=3.0, noise_scale=0.3):
        self.num_steps = num_steps
        self.tmax = tmax
        self.sigma_t = sigma_t
        self.noise_scale = noise_scale

    def encode(self, data, delta=None, device='cpu'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        data_np = data.cpu().numpy()
        batch_size, num_features = data_np.shape
        encoded = np.zeros((batch_size, self.num_steps, num_features * 3))

        kernel_size = int(6 * self.sigma_t) + 1
        kernel_x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (kernel_x / self.sigma_t) ** 2)
        kernel = kernel / kernel.sum()

        for b_idx in range(batch_size):
            sample = data_np[b_idx]
            sample_max = np.max(np.abs(sample))
            if sample_max > 0:
                sample = sample / sample_max

            for f_idx in range(num_features):
                x_i = sample[f_idx]
                inits = [x_i, 0.2 * x_i, -x_i]

                for ch, init_val in enumerate(inits):
                    seed = abs(int(x_i * 1e6 + f_idx * 1000 + ch * 100 + b_idx)) % (2 ** 31)
                    rng = np.random.RandomState(seed)

                    noise = rng.randn(self.num_steps + kernel_size) * self.noise_scale * (abs(init_val) + 1e-8)
                    smoothed = np.convolve(noise, kernel, mode='valid')[:self.num_steps]
                    encoded[b_idx, :, f_idx * 3 + ch] = init_val + smoothed

        return torch.from_numpy(encoded).float().to(device)


# ============================================================
# Trainer (faithful to original Exp1 / run_lowpass_control.py)
# ============================================================

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_acc': [], 'val_acc': []}

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            output, _ = self.model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        return total_loss / len(train_loader), 100.0 * correct / total

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output, _ = self.model(batch_x)
                total_loss += criterion(output, batch_y).item()
                _, predicted = output.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        return total_loss / len(val_loader), 100.0 * correct / total

    def fit(self, train_loader, val_loader, epochs, lr=1e-4, patience=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0
        patience_counter = 0

        pbar = tqdm(range(epochs), desc='Training SNN', leave=False)
        for epoch in pbar:
            _, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            _, val_acc = self.validate(val_loader, criterion)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            pbar.set_postfix({'train': f'{train_acc:.1f}%', 'val': f'{val_acc:.1f}%'})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        return best_val_acc


def evaluate_on_encoded(model, encoded_data, labels, device):
    """Accuracy of a trained model on a pre-encoded test tensor."""
    model.eval()
    loader = DataLoader(TensorDataset(encoded_data, labels), batch_size=64, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out, _ = model(bx)
            _, pred = out.max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
    return 100.0 * correct / total


def encode_dataset_sample_by_sample(encoder, raw_data, param, device, desc="Encoding"):
    """Encode each sample independently (per-sample normalization, as in Exp1)."""
    encoded_list = []
    for i in tqdm(range(raw_data.shape[0]), desc=desc, leave=False):
        sample = raw_data[i].unsqueeze(0)
        enc = encoder.encode(sample, param, device='cpu')
        encoded_list.append(enc.squeeze(0))
    return torch.stack(encoded_list)


# ============================================================
# Result persistence
# ============================================================

def save_detailed(path, config, records, status):
    payload = {
        'experiment': 'ExpH_CrossFamily_Transfer',
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'records': records,
    }
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def write_summary(csv_path, records, train_names, families):
    """Aggregate train_condition x test_family: per-run family mean -> mean +- std."""
    rows = []
    for cond in train_names:
        cond_recs = [r for r in records if r['condition'] == cond]
        for fam in families:
            per_run_means, per_run_means_excl_id = [], []
            for r in cond_recs:
                cells = r['test_accs'].get(fam, {})
                if not cells:
                    continue
                vals = list(cells.values())
                per_run_means.append(float(np.mean(vals)))
                vals_excl = [v for k, v in cells.items()
                             if not (fam == r['id_family'] and k == r['id_param'])]
                if vals_excl:
                    per_run_means_excl_id.append(float(np.mean(vals_excl)))
            if not per_run_means:
                continue
            rows.append({
                'train_condition': cond,
                'test_family': fam,
                'n_runs': len(per_run_means),
                'n_cells': len(cond_recs[0]['test_accs'].get(fam, {})),
                'acc_mean': round(float(np.mean(per_run_means)), 2),
                'acc_std': round(float(np.std(per_run_means)), 2),
                'acc_mean_excl_id': round(float(np.mean(per_run_means_excl_id)), 2)
                if per_run_means_excl_id else None,
                'acc_std_excl_id': round(float(np.std(per_run_means_excl_id)), 2)
                if per_run_means_excl_id else None,
            })
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def print_matrix(summary_rows, train_names, families):
    print("\n" + "=" * 78)
    print("Cross-family generalization matrix (family-mean accuracy, mean +- std over runs)")
    print("=" * 78)
    header = f"{'train \\ test':24s}" + "".join(f"{fam:>18s}" for fam in families)
    print(header)
    print("-" * len(header))
    lookup = {(r['train_condition'], r['test_family']): r for r in summary_rows}
    for cond in train_names:
        cells = []
        for fam in families:
            r = lookup.get((cond, fam))
            cells.append(f"{r['acc_mean']:5.1f}+-{r['acc_std']:4.1f}" if r else "     --   ")
        print(f"{cond:24s}" + "".join(f"{c:>18s}" for c in cells))
    key = lookup.get(('Lorenz_dissipative', 'Duffing'))
    if key:
        print(f"\nKiller cell  Lorenz_dissipative -> Duffing family: "
              f"{key['acc_mean']:.1f} +- {key['acc_std']:.1f} % "
              f"(fully out-of-family; compare Gaussian row)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='ExpH cross-family transfer matrix')
    parser.add_argument('--smoke', action='store_true',
                        help='1 run, 2 conditions, 2 test points/family, 3 epochs')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Device: {DEVICE}")

    # --- Config (identical to original Exp1 / run_lowpass_control.py) ---
    NUM_STEPS = 30
    HIDDEN_DIM = 32
    BATCH_SIZE = 32
    LR = 1e-4
    PATIENCE = 10
    SNN_BETA = 0.95

    N_RUNS = 1 if args.smoke else args.runs
    EPOCHS = 3 if args.smoke else args.epochs

    duffing_enc = DuffingEncoder(NUM_STEPS, tmax=4.0)
    lorenz_enc = LorenzEncoder(NUM_STEPS, tmax=2.0)   # repo default tmax for lorenz_encode
    expdecay_enc = ExponentialDecayEncoder(NUM_STEPS, tmax=4.0)
    gauss_enc = GaussianSmoothedEncoder(NUM_STEPS, tmax=4.0)

    # --- Test grids ---
    if args.smoke:
        DUFFING_DELTAS = [2.0, 10.0]
        EXPDECAY_DELTAS = [2.0, 10.0]
        LORENZ_RHOS = [1.0, 28.0]
    else:
        DUFFING_DELTAS = [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.0, 2.5, 5.0, 7.0, 10.0]
        EXPDECAY_DELTAS = [0.5, 1.0, 2.0, 5.0, 10.0]
        LORENZ_RHOS = [0.5, 1.0, 1.5, 10.0, 28.0]

    FAMILIES = ['Duffing', 'ExpDecay', 'Lorenz']
    # (family, param, encoder); Gaussian own-encoding cell added separately for ID acc
    TEST_GRID = ([('Duffing', d, duffing_enc) for d in DUFFING_DELTAS]
                 + [('ExpDecay', d, expdecay_enc) for d in EXPDECAY_DELTAS]
                 + [('Lorenz', r, lorenz_enc) for r in LORENZ_RHOS])

    # --- Training conditions: (name, encoder, param, id_family, id_param) ---
    ALL_CONDITIONS = [
        ('Duffing_transition', duffing_enc, 2.0, 'Duffing', '2.0'),
        ('Lorenz_dissipative', lorenz_enc, 1.0, 'Lorenz', '1.0'),
        ('ExpDecay', expdecay_enc, 2.0, 'ExpDecay', '2.0'),
        ('Gaussian', gauss_enc, None, 'Gaussian', 'own'),
    ]
    if args.smoke:
        TRAIN_CONDITIONS = [c for c in ALL_CONDITIONS
                            if c[0] in ('Duffing_transition', 'Gaussian')]
    else:
        TRAIN_CONDITIONS = ALL_CONDITIONS
    train_names = [c[0] for c in TRAIN_CONDITIONS]

    suffix = '_smoke' if args.smoke else ''
    detailed_path = os.path.join(OUTPUT_DIR, f'crossfamily_detailed{suffix}.json')
    summary_path = os.path.join(OUTPUT_DIR, f'crossfamily_summary{suffix}.csv')

    config = {
        'n_runs': N_RUNS, 'epochs': EPOCHS, 'patience': PATIENCE, 'lr': LR,
        'batch_size': BATCH_SIZE, 'num_steps': NUM_STEPS, 'hidden_dim': HIDDEN_DIM,
        'snn_beta': SNN_BETA, 'smoke': args.smoke,
        'duffing': {'alpha': 2.0, 'beta': 0.1, 'gamma': 0.1, 'omega': 1.0,
                    'drive': 0.0, 'tmax': 4.0, 'test_deltas': DUFFING_DELTAS},
        'lorenz': {'sigma': 10.0, 'beta': 8.0 / 3.0, 'tmax': 2.0, 'h': 0.01,
                   'init': '(v, 0.2v, -v)', 'train_rho': 1.0, 'test_rhos': LORENZ_RHOS},
        'expdecay': {'train_delta': 2.0, 'test_deltas': EXPDECAY_DELTAS},
        'gaussian': {'sigma_t': 3.0, 'noise_scale': 0.3},
        'train_conditions': train_names,
        'device': str(DEVICE),
    }

    # --- Data (identical to original Exp1) ---
    print("Loading sklearn digits...")
    digits = load_digits()
    X_np = StandardScaler().fit_transform(digits.data)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    N_SAMPLES, N_FEATURES = X.shape
    N_CLASSES = len(torch.unique(y))
    INPUT_DIM = N_FEATURES * 3  # 192
    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features, {N_CLASSES} classes")
    print(f"Conditions: {train_names}")
    print(f"Test cells/run: {len(TEST_GRID)} (+1 Gaussian ID cell)")

    records = []
    t_start = time.time()

    for run in range(1, N_RUNS + 1):
        print(f"\n{'=' * 25} RUN {run} / {N_RUNS} {'=' * 25}")
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)

        # Split identical to original Exp1
        n_train = int(0.7 * N_SAMPLES)
        n_val = int(0.15 * N_SAMPLES)
        n_test = N_SAMPLES - n_train - n_val
        train_data, val_data, test_data = random_split(
            list(zip(X, y)), [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42 + run)
        )
        train_X = torch.stack([a for a, _ in train_data])
        train_y = torch.stack([b for _, b in train_data])
        val_X = torch.stack([a for a, _ in val_data])
        val_y = torch.stack([b for _, b in val_data])
        test_X = torch.stack([a for a, _ in test_data])
        test_y = torch.stack([b for _, b in test_data])
        print(f"Split: train={n_train}, val={n_val}, test={n_test}")

        # --- Cache all test-cell encodings once per run (shared by all models) ---
        t0 = time.time()
        test_cache = {}
        for fam, param, enc in TEST_GRID:
            key = (fam, str(param))
            test_cache[key] = encode_dataset_sample_by_sample(
                enc, test_X, param, DEVICE, desc=f"Test enc {fam} {param}")
        test_cache[('Gaussian', 'own')] = encode_dataset_sample_by_sample(
            gauss_enc, test_X, None, DEVICE, desc="Test enc Gaussian own")
        test_enc_time = time.time() - t0
        print(f"Test encodings cached: {len(test_cache)} cells in {test_enc_time:.1f}s")

        for name, enc, param, id_family, id_param in TRAIN_CONDITIONS:
            print(f"\n--- {name} (run {run}) ---")
            t0 = time.time()
            train_enc = encode_dataset_sample_by_sample(
                enc, train_X, param, DEVICE, desc="Train enc")
            val_enc = encode_dataset_sample_by_sample(
                enc, val_X, param, DEVICE, desc="Val enc")
            enc_time = time.time() - t0

            train_loader = DataLoader(TensorDataset(train_enc, train_y),
                                      BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_enc, val_y),
                                    BATCH_SIZE, shuffle=False)

            t0 = time.time()
            model = SimpleSNN(INPUT_DIM, HIDDEN_DIM, N_CLASSES,
                              num_steps=NUM_STEPS, beta=SNN_BETA)
            trainer = Trainer(model, DEVICE)
            best_val = trainer.fit(train_loader, val_loader, EPOCHS,
                                   lr=LR, patience=PATIENCE)
            train_time = time.time() - t0

            # --- Evaluate on all cached test cells ---
            t0 = time.time()
            test_accs = {fam: {} for fam in FAMILIES}
            for (fam, param_str), enc_data in test_cache.items():
                acc = evaluate_on_encoded(model, enc_data, test_y, DEVICE)
                if fam in test_accs:
                    test_accs[fam][param_str] = acc
                elif fam == 'Gaussian':
                    test_accs.setdefault('Gaussian', {})[param_str] = acc
            eval_time = time.time() - t0

            id_acc = test_accs.get(id_family, {}).get(id_param, float('nan'))

            record = {
                'run': run,
                'condition': name,
                'train_param': param,
                'id_family': id_family,
                'id_param': id_param,
                'best_val_acc': best_val,
                'id_acc': id_acc,
                'epochs_trained': len(trainer.history['val_acc']),
                'encode_time_s': round(enc_time, 1),
                'train_time_s': round(train_time, 1),
                'eval_time_s': round(eval_time, 1),
                'test_enc_time_s': round(test_enc_time, 1),
                'test_accs': test_accs,
            }
            records.append(record)

            fam_means = {f: np.mean(list(v.values()))
                         for f, v in test_accs.items() if f in FAMILIES and v}
            print(f"  BestVal={best_val:.1f}%  ID={id_acc:.1f}%  "
                  + "  ".join(f"{f}={m:.1f}%" for f, m in fam_means.items())
                  + f"  [enc {enc_time:.0f}s, train {train_time:.0f}s, eval {eval_time:.0f}s]")

            # Crash safety: persist after every condition
            save_detailed(detailed_path, config, records, status='partial')

        # Per-run summary refresh
        write_summary(summary_path, records, train_names, FAMILIES)

    # --- Final aggregation ---
    save_detailed(detailed_path, config, records, status='complete')
    summary_rows = write_summary(summary_path, records, train_names, FAMILIES)
    print_matrix(summary_rows, train_names, FAMILIES)

    total_time = time.time() - t_start
    print(f"\nDetailed JSON: {detailed_path}")
    print(f"Summary CSV:   {summary_path}")
    print(f"Total wall time: {total_time / 60:.1f} min")


if __name__ == '__main__':
    main()
