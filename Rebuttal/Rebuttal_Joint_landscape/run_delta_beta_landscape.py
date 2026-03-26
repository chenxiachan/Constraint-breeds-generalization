#!/usr/bin/env python3
"""
Rebuttal Experiment 3: delta-beta 2D Joint Landscape

Key question (mLxH Q2):
  What if we simultaneously adjust delta (encoding-level dissipation) and beta
  (architecture-level dissipation) in Experiment 1? Does the "transition regime"
  remain valid for all betas, or for a particular beta?

Paradigm: Based on Exp1 (cross-encoding classification on sklearn digits)
  - Duffing encoding with variable delta
  - SimpleSNN with variable beta (membrane leak)
  - Same training protocol: early stopping (patience=10), Adam lr=1e-4
  - Same cross-encoding test on full delta range
  - Metric: Mean OOD accuracy across 12 test deltas

Grid: 5 delta x 5 beta x 5 runs
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import csv
import json
from datetime import datetime

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..', 'Constraint-breeds-generalization-main')
EXP1_CODE_DIR = os.path.join(REPO_DIR, '0_Experiment 1', 'code')
sys.path.insert(0, EXP1_CODE_DIR)

from core.model_dup import SimpleSNN
from core.encoding import mixed_oscillator_encode

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Encoder (same as Exp1)
# ============================================================

class DuffingEncoder:
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


# ============================================================
# Trainer (faithful to original Exp1)
# ============================================================

class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(self.device), by.to(self.device)
            optimizer.zero_grad()
            out, _ = self.model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = out.max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
        return total_loss / len(loader), 100.0 * correct / total

    def validate(self, loader, criterion):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out, _ = self.model(bx)
                loss = criterion(out, by)
                total_loss += loss.item()
                _, pred = out.max(1)
                total += by.size(0)
                correct += pred.eq(by).sum().item()
        return total_loss / len(loader), 100.0 * correct / total

    def fit(self, train_loader, val_loader, epochs=200, lr=1e-4, patience=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            self.train_epoch(train_loader, criterion, optimizer)
            _, val_acc = self.validate(val_loader, criterion)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_val_acc


# ============================================================
# Encoding & Testing (sample-by-sample, matches Exp1)
# ============================================================

def encode_dataset(encoder, raw_data, raw_labels, delta, device):
    """Sample-by-sample encoding (consistent with Exp1)."""
    encoded_list = []
    for i in range(raw_data.shape[0]):
        sample = raw_data[i].unsqueeze(0)
        enc = encoder.encode(sample, delta, device=device)
        encoded_list.append(enc.squeeze(0))
    return TensorDataset(torch.stack(encoded_list), raw_labels)


def cross_encoding_test(model, test_X, test_y, encoder, test_deltas, device):
    """Test on all Duffing delta values (sample-by-sample)."""
    model.eval()
    results = {}

    for delta in test_deltas:
        encoded_list = []
        with torch.no_grad():
            for i in range(test_X.shape[0]):
                sample = test_X[i].unsqueeze(0)
                enc = encoder.encode(sample, delta, device=device)
                encoded_list.append(enc.squeeze(0))

        encoded = torch.stack(encoded_list).to(device)
        loader = DataLoader(
            TensorDataset(encoded, test_y.to(device)), batch_size=64, shuffle=False
        )

        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader:
                out, _ = model(bx)
                _, pred = out.max(1)
                total += by.size(0)
                correct += pred.eq(by).sum().item()

        results[delta] = 100.0 * correct / total

    return results


# ============================================================
# Main
# ============================================================

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Device: {DEVICE}")

    # --- Config (identical to Exp1) ---
    NUM_STEPS = 30
    TMAX = 4.0
    HIDDEN_DIM = 32
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-4
    PATIENCE = 10

    # Grid
    DELTA_VALUES = [-1.5, 0.0, 2.0, 5.0, 10.0]
    BETA_VALUES = [0.3, 0.5, 0.7, 0.9, 0.95]
    N_RUNS = 5
    TEST_DELTAS = [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.0, 2.5, 5.0, 7.0, 10.0]

    total_conditions = len(DELTA_VALUES) * len(BETA_VALUES) * N_RUNS
    print(f"\nGrid: {len(DELTA_VALUES)} deltas x {len(BETA_VALUES)} betas x {N_RUNS} runs = {total_conditions}")
    print(f"Delta: {DELTA_VALUES}")
    print(f"Beta:  {BETA_VALUES}")

    # --- Load data (identical to Exp1) ---
    print("\nLoading Sklearn Digits dataset...")
    digits = load_digits()
    X_np = StandardScaler().fit_transform(digits.data)
    y_np = digits.target
    N_SAMPLES = X_np.shape[0]
    N_FEATURES = X_np.shape[1]   # 64
    N_CLASSES = len(np.unique(y_np))  # 10
    INPUT_DIM = N_FEATURES * 3   # 192

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features, {N_CLASSES} classes")

    encoder = DuffingEncoder(NUM_STEPS, TMAX)

    # --- Run grid ---
    all_results = []
    progress = tqdm(total=total_conditions, desc="Training")

    for run in range(1, N_RUNS + 1):
        # Data split (identical to Exp1)
        n_train = int(0.7 * N_SAMPLES)
        n_val = int(0.15 * N_SAMPLES)
        n_test = N_SAMPLES - n_train - n_val
        if (n_train + n_val + n_test) != N_SAMPLES:
            n_train = N_SAMPLES - n_val - n_test

        all_data = list(zip(X, y))
        train_data, val_data, test_data = random_split(
            all_data, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42 + run)
        )

        train_X = torch.stack([x for x, _ in train_data])
        train_y = torch.stack([yl for _, yl in train_data])
        val_X = torch.stack([x for x, _ in val_data])
        val_y = torch.stack([yl for _, yl in val_data])
        test_X = torch.stack([x for x, _ in test_data])
        test_y = torch.stack([yl for _, yl in test_data])

        for delta in DELTA_VALUES:
            # Encode train/val once per delta (shared across betas)
            train_dataset = encode_dataset(encoder, train_X, train_y, delta, DEVICE)
            val_dataset = encode_dataset(encoder, val_X, val_y, delta, DEVICE)

            train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

            for beta in BETA_VALUES:
                # Train SNN with this (delta, beta)
                model = SimpleSNN(INPUT_DIM, HIDDEN_DIM, N_CLASSES,
                                  num_steps=NUM_STEPS, beta=beta)
                trainer = Trainer(model, DEVICE)
                best_val = trainer.fit(train_loader, val_loader, EPOCHS, LR, PATIENCE)

                # Cross-encoding test
                cross_results = cross_encoding_test(
                    model, test_X, test_y, encoder, TEST_DELTAS, DEVICE
                )

                mean_ood = np.mean(list(cross_results.values()))
                id_acc = cross_results.get(delta, mean_ood)

                all_results.append({
                    'delta': delta,
                    'beta': beta,
                    'run': run,
                    'best_val': best_val,
                    'id_acc': id_acc,
                    'mean_ood': mean_ood,
                    'cross_results': cross_results,
                })

                progress.set_postfix({
                    'd': delta, 'b': beta, 'run': run,
                    'ood': f"{mean_ood:.1f}"
                })
                progress.update(1)

    progress.close()

    # --- Aggregate ---
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    grid_ood = {}
    grid_id = {}

    for delta in DELTA_VALUES:
        for beta in BETA_VALUES:
            runs = [r for r in all_results
                    if r['delta'] == delta and r['beta'] == beta]
            oods = [r['mean_ood'] for r in runs]
            ids = [r['id_acc'] for r in runs]
            grid_ood[(delta, beta)] = (np.mean(oods), np.std(oods))
            grid_id[(delta, beta)] = (np.mean(ids), np.std(ids))

    # Print OOD table
    header = f"{'d\\b':>8s}" + "".join(f"{'b='+str(b):>14s}" for b in BETA_VALUES)
    print(f"\nMean OOD Accuracy (%):")
    print(header)
    print("-" * (10 + 14 * len(BETA_VALUES)))
    for delta in DELTA_VALUES:
        row = f"{delta:8.1f}"
        for beta in BETA_VALUES:
            m, s = grid_ood[(delta, beta)]
            row += f"  {m:5.1f}+-{s:4.1f}"
        print(row)

    # --- Save ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(OUTPUT_DIR, f'delta_beta_landscape_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['delta', 'beta', 'ood_mean', 'ood_std', 'id_mean', 'id_std'])
        for delta in DELTA_VALUES:
            for beta in BETA_VALUES:
                om, os_ = grid_ood[(delta, beta)]
                im, is_ = grid_id[(delta, beta)]
                writer.writerow([delta, beta, f"{om:.2f}", f"{os_:.2f}",
                                 f"{im:.2f}", f"{is_:.2f}"])
    print(f"\nCSV saved: {csv_path}")

    json_path = os.path.join(OUTPUT_DIR, f'delta_beta_landscape_full_{timestamp}.json')
    json_data = []
    for r in all_results:
        json_data.append({
            'delta': r['delta'], 'beta': r['beta'], 'run': r['run'],
            'best_val': float(r['best_val']),
            'id_acc': float(r['id_acc']),
            'mean_ood': float(r['mean_ood']),
            'cross_results': {str(k): float(v) for k, v in r['cross_results'].items()},
        })
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved: {json_path}")

    # --- Figure ---
    try:
        generate_figure(grid_ood, grid_id, DELTA_VALUES, BETA_VALUES, timestamp)
    except Exception as e:
        print(f"Figure failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nExperiment complete.")


def generate_figure(grid_ood, grid_id, deltas, betas, timestamp):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    for ax, grid, title, cmap in [
        (axes[0], grid_ood, 'Mean OOD Accuracy (%)\n(higher = better generalization)', 'RdYlGn'),
        (axes[1], grid_id, 'ID Accuracy (%)\n(in-distribution performance)', 'RdYlGn'),
    ]:
        matrix = np.zeros((len(deltas), len(betas)))
        for i, d in enumerate(deltas):
            for j, b in enumerate(betas):
                matrix[i, j] = grid[(d, b)][0]

        im = ax.imshow(matrix, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"{b}" for b in betas])
        ax.set_yticks(range(len(deltas)))
        ax.set_yticklabels([f"{d}" for d in deltas])
        ax.set_xlabel(r'$\beta$ (membrane leak)')
        ax.set_ylabel(r'$\delta$ (encoding dissipation)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Accuracy (%)')

        for i in range(len(deltas)):
            for j in range(len(betas)):
                val = matrix[i, j]
                color = 'white' if val < matrix.mean() else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')

    plt.suptitle(r'$\delta$-$\beta$ Joint Landscape (Exp1 Classification)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, f'delta_beta_landscape_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
