#!/usr/bin/env python3
"""
NeurIPS Rebuttal ExpA: Matched Temporal Filtering Controls

Addresses AC priority #1 (Reviewer oBwj):
  "Because LIF dynamics implement low-pass filtering, please separate this smoothing
   effect from spiking/recurrence/temporal pooling - e.g., compare against MLP/RNN/LSTM
   baselines with matched temporal filtering - and justify why SNNs are necessary."

Design: all architectures receive IDENTICAL Duffing-encoded input (B, T, 192) and
identical training protocol (faithful to original 0_main_Fig1.py / ICML
run_lowpass_control.py paradigm). The architecture ladder decomposes the LIF neuron:

  1. SNN            : leaky integration + spike nonlinearity + reset   (reference)
  2. Leaky-MLP      : leaky integration + ReLU, NO spike/reset.
                      h_t = beta*h_{t-1} + W x_t  (exact snnTorch Leaky update,
                      beta matched = 0.95), readout = sum_t h3_t (non-spiking
                      readout membrane). Isolates {spiking, reset}.
  3. EMA-MLP        : input prefiltered by the SAME exponential filter
                      y_t = beta*y_{t-1} + x_t, then AvgPool-MLP.
                      "MLP with matched temporal filtering" applied at the INPUT.
                      Isolates input spectral smoothing from model-internal state.
  4. AvgPool-MLP    : uniform temporal averaging (boxcar filter), no exp filter.
  5. LastT-MLP      : no temporal integration at all.

RNN/LSTM baselines are not rerun here: full cross-encoding data already exists in
the paper (Fig. 1 d,e; Appendix Tables 6-9).

Per-run results are saved to JSON for significance testing (Reviewer 3zEC).
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
from tqdm import tqdm
import csv
import json
from datetime import datetime

# ============================================================
# Path setup: import from existing Experiment 1 codebase
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..', '..', '..')
EXP1_CODE_DIR = os.path.join(REPO_DIR, '0_Experiment 1', 'code')
sys.path.insert(0, EXP1_CODE_DIR)

from core.model_dup import SimpleSNN, SimpleANN, TemporalANN_Avg
from core.encoding import mixed_oscillator_encode

OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Duffing encoder (verbatim from ICML run_lowpass_control.py,
# which itself mirrors original DynamicEncoder / EncodingConfig)
# ============================================================

class DuffingEncoder:
    """Full nonlinear Duffing encoder (original from paper). alpha=2.0, beta=0.1
    matches Exp1's encoding_wrapper.py EncodingConfig."""

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
# New matched-filtering architectures
# ============================================================

class LeakyMLP(nn.Module):
    """
    LIF minus {spiking, reset}: identical 3-layer topology to SimpleSNN,
    identical beta, identical time-summed readout.

    Per layer, exact snnTorch Leaky membrane update WITHOUT spike/reset:
        h_t = beta * h_{t-1} + cur_t
    Hidden layers apply ReLU to the leaky state (nonlinearity in the same
    position as the spike threshold); the output layer is a non-spiking
    readout membrane summed over time (standard non-spiking readout control
    in the SNN literature).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_steps=30, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, num_steps, features)
        B = x.shape[0]
        dev = x.device
        h1 = torch.zeros(B, self.fc1.out_features, device=dev)
        h2 = torch.zeros(B, self.fc2.out_features, device=dev)
        h3 = torch.zeros(B, self.fc3.out_features, device=dev)
        out_sum = torch.zeros(B, self.fc3.out_features, device=dev)

        for t in range(self.num_steps):
            x_t = x[:, t, :]
            h1 = self.beta * h1 + self.fc1(x_t)
            a1 = self.relu(h1)
            h2 = self.beta * h2 + self.fc2(a1)
            a2 = self.relu(h2)
            h3 = self.beta * h3 + self.fc3(a2)
            out_sum = out_sum + h3

        return out_sum


class EMAPrefilterAvgMLP(nn.Module):
    """
    "MLP with matched temporal filtering" applied at the INPUT:
    the input sequence is passed through the same first-order exponential
    filter as the LIF membrane (y_t = beta*y_{t-1} + x_t, y_0 = 0, matching
    snnTorch init_leaky zeros), then averaged over time and fed to the
    standard 3-layer MLP. Tests whether input-side spectral smoothing alone
    reproduces SNN generalization.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_steps=30, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, num_steps, features)
        B, T, F = x.shape
        y = torch.zeros(B, F, device=x.device)
        y_sum = torch.zeros(B, F, device=x.device)
        for t in range(T):
            y = self.beta * y + x[:, t, :]
            y_sum = y_sum + y
        y_avg = y_sum / T

        z = self.relu(self.fc1(y_avg))
        z = self.relu(self.fc2(z))
        return self.fc3(z)


# ============================================================
# Generic trainer (mirrors ICML lowpass Trainer; handles both
# SNN-style (out, spikes) and plain-out models)
# ============================================================

def model_forward(model, x):
    out = model(x)
    if isinstance(out, tuple):
        return out[0]
    return out


class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def run_epoch(self, loader, criterion, optimizer=None):
        training = optimizer is not None
        self.model.train() if training else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                if training:
                    optimizer.zero_grad()
                out = model_forward(self.model, bx)
                loss = criterion(out, by)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                _, pred = out.max(1)
                total += by.size(0)
                correct += pred.eq(by).sum().item()

        return total_loss / len(loader), 100.0 * correct / total

    def fit(self, train_loader, val_loader, epochs, lr=1e-4, patience=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc, patience_counter = 0.0, 0

        for epoch in range(epochs):
            self.run_epoch(train_loader, criterion, optimizer)
            _, val_acc = self.run_epoch(val_loader, criterion)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_val_acc


def evaluate(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model_forward(model, bx)
            _, pred = out.max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
    return 100.0 * correct / total


# ============================================================
# Sample-by-sample encoding with caching (per-sample normalization
# semantics identical to original Exp1)
# ============================================================

def encode_split(encoder, data, delta, device):
    encoded_list = []
    for i in tqdm(range(data.shape[0]), desc=f"Encoding d={delta}", leave=False):
        sample = data[i].unsqueeze(0)
        enc = encoder.encode(sample, delta, device=device)
        encoded_list.append(enc.squeeze(0))
    return torch.stack(encoded_list)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='1 run, 1 delta, 3 epochs')
    parser.add_argument('--start-run', type=int, default=1,
                        help='resume from this run (seeds are deterministic per run)')
    parser.add_argument('--end-run', type=int, default=None)
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Device: {DEVICE}", flush=True)

    # --- Config (identical to original Exp1 / ICML lowpass control) ---
    NUM_STEPS = 30
    TMAX = 4.0
    HIDDEN_DIM = 32
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-4
    PATIENCE = 10
    BETA = 0.95           # matched across SNN / LeakyMLP / EMA prefilter
    N_RUNS = 10
    TRAIN_DELTAS = [-1.5, 0.0, 2.0, 10.0]
    TEST_DELTAS = [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.0, 2.5, 5.0, 7.0, 10.0]

    if args.smoke:
        N_RUNS, TRAIN_DELTAS, EPOCHS = 1, [2.0], 3
        TEST_DELTAS = [-1.5, 2.0, 10.0]
        print("SMOKE TEST MODE", flush=True)

    MODEL_SPECS = ['SNN', 'LeakyMLP', 'EMA_MLP', 'AvgPool_MLP', 'LastT_MLP']

    def build_model(name, input_dim, n_classes):
        if name == 'SNN':
            return SimpleSNN(input_dim, HIDDEN_DIM, n_classes,
                             num_steps=NUM_STEPS, beta=BETA)
        if name == 'LeakyMLP':
            return LeakyMLP(input_dim, HIDDEN_DIM, n_classes,
                            num_steps=NUM_STEPS, beta=BETA)
        if name == 'EMA_MLP':
            return EMAPrefilterAvgMLP(input_dim, HIDDEN_DIM, n_classes,
                                      num_steps=NUM_STEPS, beta=BETA)
        if name == 'AvgPool_MLP':
            return TemporalANN_Avg(input_dim, HIDDEN_DIM, n_classes)
        if name == 'LastT_MLP':
            return SimpleANN(input_dim, HIDDEN_DIM, n_classes)
        raise ValueError(name)

    # --- Data (identical to original Exp1) ---
    digits = load_digits()
    scaler = StandardScaler()
    X_np = scaler.fit_transform(digits.data)
    y_np = digits.target
    N_SAMPLES, N_FEATURES = X_np.shape
    N_CLASSES = len(np.unique(y_np))
    INPUT_DIM = N_FEATURES * 3   # 192

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} feats, {N_CLASSES} classes",
          flush=True)

    encoder = DuffingEncoder(NUM_STEPS, TMAX)

    # results[model][train_delta] = list over runs of
    #   {'cross': {test_delta: acc}, 'id_acc': .., 'best_val': ..}
    results = {m: {d: [] for d in TRAIN_DELTAS} for m in MODEL_SPECS}

    START_RUN = max(1, args.start_run)
    END_RUN = args.end_run if args.end_run else N_RUNS
    global RESULT_SUFFIX
    RESULT_SUFFIX = f'_runs{START_RUN}-{END_RUN}' if START_RUN > 1 else ''

    for run in range(START_RUN, END_RUN + 1):
        print(f"\n{'=' * 25} RUN {run}/{N_RUNS} {'=' * 25}", flush=True)

        # Split identical to original Exp1 (seed 42+run)
        n_train = int(0.7 * N_SAMPLES)
        n_val = int(0.15 * N_SAMPLES)
        n_test = N_SAMPLES - n_train - n_val
        all_data_list = list(zip(X, y))
        train_data, val_data, test_data = random_split(
            all_data_list, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42 + run)
        )
        train_X = torch.stack([a for a, _ in train_data])
        train_y = torch.stack([b for _, b in train_data])
        val_X = torch.stack([a for a, _ in val_data])
        val_y = torch.stack([b for _, b in val_data])
        test_X = torch.stack([a for a, _ in test_data])
        test_y = torch.stack([b for _, b in test_data])

        # ---- Encode test set once per test delta (shared by all models) ----
        test_encoded = {}
        for d in TEST_DELTAS:
            test_encoded[d] = TensorDataset(
                encode_split(encoder, test_X, d, DEVICE), test_y.to(DEVICE))

        for train_delta in TRAIN_DELTAS:
            print(f"\n--- train delta = {train_delta} ---", flush=True)

            # ---- Encode train/val once (shared by all models) ----
            train_enc = encode_split(encoder, train_X, train_delta, DEVICE)
            val_enc = encode_split(encoder, val_X, train_delta, DEVICE)
            train_loader = DataLoader(TensorDataset(train_enc, train_y.to(DEVICE)),
                                      BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_enc, val_y.to(DEVICE)),
                                    BATCH_SIZE, shuffle=False)

            for mname in MODEL_SPECS:
                torch.manual_seed(1000 + run)   # same init seed across models
                model = build_model(mname, INPUT_DIM, N_CLASSES)
                trainer = Trainer(model, DEVICE)
                best_val = trainer.fit(train_loader, val_loader, EPOCHS,
                                       lr=LR, patience=PATIENCE)

                cross = {d: evaluate(model, test_encoded[d], DEVICE)
                         for d in TEST_DELTAS}
                id_acc = cross.get(train_delta, None)
                mean_ood = float(np.mean(list(cross.values())))

                results[mname][train_delta].append({
                    'run': run, 'best_val': best_val,
                    'id_acc': id_acc, 'mean_ood': mean_ood,
                    'cross': {str(k): v for k, v in cross.items()},
                })
                print(f"  {mname:12s} | BestVal={best_val:5.1f} | "
                      f"ID={id_acc:5.1f} | MeanOOD={mean_ood:5.1f}", flush=True)

        # ---- checkpoint partial results after every run ----
        save_results(results, TRAIN_DELTAS, TEST_DELTAS, MODEL_SPECS, partial=True)

    save_results(results, TRAIN_DELTAS, TEST_DELTAS, MODEL_SPECS, partial=False)
    print("\nExperiment complete.", flush=True)


RESULT_SUFFIX = ''


def save_results(results, train_deltas, test_deltas, model_specs, partial):
    suffix = RESULT_SUFFIX + ('_partial' if partial else '')

    json_path = os.path.join(OUTPUT_DIR, f'matched_filtering_detailed{suffix}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=1, default=float)

    rows = []
    for m in model_specs:
        for d in train_deltas:
            runs = results[m][d]
            if not runs:
                continue
            oods = [r['mean_ood'] for r in runs]
            ids_ = [r['id_acc'] for r in runs if r['id_acc'] is not None]
            rows.append({
                'model': m, 'train_delta': d, 'n_runs': len(runs),
                'id_acc_mean': round(float(np.mean(ids_)), 2) if ids_ else None,
                'id_acc_std': round(float(np.std(ids_)), 2) if ids_ else None,
                'ood_acc_mean': round(float(np.mean(oods)), 2),
                'ood_acc_std': round(float(np.std(oods)), 2),
            })
    if rows:
        csv_path = os.path.join(OUTPUT_DIR, f'matched_filtering_summary{suffix}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    if not partial:
        print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
