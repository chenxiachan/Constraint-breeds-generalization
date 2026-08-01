#!/usr/bin/env python3
"""
Control experiment: Disentangling Dissipative Dynamics from Generic Temporal Smoothing

Key question:
  Does generalization arise from the specific nonlinear dynamics, or from generic temporal
  smoothing / low-pass filtering / signal degeneration?

Control encoders:
  1. Exponential Decay: 3-channel exponential decay (simplest dissipative dynamics).
  2. Gaussian Smoothed: Static input + temporally smoothed noise. Tests generic smoothing.

Protocol (faithful to original 0_main_Fig1.py):
  - Sample-by-sample encoding (each sample normalized independently)
  - SNN with early stopping (patience=10) on validation accuracy
  - Cross-encoding test on full Duffing delta range (12 delta values)
  - 10 runs with same data split strategy (seed 42+run)
  - 6 training conditions: 4 Duffing delta values + 2 control encoders
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
from scipy.signal import welch
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

from core.model_dup import SimpleSNN
from core.encoding import mixed_oscillator_encode

OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Encoding Classes
# ============================================================

class DuffingEncoder:
    """Full nonlinear Duffing encoder (original from paper)."""

    def __init__(self, num_steps=30, tmax=4.0):
        self.num_steps = num_steps
        self.tmax = tmax

    def encode(self, data, delta, device='cpu'):
        """Encode data. Matches original DynamicEncoder.encode() signature."""
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

    def get_encoding_name(self, delta):
        if delta < -0.5:
            return "expansive"
        elif delta > 5.0:
            return "dissipative"
        else:
            return "transition"


class ExponentialDecayEncoder:
    """
    3-channel exponential decay from initial conditions matching Duffing.
    Directly tests the degeneration hypothesis.
    """

    def __init__(self, num_steps=30, tmax=4.0):
        self.num_steps = num_steps
        self.tmax = tmax

    def encode(self, data, delta=2.0, device='cpu'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        data_np = data.cpu().numpy()
        batch_size, num_features = data_np.shape

        # Per-sample normalization (matches original encoding behavior)
        encoded = np.zeros((batch_size, self.num_steps, num_features * 3))
        t = np.linspace(0, self.tmax, self.num_steps)

        for b_idx in range(batch_size):
            sample = data_np[b_idx]
            sample_max = np.max(np.abs(sample))
            if sample_max > 0:
                sample = sample / sample_max

            # Decay rates derived from delta (matching Duffing's damping timescale)
            taus = [1.0 / max(delta, 0.1),
                    1.0 / max(delta * 0.5, 0.1),
                    1.0 / max(delta * 2.0, 0.1)]

            for f_idx in range(num_features):
                x_i = sample[f_idx]
                inits = [x_i, 0.2 * x_i, -x_i]  # Same init as Duffing
                for ch, (init_val, tau) in enumerate(zip(inits, taus)):
                    encoded[b_idx, :, f_idx * 3 + ch] = init_val * np.exp(-t / tau)

        return torch.from_numpy(encoded).float().to(device)

    def get_encoding_name(self, delta=2.0):
        return "exp_decay"


class GaussianSmoothedEncoder:
    """
    Static input + Gaussian-smoothed temporal noise.
    Tests whether generic temporal smoothing is sufficient.
    """

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

        # Gaussian smoothing kernel
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
                    # Deterministic seed from input for reproducibility
                    seed = abs(int(x_i * 1e6 + f_idx * 1000 + ch * 100 + b_idx)) % (2**31)
                    rng = np.random.RandomState(seed)

                    noise = rng.randn(self.num_steps + kernel_size) * self.noise_scale * (abs(init_val) + 1e-8)
                    smoothed = np.convolve(noise, kernel, mode='valid')[:self.num_steps]
                    encoded[b_idx, :, f_idx * 3 + ch] = init_val + smoothed

        return torch.from_numpy(encoded).float().to(device)

    def get_encoding_name(self, delta=None):
        return "gauss_smooth"


# ============================================================
# Trainer (faithful to original 0_main_Fig1.py Trainer class)
# ============================================================

class Trainer:
    """SNN Trainer - mirrors original Exp1 Trainer for SNN."""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'spike_count': []
        }

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss, correct, total, total_spikes = 0, 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            output, spike_records = self.model(batch_x)
            total_spikes += self.model.count_total_spikes(spike_records)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        avg_spikes = total_spikes / total
        return avg_loss, accuracy, avg_spikes

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss, correct, total, total_spikes = 0, 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output, spike_records = self.model(batch_x)
                total_spikes += self.model.count_total_spikes(spike_records)
                loss = criterion(output, batch_y)
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        avg_spikes = total_spikes / total
        return avg_loss, accuracy, avg_spikes

    def fit(self, train_loader, val_loader, epochs, lr=1e-4, patience=10):
        """Train with early stopping - matches original Exp1."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0
        patience_counter = 0

        pbar = tqdm(range(epochs), desc='Training SNN')
        for epoch in pbar:
            train_loss, train_acc, train_spikes = self.train_epoch(
                train_loader, criterion, optimizer
            )
            val_loss, val_acc, val_spikes = self.validate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['spike_count'].append(val_spikes)

            pbar.set_postfix({
                'train_acc': f'{train_acc:.2f}%',
                'val_acc': f'{val_acc:.2f}%',
                'spikes': f'{val_spikes:.0f}'
            })

            # Early stopping (same logic as original)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        return best_val_acc


# ============================================================
# Cross-Encoding Tester (faithful to original)
# ============================================================

class CrossEncodingTester:
    """Mirrors original Exp1 CrossEncodingTester."""

    def __init__(self, duffing_encoder, device):
        self.encoder = duffing_encoder
        self.device = device

    def test_single_encoding(self, model, test_data, test_labels, delta):
        """Test model on test data re-encoded with a specific Duffing delta."""
        model.eval()

        # Sample-by-sample encoding (matches original)
        encoded_list = []
        with torch.no_grad():
            for i in range(test_data.shape[0]):
                sample = test_data[i].unsqueeze(0)
                enc = self.encoder.encode(sample, delta, device=self.device)
                encoded_list.append(enc.squeeze(0))

        encoded_data = torch.stack(encoded_list).to(self.device)
        test_dataset = TensorDataset(encoded_data, test_labels.to(self.device))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output, _ = model(batch_x)
                _, predicted = output.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        return 100.0 * correct / total

    def run_cross_encoding_test(self, model, test_data, test_labels, test_deltas):
        """Test on all Duffing delta values. Returns {delta: accuracy}."""
        results = {}
        for delta in test_deltas:
            acc = self.test_single_encoding(model, test_data, test_labels, delta)
            results[delta] = acc
        return results

    def test_custom_encoding(self, model, test_data, test_labels, custom_encoder, delta=None):
        """Test on data encoded with a custom (non-Duffing) encoder."""
        model.eval()

        encoded_list = []
        with torch.no_grad():
            for i in range(test_data.shape[0]):
                sample = test_data[i].unsqueeze(0)
                enc = custom_encoder.encode(sample, delta, device=self.device)
                encoded_list.append(enc.squeeze(0))

        encoded_data = torch.stack(encoded_list).to(self.device)
        loader = DataLoader(
            TensorDataset(encoded_data, test_labels.to(self.device)),
            batch_size=64, shuffle=False
        )

        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in loader:
                out, _ = model(bx)
                _, pred = out.max(1)
                total += by.size(0)
                correct += pred.eq(by).sum().item()

        return 100.0 * correct / total


# ============================================================
# Spectral Analysis
# ============================================================

def compute_spectral_properties(encoded_data, fs=1.0):
    """Compute spectral centroid and entropy (matches paper Fig 5B analysis)."""
    data_np = encoded_data.cpu().numpy()
    batch_size, num_steps, num_features = data_np.shape

    centroids, entropies = [], []

    for b in range(min(batch_size, 50)):
        for f in range(min(num_features, 30)):
            signal = data_np[b, :, f]
            if np.std(signal) < 1e-10 or num_steps < 8:
                continue

            nperseg = min(num_steps, max(4, num_steps // 2))
            freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

            psd_sum = psd.sum()
            if psd_sum < 1e-15:
                continue
            psd_norm = psd / psd_sum

            centroids.append(np.sum(freqs * psd_norm))

            psd_nz = psd_norm[psd_norm > 0]
            entropies.append(-np.sum(psd_nz * np.log2(psd_nz)))

    return {
        'spectral_centroid': np.mean(centroids) if centroids else 0.0,
        'spectral_centroid_std': np.std(centroids) if centroids else 0.0,
        'spectral_entropy': np.mean(entropies) if entropies else 0.0,
        'spectral_entropy_std': np.std(entropies) if entropies else 0.0,
    }


# ============================================================
# Sample-by-sample encoding (matches original Exp1 paradigm)
# ============================================================

def encode_dataset_sample_by_sample(encoder, raw_data, raw_labels, delta, device):
    """
    Encode each sample independently (same as original Exp1).
    This ensures per-sample normalization consistency.
    """
    encoded_list = []
    for i in tqdm(range(raw_data.shape[0]), desc="Encoding", leave=False):
        sample = raw_data[i].unsqueeze(0)
        enc = encoder.encode(sample, delta, device=device)
        encoded_list.append(enc.squeeze(0))
    encoded = torch.stack(encoded_list)
    return TensorDataset(encoded, raw_labels)


# ============================================================
# Main Experiment
# ============================================================

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Device: {DEVICE}")

    # --- Config (identical to original Exp1) ---
    NUM_STEPS = 30
    TMAX = 4.0
    HIDDEN_DIM = 32
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-4
    PATIENCE = 10
    BETA = 0.95
    N_RUNS = 10  # Same as original

    # Full test delta range (same as paper Table 5)
    TEST_DELTAS = [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.0, 2.5, 5.0, 7.0, 10.0]

    # --- Encoders ---
    duffing_enc = DuffingEncoder(NUM_STEPS, TMAX)
    expdecay_enc = ExponentialDecayEncoder(NUM_STEPS, TMAX)
    gauss_enc = GaussianSmoothedEncoder(NUM_STEPS, TMAX)

    # Training conditions: (encoder, delta, label)
    TRAIN_CONDITIONS = [
        # Full nonlinear Duffing (reference)
        (duffing_enc,   -1.5, "Duffing Expansive (d=-1.5)"),
        (duffing_enc,    0.0, "Duffing Critical (d=0.0)"),
        (duffing_enc,    2.0, "Duffing Transition (d=2.0)"),
        (duffing_enc,   10.0, "Duffing Dissipative (d=10.0)"),
        # Control encoders
        (expdecay_enc,   2.0, "Exponential Decay"),
        (gauss_enc,     None, "Gaussian Smoothed"),
    ]

    # --- Load data (identical to original Exp1) ---
    print("\nLoading Sklearn Digits dataset...")
    digits = load_digits()
    scaler = StandardScaler()
    X_np = scaler.fit_transform(digits.data)
    y_np = digits.target

    N_SAMPLES = X_np.shape[0]
    N_FEATURES = X_np.shape[1]   # 64
    N_CLASSES = len(np.unique(y_np))  # 10
    INPUT_DIM = N_FEATURES * 3   # 192

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features, {N_CLASSES} classes")

    # --- Step 1: Spectral Analysis ---
    print("\n" + "=" * 60)
    print("Step 1: Spectral Analysis of All Encodings")
    print("=" * 60)

    # Use first 100 samples, encode sample-by-sample for consistency
    sample_data = X[:100]
    spectral_results = {}
    for enc, delta, label in TRAIN_CONDITIONS:
        enc_list = []
        for i in range(sample_data.shape[0]):
            s = sample_data[i].unsqueeze(0)
            enc_list.append(enc.encode(s, delta, device='cpu').squeeze(0))
        encoded = torch.stack(enc_list)
        props = compute_spectral_properties(encoded)
        spectral_results[label] = props
        print(f"  {label:40s} | Centroid: {props['spectral_centroid']:.4f} | "
              f"Entropy: {props['spectral_entropy']:.4f}")

    # --- Step 2: Cross-Encoding Generalization ---
    print("\n" + "=" * 60)
    print(f"Step 2: Cross-Encoding Generalization ({N_RUNS} runs)")
    print("=" * 60)

    all_cross_results = {label: [] for _, _, label in TRAIN_CONDITIONS}
    all_id_accs = {label: [] for _, _, label in TRAIN_CONDITIONS}

    for run in range(1, N_RUNS + 1):
        print(f"\n{'=' * 25} RUN {run} / {N_RUNS} {'=' * 25}")

        # Data split (identical to original Exp1)
        n_train = int(0.7 * N_SAMPLES)
        n_val = int(0.15 * N_SAMPLES)
        n_test = N_SAMPLES - n_train - n_val
        if (n_train + n_val + n_test) != N_SAMPLES:
            n_train = N_SAMPLES - n_val - n_test

        all_data_list = list(zip(X, y))
        train_data, val_data, test_data = random_split(
            all_data_list, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42 + run)
        )

        train_X = torch.stack([x for x, _ in train_data])
        train_y = torch.stack([y_lbl for _, y_lbl in train_data])
        val_X = torch.stack([x for x, _ in val_data])
        val_y = torch.stack([y_lbl for _, y_lbl in val_data])
        test_X = torch.stack([x for x, _ in test_data])
        test_y = torch.stack([y_lbl for _, y_lbl in test_data])

        print(f"Split: Train={n_train}, Val={n_val}, Test={n_test}")

        for enc, delta, label in TRAIN_CONDITIONS:
            print(f"\n--- {label} ---")

            # Encode train/val sample-by-sample (matches original)
            print("Encoding train data...")
            train_dataset = encode_dataset_sample_by_sample(
                enc, train_X, train_y, delta, DEVICE
            )
            print("Encoding val data...")
            val_dataset = encode_dataset_sample_by_sample(
                enc, val_X, val_y, delta, DEVICE
            )

            train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

            # Train SNN (same architecture & hyperparams as original)
            model = SimpleSNN(INPUT_DIM, HIDDEN_DIM, N_CLASSES,
                              num_steps=NUM_STEPS, beta=BETA)
            trainer = Trainer(model, DEVICE)
            best_val = trainer.fit(train_loader, val_loader, EPOCHS, lr=LR,
                                   patience=PATIENCE)

            # Cross-encoding test: test on all Duffing deltas
            tester = CrossEncodingTester(duffing_enc, DEVICE)
            print("Cross-encoding test...")
            cross_results = tester.run_cross_encoding_test(
                model, test_X, test_y, TEST_DELTAS
            )
            all_cross_results[label].append(cross_results)

            # ID accuracy: for Duffing encoders, it's cross_results[delta]
            # For non-Duffing encoders, test on their own encoding
            if isinstance(enc, DuffingEncoder) and delta in cross_results:
                id_acc = cross_results[delta]
            else:
                id_acc = tester.test_custom_encoding(
                    model, test_X, test_y, enc, delta
                )
            all_id_accs[label].append(id_acc)

            mean_ood = np.mean(list(cross_results.values()))
            print(f"  BestVal={best_val:.1f}%, ID={id_acc:.1f}%, MeanOOD={mean_ood:.1f}%")

            # Print per-delta breakdown
            for d in TEST_DELTAS:
                marker = " (ID)" if isinstance(enc, DuffingEncoder) and abs(d - delta) < 0.1 else ""
                print(f"    delta={d:5.1f}: {cross_results[d]:.1f}%{marker}")

    # --- Step 3: Aggregate & Save Results ---
    print("\n" + "=" * 60)
    print("Step 3: Results Summary")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_rows = []
    print(f"\n{'Encoding':40s} | {'ID Acc':>14s} | {'Mean OOD Acc':>14s} | "
          f"{'Centroid':>10s} | {'Entropy':>10s}")
    print("-" * 100)

    for enc, delta, label in TRAIN_CONDITIONS:
        id_accs = all_id_accs[label]
        cross_runs = all_cross_results[label]

        ood_per_run = [np.mean(list(r.values())) for r in cross_runs]

        mean_id = np.mean(id_accs)
        std_id = np.std(id_accs)
        mean_ood = np.mean(ood_per_run)
        std_ood = np.std(ood_per_run)

        sp = spectral_results.get(label, {})
        centroid = sp.get('spectral_centroid', 0)
        entropy = sp.get('spectral_entropy', 0)

        print(f"{label:40s} | {mean_id:5.1f} +- {std_id:4.1f} | "
              f"{mean_ood:5.1f} +- {std_ood:4.1f} | "
              f"{centroid:10.4f} | {entropy:10.4f}")

        summary_rows.append({
            'encoding': label,
            'id_acc_mean': round(mean_id, 2),
            'id_acc_std': round(std_id, 2),
            'ood_acc_mean': round(mean_ood, 2),
            'ood_acc_std': round(std_ood, 2),
            'spectral_centroid': round(centroid, 4),
            'spectral_entropy': round(entropy, 4),
        })

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f'lowpass_control_results_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nCSV saved: {csv_path}")

    # Save detailed JSON (per-run, per-delta)
    json_path = os.path.join(OUTPUT_DIR, f'lowpass_control_detailed_{timestamp}.json')
    def to_native(obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    json_data = {}
    for label in all_cross_results:
        json_data[label] = to_native({
            'cross_encoding_runs': [
                {str(k): v for k, v in run.items()}
                for run in all_cross_results[label]
            ],
            'id_accs': all_id_accs[label],
            'spectral': spectral_results.get(label, {}),
        })
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved: {json_path}")

    # --- Step 4: Generate figure ---
    try:
        generate_figure(all_cross_results, all_id_accs, spectral_results,
                        TRAIN_CONDITIONS, TEST_DELTAS, timestamp)
    except Exception as e:
        print(f"Figure generation failed: {e}")

    print("\nExperiment complete.")


def generate_figure(all_cross_results, all_id_accs, spectral_results,
                    train_conditions, test_deltas, timestamp):
    """Generate comparison figures for rebuttal."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # --- Panel A: Mean OOD Accuracy bar chart ---
    ax = axes[0]
    color_map = {
        'Duffing Transition': '#2196F3',
        'Duffing Expansive': '#F44336',
        'Duffing Critical': '#FF9800',
        'Duffing Dissipative': '#9C27B0',
        'Exponential': '#795548',
        'Gaussian': '#607D8B',
    }

    labels_short, ood_means, ood_stds, colors = [], [], [], []
    for enc, delta, label in train_conditions:
        cross_runs = all_cross_results[label]
        ood_per_run = [np.mean(list(r.values())) for r in cross_runs]
        ood_means.append(np.mean(ood_per_run))
        ood_stds.append(np.std(ood_per_run))

        if 'Duffing' in label:
            labels_short.append(f"Duf d={delta}")
        elif 'Exp' in label:
            labels_short.append("ExpDec")
        else:
            labels_short.append("Gauss")

        c = '#999999'
        for key, val in color_map.items():
            if key in label:
                c = val
                break
        colors.append(c)

    x_pos = np.arange(len(labels_short))
    ax.bar(x_pos, ood_means, yerr=ood_stds, color=colors, alpha=0.8, capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean OOD Accuracy (%)')
    ax.set_title('(a) Cross-Encoding Generalization')
    ax.grid(axis='y', alpha=0.3)

    # --- Panel B: Spectral Centroid vs Entropy scatter ---
    ax = axes[1]
    for enc, delta, label in train_conditions:
        sp = spectral_results.get(label, {})
        centroid = sp.get('spectral_centroid', 0)
        entropy = sp.get('spectral_entropy', 0)
        cross_runs = all_cross_results[label]
        ood = np.mean([np.mean(list(r.values())) for r in cross_runs])

        c = '#999999'
        for key, val in color_map.items():
            if key in label:
                c = val
                break

        ax.scatter(centroid, entropy, s=max(ood * 2, 10), c=c, alpha=0.7,
                   edgecolors='k', linewidth=0.5)
        if 'Transition' in label or 'Gaussian' in label or 'Exponential' in label:
            short = label.split('(')[0].strip() if '(' in label else label
            ax.annotate(short, (centroid, entropy), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Spectral Centroid')
    ax.set_ylabel('Spectral Entropy')
    ax.set_title('(b) Spectral Properties\n(size = OOD accuracy)')

    # --- Panel C: Cross-encoding heatmap for key conditions ---
    ax = axes[2]
    key_labels = [
        "Duffing Transition (d=2.0)",
        "Exponential Decay",
        "Gaussian Smoothed",
    ]
    heatmap_data, row_labels = [], []
    for label in key_labels:
        if label in all_cross_results and all_cross_results[label]:
            cross_runs = all_cross_results[label]
            mean_accs = {}
            for d in test_deltas:
                accs = [r[d] for r in cross_runs if d in r]
                mean_accs[d] = np.mean(accs) if accs else 0
            heatmap_data.append([mean_accs[d] for d in test_deltas])
            row_labels.append(label.replace('(d=', 'd=').replace(')', ''))

    if heatmap_data:
        hm = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        ax.set_xticks(range(len(test_deltas)))
        ax.set_xticklabels([str(d) for d in test_deltas], rotation=45, fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel('Test Delta')
        ax.set_title('(c) OOD Accuracy by Test Delta')
        plt.colorbar(hm, ax=ax, label='Accuracy (%)')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'lowpass_control_figure_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
