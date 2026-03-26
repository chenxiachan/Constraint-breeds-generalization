#!/usr/bin/env python3
"""
Rebuttal Experiment 2: Transition Regime Channel Statistics Analysis

Key question (o7Bu core critique):
  At high delta, y and z channels decay rapidly (timescale ~1/delta), leaving x as the
  dominant slowly-decaying variable. Could cross-encoding generalization simply arise
  from this signal convergence toward a quasi-static state?

This analysis provides quantitative evidence that:
  1. At delta=2.0 (transition), ALL THREE channels maintain meaningful variance
     → the signal is NOT degenerate
  2. At delta=10.0 (dissipative), channels DO degenerate toward single-channel
     → but this regime achieves LOWER generalization than transition
  3. The transition regime uniquely maintains high effective dimensionality
     combined with structured (non-chaotic) dynamics

Metrics:
  - Per-channel temporal variance (averaged across features and samples)
  - Channel contribution ratio (how much each channel contributes to total variance)
  - Effective dimensionality (PCA on 3 channels)
  - Cross-channel correlation
  - Per-channel spectral entropy
  - Signal waveform visualizations

No model training required — pure signal analysis.
"""

import sys
import os
import numpy as np
import torch
from scipy.signal import welch
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import csv
from datetime import datetime

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.join(BASE_DIR, '..', 'Constraint-breeds-generalization-main')
EXP1_CODE_DIR = os.path.join(REPO_DIR, '0_Experiment 1', 'code')
sys.path.insert(0, EXP1_CODE_DIR)

from core.encoding import mixed_oscillator_encode

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def encode_single_sample(feature_val, delta, num_steps=30, tmax=4.0):
    """Encode a single scalar feature value, return (num_steps, 3) for x/y/z channels."""
    sample = torch.tensor([[feature_val]], dtype=torch.float32)
    params = {
        'alpha': 2.0, 'beta': 0.1, 'delta': delta,
        'gamma': 0.1, 'omega': 1.0, 'drive': 0.0
    }
    enc = mixed_oscillator_encode(sample, num_steps=num_steps, tmax=tmax, params=params)
    # enc shape: (1, num_steps, 3) for 1 feature
    return enc[0].numpy()  # (num_steps, 3)


def encode_dataset_channels(data_np, delta, num_steps=30, tmax=4.0):
    """
    Encode dataset and separate x/y/z channels.

    Args:
        data_np: (n_samples, n_features) normalized numpy array
        delta: dissipation parameter

    Returns:
        x_ch, y_ch, z_ch: each (n_samples, num_steps, n_features)
    """
    n_samples, n_features = data_np.shape
    data_tensor = torch.tensor(data_np, dtype=torch.float32)

    # Encode sample-by-sample (consistent with original paradigm)
    params = {
        'alpha': 2.0, 'beta': 0.1, 'delta': delta,
        'gamma': 0.1, 'omega': 1.0, 'drive': 0.0
    }

    x_ch = np.zeros((n_samples, num_steps, n_features))
    y_ch = np.zeros((n_samples, num_steps, n_features))
    z_ch = np.zeros((n_samples, num_steps, n_features))

    for i in range(n_samples):
        sample = data_tensor[i:i+1]
        enc = mixed_oscillator_encode(sample, num_steps=num_steps, tmax=tmax, params=params)
        enc_np = enc[0].numpy()  # (num_steps, n_features * 3)

        for f in range(n_features):
            x_ch[i, :, f] = enc_np[:, f * 3]
            y_ch[i, :, f] = enc_np[:, f * 3 + 1]
            z_ch[i, :, f] = enc_np[:, f * 3 + 2]

    return x_ch, y_ch, z_ch


def compute_channel_metrics(x_ch, y_ch, z_ch):
    """
    Compute comprehensive per-channel metrics.

    Args:
        x_ch, y_ch, z_ch: each (n_samples, num_steps, n_features)

    Returns:
        dict of metrics
    """
    # 1. Per-channel temporal variance (variance over time, averaged across samples & features)
    x_var = np.mean(np.var(x_ch, axis=1))  # var over time, mean over samples & features
    y_var = np.mean(np.var(y_ch, axis=1))
    z_var = np.mean(np.var(z_ch, axis=1))
    total_var = x_var + y_var + z_var

    # Per-channel std versions too
    x_var_std = np.std(np.var(x_ch, axis=1).mean(axis=1))
    y_var_std = np.std(np.var(y_ch, axis=1).mean(axis=1))
    z_var_std = np.std(np.var(z_ch, axis=1).mean(axis=1))

    # 2. Channel contribution ratio
    if total_var > 1e-15:
        x_ratio = x_var / total_var
        y_ratio = y_var / total_var
        z_ratio = z_var / total_var
    else:
        x_ratio = y_ratio = z_ratio = 1.0 / 3

    # 3. Effective dimensionality via channel variance ratios
    # Using participation ratio: (sum(var))^2 / sum(var^2)
    vars_arr = np.array([x_var, y_var, z_var])
    if np.sum(vars_arr ** 2) > 1e-30:
        eff_dim = (np.sum(vars_arr)) ** 2 / np.sum(vars_arr ** 2)
    else:
        eff_dim = 0.0

    # 4. Cross-channel correlation (averaged across samples and features)
    n_samples, num_steps, n_features = x_ch.shape
    xy_corrs, xz_corrs, yz_corrs = [], [], []

    for i in range(min(n_samples, 100)):
        for f in range(min(n_features, 32)):
            x_sig = x_ch[i, :, f]
            y_sig = y_ch[i, :, f]
            z_sig = z_ch[i, :, f]

            # Skip near-constant signals
            if np.std(x_sig) < 1e-10 or np.std(y_sig) < 1e-10 or np.std(z_sig) < 1e-10:
                continue

            xy_corrs.append(abs(np.corrcoef(x_sig, y_sig)[0, 1]))
            xz_corrs.append(abs(np.corrcoef(x_sig, z_sig)[0, 1]))
            yz_corrs.append(abs(np.corrcoef(y_sig, z_sig)[0, 1]))

    # 5. Per-channel spectral entropy
    def channel_spectral_entropy(ch_data):
        entropies = []
        for i in range(min(ch_data.shape[0], 50)):
            for f in range(min(ch_data.shape[2], 20)):
                sig = ch_data[i, :, f]
                if np.std(sig) < 1e-10 or ch_data.shape[1] < 8:
                    continue
                nperseg = min(ch_data.shape[1], max(4, ch_data.shape[1] // 2))
                _, psd = welch(sig, fs=1.0, nperseg=nperseg)
                psd_sum = psd.sum()
                if psd_sum < 1e-15:
                    continue
                psd_n = psd / psd_sum
                psd_nz = psd_n[psd_n > 0]
                entropies.append(-np.sum(psd_nz * np.log2(psd_nz)))
        return np.mean(entropies) if entropies else 0.0

    x_entropy = channel_spectral_entropy(x_ch)
    y_entropy = channel_spectral_entropy(y_ch)
    z_entropy = channel_spectral_entropy(z_ch)

    return {
        'x_var': x_var, 'y_var': y_var, 'z_var': z_var,
        'x_var_std': x_var_std, 'y_var_std': y_var_std, 'z_var_std': z_var_std,
        'total_var': total_var,
        'x_ratio': x_ratio, 'y_ratio': y_ratio, 'z_ratio': z_ratio,
        'eff_dim': eff_dim,
        'xy_corr': np.mean(xy_corrs) if xy_corrs else 0.0,
        'xz_corr': np.mean(xz_corrs) if xz_corrs else 0.0,
        'yz_corr': np.mean(yz_corrs) if yz_corrs else 0.0,
        'x_entropy': x_entropy, 'y_entropy': y_entropy, 'z_entropy': z_entropy,
    }


def main():
    print("=" * 60)
    print("Rebuttal Exp 2: Channel Statistics Analysis")
    print("=" * 60)

    # --- Config ---
    NUM_STEPS = 30
    TMAX = 4.0
    # Fine-grained delta range for smooth curves
    DELTAS = [-1.5, -1.0, -0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6,
              1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    N_SAMPLES = 200  # Use subset for efficiency

    # --- Load data ---
    digits = load_digits()
    X_np = StandardScaler().fit_transform(digits.data)
    X_np = X_np[:N_SAMPLES]
    print(f"Using {N_SAMPLES} samples, {X_np.shape[1]} features")

    # --- Compute metrics for each delta ---
    all_metrics = {}
    for delta in DELTAS:
        print(f"\nProcessing delta={delta:6.2f}...", end=" ", flush=True)
        x_ch, y_ch, z_ch = encode_dataset_channels(X_np, delta, NUM_STEPS, TMAX)
        metrics = compute_channel_metrics(x_ch, y_ch, z_ch)
        all_metrics[delta] = metrics
        print(f"eff_dim={metrics['eff_dim']:.2f}, "
              f"x_var={metrics['x_var']:.4f}, y_var={metrics['y_var']:.4f}, z_var={metrics['z_var']:.4f}")

    # --- Print summary table ---
    print("\n" + "=" * 120)
    print(f"{'delta':>6s} | {'x_var':>8s} {'y_var':>8s} {'z_var':>8s} | "
          f"{'x_ratio':>7s} {'y_ratio':>7s} {'z_ratio':>7s} | "
          f"{'eff_dim':>7s} | {'xy_corr':>7s} {'xz_corr':>7s} {'yz_corr':>7s} | "
          f"{'x_entr':>7s} {'y_entr':>7s} {'z_entr':>7s}")
    print("-" * 120)

    for delta in DELTAS:
        m = all_metrics[delta]
        print(f"{delta:6.2f} | "
              f"{m['x_var']:8.4f} {m['y_var']:8.4f} {m['z_var']:8.4f} | "
              f"{m['x_ratio']:7.3f} {m['y_ratio']:7.3f} {m['z_ratio']:7.3f} | "
              f"{m['eff_dim']:7.3f} | "
              f"{m['xy_corr']:7.3f} {m['xz_corr']:7.3f} {m['yz_corr']:7.3f} | "
              f"{m['x_entropy']:7.3f} {m['y_entropy']:7.3f} {m['z_entropy']:7.3f}")

    # --- Save CSV ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'channel_stats_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        fields = ['delta'] + list(all_metrics[DELTAS[0]].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for delta in DELTAS:
            row = {'delta': delta, **all_metrics[delta]}
            writer.writerow(row)
    print(f"\nCSV saved: {csv_path}")

    # --- Generate figures ---
    try:
        generate_figures(all_metrics, DELTAS, timestamp)
    except Exception as e:
        print(f"Figure generation failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Generate example waveforms ---
    try:
        generate_waveform_figure(X_np, NUM_STEPS, TMAX, timestamp)
    except Exception as e:
        print(f"Waveform figure failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nExperiment complete.")


def generate_figures(all_metrics, deltas, timestamp):
    """Generate analysis figures."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    deltas_arr = np.array(deltas)

    x_vars = [all_metrics[d]['x_var'] for d in deltas]
    y_vars = [all_metrics[d]['y_var'] for d in deltas]
    z_vars = [all_metrics[d]['z_var'] for d in deltas]
    x_ratios = [all_metrics[d]['x_ratio'] for d in deltas]
    y_ratios = [all_metrics[d]['y_ratio'] for d in deltas]
    z_ratios = [all_metrics[d]['z_ratio'] for d in deltas]
    eff_dims = [all_metrics[d]['eff_dim'] for d in deltas]
    xy_corrs = [all_metrics[d]['xy_corr'] for d in deltas]
    xz_corrs = [all_metrics[d]['xz_corr'] for d in deltas]
    yz_corrs = [all_metrics[d]['yz_corr'] for d in deltas]
    x_entrs = [all_metrics[d]['x_entropy'] for d in deltas]
    y_entrs = [all_metrics[d]['y_entropy'] for d in deltas]
    z_entrs = [all_metrics[d]['z_entropy'] for d in deltas]

    # Transition region shading
    def shade_transition(ax):
        ax.axvspan(0, 2.5, alpha=0.1, color='blue', label='Transition regime')
        ax.axvline(x=2.0, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # --- Panel A: Per-channel variance (log scale) ---
    ax = axes[0, 0]
    # Clip negative variances for log scale (shouldn't happen but safety)
    ax.semilogy(deltas_arr, np.clip(x_vars, 1e-10, None), 'o-', label='x channel', color='#E53935', linewidth=2)
    ax.semilogy(deltas_arr, np.clip(y_vars, 1e-10, None), 's-', label='y channel', color='#43A047', linewidth=2)
    ax.semilogy(deltas_arr, np.clip(z_vars, 1e-10, None), '^-', label='z channel', color='#1E88E5', linewidth=2)
    shade_transition(ax)
    ax.set_xlabel('Delta (dissipation)')
    ax.set_ylabel('Temporal Variance (log)')
    ax.set_title('(a) Per-Channel Temporal Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel B: Channel contribution ratio (stacked) ---
    ax = axes[0, 1]
    ax.stackplot(deltas_arr, x_ratios, y_ratios, z_ratios,
                 labels=['x channel', 'y channel', 'z channel'],
                 colors=['#E53935', '#43A047', '#1E88E5'], alpha=0.7)
    ax.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5, label='Equal (1/3)')
    shade_transition(ax)
    ax.set_xlabel('Delta (dissipation)')
    ax.set_ylabel('Variance Contribution Ratio')
    ax.set_title('(b) Channel Contribution Ratio')
    ax.legend(loc='center right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Effective dimensionality ---
    ax = axes[0, 2]
    ax.plot(deltas_arr, eff_dims, 'ko-', linewidth=2, markersize=6)
    shade_transition(ax)
    ax.axhline(y=3.0, color='gray', linestyle=':', alpha=0.5, label='Max (3.0)')
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Single channel')
    ax.set_xlabel('Delta (dissipation)')
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title('(c) Effective Dimensionality\n(participation ratio)')
    ax.legend()
    ax.set_ylim(0.5, 3.2)
    ax.grid(True, alpha=0.3)

    # --- Panel D: Cross-channel correlation ---
    ax = axes[1, 0]
    ax.plot(deltas_arr, xy_corrs, 'o-', label='|corr(x,y)|', color='#E53935', linewidth=2)
    ax.plot(deltas_arr, xz_corrs, 's-', label='|corr(x,z)|', color='#43A047', linewidth=2)
    ax.plot(deltas_arr, yz_corrs, '^-', label='|corr(y,z)|', color='#1E88E5', linewidth=2)
    shade_transition(ax)
    ax.set_xlabel('Delta (dissipation)')
    ax.set_ylabel('Absolute Correlation')
    ax.set_title('(d) Cross-Channel Correlation')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # --- Panel E: Per-channel spectral entropy ---
    ax = axes[1, 1]
    ax.plot(deltas_arr, x_entrs, 'o-', label='x channel', color='#E53935', linewidth=2)
    ax.plot(deltas_arr, y_entrs, 's-', label='y channel', color='#43A047', linewidth=2)
    ax.plot(deltas_arr, z_entrs, '^-', label='z channel', color='#1E88E5', linewidth=2)
    shade_transition(ax)
    ax.set_xlabel('Delta (dissipation)')
    ax.set_ylabel('Spectral Entropy (bits)')
    ax.set_title('(e) Per-Channel Spectral Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel F: Key regime comparison summary ---
    ax = axes[1, 2]
    regimes = ['Expansive\n(d=-1.5)', 'Critical\n(d=0.0)', 'Transition\n(d=2.0)', 'Dissipative\n(d=10.0)']
    key_deltas = [-1.5, 0.0, 2.0, 10.0]
    key_eff_dims = [all_metrics[d]['eff_dim'] for d in key_deltas]
    key_total_vars = [min(all_metrics[d]['total_var'], 100) for d in key_deltas]  # cap for display

    x_pos = np.arange(len(regimes))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, key_eff_dims, width, label='Eff. Dim.', color='#1E88E5', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + width/2, key_total_vars, width, label='Total Var.', color='#FF9800', alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylabel('Effective Dimensionality', color='#1E88E5')
    ax2.set_ylabel('Total Variance (capped)', color='#FF9800')
    ax.set_title('(f) Regime Comparison Summary')
    ax.set_ylim(0, 3.5)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'channel_stats_figure_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    plt.close()


def generate_waveform_figure(X_np, num_steps, tmax, timestamp):
    """Generate example waveform visualizations for key delta values."""
    import matplotlib.pyplot as plt

    key_deltas = [-1.5, 0.0, 2.0, 10.0]
    regime_names = ['Expansive (d=-1.5)', 'Critical (d=0.0)',
                    'Transition (d=2.0)', 'Dissipative (d=10.0)']

    # Pick 3 representative features from one sample
    sample = X_np[0]
    feature_indices = [0, 16, 32]  # Spread across feature space
    t = np.linspace(0, tmax, num_steps)

    fig, axes = plt.subplots(len(key_deltas), len(feature_indices),
                             figsize=(13, 10), dpi=150)

    for row, (delta, name) in enumerate(zip(key_deltas, regime_names)):
        for col, f_idx in enumerate(feature_indices):
            ax = axes[row, col]
            feature_val = sample[f_idx]

            trajectory = encode_single_sample(feature_val, delta, num_steps, tmax)
            # trajectory: (num_steps, 3)

            ax.plot(t, trajectory[:, 0], '-', label='x', color='#E53935', linewidth=1.5)
            ax.plot(t, trajectory[:, 1], '-', label='y', color='#43A047', linewidth=1.5)
            ax.plot(t, trajectory[:, 2], '-', label='z', color='#1E88E5', linewidth=1.5)

            if row == 0:
                ax.set_title(f'Feature {f_idx}\n(val={feature_val:.2f})', fontsize=9)
            if col == 0:
                ax.set_ylabel(f'{name}\n', fontsize=9)
            if row == len(key_deltas) - 1:
                ax.set_xlabel('Time')
            if row == 0 and col == len(feature_indices) - 1:
                ax.legend(fontsize=7, loc='upper right')

            ax.grid(True, alpha=0.2)

    plt.suptitle('Signal Waveforms: x/y/z Channels Across Dynamical Regimes',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f'channel_waveforms_{timestamp}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Waveform figure saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
