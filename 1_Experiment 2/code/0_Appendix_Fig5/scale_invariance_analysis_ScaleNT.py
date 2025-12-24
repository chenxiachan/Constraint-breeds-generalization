"""
Advanced visualization: Delta vs N_steps across multiple T_max regimes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import entropy
from tqdm import tqdm
from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as ticker
from encoding_wrapper import DynamicEncoder, create_synthetic_dataset

# --- Style Settings ---
plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams['font.family'] = 'serif'

def compute_spectral_metrics(encoded_signal, sampling_rate):
    """Compute spectral centroid and entropy for a single sample."""
    signal_1d = encoded_signal[0, :, 0].cpu().numpy()

    nperseg = min(len(signal_1d), 128)
    if nperseg < 4: return 0.0, 1.0

    freqs, psd = signal.welch(signal_1d, fs=sampling_rate, nperseg=nperseg)

    psd_sum = np.sum(psd)
    if psd_sum == 0: return 0.0, 1.0
    psd_norm = psd / psd_sum

    div = np.log2(len(psd))
    spec_entropy = entropy(psd_norm, base=2) / div if div > 0 else 1.0
    centroid = np.sum(freqs * psd_norm)

    return spec_entropy, centroid

def run_single_tmax_analysis(sample_data, tmax, deltas, n_steps_list):
    """Runs the scan for a single fixed Tmax."""
    entropy_map = np.zeros((len(n_steps_list), len(deltas)))
    centroid_map = np.zeros((len(n_steps_list), len(deltas)))

    # Create a concise description for tqdm
    desc = f"Scanning Tmax={tmax}"

    for i, n in enumerate(tqdm(n_steps_list, desc=desc, leave=False)):
        # Initialize encoder with current N and specific Tmax
        encoder = DynamicEncoder(num_steps=int(n), tmax=tmax)
        sampling_rate = n / tmax

        for j, d in enumerate(deltas):
            with torch.no_grad():
                encoded = encoder.encode(sample_data, d)

            ent, cent = compute_spectral_metrics(encoded, sampling_rate)
            entropy_map[i, j] = ent
            centroid_map[i, j] = cent

    return entropy_map, centroid_map

def plot_tmax_comparison(results_dict, deltas, n_steps_list):
    """
    Generates a Facet Grid Plot.
    Rows: Metrics (Entropy, Centroid)
    Columns: Different Tmax values
    """
    tmax_values = sorted(results_dict.keys())
    n_cols = len(tmax_values)

    # Setup Figure: 2 Rows (Entropy, Centroid) x N Columns (Tmax)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 10), sharex=True, sharey=True)

    # Meshgrid
    X, Y = np.meshgrid(deltas, n_steps_list)

    # Pre-calculate global vmin/vmax for consistent colorbars
    # Centroid needs careful log scale handling
    all_centroids = [res['centroid'] for res in results_dict.values()]
    c_vmin = max(np.min([np.min(c[c>0]) for c in all_centroids]), 1e-2)
    c_vmax = np.max([np.max(c) for c in all_centroids])

    # Iterate over columns (Tmax values)
    for idx, tmax in enumerate(tmax_values):
        data = results_dict[tmax]
        ent_map = data['entropy']
        cent_map = data['centroid']

        # --- Row 1: Entropy ---
        ax_ent = axes[0, idx]
        im1 = ax_ent.pcolormesh(X, Y, ent_map, cmap='RdYlBu_r', shading='auto', vmin=0, vmax=1)

        ax_ent.set_yscale('log')
        ax_ent.set_title(f"$T_{{max}} = {tmax}$", fontsize=13, fontweight='bold')

        # Only label Y-axis on the first column
        if idx == 0:
            ax_ent.set_ylabel('Scale $N$ (Entropy)', fontweight='bold')

        # Annotate the "Valley"
        ax_ent.axvline(x=0.0, color='black', linestyle='--', alpha=0.3)
        ax_ent.axvline(x=2.0, color='black', linestyle='--', alpha=0.3)

        # --- Row 2: Centroid (LogNorm) ---
        ax_cent = axes[1, idx]
        norm = LogNorm(vmin=c_vmin, vmax=c_vmax)
        im2 = ax_cent.pcolormesh(X, Y, cent_map, cmap='magma', shading='auto', norm=norm)

        # Contours
        levels = np.logspace(np.log10(c_vmin), np.log10(c_vmax), 6)
        ax_cent.contour(X, Y, cent_map, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

        ax_cent.set_xlabel(r'$\delta$ (Control)', fontweight='bold')
        if idx == 0:
            ax_cent.set_ylabel('Scale $N$ (Freq)', fontweight='bold')

    # --- Add Global Colorbars ---
    # Entropy Colorbar
    cbar_ax1 = fig.add_axes([0.92, 0.53, 0.015, 0.35]) # Right side upper
    fig.colorbar(im1, cax=cbar_ax1, label='Spectral Entropy')

    # Centroid Colorbar
    cbar_ax2 = fig.add_axes([0.92, 0.11, 0.015, 0.35]) # Right side lower
    fig.colorbar(im2, cax=cbar_ax2, label='Spectral Centroid (Hz)', format=ticker.LogFormatterMathtext())

    # plt.suptitle('Scale-Space Invariance: $T_{max}$ Sweep Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.subplots_adjust(right=0.90, wspace=0.1, hspace=0.15)

    save_path = 'tmax_sweep_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Analysis complete. Comparison plot saved to '{save_path}'")
    plt.show()

def main():
    # 1. Data Generation
    print("Generating synthetic data...")
    X, _ = create_synthetic_dataset(n_samples=10, n_features=20, n_classes=2)

    # 2. Parameter Setup
    # T_max range to scan
    tmax_list = [4.0, 8.0, 12.0, 16.0]

    # Delta range (Same as before)
    deltas = np.concatenate([
        np.linspace(-1.0, -0.1, 15),
        np.linspace(0.0, 3.0, 40),
        np.linspace(3.1, 6.0, 15)
    ])
    deltas = np.sort(np.unique(deltas))

    # N_steps range (Log scale)
    n_steps_list = np.unique(np.logspace(np.log10(10), np.log10(100), 20).astype(int))

    # 3. Run Sweep
    results = {}
    print(f"Starting Tmax Sweep: {tmax_list}...")

    for t in tmax_list:
        ent_map, cent_map = run_single_tmax_analysis(X, t, deltas, n_steps_list)
        results[t] = {
            'entropy': ent_map,
            'centroid': cent_map
        }

    # 4. Visualize
    plot_tmax_comparison(results, deltas, n_steps_list)

if __name__ == "__main__":
    main()