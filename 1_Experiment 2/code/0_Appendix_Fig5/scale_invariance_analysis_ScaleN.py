"""
Rigorous visualization of scale-invariance in the transition regime.
Generates Scale-Space Heatmaps: Delta (control) vs. N_steps (observation scale).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from tqdm import tqdm
import seaborn as sns
from encoding_wrapper import DynamicEncoder, create_synthetic_dataset

# --- Style Settings ---
plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def compute_spectral_metrics(encoded_signal, sampling_rate):
    """Compute spectral centroid and entropy for a single sample."""
    signal_1d = encoded_signal[0, :, 0].cpu().numpy()

    # Dynamic nperseg based on signal length to avoid warnings
    nperseg = min(len(signal_1d), 128)
    if nperseg < 4: return 0.0, 1.0 # Fallback for extremely short signals

    freqs, psd = signal.welch(signal_1d, fs=sampling_rate, nperseg=nperseg)

    # Normalize PSD
    psd_sum = np.sum(psd)
    if psd_sum == 0: return 0.0, 1.0
    psd_norm = psd / psd_sum

    # 1. Spectral Entropy (Normalized [0, 1])
    # Handle log2(0) case if len(psd) is 1
    div = np.log2(len(psd))
    spec_entropy = entropy(psd_norm, base=2) / div if div > 0 else 1.0

    # 2. Spectral Centroid
    centroid = np.sum(freqs * psd_norm)

    return spec_entropy, centroid

def run_scale_space_analysis(sample_data):
    # --- 1. Define the Scale Space ---
    # Delta range (X-axis): Fine-grained sweep
    deltas = np.concatenate([
        np.linspace(-2.0, -0.1, 30),  # Expansive
        np.linspace(0.0, 3.0, 50),    # Transition (Dense sampling)
        np.linspace(3.1, 8.0, 20)     # Dissipative
    ])
    deltas = np.sort(np.unique(deltas))

    # N_steps range (Y-axis): Log-linear scale from coarse to fine
    # We want to see the evolution from N=5 up to N=100
    n_steps_list = np.unique(np.logspace(np.log10(5), np.log10(100), 20).astype(int))

    # Fixed physical time Tmax (to isolate sampling rate effects)
    fixed_tmax = 8.0

    # Storage for heatmaps
    # Matrix shape: [len(n_steps), len(deltas)]
    entropy_map = np.zeros((len(n_steps_list), len(deltas)))
    centroid_map = np.zeros((len(n_steps_list), len(deltas)))

    print(f"Starting Scale-Space Analysis...")
    print(f"X-axis: {len(deltas)} delta points")
    print(f"Y-axis: {len(n_steps_list)} scales (N={n_steps_list[0]} to {n_steps_list[-1]})")

    # --- 2. Main Loop ---
    for i, n in enumerate(tqdm(n_steps_list, desc="Scanning Scales")):
        # Initialize encoder for this scale
        encoder = DynamicEncoder(num_steps=int(n), tmax=fixed_tmax)
        sampling_rate = n / fixed_tmax

        for j, d in enumerate(deltas):
            with torch.no_grad():
                encoded = encoder.encode(sample_data, d)

            ent, cent = compute_spectral_metrics(encoded, sampling_rate)

            entropy_map[i, j] = ent
            centroid_map[i, j] = cent

    return deltas, n_steps_list, entropy_map, centroid_map


from matplotlib.colors import LogNorm, Normalize
import matplotlib.ticker as ticker


def plot_invariance_heatmaps(deltas, n_steps, entropy_map, centroid_map):
    """
    Plot the Scale-Space Heatmaps with LogNorm for better visibility of the valley.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(deltas, n_steps)

    # =================================================================
    # Plot A: Spectral Entropy (The Order Map)
    # =================================================================
    # Use a diverging colormap: Blue (Low Entropy/Order) <-> Red (High Entropy/Chaos)
    # cmap='RdYlBu_r' is excellent for this: Red=High, Blue=Low
    im1 = axes[0].pcolormesh(X, Y, entropy_map, cmap='RdYlBu_r', shading='auto', vmin=0, vmax=1)

    axes[0].set_yscale('log')
    axes[0].set_ylabel('Observation Scale $N$ (steps)', fontweight='bold', fontsize=14)
    # axes[0].set_title('Scale-Space Entropy', fontweight='bold', fontsize=16)

    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label('Normalized Spectral Entropy', fontsize=12)

    # --- Visual Guides for Invariance ---
    # 1. Vertical dashed lines to mark the Transition Regime
    axes[0].axvline(x=0.0, color='black', linestyle='--', alpha=0.4, linewidth=1.5)
    axes[0].axvline(x=2, color='black', linestyle='--', alpha=0.4, linewidth=1.5)

    # 2. Add Text Annotations
    axes[0].text(-1.0, n_steps[len(n_steps) // 2], 'Scale-Dependent\nChaos', ha='center', va='center',
                 color='white', fontweight='bold', fontsize=12,
                 bbox=dict(boxstyle="round", fc="firebrick", ec="none", alpha=0.6))

    axes[0].text(1, n_steps[len(n_steps) // 2], 'Scale-Invariant\nStructure', ha='center', va='center',
                 color='white', fontweight='bold', fontsize=12,
                 bbox=dict(boxstyle="round", fc="royalblue", ec="none", alpha=0.7))

    axes[0].text(5, n_steps[len(n_steps) // 2], 'Signal\nConvergence', ha='center', va='center',
                 color='white', fontweight='bold', fontsize=12,
                 bbox=dict(boxstyle="round", fc="gray", ec="none", alpha=0.6))

    # =================================================================
    # Plot B: Spectral Centroid (The Frequency Map) - WITH LOGNORM
    # =================================================================
    centroid_vmin = np.max([np.min(centroid_map[centroid_map > 0]), 1e-2])
    centroid_vmax = np.max(centroid_map)

    norm = LogNorm(vmin=centroid_vmin, vmax=centroid_vmax)

    # Use 'magma' or 'inferno' for high contrast in dark regions
    im2 = axes[1].pcolormesh(X, Y, centroid_map, cmap='magma', shading='auto', norm=norm)

    # --- Add Contour Lines to Highlight the Valley Floor ---
    # This draws lines at specific frequency levels, making the "shape" of the valley explicit
    # We focus levels on the lower end
    levels = np.logspace(np.log10(centroid_vmin), np.log10(centroid_vmax), 8)
    cs = axes[1].contour(X, Y, centroid_map, levels=levels, colors='white', alpha=0.3, linewidths=0.8)
    axes[1].clabel(cs, inline=True, fontsize=8, fmt='%.1f Hz')

    axes[1].set_yscale('log')
    axes[1].set_xlabel(r'Control Parameter $\delta$', fontweight='bold', fontsize=14)
    axes[1].set_ylabel('Observation Scale $N$ (steps)', fontweight='bold', fontsize=14)
    # axes[1].set_title('Scale-Space Frequency', fontweight='bold', fontsize=16)

    # Add colorbar with Log Formatting
    cbar2 = fig.colorbar(im2, ax=axes[1], pad=0.02, format=ticker.LogFormatterMathtext())
    cbar2.set_label('Spectral Centroid (Hz) - Log Scale', fontsize=12)

    # Region Annotations (Bottom)
    axes[1].axvline(x=0.0, color='white', linestyle='--', alpha=0.3)
    axes[1].axvline(x=2, color='white', linestyle='--', alpha=0.3)

    trans = axes[1].get_xaxis_transform()  # x in data coords, y in axes coords


    plt.tight_layout()
    # Adjust bottom margin for text
    plt.subplots_adjust(bottom=0.15)

    save_path = 'scale_space_invariance_log.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Scale-Space Heatmaps (LogNorm) saved to '{save_path}'")
    plt.show()

def main():
    # Generate sample
    print("Generating synthetic data...")
    X, _ = create_synthetic_dataset(n_samples=10, n_features=20, n_classes=2)

    # Run Analysis
    deltas, n_steps, entropy_map, centroid_map = run_scale_space_analysis(X)

    # Plot
    plot_invariance_heatmaps(deltas, n_steps, entropy_map, centroid_map)

    # Optional: Save data for later
    np.savez('scale_space_data.npz', deltas=deltas, n_steps=n_steps,
             entropy_map=entropy_map, centroid_map=centroid_map)

if __name__ == "__main__":
    main()