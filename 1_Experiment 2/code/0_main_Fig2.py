"""
v1_ablation_study_with_logging.py - Ablation Study: Dynamic Encoding vs Static Input

Features:
1. Training metrics logging per encoder/epoch (CSV)
2. Comprehensive comparison table (CSV)
3. Statistical analysis report

Comparison Design:
1. Baseline: Static patch input
2. Random Temporal: Random temporal expansion
3. Linear Temporal: Linear interpolation
4. Poisson Encoding: Standard Poisson spike coding
5. Dynamic Dissipative (delta=10)
6. Dynamic Critical (delta=0)
7. Dynamic Expansive (delta=-1.5)
"""

import os
import csv
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Reuse existing tools (removed analyze_receptive_fields)
from core.v1_receptive_field_learning import (
    NaturalImagePatches,
    V1_SNN_Autoencoder,
    ensure_dir,
    align_encoded_features
)


# ============================================================
# 1. Input Encoders
# ============================================================

class BaselineEncoder:
    """Baseline: Static input repeated over time."""

    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def encode(self, patches, device='cuda'):
        encoded = patches.unsqueeze(1).repeat(1, self.num_steps, 1)
        return encoded

    def get_encoding_name(self, delta=None):
        return "Baseline (Static)"


class RandomTemporalEncoder:
    """Random Temporal: Adds random temporal noise."""

    def __init__(self, num_steps=10, noise_std=0.1):
        self.num_steps = num_steps
        self.noise_std = noise_std

    def encode(self, patches, device='cuda'):
        B, F = patches.shape
        encoded = patches.unsqueeze(1).repeat(1, self.num_steps, 1)
        noise = torch.randn(B, self.num_steps, F, device=device) * self.noise_std
        encoded = encoded + noise
        return encoded

    def get_encoding_name(self, delta=None):
        return f"Random Temporal (std={self.noise_std})"


class LinearTemporalEncoder:
    """Linear Temporal: Linear interpolation."""

    def __init__(self, num_steps=10):
        self.num_steps = num_steps

    def encode(self, patches, device='cuda'):
        weights = torch.linspace(0, 1, self.num_steps, device=device)
        weights = weights.view(1, self.num_steps, 1)
        encoded = patches.unsqueeze(1) * weights
        return encoded

    def get_encoding_name(self, delta=None):
        return "Linear Temporal"


class PoissonSpikeEncoder:
    """Poisson Spike Encoding."""

    def __init__(self, num_steps=10, max_rate=1.0):
        self.num_steps = num_steps
        self.max_rate = max_rate

    def encode(self, patches, device='cuda'):
        rates = (patches + 1.0) / 2.0 * self.max_rate
        rates = rates.unsqueeze(1).repeat(1, self.num_steps, 1)
        spikes = torch.rand_like(rates, device=device) < rates
        encoded = spikes.float()
        return encoded

    def get_encoding_name(self, delta=None):
        return f"Poisson (rate={self.max_rate})"


# ============================================================
# 2. Training Function
# ============================================================

def train_v1_with_encoder(model,
                          dataloader,
                          encoder,
                          delta=None,
                          num_epochs=30,
                          lambda_sparse=0.1,
                          device='cuda',
                          enc_align_mode='mean',
                          save_dir='./results'):
    """Unified training loop with logging."""
    from v1_receptive_field_learning import align_encoded_features

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()

    history = {
        'epoch': [],
        'recon_loss': [],
        'sparse_loss': [],
        'total_loss': []
    }

    # Create CSV
    ensure_dir(save_dir)
    csv_path = os.path.join(save_dir, 'training_history.csv')

    for epoch in range(num_epochs):
        model.train()
        epoch_recon_loss = 0.0
        epoch_sparse_loss = 0.0
        epoch_total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (patches, _) in enumerate(pbar):
            patches = patches.to(device)

            # Encode
            if hasattr(encoder, 'encode'):
                if delta is not None:
                    try:
                        encoded = encoder.encode(patches, delta, device=device)
                    except TypeError:
                        encoded = encoder.encode(patches, device=device)
                else:
                    encoded = encoder.encode(patches, device=device)
            else:
                raise ValueError("Encoder must have 'encode' method")

            # Ensure shape (B, T, F)
            if encoded.dim() == 2:
                encoded = encoded.unsqueeze(1)

            # Align dimensions
            if encoded.shape[-1] != patches.shape[-1]:
                encoded = align_encoded_features(encoded, patches, mode=enc_align_mode)

            # Forward
            reconstructed, spikes_sum = model(encoded)

            # Loss
            recon_loss = recon_criterion(reconstructed, patches)
            sparse_loss = lambda_sparse * torch.mean(torch.abs(spikes_sum))
            total_loss = recon_loss + sparse_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log
            epoch_recon_loss += recon_loss.item()
            epoch_sparse_loss += sparse_loss.item()
            epoch_total_loss += total_loss.item()

            pbar.set_postfix({
                'recon': f'{recon_loss.item():.4f}',
                'sparse': f'{sparse_loss.item():.4f}'
            })

        # Epoch Stats
        n_batches = len(dataloader)
        avg_recon = epoch_recon_loss / n_batches
        avg_sparse = epoch_sparse_loss / n_batches
        avg_total = epoch_total_loss / n_batches

        history['epoch'].append(epoch + 1)
        history['recon_loss'].append(avg_recon)
        history['sparse_loss'].append(avg_sparse)
        history['total_loss'].append(avg_total)

        print(f"Epoch {epoch + 1}: Recon={avg_recon:.6f}, Sparse={avg_sparse:.6f}")

    # Save history
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_path, index=False)
    print(f"✓ Training history saved to {csv_path}")

    return history


# ============================================================
# 3. Main Ablation Study
# ============================================================

def run_ablation_study(patch_size=16,
                       hidden_dim=128,
                       num_patches=5000,
                       num_epochs=30,
                       num_steps=30,
                       batch_size=64,
                       base_dataset='cifar10'):
    """Main execution function for ablation study."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_results_dir = f'./results/ablation_{timestamp}'
    ensure_dir(main_results_dir)

    # Save Config
    config = {
        'timestamp': timestamp,
        'patch_size': patch_size,
        'hidden_dim': hidden_dim,
        'num_patches': num_patches,
        'num_epochs': num_epochs,
        'num_steps': num_steps,
        'batch_size': batch_size,
        'base_dataset': base_dataset,
        'device': str(device)
    }

    config_path = os.path.join(main_results_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to {config_path}")

    # 1. Prepare Data
    print("\n" + "=" * 60)
    print("Preparing Dataset")
    print("=" * 60)

    dataset = NaturalImagePatches(
        num_patches=num_patches,
        patch_size=patch_size,
        use_whitening=False,
        dataset=base_dataset
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 2. Define Encoders
    encoders = {
        'Baseline': BaselineEncoder(num_steps=num_steps),
        'Random': RandomTemporalEncoder(num_steps=num_steps, noise_std=0.1),
        'Linear': LinearTemporalEncoder(num_steps=num_steps),
        'Poisson': PoissonSpikeEncoder(num_steps=num_steps, max_rate=0.5),
    }

    # Add DynamicEncoder if available
    try:
        encoders['Dynamic_Dissipative'] = DynamicEncoder(num_steps=num_steps, tmax=8.0)
        encoders['Dynamic_Critical'] = DynamicEncoder(num_steps=num_steps, tmax=8.0)
        encoders['Dynamic_Transition'] = DynamicEncoder(num_steps=num_steps, tmax=8.0)
        encoders['Dynamic_Expansive'] = DynamicEncoder(num_steps=num_steps, tmax=8.0)
        print("✓ DynamicEncoder loaded")
    except ImportError:
        print("⚠ DynamicEncoder not found, skipping dynamic encoding")

    # 3. Run Experiments
    results = {}
    input_dim = patch_size * patch_size

    for name, encoder in encoders.items():
        print(f"\n" + "=" * 60)
        print(f"Training with encoder: {name}")
        print("=" * 60)

        # Init Model
        model = V1_SNN_Autoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            beta=0.9
        )

        # Set Delta
        delta = None
        if name == 'Dynamic_Dissipative':
            delta = 10.0
        elif name == 'Dynamic_Transition':
            delta = 2.0
        elif name == 'Dynamic_Critical':
            delta = 0.0
        elif name == 'Dynamic_Expansive':
            delta = -1.5

        # Directory per encoder
        save_dir = os.path.join(main_results_dir, name)
        ensure_dir(save_dir)

        # Train
        history = train_v1_with_encoder(
            model=model,
            dataloader=dataloader,
            encoder=encoder,
            delta=delta,
            num_epochs=num_epochs,
            lambda_sparse=0.1,
            device=device,
            enc_align_mode='mean',
            save_dir=save_dir
        )

        # Save Complete Model
        model_path = os.path.join(save_dir, 'model_complete.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_steps': num_steps,
                'beta': 0.9
            },
            'encoder_name': name,
            'delta': delta,
            'final_recon_loss': history['recon_loss'][-1],
            'final_sparse_loss': history['sparse_loss'][-1],
            'epoch': num_epochs
        }, model_path)
        print(f"✓ Model saved to {model_path}")

        # Save Weights Only
        weights_path = os.path.join(save_dir, 'model_weights.pth')
        torch.save(model.state_dict(), weights_path)
        print(f"✓ Model weights saved to {weights_path}")

        # Get Weights
        weights = model.get_receptive_fields()

        # Save Results
        results[name] = {
            'model': model,
            'history': history,
            'weights': weights,
            'encoder': encoder,
            'delta': delta
        }

        # Plot Curves
        plot_training_curves(history, name, save_dir)

    # 4. Comprehensive Comparison
    print("\n" + "=" * 60)
    print("Comprehensive Comparison")
    print("=" * 60)

    compare_all_encoders_with_logging(results, patch_size, main_results_dir)

    # 5. Generate Report
    generate_ablation_report(results, main_results_dir, config)

    print(f"\n✓ All results saved to: {main_results_dir}")
    return results


# ============================================================
# 4. Visualization and Comparison Tools
# ============================================================

def plot_training_curves(history, encoder_name, save_dir):
    """Plots training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['recon_loss'], label='Reconstruction Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{encoder_name}: Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['sparse_loss'], label='Sparsity Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{encoder_name}: Sparsity Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=150)
    plt.close()


def compare_all_encoders_with_logging(results, patch_size, save_dir):
    """Compares all encoders with logging."""
    ensure_dir(save_dir)

    n_encoders = len(results)

    # 1. Receptive Fields Comparison
    fig, axes = plt.subplots(2, (n_encoders + 1) // 2, figsize=(5 * ((n_encoders + 1) // 2), 10))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        weights = result['weights']
        n_display = min(16, weights.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_display)))

        ax = axes[idx]
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axis('off')

        for i in range(n_display):
            rf = weights[i].reshape(patch_size, patch_size)
            vmax = np.abs(rf).max()

            row = i // grid_size
            col = i % grid_size
            gs = 1.0 / grid_size
            left = col * gs
            bottom = 1.0 - (row + 1) * gs

            inset = ax.inset_axes([left, bottom, gs, gs])
            inset.imshow(rf, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            inset.axis('off')

    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Receptive Fields Comparison: Different Encoders', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_all_rfs.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Training Curves Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, result in results.items():
        history = result['history']
        axes[0].plot(history['recon_loss'], label=name, linewidth=2, alpha=0.8)
        axes[1].plot(history['sparse_loss'], label=name, linewidth=2, alpha=0.8)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[0].set_title('Reconstruction Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Sparsity Loss', fontsize=12)
    axes[1].set_title('Sparsity Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_training_curves.png', dpi=150)
    plt.close()

    # 3. Save Detailed CSV
    save_detailed_comparison_csv(results, save_dir)


def save_detailed_comparison_csv(results, save_dir):
    """Saves detailed comparison results to CSV."""

    # CSV 1: Comparison Summary
    comparison_data = []

    for name, result in results.items():
        history = result['history']
        weights = result['weights']
        delta = result.get('delta', None)

        # Basic Metrics
        final_recon = history['recon_loss'][-1]
        final_sparse = history['sparse_loss'][-1]
        final_total = history['total_loss'][-1]

        # RF Metrics
        rf_std = np.std(weights)
        rf_mean = np.mean(np.abs(weights))
        rf_max = np.max(np.abs(weights))
        rf_min = np.min(np.abs(weights))

        comparison_data.append({
            'encoder': name,
            'delta': delta if delta is not None else '',
            'final_recon_loss': final_recon,
            'final_sparse_loss': final_sparse,
            'final_total_loss': final_total,
            'rf_std': rf_std,
            'rf_mean_abs': rf_mean,
            'rf_max_abs': rf_max,
            'rf_min_abs': rf_min,
        })

    df_comparison = pd.DataFrame(comparison_data)

    # Save summary
    csv_path = os.path.join(save_dir, 'comparison_summary.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison summary saved to {csv_path}")

    # Print Table
    print("\n" + "=" * 80)
    print("QUANTITATIVE COMPARISON")
    print("=" * 80)
    df_display = df_comparison[['encoder', 'final_recon_loss', 'final_sparse_loss', 'rf_std']]
    df_display.columns = ['Encoder', 'Final Recon', 'Final Sparse', 'RF Quality']
    print(df_display.to_string(index=False))
    print("=" * 80)

    # CSV 2: Training Summary (Last 5 epochs)
    training_summary = []
    for name, result in results.items():
        history = result['history']
        for i in range(max(0, len(history['epoch']) - 5), len(history['epoch'])):
            training_summary.append({
                'encoder': name,
                'epoch': history['epoch'][i],
                'recon_loss': history['recon_loss'][i],
                'sparse_loss': history['sparse_loss'][i],
                'total_loss': history['total_loss'][i]
            })

    df_training = pd.DataFrame(training_summary)
    training_csv_path = os.path.join(save_dir, 'training_summary_last5epochs.csv')
    df_training.to_csv(training_csv_path, index=False)
    print(f"✓ Training summary saved to {training_csv_path}")


def generate_ablation_report(results, save_dir, config):
    """Generates the final ablation report."""
    report_path = os.path.join(save_dir, 'ABLATION_REPORT.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY: DYNAMIC ENCODING VS STATIC INPUT - FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Config
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key:20s}: {value}\n")
        f.write("\n")

        # Encoders
        f.write("ENCODERS TESTED:\n")
        f.write("-" * 80 + "\n")
        for idx, name in enumerate(results.keys(), 1):
            delta = results[name].get('delta', None)
            delta_str = f" (delta={delta})" if delta is not None else ""
            f.write(f"{idx}. {name}{delta_str}\n")
        f.write("\n")

        # Findings
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n")

        # Best Encoder metrics
        best_rf_quality = max(results.items(),
                              key=lambda x: np.std(x[1]['weights']))
        best_recon = min(results.items(),
                         key=lambda x: x[1]['history']['recon_loss'][-1])
        best_sparse = min(results.items(),
                          key=lambda x: x[1]['history']['sparse_loss'][-1])

        f.write(f"\n1. Best RF Quality (highest std): {best_rf_quality[0]}\n")
        f.write(f"   RF Std Dev = {np.std(best_rf_quality[1]['weights']):.6f}\n")

        f.write(f"\n2. Best Reconstruction: {best_recon[0]}\n")
        f.write(f"   Recon Loss = {best_recon[1]['history']['recon_loss'][-1]:.6f}\n")

        f.write(f"\n3. Best Sparsity: {best_sparse[0]}\n")
        f.write(f"   Sparse Loss = {best_sparse[1]['history']['sparse_loss'][-1]:.6f}\n")

        # Detailed Results
        f.write("\n" + "-" * 80 + "\n")
        f.write("DETAILED RESULTS FOR EACH ENCODER:\n")
        f.write("-" * 80 + "\n")

        for name, result in results.items():
            f.write(f"\n{name}:\n")
            history = result['history']
            weights = result['weights']

            f.write(f"  Final Reconstruction Loss: {history['recon_loss'][-1]:.6f}\n")
            f.write(f"  Final Sparsity Loss:       {history['sparse_loss'][-1]:.6f}\n")
            f.write(f"  RF Std Dev:                {np.std(weights):.6f}\n")
            f.write(f"  RF Mean Abs:               {np.mean(np.abs(weights)):.6f}\n")

    print(f"\n✓ Final ablation report saved to {report_path}")


# ============================================================
# 5. GPU Encoding Logic (Self-contained)
# ============================================================

class DynamicEncoder:
    def __init__(self, num_steps=5, tmax=8.0, use_gpu=True):
        self.num_steps = num_steps
        self.tmax = tmax
        self.use_gpu = use_gpu

        if use_gpu:
            try:
                self.encode_func = mixed_oscillator_encode_gpu
                print("✓ Using GPU-accelerated encoding")
            except ImportError:
                # Fallback mechanism if imports fail in standalone script
                from core.encoding import mixed_oscillator_encode
                self.encode_func = mixed_oscillator_encode
                self.use_gpu = False
                print("⚠ Falling back to CPU encoding")
        else:
            from core.encoding import mixed_oscillator_encode
            self.encode_func = mixed_oscillator_encode

    def encode(self, data, delta, device='cuda'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)

        params = {
            'alpha': 1.0,
            'beta': 0.2,
            'delta': delta,
            'gamma': 0.1,
            'omega': 1.0,
            'drive': 0.0
        }

        if self.use_gpu:
            # Direct GPU encoding
            encoded = self.encode_func(
                data,
                num_steps=self.num_steps,
                tmax=self.tmax,
                params=params
            )
        else:
            # CPU fallback
            data_cpu = data.cpu()
            encoded = self.encode_func(
                data_cpu,
                num_steps=self.num_steps,
                tmax=self.tmax,
                params=params
            )
            encoded = encoded.to(device)

        encoded = encoded.detach().clone().float()
        encoded.requires_grad_(False)

        return encoded


def mixed_oscillator_derivatives_gpu(state, alpha, beta, delta, gamma, omega):
    """
    Computes derivatives for Mixed Oscillator (Vectorized).
    state: (batch, features, 3) where 3 = [x, y, z]
    """
    x, y, z = state[..., 0], state[..., 1], state[..., 2]

    dx = y
    dy = -alpha * x - beta * x ** 3 - delta * y + gamma * z
    dz = -omega * x - delta * z + gamma * x * y

    return torch.stack([dx, dy, dz], dim=-1)


def mixed_oscillator_rk4_gpu(x0, alpha=1.0, beta=0.2, delta=10.0,
                             gamma=0.1, omega=1.0, tmax=8.0, num_steps=5):
    """
    GPU-accelerated RK4 integration for Mixed Oscillator.
    Returns: (batch, num_steps, features, 3)
    """
    device = x0.device
    B, F = x0.shape

    # High precision step size
    h = 0.01
    n_internal = int(tmax / h)

    if n_internal < num_steps:
        raise ValueError(f"tmax/h ({n_internal}) must be greater than num_steps ({num_steps})")

    sample_every = n_internal // num_steps

    # Init state [x, y, z]
    state = torch.zeros(B, F, 3, device=device, dtype=torch.float32)
    state[:, :, 0] = x0
    state[:, :, 1] = x0 * 0.2  # y0
    state[:, :, 2] = -x0       # z0

    trajectory = torch.zeros(B, num_steps, F, 3, device=device, dtype=torch.float32)
    sample_idx = 0

    for step in range(n_internal):
        # Vectorized RK4
        k1 = h * mixed_oscillator_derivatives_gpu(state, alpha, beta, delta, gamma, omega)
        k2 = h * mixed_oscillator_derivatives_gpu(state + k1 / 2, alpha, beta, delta, gamma, omega)
        k3 = h * mixed_oscillator_derivatives_gpu(state + k2 / 2, alpha, beta, delta, gamma, omega)
        k4 = h * mixed_oscillator_derivatives_gpu(state + k3, alpha, beta, delta, gamma, omega)

        state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Sampling
        if (step + 1) % sample_every == 0 and sample_idx < num_steps:
            trajectory[:, sample_idx] = state
            sample_idx += 1

    return trajectory


def mixed_oscillator_encode_gpu(data, num_steps=5, tmax=8.0, params=None):
    """
    GPU-accelerated Dynamic Encoding.
    """
    if params is None:
        params = {'alpha': 1.0, 'beta': 0.2, 'delta': 10.0,
                  'gamma': 0.1, 'omega': 1.0}

    # Normalize input
    data_max = torch.abs(data).max()
    if data_max > 1e-8:
        data_norm = data / (data_max + 1e-8)
    else:
        data_norm = data

    # GPU RK4 Integration
    trajectory = mixed_oscillator_rk4_gpu(
        data_norm,
        alpha=params['alpha'],
        beta=params['beta'],
        delta=params['delta'],
        gamma=params['gamma'],
        omega=params['omega'],
        tmax=tmax,
        num_steps=num_steps
    )

    # Reshape to (batch, num_steps, features*3)
    B, T, F, D = trajectory.shape
    encoded = trajectory.reshape(B, T, F * D)

    return encoded


# ============================================================
# 6. Main Entry
# ============================================================

if __name__ == "__main__":

    results = run_ablation_study(
        patch_size=16,
        hidden_dim=128,
        num_patches=5000,
        num_epochs=30,
        num_steps=5,
        batch_size=64,
        base_dataset='cifar10'
    )