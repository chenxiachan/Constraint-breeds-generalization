"""
v1_delta_scan.py - Dynamics Parameter Scan (Delta Scan)

Scans the Duffing oscillator dissipation parameter (delta) to observe 
structural changes in learned Receptive Fields.

Features:
1. Automated parameter scanning
2. Trend analysis (Metrics vs Delta)
3. Receptive Field visualization grid
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from v1_receptive_field_learning import (
    NaturalImagePatches,
    V1_SNN_Autoencoder,
    ensure_dir,
    analyze_receptive_fields,
    align_encoded_features
)

# ============================================================
# 1. GPU Accelerated Dynamic Encoder
# ============================================================

def mixed_oscillator_derivatives_gpu(state, alpha, beta, delta, gamma, omega):
    """Compute derivatives (Vectorized)."""
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = y
    dy = -alpha * x - beta * x**3 - delta * y + gamma * z
    dz = -omega * x - delta * z + gamma * x * y 
    return torch.stack([dx, dy, dz], dim=-1)

def mixed_oscillator_rk4_gpu(x0, alpha=2.0, beta=0.1, delta=10.0, 
                             gamma=0.1, omega=1.0, tmax=8.0, num_steps=5):
    """GPU-accelerated RK4 integration."""
    device = x0.device
    B, F = x0.shape
    
    h = 0.01 
    n_internal = int(tmax / h)
    
    if n_internal < num_steps:
        raise ValueError(f"tmax/h ({n_internal}) must be > num_steps ({num_steps})")
    
    sample_every = n_internal // num_steps
    
    # Init state [x, y, z]
    state = torch.zeros(B, F, 3, device=device, dtype=torch.float32)
    state[:, :, 0] = x0
    state[:, :, 1] = x0 * 0.2  
    state[:, :, 2] = -x0      
    
    trajectory = torch.zeros(B, num_steps, F, 3, device=device, dtype=torch.float32)
    sample_idx = 0
    
    for step in range(n_internal):
        k1 = h * mixed_oscillator_derivatives_gpu(state, alpha, beta, delta, gamma, omega)
        k2 = h * mixed_oscillator_derivatives_gpu(state + k1/2, alpha, beta, delta, gamma, omega)
        k3 = h * mixed_oscillator_derivatives_gpu(state + k2/2, alpha, beta, delta, gamma, omega)
        k4 = h * mixed_oscillator_derivatives_gpu(state + k3, alpha, beta, delta, gamma, omega)
        
        state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        if (step + 1) % sample_every == 0 and sample_idx < num_steps:
            trajectory[:, sample_idx] = state
            sample_idx += 1
            
    return trajectory

class DynamicEncoder:
    def __init__(self, num_steps=5, tmax=8.0):
        self.num_steps = num_steps
        self.tmax = tmax
        print("✓ DynamicEncoder initialized (GPU Optimized)")

    def encode(self, data, delta, device='cuda'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)
        
        params = {
            'alpha': 1.0, 'beta': 0.2, 'delta': delta,
            'gamma': 0.1, 'omega': 1.0
        }
        
        # Normalize
        data_max = torch.abs(data).max()
        if data_max > 1e-8:
            data_norm = data / (data_max + 1e-8)
        else:
            data_norm = data
        
        # RK4 Integration
        trajectory = mixed_oscillator_rk4_gpu(
            data_norm, 
            alpha=params['alpha'], beta=params['beta'], delta=params['delta'],
            gamma=params['gamma'], omega=params['omega'],
            tmax=self.tmax, num_steps=self.num_steps
        )
        
        # Flatten
        B, T, F, D = trajectory.shape
        encoded = trajectory.reshape(B, T, F * D)
        
        return encoded.detach()

# ============================================================
# 2. Training Function
# ============================================================

def precompute_encoded_data(dataloader, encoder, delta, device):
    """
    Pre-compute encoding to avoid redundant ODE solving during training.
    """
    encoded_list = []
    targets_list = []
    
    print(f"  ⚡ Pre-computing encoding for δ={delta}...")
    
    with torch.no_grad():
        for patches, _ in dataloader:
            patches = patches.to(device)
            
            # 1. Encode
            encoded = encoder.encode(patches, delta=delta, device=device)
            
            # 2. Align
            if encoded.shape[-1] != patches.shape[-1]:
                encoded = align_encoded_features(encoded, patches, mode='mean')
                
            encoded_list.append(encoded)
            targets_list.append(patches)
            
    full_encoded = torch.cat(encoded_list, dim=0)
    full_targets = torch.cat(targets_list, dim=0)
    
    cached_dataset = torch.utils.data.TensorDataset(full_encoded, full_targets)
    
    cached_loader = torch.utils.data.DataLoader(
        cached_dataset, 
        batch_size=dataloader.batch_size, 
        shuffle=True
    )
    
    return cached_loader

def train_one_delta(model, dataloader, encoder, delta, num_epochs, lambda_sparse, device, save_dir):
    """
    Train model for a specific Delta value.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()

    history = {'epoch': [], 'recon_loss': [], 'sparse_loss': [], 'total_loss': []}
    csv_path = os.path.join(save_dir, 'training_history.csv')

    # Step 1: Pre-compute (Optimization)
    train_loader = precompute_encoded_data(dataloader, encoder, delta, device)

    # Step 2: Fast Training Loop
    epoch_pbar = tqdm(range(num_epochs), desc=f"  Train δ={delta:<4}", leave=False, position=1)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_losses = {'recon': 0.0, 'sparse': 0.0, 'total': 0.0}
        
        for encoded_batch, target_batch in train_loader:
            reconstructed, spikes_sum = model(encoded_batch)

            recon_loss = recon_criterion(reconstructed, target_batch)
            sparse_loss = lambda_sparse * torch.mean(torch.abs(spikes_sum))
            total_loss = recon_loss + sparse_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['sparse'] += sparse_loss.item()
            epoch_losses['total'] += total_loss.item()

        n_batches = len(train_loader)
        avg_recon = epoch_losses['recon'] / n_batches
        avg_sparse = epoch_losses['sparse'] / n_batches
        avg_total = epoch_losses['total'] / n_batches
        
        history['epoch'].append(epoch + 1)
        history['recon_loss'].append(avg_recon)
        history['sparse_loss'].append(avg_sparse)
        history['total_loss'].append(avg_total)

        epoch_pbar.set_postfix({
            'Recon': f'{avg_recon:.5f}', 
            'Sparse': f'{avg_sparse:.5f}'
        })
    
    final_recon = history['recon_loss'][-1]
    final_sparse = history['sparse_loss'][-1]
    tqdm.write(f"  ✓ Finished δ={delta:<4} | Final Recon: {final_recon:.5f} | Sparse: {final_sparse:.5f}")

    pd.DataFrame(history).to_csv(csv_path, index=False)
    return history

# ============================================================
# 3. Main Scan Logic
# ============================================================

def run_delta_scan(
    start=0.0, end=2.4, step=0.2,
    patch_size=16, hidden_dim=128, num_patches=5000,
    num_epochs=30, num_steps=5, batch_size=64,
    base_dataset='cifar10'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_dir = f'./results/scan_delta_{timestamp}'
    ensure_dir(main_dir)

    deltas = [round(x, 1) for x in np.arange(start, end + 0.01, step)]
    print(f"Scanning Deltas: {deltas}")

    dataset = NaturalImagePatches(
        num_patches=num_patches, 
        patch_size=patch_size,
        dataset=base_dataset 
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    encoder = DynamicEncoder(num_steps=num_steps, tmax=8.0)

    results = {}

    # --- Scan Loop ---
    pbar = tqdm(deltas, desc="Scanning Deltas")
    for delta in pbar:
        name = f"Delta_{delta:.1f}"
        pbar.set_description(f"Scanning {name}")
        
        sub_dir = os.path.join(main_dir, name)
        ensure_dir(sub_dir)

        model = V1_SNN_Autoencoder(
            input_dim=patch_size**2,
            hidden_dim=hidden_dim,
            num_steps=num_steps
        )

        history = train_one_delta(
            model, dataloader, encoder, delta, 
            num_epochs, 1.0, device, sub_dir
        )

        weights = model.get_receptive_fields()
        
        # Analyze RF (Visuals only, no Gabor)
        _ = analyze_receptive_fields(weights, patch_size, save_dir=sub_dir)

        results[delta] = {
            'name': name,
            'history': history,
            'weights': weights
        }

    print("\nGenerating Report...")
    analyze_scan_results(results, deltas, main_dir, patch_size)
    
    return results

# ============================================================
# 4. Results Analysis & Plotting
# ============================================================

def analyze_scan_results(results, deltas, save_dir, patch_size):
    """Analyzes metrics across the delta scan."""
    
    summary_data = []
    
    for delta in deltas:
        res = results[delta]
        history = res['history']
        weights = res['weights']
        
        final_recon = history['recon_loss'][-1]
        final_sparse = history['sparse_loss'][-1]
        rf_std = np.std(weights) # Metric for RF Structure
        
        summary_data.append({
            'delta': delta,
            'recon_loss': final_recon,
            'sparse_loss': final_sparse,
            'rf_std': rf_std
        })

    # 1. Save CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(save_dir, 'delta_scan_summary.csv'), index=False)
    print("✓ Summary CSV saved")

    # 2. Plot Trend
    plot_delta_trends(df, save_dir)
    
    # 3. Plot RF Grid
    plot_rf_grid(results, deltas, patch_size, save_dir)

def plot_delta_trends(df, save_dir):
    """Plots metrics vs. Delta."""
    fig, ax1 = plt.subplots(figsize=(8, 5.8))

    color = 'tab:red'
    ax1.set_xlabel('Delta (Dissipation)', fontsize=14)
    ax1.set_ylabel('RF Structure (Std Dev)', color=color, fontsize=14)
    line1 = ax1.plot(df['delta'], df['rf_std'], color=color, marker='o', linewidth=2, label='RF Structure (Std)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Dual axis for Reconstruction Loss
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Reconstruction Loss (MSE)', color=color, fontsize=14)
    line2 = ax2.plot(df['delta'], df['recon_loss'], color=color, marker='s', linestyle='--', label='Recon Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Phase Transition: Structure vs Reconstruction', fontsize=14)
    plt.savefig(os.path.join(save_dir, 'trend_structure_vs_recon.png'), dpi=150)
    plt.close()

def plot_rf_grid(results, deltas, patch_size, save_dir):
    """Plots overview grid of RFs for all Deltas."""
    n_plots = len(deltas)
    cols = 6
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()
    
    for idx, delta in enumerate(deltas):
        ax = axes[idx]
        weights = results[delta]['weights']
        
        # Show 4x4 RFs per subplot
        n_show = 4
        grid_n = 2
        
        canvas = np.zeros((patch_size * grid_n, patch_size * grid_n))
        for i in range(n_show):
            if i < len(weights):
                rf = weights[i].reshape(patch_size, patch_size)
                r = i // grid_n
                c = i % grid_n
                canvas[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = rf
                
        vmax = np.abs(canvas).max()
        ax.imshow(canvas, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f"δ = {delta:.1f}", fontsize=22, fontweight='bold')
        ax.axis('off')
        
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.suptitle("Receptive Field Evolution across Delta", y=1.02, fontsize=16)
    plt.savefig(os.path.join(save_dir, 'scan_rf_grid.png'), bbox_inches='tight', dpi=200)
    plt.close()

# ============================================================
# 5. Execution
# ============================================================

if __name__ == "__main__":
    run_delta_scan(
        start=-0.8, 
        end=5, 
        step=0.2, 
        num_epochs=30,
        base_dataset='cifar10'
    )