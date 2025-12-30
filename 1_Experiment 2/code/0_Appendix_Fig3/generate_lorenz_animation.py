#!/usr/bin/env python3
"""
generate_lorenz_animation.py

Train SNN models with Lorenz System dynamics (Scanning rho) 
to capture Receptive Field evolution and generate GIFs.

System:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Parameters:
    sigma = 10.0
    beta = 8.0/3.0
    rho = [Scan]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from v1_receptive_field_learning import (
    NaturalImagePatches,
    V1_SNN_Autoencoder,
    ensure_dir,
    align_encoded_features
)

# ============================================================
# Lorenz Dynamics on GPU
# ============================================================

def lorenz_derivatives_gpu(state, sigma, rho, beta):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)

def lorenz_rk4_gpu(x0, sigma=10.0, rho=28.0, beta=8.0/3.0, 
                   tmax=8.0, num_steps=5):
    device = x0.device
    B, F = x0.shape
    h = 0.01 # Standard step size for Lorenz
    n_internal = int(tmax / h)
    sample_every = n_internal // num_steps
    
    # Initialize state (3D) to match Duffing scheme style
    # x = input, y = 0.2 * input, z = -input
    state = torch.zeros(B, F, 3, device=device, dtype=torch.float32)
    state[:, :, 0] = x0
    state[:, :, 1] = 0.2 * x0 
    state[:, :, 2] = -x0
    
    trajectory = torch.zeros(B, num_steps, F, 3, device=device, dtype=torch.float32)
    sample_idx = 0
    
    for step in range(n_internal):
        k1 = h * lorenz_derivatives_gpu(state, sigma, rho, beta)
        k2 = h * lorenz_derivatives_gpu(state + k1/2, sigma, rho, beta)
        k3 = h * lorenz_derivatives_gpu(state + k2/2, sigma, rho, beta)
        k4 = h * lorenz_derivatives_gpu(state + k3, sigma, rho, beta)
        state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        if (step + 1) % sample_every == 0 and sample_idx < num_steps:
            trajectory[:, sample_idx] = state
            sample_idx += 1
            
    return trajectory

class LorenzEncoder:
    def __init__(self, num_steps=5, tmax=8.0):
        self.num_steps = num_steps
        self.tmax = tmax

    def encode(self, data, rho, device='cuda'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)
        
        data_max = torch.abs(data).max()
        if data_max > 1e-8:
            data_norm = data / (data_max + 1e-8)
        else:
            data_norm = data
        
        trajectory = lorenz_rk4_gpu(
            data_norm, 
            sigma=10.0, rho=rho, beta=8.0/3.0,
            tmax=self.tmax, num_steps=self.num_steps
        )
        
        B, T, F, D = trajectory.shape
        encoded = trajectory.reshape(B, T, F * D)
        return encoded.detach()

# ============================================================
# Training & Animation Util
# ============================================================

def train_and_capture(model, train_loader, num_epochs, lambda_sparse, device, save_dir, title_prefix):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()

    history_weights = []
    
    # Capture initial
    history_weights.append(model.get_receptive_fields().copy())

    pbar = tqdm(range(num_epochs), desc=f"Training {title_prefix}", leave=False)
    for epoch in pbar:
        model.train()
        avg_loss = 0
        
        for encoded_batch, target_batch in train_loader:
            reconstructed, spikes_sum = model(encoded_batch)
            recon_loss = recon_criterion(reconstructed, target_batch)
            sparse_loss = lambda_sparse * torch.mean(torch.abs(spikes_sum))
            total_loss = recon_loss + sparse_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            avg_loss += total_loss.item()
            
        history_weights.append(model.get_receptive_fields().copy())
        pbar.set_postfix({'Loss': avg_loss / len(train_loader)})
        
    return history_weights

def create_lorenz_animation(all_histories, save_path, patch_size=16, fps=4):
    print(f"Generating Lorenz animation: {save_path}")
    
    n_epochs = len(all_histories[0][1])
    n_scenarios = len(all_histories)
    
    # Layout: 2 rows x 4 cols = 8 scenarios (1 empty)
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
    axes = axes.flatten()
    
    n_rf_show = 16 
    rf_grid_dim = 4
    
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.1, hspace=0.2)
    
    def update(frame):
        for idx in range(len(axes)):
            ax = axes[idx]
            ax.clear()
            
            if idx < n_scenarios:
                name, history = all_histories[idx]
                weights = history[frame]
                
                canvas = np.zeros((patch_size * rf_grid_dim, patch_size * rf_grid_dim))
                
                for i in range(n_rf_show):
                    if i < len(weights):
                        rf = weights[i].reshape(patch_size, patch_size)
                        r = i // rf_grid_dim
                        c = i % rf_grid_dim
                        canvas[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = rf
                
                vmax = np.abs(canvas).max() + 1e-5
                ax.imshow(canvas, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax.set_title(name, fontsize=15, fontweight='bold')
            
            ax.axis('off')
            
        plt.suptitle(f"Lorenz System RF Evolution - Epoch {frame}", fontsize=18, y=0.98)
        
    ani = animation.FuncAnimation(fig, update, frames=n_epochs, interval=200)
    ani.save(save_path, writer='pillow', fps=fps)
    
    # Save last frame as PNG
    png_path = save_path.replace('.gif', '.png')
    update(n_epochs - 1) # Render last frame
    plt.savefig(png_path)
    print(f"Saved final frame to: {png_path}")
    
    plt.close()

def run_lorenz_experiment():
    # Settings
    patch_size = 16
    hidden_dim = 128
    num_epochs = 30
    batch_size = 64
    num_steps = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    dataset = NaturalImagePatches(num_patches=2000, patch_size=patch_size, dataset='cifar10')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'./results/lorenz_{timestamp}'
    ensure_dir(save_dir)
    
    # Scan params (rho)
    # 10 points low rho scan: 0.1 to 10.0
    rhos = [0.1, 0.5, 1.0, 1.2, 1.5, 3.0, 5.0, 28.0]
    
    encoder = LorenzEncoder(num_steps=num_steps, tmax=8.0)
    all_histories = []

    for rho in rhos:
        name = f"Lorenz (ρ={rho})"
        print(f"\n=== Processing: {name} ===")
        
        # Precompute
        input_dim = patch_size**2
        encoded_list = []
        targets_list = []
        
        with torch.no_grad():
            for patches, _ in dataloader:
                patches = patches.to(device)
                encoded = encoder.encode(patches, rho=rho, device=device)
                
                if encoded.shape[-1] != input_dim:
                    encoded = align_encoded_features(encoded, patches, mode='mean')
                
                encoded_list.append(encoded)
                targets_list.append(patches)
                
        full_encoded = torch.cat(encoded_list, dim=0)
        full_targets = torch.cat(targets_list, dim=0)
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(full_encoded, full_targets), 
            batch_size=batch_size, shuffle=True
        )
        
        model = V1_SNN_Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, num_steps=num_steps)
        
        weights_history = train_and_capture(
            model, train_loader, num_epochs, 0.1, device, save_dir, name
        )
        
        all_histories.append((name, weights_history))
        
    create_lorenz_animation(
        all_histories, 
        os.path.join(save_dir, 'lorenz_evolution.gif'),
        patch_size=patch_size
    )

if __name__ == "__main__":
    run_lorenz_experiment()
