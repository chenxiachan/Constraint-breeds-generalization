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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 0. Utility Functions
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def align_encoded_features(encoded: torch.Tensor,
                           patches: torch.Tensor,
                           mode: str = "mean") -> torch.Tensor:
    """
    Aligns DynamicEncoder output dimensions with original patch dimensions.
    encoded: (B, T, F_enc)
    patches: (B, F_raw)
    mode: "mean" | "first" | "truncate"
    """
    B, T, F_enc = encoded.shape
    F_raw = patches.shape[1]

    if F_enc == F_raw:
        return encoded

    if F_enc % F_raw == 0:
        k = F_enc // F_raw
        encoded = encoded.view(B, T, F_raw, k)
        if mode == "mean":
            encoded = encoded.mean(dim=-1).contiguous()
        elif mode == "first":
            encoded = encoded[:, :, :, 0].contiguous()
        else:
            encoded = encoded[:, :, :F_raw, 0].contiguous()
        return encoded

    # Fallback: truncate
    return encoded[:, :, :F_raw].contiguous()


# ============================================================
# 1. Natural Image Dataset
# ============================================================

class NaturalImagePatches(Dataset):
    """
    Extracts patches from natural images for unsupervised learning.
    Uses CIFAR-10 or STL-10 as a proxy for van Hateren dataset.
    """

    def __init__(self,
                 num_patches=10000,
                 patch_size=16,
                 use_whitening=False,
                 dataset='cifar10'):
        """
        Args:
            num_patches: Number of patches to extract
            patch_size: Size of square patch
            use_whitening: Apply ZCA whitening
            dataset: 'cifar10' or 'stl10'
        """
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.use_whitening = use_whitening

        print(f"Loading {dataset} dataset...")
        if dataset == 'cifar10':
            base_dataset = datasets.CIFAR10(
                root='./data', train=True, download=True,
                transform=transforms.ToTensor()
            )
        elif dataset == 'stl10':
            base_dataset = datasets.STL10(
                root='./data', split='unlabeled', download=True,
                transform=transforms.ToTensor()
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        print(f"Extracting {num_patches} patches of size {patch_size}x{patch_size}...")
        self.patches = self._extract_patches(base_dataset)
        self.patches = self._preprocess_patches(self.patches)

        print(f"Dataset ready: {self.patches.shape}")

    def _extract_patches(self, dataset):
        """Randomly extracts patches from dataset."""
        patches = []
        P = self.patch_size

        while len(patches) < self.num_patches:
            idx = np.random.randint(len(dataset))
            img, _ = dataset[idx]

            # Convert to grayscale if RGB
            if img.shape[0] == 3:
                img = img.mean(dim=0)

            h, w = img.shape
            if h < P or w < P:
                continue

            top = np.random.randint(0, h - P + 1)
            left = np.random.randint(0, w - P + 1)

            patch = img[top:top + P, left:left + P]
            patches.append(patch.numpy().astype(np.float32).flatten())

        return np.array(patches[:self.num_patches], dtype=np.float32)

    def _preprocess_patches(self, patches):
        """Preprocessing: Demean + Normalize + Optional Whitening."""
        # 1. Remove mean
        patches = patches - patches.mean(axis=1, keepdims=True)

        # 2. Normalize norm
        norms = np.linalg.norm(patches, axis=1, keepdims=True)
        patches = patches / (norms + 1e-8)

        # 3. Whitening
        if self.use_whitening:
            patches = self._whiten(patches)

        return patches.astype(np.float32)

    def _whiten(self, patches):
        """ZCA Whitening."""
        print("Applying ZCA whitening...")
        cov = np.cov(patches.T)
        U, S, _ = np.linalg.svd(cov)
        epsilon = 1e-5
        ZCA = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
        return patches @ ZCA

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx])
        return patch, patch  # Self-supervised


# ============================================================
# 2. Simplified SNN Autoencoder
# ============================================================

class V1_SNN_Autoencoder(nn.Module):
    """
    SNN Autoencoder for learning V1-like receptive fields.
    Structure: Encoder -> LIF -> Decoder
    """

    def __init__(self, input_dim, hidden_dim, num_steps=10, beta=0.9):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        # Encoder (Learns RFs)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        # Decoder (Reconstruction)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # LIF Neuron
        import snntorch as snn
        from snntorch import surrogate
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """
        Args:
            x: (batch, num_steps, input_dim)
        Returns:
            reconstructed: (batch, input_dim)
            spikes_sum: (batch, hidden_dim) - For sparsity constraint
        """
        mem = self.lif.init_leaky()
        spikes_list = []

        for t in range(self.num_steps):
            x_t = x[:, t, :]
            cur = self.encoder(x_t)
            spk, mem = self.lif(cur, mem)
            spikes_list.append(spk)

        # Accumulate spikes
        spikes_sum = torch.stack(spikes_list, dim=1).sum(dim=1)

        # Decode
        reconstructed = self.decoder(spikes_sum)

        return reconstructed, spikes_sum

    def get_receptive_fields(self):
        """Returns learned weights (RFs)."""
        return self.encoder.weight.data.detach().cpu().numpy()


# ============================================================
# 3. Training Function
# ============================================================

def train_v1_model(model,
                   dataloader,
                   encoder_func,
                   delta,
                   num_epochs=50,
                   lambda_sparse=0.1,
                   device='cuda',
                   enc_align_mode: str = "mean",
                   save_dir='./results'):
    """
    Trains V1 model to learn receptive fields.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()

    history = {
        'epoch': [], 'recon_loss': [], 'sparse_loss': [], 'total_loss': []
    }

    ensure_dir(save_dir)
    csv_path = os.path.join(save_dir, 'training_history.csv')
    
    for epoch in range(num_epochs):
        model.train()
        ep_recon, ep_sparse, ep_total = 0.0, 0.0, 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (patches, _) in enumerate(pbar):
            patches = patches.to(device)

            # 1. Dynamic Encoding
            encoded = encoder_func.encode(patches, delta, device=device)
            encoded = align_encoded_features(encoded, patches, mode=enc_align_mode)

            # 2. Forward
            reconstructed, spikes_sum = model(encoded)

            # 3. Loss
            recon_loss = recon_criterion(reconstructed, patches)
            sparse_loss = lambda_sparse * torch.mean(torch.abs(spikes_sum))
            total_loss = recon_loss + sparse_loss

            # 4. Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log
            ep_recon += recon_loss.item()
            ep_sparse += sparse_loss.item()
            ep_total += total_loss.item()

            pbar.set_postfix({'recon': f'{recon_loss.item():.4f}', 'sparse': f'{sparse_loss.item():.4f}'})

        # Epoch Stats
        n_batches = len(dataloader)
        avg_recon = ep_recon / n_batches
        avg_sparse = ep_sparse / n_batches
        avg_total = ep_total / n_batches
        
        history['epoch'].append(epoch + 1)
        history['recon_loss'].append(avg_recon)
        history['sparse_loss'].append(avg_sparse)
        history['total_loss'].append(avg_total)

        print(f"Epoch {epoch+1}: Recon={avg_recon:.6f}, Sparse={avg_sparse:.6f}")

    # Save History
    df_history = pd.DataFrame(history)
    df_history.to_csv(csv_path, index=False)
    print(f"✓ Training history saved to {csv_path}")

    return history


# ============================================================
# 4. Analysis Tools
# ============================================================

def analyze_receptive_fields(weights, patch_size, save_dir='./rf_analysis'):
    """Visualizes learned receptive fields."""
    ensure_dir(save_dir)

    print("Visualizing receptive fields...")
    visualize_receptive_fields(weights, patch_size,
                               save_path=f'{save_dir}/receptive_fields.png')

    print(f"Analysis complete. Results saved to {save_dir}")
    return None # No Gabor params returned


def visualize_receptive_fields(weights, patch_size, n_display=64, save_path=None):
    """Visualizes RF weights (Grid plot)."""
    n_display = min(n_display, weights.shape[0])
    grid_size = int(np.ceil(np.sqrt(n_display)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(n_display):
        rf = weights[i].reshape(patch_size, patch_size)
        vmax = np.abs(rf).max()
        axes[i].imshow(rf, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i].axis('off')

    for i in range(n_display, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Learned Receptive Fields', fontsize=16)
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Receptive fields saved to {save_path}")

    plt.close(fig)


# ============================================================
# 5. Result Comparison
# ============================================================

def compare_results(results, patch_size, save_dir='./results'):
    """Compares results across different deltas (RF visualization)."""
    n_cols = len(results)
    fig_rfs, axes_rfs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes_rfs = [axes_rfs]

    # Display RFs
    for idx, (delta, result) in enumerate(results.items()):
        weights = result['weights']
        n_display = min(16, weights.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_display)))

        ax = axes_rfs[idx]
        ax.set_title(f'Delta={delta}')
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

    ensure_dir(save_dir)
    fig_rfs.tight_layout()
    fig_rfs.savefig(f'{save_dir}/comparison_rfs.png', dpi=150)
    plt.close(fig_rfs)

    # Save CSV
    save_comparison_csv(results, save_dir)


def save_comparison_csv(results, save_dir):
    """Saves comparison results to CSV (Basic Metrics)."""
    comparison_data = []
    
    for delta, result in results.items():
        history = result['history']
        weights = result['weights']
        
        # Metrics
        final_recon = history['recon_loss'][-1]
        final_sparse = history['sparse_loss'][-1]
        final_total = history['total_loss'][-1]
        
        # RF Stats
        rf_std = np.std(weights)
        rf_mean = np.mean(np.abs(weights))
        
        comparison_data.append({
            'delta': delta,
            'final_recon_loss': final_recon,
            'final_sparse_loss': final_sparse,
            'final_total_loss': final_total,
            'rf_std': rf_std,
            'rf_mean_abs': rf_mean
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    csv_path = os.path.join(save_dir, 'comparison_summary.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison summary saved to {csv_path}")
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(df_comparison.to_string(index=False))
    print("=" * 60)


# ============================================================
# 6. Main Experiment
# ============================================================

def run_v1_experiment(delta_values=[10.0, 0.0, -1.5],
                      patch_size=16,
                      hidden_dim=128,
                      num_patches=5000,
                      num_epochs=30,
                      num_steps=10,
                      batch_size=64,
                      enc_align_mode="mean",
                      base_dataset='cifar10',
                      use_whitening=False):
    """
    Runs full V1 RF learning experiment.
    """
    from core.encoding_wrapper import DynamicEncoder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_results_dir = f'./results/experiment_{timestamp}'
    ensure_dir(main_results_dir)
    
    config = {
        'timestamp': timestamp,
        'delta_values': delta_values,
        'patch_size': patch_size,
        'hidden_dim': hidden_dim,
        'num_patches': num_patches,
        'num_epochs': num_epochs,
        'num_steps': num_steps,
        'batch_size': batch_size,
        'enc_align_mode': enc_align_mode,
        'base_dataset': base_dataset,
        'use_whitening': use_whitening,
        'device': str(device)
    }
    
    config_path = os.path.join(main_results_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Experiment config saved to {config_path}")

    # 1. Prepare Data
    print("\n" + "=" * 60)
    print("Preparing Natural Image Patches Dataset")
    print("=" * 60)

    dataset = NaturalImagePatches(
        num_patches=num_patches,
        patch_size=patch_size,
        use_whitening=use_whitening,
        dataset=base_dataset
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 2. Create Encoder
    encoder = DynamicEncoder(num_steps=num_steps, tmax=8.0)

    # 3. Run for each delta
    results = {}

    for delta in delta_values:
        print(f"\n" + "=" * 60)
        try:
            enc_name = encoder.get_encoding_name(delta)
        except Exception:
            enc_name = f"delta={delta}"
        print(f"Training with delta={delta} ({enc_name})")
        print("=" * 60)

        input_dim = patch_size * patch_size
        model = V1_SNN_Autoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            beta=0.9
        )

        save_dir = os.path.join(main_results_dir, f'delta_{delta:.1f}')
        ensure_dir(save_dir)

        # Train
        history = train_v1_model(
            model=model,
            dataloader=dataloader,
            encoder_func=encoder,
            delta=delta,
            num_epochs=num_epochs,
            lambda_sparse=0.1,
            device=device,
            enc_align_mode=enc_align_mode,
            save_dir=save_dir
        )

        # Analyze RFs
        weights = model.get_receptive_fields()
        _ = analyze_receptive_fields(weights, patch_size, save_dir=save_dir)

        # Save Results
        results[delta] = {
            'model': model,
            'history': history,
            'weights': weights,
            'gabor_params': None # Removed
        }

        # Plot Curves
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['recon_loss'], label='Reconstruction Loss')
        plt.title(f'Training Loss (delta={delta})')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history['sparse_loss'], label='Sparsity Loss')
        plt.title(f'Sparsity Loss (delta={delta})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_curves.png', dpi=150)
        plt.close()

    # 4. Compare Results
    print("\n" + "=" * 60)
    print("Comparing Results Across Different Deltas")
    print("=" * 60)

    compare_results(results, patch_size, save_dir=main_results_dir)
    generate_final_report(results, main_results_dir, config)

    print(f"\n✓ All results saved to: {main_results_dir}")
    
    return results


def generate_final_report(results, save_dir, config):
    """Generates final experiment report."""
    report_path = os.path.join(save_dir, 'EXPERIMENT_REPORT.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("V1 RECEPTIVE FIELD LEARNING EXPERIMENT - FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in config.items():
            f.write(f"{key:20s}: {value}\n")
        f.write("\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 80 + "\n")
        
        for delta, result in results.items():
            f.write(f"\nDelta = {delta}:\n")
            history = result['history']
            weights = result['weights']
            
            f.write(f"  Final Recon Loss:    {history['recon_loss'][-1]:.6f}\n")
            f.write(f"  Final Sparsity Loss: {history['sparse_loss'][-1]:.6f}\n")
            f.write(f"  RF Std Dev:          {np.std(weights):.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FILES GENERATED:\n")
        f.write("-" * 80 + "\n")
        f.write("  - experiment_config.json: Experiment configuration\n")
        f.write("  - comparison_summary.csv: Summary metrics\n")
        f.write("  - comparison_rfs.png: RF comparison\n")
        f.write("  - delta_X.X/: Individual results\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Final report saved to {report_path}")


# ============================================================
# 7. Main Entry
# ============================================================

if __name__ == "__main__":
    results = run_v1_experiment(
        delta_values=[10.0, 0.0, -1.5],  # Dissipative, Critical, Expansive
        patch_size=16,
        hidden_dim=128,
        num_patches=5000,
        num_epochs=1,
        num_steps=5,
        batch_size=64,
        enc_align_mode="mean",
        base_dataset='cifar10',
        use_whitening=False
    )
