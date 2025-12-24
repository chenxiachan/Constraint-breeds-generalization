"""
encoding_wrapper.py - Dynamic Encoding Wrapper
Uses mixed_oscillator_encode directly from encoding.py
"""
import torch
import numpy as np
from .encoding import mixed_oscillator_encode


class DynamicEncoder:
    """
    Dynamic Encoder - Uses Duffing oscillators (mixed oscillator) for encoding.
    """

    def __init__(self, num_steps=5, tmax=8.0):
        """
        Args:
            num_steps: Number of time steps
            tmax: Evolution time
        """
        self.num_steps = num_steps
        self.tmax = tmax

    def encode(self, data, delta, device='cpu'):
        """
        Encodes data using the specified delta parameter.

        Args:
            data: (batch, features) - Input data
            delta: float - Duffing dissipation parameter
            device: torch device

        Returns:
            encoded: (batch, num_steps, features*3) - Encoded time-series
        """
        # Ensure input is a tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        # Move to CPU (encoding logic uses numpy)
        data_cpu = data.cpu()

        # Set parameters (Duffing oscillator)
        params = {
            'alpha': 2.0,    # Linear restoring force
            'beta': 0.1,     # Nonlinear coefficient
            'delta': delta,  # Dissipation/Injection (Core control param)
            'gamma': 0.1,    # Coupling coefficient
            'omega': 1.0,    # Natural frequency
            'drive': 0.0     # Drive force
        }

        # Call the encoding function
        encoded = mixed_oscillator_encode(
            data_cpu,
            num_steps=self.num_steps,
            tmax=self.tmax,
            params=params
        )

        # Ensure correct dtype/device, detach to block gradients
        # Note: Encoding itself does not require gradients
        encoded = encoded.detach().clone().to(device).float()
        encoded.requires_grad_(False)

        return encoded

    def get_encoding_name(self, delta):
        """Returns the name of the encoding mode."""
        if delta < -0.5:
            return "expansive"
        elif delta > 5.0:
            return "dissipative"
        else:
            return "transition"


class EncodingConfig:
    """Pre-defined encoding configurations."""

    # Standard configurations
    EXPANSIVE = -1.5      # Expansive mode: High performance, high energy
    DISSIPATIVE = 10.0    # Dissipative mode: Low energy, stable
    TRANSITION = 2.0      # Transition mode: Intermediate state
    CRITICAL = 0.0


    # Full test range
    TEST_RANGE = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    @classmethod
    def get_config_name(cls, delta):
        """Get configuration name."""
        if abs(delta - cls.EXPANSIVE) < 0.1:
            return "expansive"
        elif abs(delta - cls.DISSIPATIVE) < 0.1:
            return "dissipative"
        elif abs(delta - cls.TRANSITION) < 0.1:
            return "transition"
        else:
            return f"delta_{delta:.1f}"


def create_synthetic_dataset(n_samples=1000, n_features=20, n_classes=3,
                             noise_level=0.2, random_state=42):
    """
    Creates synthetic dataset for testing.

    Args:
        n_samples: Number of samples
        n_features: Feature dimensions
        n_classes: Number of classes
        noise_level: Noise level
        random_state: Random seed

    Returns:
        X: (n_samples, n_features) - Features
        y: (n_samples,) - Labels
    """
    np.random.seed(random_state)

    X = []
    y = []

    samples_per_class = n_samples // n_classes

    for class_idx in range(n_classes):
        # Create class center
        center = np.random.randn(n_features) * 2

        # Generate samples (Gaussian)
        class_samples = center + np.random.randn(samples_per_class, n_features) * noise_level

        X.append(class_samples)
        y.append(np.full(samples_per_class, class_idx))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Normalize to [-1, 1]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X = np.clip(X, -3, 3) / 3

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return torch.FloatTensor(X), torch.LongTensor(y)


def visualize_encoding(encoder, sample_data, delta_values, save_path=None):
    """
    Visualizes encoding results for different deltas.

    Args:
        encoder: DynamicEncoder instance
        sample_data: (1, features) - Single sample
        delta_values: list - Deltas to visualize
        save_path: Optional save path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(delta_values), 1, figsize=(12, 3*len(delta_values)))

    if len(delta_values) == 1:
        axes = [axes]

    for idx, delta in enumerate(delta_values):
        encoded = encoder.encode(sample_data, delta)

        # Flatten time and feature dimensions
        encoded_2d = encoded[0].cpu().numpy()  # (num_steps, features*3)

        # Visualize
        im = axes[idx].imshow(encoded_2d.T, aspect='auto', cmap='viridis')
        axes[idx].set_title(f'Delta = {delta:.1f} ({encoder.get_encoding_name(delta)})')
        axes[idx].set_xlabel('Time Step')
        axes[idx].set_ylabel('Feature Dimension')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test Encoder
    print("=" * 50)
    print("Testing Dynamic Encoder")
    print("=" * 50)

    encoder = DynamicEncoder(num_steps=5, tmax=8.0)
    test_data = torch.randn(4, 10)  # (batch=4, features=10)

    # Test 3 modes
    for delta in [EncodingConfig.EXPANSIVE, EncodingConfig.TRANSITION, EncodingConfig.DISSIPATIVE]:
        encoded = encoder.encode(test_data, delta)
        print(f"\nDelta = {delta} ({encoder.get_encoding_name(delta)})")
        print(f"  Input shape: {test_data.shape}")
        print(f"  Encoded shape: {encoded.shape}")
        print(f"  Range: [{encoded.min():.2f}, {encoded.max():.2f}]")

    # Test Synthetic Dataset
    print("\n" + "=" * 50)
    print("Testing Synthetic Dataset")
    print("=" * 50)

    X, y = create_synthetic_dataset(n_samples=300, n_features=10, n_classes=3)
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {torch.bincount(y)}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")

    # Visualize Encoding
    try:
        sample = X[0:1]  # (1, features)
        visualize_encoding(
            encoder,
            sample,
            delta_values=[EncodingConfig.EXPANSIVE, EncodingConfig.DISSIPATIVE],
            save_path='encoding_comparison.png'
        )
    except Exception as e:
        print(f"\nVisualization skipped (matplotlib missing): {e}")