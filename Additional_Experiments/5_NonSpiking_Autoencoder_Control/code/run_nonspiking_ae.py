"""
ExpF — Non-spiking (ReLU) autoencoder control for Experiment 2 (receptive-field
emergence).

Reviewer question addressed (NeurIPS Reviewer oBwj):
    "It is also not clear why SNNs are necessary for this experiment [Exp 2].
     Could similar receptive-field structure be obtained with a feedforward
     autoencoder or an RNN using comparable temporal prefilterings?"

Design — change exactly ONE variable
------------------------------------
We take the Experiment-2 receptive-field-learning pipeline verbatim (same
Duffing/static encodings, same CIFAR-10 patches, same training protocol, same
loss, same RF metrics) and swap ONLY the neuron:

  * Original  V1_SNN_Autoencoder : Linear(256,128) -> LIF(beta=0.9) unrolled over
                                   T timesteps -> decode from Sum_t spk(t).
  * New       V1_ReLU_Autoencoder: Linear(256,128) -> ReLU  unrolled over T
                                   timesteps -> decode from Sum_t a(t).
    (no membrane potential, no leak, no threshold; read-out aligned with the
     original: x_recon = W_dec . Sum_t a(t).)

Everything else — encoder/decoder shapes, forward-over-time structure, the
reconstruction + sparsity loss with lambda=0.1, and the RF metrics
(sigma_RF = std of encoder weights; OSI = 2D-FFT orientation selectivity) — is
IMPORTED and reused, not reimplemented, from the authoritative feature-
quality script.

Sources reused verbatim (imported, not copied):
    core.v1_receptive_field_learning : NaturalImagePatches, V1_SNN_Autoencoder,
                                        align_encoded_features, ensure_dir
    run_feature_quality              : train_v1_with_encoder (training loop +
                                        loss), compute_osi (OSI metric),
                                        BaselineEncoder, DynamicEncoder
                                        (self-contained GPU-RK4 Duffing encoder)

Only V1_ReLU_Autoencoder and the driver/CLI below are new.

Conditions (relu, full run): Baseline (static), Dynamic_Transition (delta=2.0),
Dynamic_Expansive (delta=-1.5), Dynamic_Dissipative (delta=10.0); 3 seeds each
= 12 models. Plus an SNN anchor (`--model snn`) on Dynamic_Transition x 3 seeds
= 3 models, to verify the imported pipeline reproduces the original result.

Usage
-----
    # Smoke test (1 condition x 1 seed x 2 epochs x 500 patches, <5 min):
    MPLBACKEND=Agg python run_nonspiking_ae.py --smoke

    # Full ReLU run (12 models):
    MPLBACKEND=Agg python run_nonspiking_ae.py --model relu

    # SNN reproducibility anchor (Transition x 3 seeds):
    MPLBACKEND=Agg python run_nonspiking_ae.py --model snn

Checkpointed: each finished model writes its weights + a CSV row immediately, so
an interrupted run resumes by skipping models already on disk.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader


# ------------------------------------------------------------------
# Paths (mirror ExpC_Exp2_Heldout/code/run_heldout_eval.py)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_FULL = os.path.normpath(os.path.join(BASE_DIR, "..", "output"))

REPO_DIR = os.path.join(BASE_DIR, "..", "..", "..")
EXP2_CODE_DIR = os.path.join(REPO_DIR, "1_Experiment 2", "code")
RFQ_CODE_DIR = os.path.join(BASE_DIR, "..", "..", "8_Feature_Quality_OSI", "code")

# CIFAR-10 root: set CIFAR_DATA_ROOT to a pre-downloaded cache (must contain
# cifar-10-batches-py/); otherwise falls back to ./data with download=True.
DATA_ROOT = os.environ.get("CIFAR_DATA_ROOT", "./data")

# EXP2 core/ path first (authoritative model + dataset), then feature-quality dir.
sys.path.insert(0, EXP2_CODE_DIR)
sys.path.insert(0, RFQ_CODE_DIR)

from core.v1_receptive_field_learning import (  # noqa: E402
    NaturalImagePatches,
    V1_SNN_Autoencoder,
    align_encoded_features,   # noqa: F401  (used indirectly by train loop)
    ensure_dir,
)
import run_feature_quality as rfq  # noqa: E402


# ------------------------------------------------------------------
# Experiment constants (identical to the feature-quality config)
# ------------------------------------------------------------------
PATCH_SIZE = 16
INPUT_DIM = PATCH_SIZE * PATCH_SIZE   # 256
HIDDEN_DIM = 128
NUM_STEPS = 5
BETA = 0.9                            # SNN anchor leak (matches the original config)
BATCH_SIZE = 64
LAMBDA_SPARSE = 0.1
NUM_PATCHES_FULL = 5000
NUM_EPOCHS_FULL = 30
SEEDS_FULL = [0, 1, 2]
DEVICE = torch.device("cpu")

# Condition -> delta (None for the static Baseline). Mirrors
# run_feature_quality.run_ablation_study delta assignment.
ALL_CONDITIONS = [
    ("Baseline", None),
    ("Dynamic_Transition", 2.0),
    ("Dynamic_Expansive", -1.5),
    ("Dynamic_Dissipative", 10.0),
]

CSV_COLUMNS = ["model", "condition", "seed", "delta",
               "sigma_rf", "osi_mean", "final_recon", "final_sparsity"]


# ==================================================================
# NEW: Non-spiking ReLU autoencoder
# ==================================================================
class V1_ReLU_Autoencoder(torch.nn.Module):
    """Non-spiking control for V1_SNN_Autoencoder.

    Identical architecture and forward-over-time structure as
    core.v1_receptive_field_learning.V1_SNN_Autoencoder, with the LIF neuron
    replaced by a plain ReLU: NO membrane potential, NO leak, NO threshold.
    The decoder reads out from the temporal SUM of ReLU activations, matching
    the original's read-out from the temporal sum of spikes
    (x_recon = W_dec . Sum_t a(t)).
    """

    def __init__(self, input_dim, hidden_dim, num_steps=10, beta=0.9):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        # beta accepted for signature parity with V1_SNN_Autoencoder; unused
        # (a ReLU has no leak).

        # Encoder (learns RFs) and decoder — same as the SNN version.
        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, num_steps, input_dim)
        Returns:
            reconstructed: (batch, input_dim)
            acts_sum:      (batch, hidden_dim) — for the sparsity constraint,
                           the ReLU analogue of the SNN's spikes_sum.
        """
        acts_list = []
        for t in range(self.num_steps):
            x_t = x[:, t, :]
            cur = self.encoder(x_t)
            a = torch.relu(cur)          # memoryless: no mem, no leak, no thresh
            acts_list.append(a)

        # Accumulate activations over time (mirrors spikes_sum in the SNN).
        acts_sum = torch.stack(acts_list, dim=1).sum(dim=1)

        # Decode from the temporal sum.
        reconstructed = self.decoder(acts_sum)

        return reconstructed, acts_sum

    def get_receptive_fields(self):
        """Returns learned encoder weights (RFs)."""
        return self.encoder.weight.data.detach().cpu().numpy()


# ==================================================================
# Cached-CIFAR patch dataset (reuse authoritative extraction/preprocessing)
# ==================================================================
class CachedPatches(NaturalImagePatches):
    """Same as NaturalImagePatches but reads the CIFAR-10 cache at root=DATA_ROOT
    (CIFAR_DATA_ROOT env var, or ./data with automatic download).

    _extract_patches and _preprocess_patches are inherited unchanged — the
    authoritative pipeline (RGB->grayscale mean, per-patch demean, L2-norm; no
    whitening, matching use_whitening=False in the original config).
    """

    def __init__(self, num_patches, patch_size, data_root):
        from torchvision import datasets, transforms
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.use_whitening = False
        base = datasets.CIFAR10(root=data_root, train=True, download=True,
                                transform=transforms.ToTensor())
        self.patches = self._extract_patches(base)
        self.patches = self._preprocess_patches(self.patches)


def make_dataloader(num_patches, seed):
    """Build a reproducible patch dataloader for a given seed."""
    np.random.seed(seed)   # controls np.random.randint inside _extract_patches
    ds = CachedPatches(num_patches=num_patches, patch_size=PATCH_SIZE,
                       data_root=DATA_ROOT)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# ==================================================================
# Model / encoder factories
# ==================================================================
def build_model(model_type):
    if model_type == "relu":
        return V1_ReLU_Autoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                                   num_steps=NUM_STEPS, beta=BETA)
    if model_type == "snn":
        return V1_SNN_Autoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                                  num_steps=NUM_STEPS, beta=BETA)
    raise ValueError(model_type)


def build_encoder(name):
    """Reconstruct the exact encoder object used at training time.
    Mirrors run_feature_quality.run_ablation_study."""
    if name == "Baseline":
        return rfq.BaselineEncoder(num_steps=NUM_STEPS)
    if name.startswith("Dynamic_"):
        return rfq.DynamicEncoder(num_steps=NUM_STEPS, tmax=8.0)
    raise ValueError(name)


# ==================================================================
# CSV checkpoint helpers
# ==================================================================
def load_results(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rows = df.to_dict("records")
    else:
        rows = []
    done = {(str(r["model"]), str(r["condition"]), int(r["seed"])) for r in rows}
    return rows, done


def save_results(rows, csv_path):
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)


# ==================================================================
# RF comparison figure (reuse the original inset-grid RF rendering)
# ==================================================================
def plot_rf_comparison(weights_by_condition, patch_size, save_path):
    """One panel per condition, each showing a 4x4 grid of learned RFs.
    Rendering mirrors run_feature_quality.compare_all_encoders_with_logging.
    Panel width kept at 3.0in @ dpi=150 so the longest edge stays < 2000px.
    """
    names = list(weights_by_condition.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2))
    if n == 1:
        axes = [axes]

    for idx, name in enumerate(names):
        weights = weights_by_condition[name]
        n_display = min(16, weights.shape[0])
        grid_size = int(np.ceil(np.sqrt(n_display)))

        ax = axes[idx]
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")

        for i in range(n_display):
            rf = weights[i].reshape(patch_size, patch_size)
            vmax = np.abs(rf).max()
            row = i // grid_size
            col = i % grid_size
            gs = 1.0 / grid_size
            left = col * gs
            bottom = 1.0 - (row + 1) * gs
            inset = ax.inset_axes([left, bottom, gs, gs])
            inset.imshow(rf, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            inset.axis("off")

    plt.suptitle("Learned Receptive Fields (ReLU control vs. dynamics)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved", save_path)


# ==================================================================
# Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="ExpF: non-spiking AE control for Exp 2")
    parser.add_argument("--model", choices=["relu", "snn"], default="relu",
                        help="relu = non-spiking control (12 models); "
                             "snn = reproduction anchor on Transition (3 models)")
    parser.add_argument("--smoke", action="store_true",
                        help="1 condition (Transition) x 1 seed x 2 epochs x 500 patches")
    args = parser.parse_args()

    model_type = args.model

    # ---- run plan ----
    if args.smoke:
        conditions = [("Dynamic_Transition", 2.0)]
        seeds = [0]
        num_epochs = 2
        num_patches = 500
        out_dir = os.path.join(OUTPUT_DIR_FULL, "smoke")
    else:
        seeds = SEEDS_FULL
        num_epochs = NUM_EPOCHS_FULL
        num_patches = NUM_PATCHES_FULL
        out_dir = OUTPUT_DIR_FULL
        if model_type == "snn":
            # Reproducibility anchor: Transition only.
            conditions = [("Dynamic_Transition", 2.0)]
        else:
            conditions = ALL_CONDITIONS

    ensure_dir(out_dir)
    weights_dir = os.path.join(out_dir, "weights")
    ensure_dir(weights_dir)
    csv_path = os.path.join(out_dir, "nonspiking_ae_results.csv")

    print("=" * 70)
    print(f"ExpF: non-spiking AE control  |  model={model_type}  smoke={args.smoke}")
    print(f"  conditions={[c for c, _ in conditions]}")
    print(f"  seeds={seeds}  epochs={num_epochs}  patches={num_patches}")
    print(f"  out_dir={out_dir}")
    print("=" * 70)

    rows, done = load_results(csv_path)

    # Loop seeds outer so patches are shared across conditions within a seed.
    for seed in seeds:
        dataloader = None  # lazily build once we know a job is pending
        for cond_name, delta in conditions:
            key = (model_type, cond_name, seed)
            wpath = os.path.join(weights_dir, f"{model_type}_{cond_name}_seed{seed}.pth")
            if key in done and os.path.exists(wpath):
                print(f"[skip] {model_type} {cond_name} seed{seed} (already done)")
                continue

            if dataloader is None:
                print(f"\n[data] building {num_patches} patches for seed {seed} ...")
                dataloader = make_dataloader(num_patches, seed)

            print(f"\n[train] {model_type} | {cond_name} (delta={delta}) | seed {seed}")
            torch.manual_seed(seed)          # model init + encoder stochasticity
            model = build_model(model_type)
            encoder = build_encoder(cond_name)

            save_dir = os.path.join(out_dir, "logs", f"{model_type}_{cond_name}_seed{seed}")
            ensure_dir(save_dir)

            # Reuse the authoritative training loop + loss (recon + lambda*|acts|).
            history = rfq.train_v1_with_encoder(
                model=model,
                dataloader=dataloader,
                encoder=encoder,
                delta=delta,
                num_epochs=num_epochs,
                lambda_sparse=LAMBDA_SPARSE,
                device=DEVICE,
                enc_align_mode="mean",
                save_dir=save_dir,
            )

            # ---- metrics (imported / reused) ----
            weights = model.get_receptive_fields()
            sigma_rf = float(np.std(weights))
            _, osi_mean, _ = rfq.compute_osi(weights, PATCH_SIZE)
            final_recon = float(history["recon_loss"][-1])
            final_sparsity = float(history["sparse_loss"][-1])

            # ---- checkpoint: weights + CSV row immediately ----
            torch.save(model.state_dict(), wpath)
            rows.append({
                "model": model_type,
                "condition": cond_name,
                "seed": seed,
                "delta": delta if delta is not None else "",
                "sigma_rf": sigma_rf,
                "osi_mean": osi_mean,
                "final_recon": final_recon,
                "final_sparsity": final_sparsity,
            })
            done.add(key)
            save_results(rows, csv_path)
            print(f"  -> sigma_rf={sigma_rf:.6f}  osi_mean={osi_mean:.6f}  "
                  f"recon={final_recon:.6f}  sparsity={final_sparsity:.6f}")
            print(f"  -> saved weights {wpath}")
            print(f"  -> updated CSV   {csv_path}")

    # ---- RF comparison figure (reference seed = first seed) ----
    ref_seed = seeds[0]
    weights_by_condition = {}
    for cond_name, _ in conditions:
        wpath = os.path.join(weights_dir, f"{model_type}_{cond_name}_seed{ref_seed}.pth")
        if not os.path.exists(wpath):
            continue
        m = build_model(model_type)
        m.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        weights_by_condition[cond_name] = m.get_receptive_fields()

    if weights_by_condition:
        fig_path = os.path.join(out_dir, f"rf_comparison_{model_type}.png")
        plot_rf_comparison(weights_by_condition, PATCH_SIZE, fig_path)

    print("\nDone.")
    print("Results CSV:", csv_path)


if __name__ == "__main__":
    main()
