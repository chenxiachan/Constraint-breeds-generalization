"""Cross-encoding generalization test comparing SNN, ANN, RNN, and LSTM."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import csv
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# --- Import models and encoder ---
from core.model_dup import SimpleSNN, SimpleANN, RecurrentNet, TemporalANN_Avg, TemporalANN_Flatten
from core.encoding_wrapper import DynamicEncoder, EncodingConfig, create_synthetic_dataset

def compute_layer_cv(spike_records, test_deltas):
    """
    Computes layer-wise CV (Coefficient of Variation) to measure activity stability across dynamic inputs.

    Args:
        spike_records: dict {delta: {layer: spikes}}. Spikes shape: [Total_Samples, Time, Neurons]
        test_deltas: list of test deltas

    Returns:
        layer_cvs: dict {layer: cv_value}
    """
    layer_names = list(list(spike_records.values())[0].keys())
    layer_cvs = {}

    for layer in layer_names:
        firing_rates = []

        # Iterate through all test deltas
        for delta in test_deltas:
            # Get spikes for this layer and delta
            spikes = spike_records[delta][layer].float()

            # Calculate "mean firing rate of average neuron" for this condition
            # mean(dim=(0, 1)) -> [Neurons]
            # .mean().item() -> scalar (average across all neurons)
            fr = spikes.mean(dim=(0, 1)).mean().item()
            firing_rates.append(fr)

        # Calculate CV of these firing rates across different dynamics
        firing_rates = np.array(firing_rates)
        cv = np.std(firing_rates) / (np.mean(firing_rates) + 1e-8)
        layer_cvs[layer] = cv

    return layer_cvs


class Trainer:
    """Trainer (supports SNN, ANN, RNN, LSTM)"""

    def __init__(self, model, device, model_type='snn'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type.upper()
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

            # Handle SNN vs others
            if self.model_type == 'SNN':
                output, spike_records = self.model(batch_x)
                total_spikes += self.model.count_total_spikes(spike_records)
            else:  # ANN, RNN, LSTM
                output = self.model(batch_x)

            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        avg_spikes = total_spikes / total if self.model_type == 'SNN' else 0

        return avg_loss, accuracy, avg_spikes

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss, correct, total, total_spikes = 0, 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                if self.model_type == 'SNN':
                    output, spike_records = self.model(batch_x)
                    total_spikes += self.model.count_total_spikes(spike_records)
                else:
                    output = self.model(batch_x)

                loss = criterion(output, batch_y)
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        avg_spikes = total_spikes / total if self.model_type == 'SNN' else 0

        return avg_loss, accuracy, avg_spikes

    def fit(self, train_loader, val_loader, epochs, lr=1e-3, patience=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_acc = 0
        patience_counter = 0

        pbar = tqdm(range(epochs), desc=f'Training {self.model_type}')

        for epoch in pbar:
            train_loss, train_acc, train_spikes = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc, val_spikes = self.validate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['spike_count'].append(val_spikes)

            postfix = {'train_acc': f'{train_acc:.2f}%', 'val_acc': f'{val_acc:.2f}%'}
            if self.model_type == 'SNN':
                postfix['spikes'] = f'{val_spikes:.0f}'
            pbar.set_postfix(postfix)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        return best_val_acc


class CrossEncodingTester:
    """Cross-Encoding Tester"""

    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device

    def test_single_encoding(self, model, test_data, test_labels, delta, model_type='snn'):
        """
        Tests model on a specific encoding delta.
        """
        model.eval()
        model_type = model_type.upper()

        # Simulate sample-by-sample encoding like validation data for consistency
        print(f" (Encoding {test_data.shape[0]} test samples...)", end='')

        encoded_data_list = []
        with torch.no_grad():
            for i in range(test_data.shape[0]):
                sample_x = test_data[i].unsqueeze(0)
                enc = self.encoder.encode(sample_x, delta, device=self.device)
                encoded_data_list.append(enc.squeeze(0))
        
        encoded_data = torch.stack(encoded_data_list).to(self.device)

        test_dataset = TensorDataset(encoded_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct, total, total_spikes = 0, 0, 0
        snn_spike_collection = {}

        # Initialize spike collection if SNN
        if model_type == 'SNN':
            dummy_input = encoded_data[0:1].to(self.device)
            _, dummy_spikes = model(dummy_input)
            for layer_name in dummy_spikes.keys():
                snn_spike_collection[layer_name] = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_y = batch_y.to(self.device)

                if model_type == 'SNN':
                    output, spike_records = model(batch_x)
                    total_spikes += model.count_total_spikes(spike_records)
                    for layer_name in snn_spike_collection.keys():
                        snn_spike_collection[layer_name].append(spike_records[layer_name].detach().cpu())
                else:
                    output = model(batch_x)

                _, predicted = output.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        accuracy = 100. * correct / total
        avg_spikes = total_spikes / total if model_type == 'SNN' else 0

        aggregated_spike_records = None
        if model_type == 'SNN':
            try:
                aggregated_spike_records = {}
                for layer_name, spike_list in snn_spike_collection.items():
                    aggregated_spike_records[layer_name] = torch.cat(spike_list, dim=0)
            except Exception as e:
                print(f"\n[Warning] Failed to aggregate spike records: {e}")
                aggregated_spike_records = None

        return accuracy, avg_spikes, aggregated_spike_records


    def test_raw_data(self, model, test_data, test_labels, model_type='ann'):
        """Test on raw data (no encoding)"""
        model.eval()
        # Expand time dimension to match model input
        test_data_expanded = test_data.unsqueeze(1).repeat(1, 5, 1)

        test_dataset = TensorDataset(test_data_expanded, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = model(batch_x)
                _, predicted = output.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        return 100. * correct / total

    def run_cross_encoding_test(self, model, test_data, test_labels,
                                train_delta, test_deltas, model_type='snn'):
        model_type = model_type.upper()
        print("\n" + "=" * 60)
        print(f"Cross-Encoding Generalization Test ({model_type})")
        print(f"Train Encoding: Delta = {train_delta} ({self.encoder.get_encoding_name(train_delta)})")
        print("=" * 60)

        results = {
            'train_delta': train_delta,
            'train_encoding': self.encoder.get_encoding_name(train_delta),
            'model_type': model_type,
            'test_results': []
        }

        all_delta_spike_records = {}

        for test_delta in test_deltas:
            print(f"\nTesting Delta = {test_delta} ({self.encoder.get_encoding_name(test_delta)})...", end=' ')

            acc, spikes, agg_spikes = self.test_single_encoding(
                model, test_data, test_labels, test_delta, model_type
            )

            if agg_spikes is not None:
                all_delta_spike_records[test_delta] = agg_spikes

            is_same = abs(test_delta - train_delta) < 0.1

            result = {
                'test_delta': test_delta,
                'test_encoding': self.encoder.get_encoding_name(test_delta),
                'accuracy': acc,
                'avg_spikes': spikes,
                'is_same_encoding': is_same,
                'accuracy_drop': 0.0 if is_same else None
            }
            results['test_results'].append(result)

            status = "✓ (ID)" if is_same else ""
            if model_type == 'SNN':
                print(f"Acc: {acc:.2f}% | Avg Spikes: {spikes:.0f} {status}")
            else:
                print(f"Acc: {acc:.2f}% {status}")

        # Calculate accuracy drop
        baseline_acc = next((r['accuracy'] for r in results['test_results'] if r['is_same_encoding']), None)
        if baseline_acc:
            for res in results['test_results']:
                if not res['is_same_encoding']:
                    res['accuracy_drop'] = baseline_acc - res['accuracy']

        # --- CV Calculation (SNN Only) ---
        layer_cvs = {}
        if model_type == 'SNN' and all_delta_spike_records:
            print("\n" + "-" * 60)
            print(f"SNN Layer-wise Stability (CV) Analysis (Train δ={train_delta}):")
            try:
                layer_cvs = compute_layer_cv(all_delta_spike_records, test_deltas)
                for layer, cv in layer_cvs.items():
                    print(f"  {layer}: CV = {cv:.4f}")
            except Exception as e:
                print(f"  CV Calculation Failed: {e}")
            print("-" * 60)

        return results, layer_cvs


def plot_comprehensive_comparison(snn_results, ann_encoded_results, rnn_results, lstm_results,
                                  ann_avg_results, ann_flatten_results,
                                  ann_raw_acc, N_RUNS,
                                  save_path='output/comprehensive_comparison.png'):
    """Plots comprehensive comparison: SNN vs ANN vs RNN vs LSTM vs ANN-Avg vs ANN-Flatten"""
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)

    # --- 1. Extract Data ---
    train_deltas = []
    snn_accs, ann_encoded_accs, rnn_accs, lstm_accs = [], [], [], []
    ann_avg_accs, ann_flatten_accs = [], []
    snn_spikes = []

    sorted_deltas = sorted([float(k) for k in snn_results.keys()])

    def get_id_acc(results_dict, delta_key):
        """Helper: Get In-Distribution accuracy"""
        if delta_key not in results_dict: return np.nan
        for res in results_dict[delta_key]['test_results']:
            if res['is_same_encoding']:
                return res['accuracy']
        return np.nan

    def get_id_spikes(results_dict, delta_key):
        """Helper: Get SNN ID spikes"""
        if delta_key not in results_dict: return np.nan
        for res in results_dict[delta_key]['test_results']:
            if res['is_same_encoding']:
                return res['avg_spikes']
        return np.nan

    for train_delta in sorted_deltas:
        train_deltas.append(train_delta)
        snn_accs.append(get_id_acc(snn_results, train_delta))
        ann_encoded_accs.append(get_id_acc(ann_encoded_results, train_delta))
        rnn_accs.append(get_id_acc(rnn_results, train_delta))
        lstm_accs.append(get_id_acc(lstm_results, train_delta))
        ann_avg_accs.append(get_id_acc(ann_avg_results, train_delta))
        ann_flatten_accs.append(get_id_acc(ann_flatten_results, train_delta))
        snn_spikes.append(get_id_spikes(snn_results, train_delta))

    train_deltas = np.array(train_deltas)
    snn_accs = np.array(snn_accs)
    ann_encoded_accs = np.array(ann_encoded_accs)
    rnn_accs = np.array(rnn_accs)
    lstm_accs = np.array(lstm_accs)
    ann_avg_accs = np.array(ann_avg_accs)
    ann_flatten_accs = np.array(ann_flatten_accs)
    snn_spikes = np.array(snn_spikes)

    # --- Plot 1: Accuracy Comparison (All Models) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_deltas, snn_accs, 'o-', linewidth=2.5, markersize=7, label='SNN (Encoded)', color='blue', zorder=10)
    ax1.plot(train_deltas, lstm_accs, 's-', linewidth=2, markersize=6, label='LSTM (Encoded)', color='purple', zorder=9)
    ax1.plot(train_deltas, rnn_accs, 'x-', linewidth=2, markersize=6, label='RNN (Encoded)', color='orange', zorder=8)
    ax1.plot(train_deltas, ann_encoded_accs, '^-', linewidth=2, markersize=6, label='ANN (Encoded, T_last)', color='green', zorder=7)
    ax1.plot(train_deltas, ann_avg_accs, 'D-', linewidth=2, markersize=6, label='ANN (Encoded, Avg-Pool)', color='cyan', zorder=6)
    ax1.plot(train_deltas, ann_flatten_accs, 'p-', linewidth=2, markersize=6, label='ANN (Encoded, Flatten)', color='magenta', zorder=5)

    ax1.axhline(y=ann_raw_acc, color='red', linestyle='--', linewidth=2.5, label=f'ANN (Raw Data): {ann_raw_acc:.2f}%', zorder=4)
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Training Delta (δ)', fontsize=12)
    ax1.set_ylabel(f'Avg. Accuracy (%) (Over {N_RUNS} runs)', fontsize=12)
    ax1.set_title('In-Distribution Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower left')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: SNN Energy Analysis ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(train_deltas, snn_spikes, 'o-', linewidth=2, markersize=6, color='orange')
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Training Delta (δ)', fontsize=12)
    ax2.set_ylabel('Avg. Spike Count', fontsize=12)
    ax2.set_title('SNN Energy Efficiency', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: SNN Efficiency (Acc/Spikes) ---
    ax3 = fig.add_subplot(gs[0, 2])
    efficiency = snn_accs / (snn_spikes + 1e-6)
    ax3.plot(train_deltas, efficiency, 'o-', linewidth=2, markersize=6, color='purple')
    ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Training Delta (δ)', fontsize=12)
    ax3.set_ylabel('Avg. Efficiency (Acc/Spikes)', fontsize=12)
    ax3.set_title('SNN Computational Efficiency', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Generalization Comparison ---
    ax4 = fig.add_subplot(gs[1, 0])
    snn_gen, ann_gen, rnn_gen, lstm_gen = [], [], [], []
    ann_avg_gen, ann_flatten_gen = [], []

    def get_cross_gen_acc(results_dict, delta_key):
        if delta_key not in results_dict: return 0
        cross_accs = [r['accuracy'] for r in results_dict[delta_key]['test_results'] if not r['is_same_encoding']]
        return np.mean(cross_accs) if cross_accs else 0

    for train_delta in sorted_deltas:
        snn_gen.append(get_cross_gen_acc(snn_results, train_delta))
        ann_gen.append(get_cross_gen_acc(ann_encoded_results, train_delta))
        rnn_gen.append(get_cross_gen_acc(rnn_results, train_delta))
        lstm_gen.append(get_cross_gen_acc(lstm_results, train_delta))
        ann_avg_gen.append(get_cross_gen_acc(ann_avg_results, train_delta))
        ann_flatten_gen.append(get_cross_gen_acc(ann_flatten_results, train_delta))

    x = np.arange(len(train_deltas))
    width = 0.14
    ax4.bar(x - 2.5 * width, snn_gen, width, label='SNN', color='blue', alpha=0.8)
    ax4.bar(x - 1.5 * width, lstm_gen, width, label='LSTM', color='purple', alpha=0.7)
    ax4.bar(x - 0.5 * width, rnn_gen, width, label='RNN', color='orange', alpha=0.7)
    ax4.bar(x + 0.5 * width, ann_gen, width, label='ANN (T_last)', color='green', alpha=0.7)
    ax4.bar(x + 1.5 * width, ann_avg_gen, width, label='ANN (Avg-Pool)', color='cyan', alpha=0.6)
    ax4.bar(x + 2.5 * width, ann_flatten_gen, width, label='ANN (Flatten)', color='magenta', alpha=0.6)

    ax4.axhline(y=ann_raw_acc, color='red', linestyle='--', linewidth=2, label='ANN (Raw)')
    ax4.set_xlabel('Training Delta (δ)', fontsize=12)
    ax4.set_ylabel('Avg Cross-Encoding Accuracy (%)', fontsize=12)
    ax4.set_title('Generalization Capability (Avg. O.O.D. Acc)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{d:.2f}' for d in train_deltas], rotation=90, fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # --- Plot 5: Performance Gap (SNN vs Best Non-Spiking) ---
    ax5 = fig.add_subplot(gs[1, 1])
    best_non_spiking_accs = np.nanmax(np.stack([
        ann_encoded_accs, rnn_accs, lstm_accs, ann_avg_accs, ann_flatten_accs
    ]), axis=0)

    performance_gap = snn_accs - best_non_spiking_accs
    colors = ['green' if g > 0 else 'red' for g in performance_gap]
    bars = ax5.bar(range(len(train_deltas)), performance_gap, color=colors, alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Training Delta (δ)', fontsize=12)
    ax5.set_ylabel('Avg. Perf. Gap (SNN - Best Non-Spiking)', fontsize=12)
    ax5.set_title('SNN vs Best Non-Spiking (In-Distr.)', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(len(train_deltas)))
    ax5.set_xticklabels([f'{d:.2f}' for d in train_deltas], rotation=90, fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, performance_gap)):
        if not np.isnan(val):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=7)

    # --- Plot 6: SNN Layer CV ---
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        sample_res = snn_results[sorted_deltas[0]]['test_results'][0]
        layer_names = sorted([k for k in sample_res.keys() if k.startswith('cv_')])

        cv_data = {lname: [] for lname in layer_names}
        for delta in sorted_deltas:
            res = snn_results[delta]['test_results'][0]
            for lname in layer_names:
                cv_data[lname].append(res.get(lname, np.nan))

        for lname, cvs in cv_data.items():
            ax6.plot(train_deltas, cvs, 'o-', label=f"SNN {lname.split('_')[1]}")

        ax6.set_xlabel('Training Delta (δ)', fontsize=12)
        ax6.set_ylabel('Layer-wise CV', fontsize=12)
        ax6.set_title('SNN Layer-wise Stability (CV)', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
    except Exception as e:
        ax6.text(0.5, 0.5, f"Cannot plot CV:\n{e}", ha='center', va='center', wrap=True)
        print(f"[Plotting Error] Failed to plot CV: {e}")

    # --- Plot 7: Summary Stats ---
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')

    def get_best(accs, deltas):
        valid_accs = accs[~np.isnan(accs)]
        valid_deltas = deltas[~np.isnan(accs)]
        if len(valid_accs) > 0:
            best_idx = np.argmax(valid_accs)
            return f"{valid_accs[best_idx]:.2f}% (at δ={valid_deltas[best_idx]:.2f})"
        return "N/A"

    summary_text = f"Performance Summary (Avg. over {N_RUNS} runs)\n" + "=" * 40 + "\n\n"
    summary_text += f"ANN (Raw Data): {ann_raw_acc:.2f}%\n\n"
    summary_text += "Best Avg. Performance (In-Distribution):\n"
    summary_text += f"  SNN:        {get_best(snn_accs, train_deltas)}\n"
    summary_text += f"  LSTM:       {get_best(lstm_accs, train_deltas)}\n"
    summary_text += f"  RNN:        {get_best(rnn_accs, train_deltas)}\n"
    summary_text += f"  ANN (Avg):  {get_best(ann_avg_accs, train_deltas)}\n"
    summary_text += f"  ANN (Flat): {get_best(ann_flatten_accs, train_deltas)}\n"
    summary_text += f"  ANN (T_last):{get_best(ann_encoded_accs, train_deltas)}\n"

    summary_text += "Overall Avg. Performance (In-Distribution):\n"
    summary_text += f"  SNN:        {np.nanmean(snn_accs):.2f}%\n"
    summary_text += f"  LSTM:       {np.nanmean(lstm_accs):.2f}%\n"
    summary_text += f"  RNN:        {np.nanmean(rnn_accs):.2f}%\n"
    summary_text += f"  ANN (Avg):  {np.nanmean(ann_avg_accs):.2f}%\n"
    summary_text += f"  ANN (Flat): {np.nanmean(ann_flatten_accs):.2f}%\n"
    summary_text += f"  ANN (T_last):{np.nanmean(ann_encoded_accs):.2f}%\n"

    summary_text += "Energy Efficiency (SNN):\n"
    valid_snn_spikes = snn_spikes[~np.isnan(snn_spikes)]
    valid_spike_deltas = train_deltas[~np.isnan(snn_spikes)]
    if len(valid_snn_spikes) > 0:
        min_spike_delta = valid_spike_deltas[np.argmin(valid_snn_spikes)]
        max_spike_delta = valid_spike_deltas[np.argmax(valid_snn_spikes)]
        summary_text += f"  Min Spikes: {min(valid_snn_spikes):.0f} (at δ={min_spike_delta:.2f})\n"
        summary_text += f"  Max Spikes: {max(valid_snn_spikes):.0f} (at δ={max_spike_delta:.2f})\n"
        summary_text += f"  Avg Spikes: {np.nanmean(snn_spikes):.0f}\n"

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.add_subplot(gs[2, 1]).axis('off')
    fig.add_subplot(gs[2, 2]).axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')


def main():
    # ================== Config ==================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    N_RUNS = 10 

    # --- 1. Load Digits Dataset ---
    print("\n" + "=" * 60)
    print("Loading Sklearn Digits dataset...")
    digits = load_digits()
    X_np = digits.data
    y_np = digits.target

    # Normalize data (StandardScaler)
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)

    # --- 2. Get Params ---
    N_SAMPLES = X_np.shape[0]
    N_FEATURES = X_np.shape[1]
    N_CLASSES = len(np.unique(y_np))

    print(f"Digits dataset: {N_SAMPLES} samples, {N_FEATURES} features, {N_CLASSES} classes")

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    print("=" * 60)

    # Model Params
    HIDDEN_DIM = 32
    NUM_STEPS = 30
    TMAX = 4.0

    # Training Params
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-4

    ALL_DELTAS = [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.0, 2.5, 5.0, 7.0, 10.0]
    TRAIN_DELTAS = ALL_DELTAS
    TEST_DELTAS = ALL_DELTAS

    print("=" * 60)
    print(f"Starting {N_RUNS} repetitions...")
    print(f"Models: SNN, ANN (T_last), RNN, LSTM, ANN (Raw), ANN (Avg), ANN (Flat)")
    print("=" * 60)

    all_results_list = []

    # ================== Start Loop ==================
    for run in range(1, N_RUNS + 1):
        print(f"\n{'=' * 25} RUN {run} / {N_RUNS} {'=' * 25}\n")

        n_train = int(0.7 * N_SAMPLES)
        n_val = int(0.15 * N_SAMPLES)
        n_test = N_SAMPLES - n_train - n_val

        if (n_train + n_val + n_test) != N_SAMPLES:
            n_train = N_SAMPLES - n_val - n_test

        all_data_list = list(zip(X, y))

        train_data, val_data, test_data = random_split(
            all_data_list,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42 + run)
        )

        test_X = torch.stack([x for x, _ in test_data])
        test_y = torch.stack([y for _, y in test_data])

        print(f"Run {run} - Train: {n_train}, Val: {n_val}, Test: {n_test}")

        encoder = DynamicEncoder(num_steps=NUM_STEPS, tmax=TMAX)

        # ================== Exp 0: ANN Raw Data ==================
        print("\n" + "=" * 60)
        print(f"Run {run} - Exp 0: Baseline ANN (Raw Data)")
        print("=" * 60)

        train_X_raw = torch.stack([x for x, _ in train_data]).unsqueeze(1).repeat(1, NUM_STEPS, 1)
        train_y_raw = torch.stack([y for _, y in train_data])
        val_X_raw = torch.stack([x for x, _ in val_data]).unsqueeze(1).repeat(1, NUM_STEPS, 1)
        val_y_raw = torch.stack([y for _, y in val_data])

        train_loader_raw = DataLoader(TensorDataset(train_X_raw, train_y_raw), BATCH_SIZE, shuffle=True)
        val_loader_raw = DataLoader(TensorDataset(val_X_raw, val_y_raw), BATCH_SIZE, shuffle=False)

        ann_raw = SimpleANN(N_FEATURES, HIDDEN_DIM, N_CLASSES)
        trainer_ann_raw = Trainer(ann_raw, DEVICE, model_type='ANN_Raw')
        trainer_ann_raw.fit(train_loader_raw, val_loader_raw, EPOCHS, lr=LR)

        tester = CrossEncodingTester(encoder, DEVICE)
        ann_raw_test_acc = tester.test_raw_data(ann_raw, test_X, test_y, model_type='ann')

        print(f"\nRun {run} - ANN (Raw) Acc: {ann_raw_test_acc:.2f}%")

        if not os.path.exists('output'):
            os.makedirs('output')

        all_results_list.append({
            'run': run,
            'model_type': 'ANN_Raw',
            'train_delta': 'N/A',
            'train_encoding': 'Raw',
            'test_delta': 'N/A',
            'test_encoding': 'Raw',
            'accuracy': ann_raw_test_acc,
            'avg_spikes': 0,
            'is_same_encoding': True,
            'accuracy_drop': 0.0
        })

        # ================== Exp 1-6: Encoded Models ==================

        snn_results_agg_run = {}
        ann_encoded_results_agg_run = {}
        rnn_results_agg_run = {}
        lstm_results_agg_run = {}
        ann_avg_results_agg_run = {}
        ann_flatten_results_agg_run = {}

        for train_delta in TRAIN_DELTAS:
            print("\n" + "=" * 60)
            print(f"Run {run} - Train Delta = {train_delta} ({encoder.get_encoding_name(train_delta)})")
            print("=" * 60)

            print("Encoding data...")
            encoded_train = []
            labels_train = []
            for x, y_label in tqdm(train_data, desc="Encoding Train"):
                enc = encoder.encode(x.unsqueeze(0), train_delta, device=DEVICE)
                encoded_train.append(enc.squeeze(0))
                labels_train.append(y_label)

            encoded_val = []
            labels_val = []
            for x, y_label in tqdm(val_data, desc="Encoding Val"):
                enc = encoder.encode(x.unsqueeze(0), train_delta, device=DEVICE)
                encoded_val.append(enc.squeeze(0))
                labels_val.append(y_label)

            train_loader = DataLoader(TensorDataset(torch.stack(encoded_train), torch.stack(labels_train)), BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.stack(encoded_val), torch.stack(labels_val)), BATCH_SIZE, shuffle=False)

            input_dim = encoded_train[0].shape[-1]

            # --- 1. Train SNN ---
            print(f"\n>>> Training SNN")
            snn = SimpleSNN(input_dim, HIDDEN_DIM, N_CLASSES, NUM_STEPS)
            trainer_snn = Trainer(snn, DEVICE, model_type='SNN')
            trainer_snn.fit(train_loader, val_loader, EPOCHS, lr=LR)

            tester_snn = CrossEncodingTester(encoder, DEVICE)
            snn_cross_results, snn_layer_cvs = tester_snn.run_cross_encoding_test(
                snn, test_X, test_y, train_delta, TEST_DELTAS, model_type='SNN'
            )
            snn_results_agg_run[float(train_delta)] = snn_cross_results

            for test_res in snn_cross_results['test_results']:
                log_entry = {'run': run, 'model_type': 'SNN', **snn_cross_results, **test_res}
                del log_entry['test_results']
                if snn_layer_cvs:
                    for layer, cv in snn_layer_cvs.items():
                        log_entry[f'cv_{layer}'] = cv
                all_results_list.append(log_entry)

            # --- 2. Train ANN (T_last) ---
            print(f"\n>>> Training ANN (Encoded, T_last)")
            ann_encoded = SimpleANN(input_dim, HIDDEN_DIM, N_CLASSES)
            trainer_ann = Trainer(ann_encoded, DEVICE, model_type='ANN')
            trainer_ann.fit(train_loader, val_loader, EPOCHS, lr=LR)

            tester_ann = CrossEncodingTester(encoder, DEVICE)
            ann_cross_results, _ = tester_ann.run_cross_encoding_test(
                ann_encoded, test_X, test_y, train_delta, TEST_DELTAS, model_type='ANN'
            )
            ann_encoded_results_agg_run[float(train_delta)] = ann_cross_results

            for test_res in ann_cross_results['test_results']:
                log_entry = {'run': run, 'model_type': 'ANN', **ann_cross_results, **test_res}
                del log_entry['test_results']
                all_results_list.append(log_entry)

            # --- 3. Train RNN ---
            print(f"\n>>> Training RNN")
            rnn_model = RecurrentNet(input_dim, HIDDEN_DIM, N_CLASSES, rnn_type='RNN')
            trainer_rnn = Trainer(rnn_model, DEVICE, model_type='RNN')
            trainer_rnn.fit(train_loader, val_loader, EPOCHS, lr=LR)

            tester_rnn = CrossEncodingTester(encoder, DEVICE)
            rnn_cross_results, _ = tester_rnn.run_cross_encoding_test(
                rnn_model, test_X, test_y, train_delta, TEST_DELTAS, model_type='RNN'
            )
            rnn_results_agg_run[float(train_delta)] = rnn_cross_results

            for test_res in rnn_cross_results['test_results']:
                log_entry = {'run': run, 'model_type': 'RNN', **rnn_cross_results, **test_res}
                del log_entry['test_results']
                all_results_list.append(log_entry)

            # --- 4. Train LSTM ---
            print(f"\n>>> Training LSTM")
            lstm_model = RecurrentNet(input_dim, HIDDEN_DIM, N_CLASSES, rnn_type='LSTM')
            trainer_lstm = Trainer(lstm_model, DEVICE, model_type='LSTM')
            trainer_lstm.fit(train_loader, val_loader, EPOCHS, lr=LR)

            tester_lstm = CrossEncodingTester(encoder, DEVICE)
            lstm_cross_results, _ = tester_lstm.run_cross_encoding_test(
                lstm_model, test_X, test_y, train_delta, TEST_DELTAS, model_type='LSTM'
            )
            lstm_results_agg_run[float(train_delta)] = lstm_cross_results

            for test_res in lstm_cross_results['test_results']:
                log_entry = {'run': run, 'model_type': 'LSTM', **lstm_cross_results, **test_res}
                del log_entry['test_results']
                all_results_list.append(log_entry)

            # --- 5. Train ANN (Avg-Pool) ---
            print(f"\n>>> Training ANN (Encoded, Avg-Pool)")
            ann_avg_model = TemporalANN_Avg(input_dim, HIDDEN_DIM, N_CLASSES)
            trainer_ann_avg = Trainer(ann_avg_model, DEVICE, model_type='ANN_Avg')
            trainer_ann_avg.fit(train_loader, val_loader, EPOCHS, lr=LR)

            tester_ann_avg = CrossEncodingTester(encoder, DEVICE)
            ann_avg_cross_results, _ = tester_ann_avg.run_cross_encoding_test(
                ann_avg_model, test_X, test_y, train_delta, TEST_DELTAS, model_type='ANN_Avg'
            )
            ann_avg_results_agg_run[float(train_delta)] = ann_avg_cross_results

            for test_res in ann_avg_cross_results['test_results']:
                log_entry = {'run': run, 'model_type': 'ANN_Avg', **ann_avg_cross_results, **test_res}
                del log_entry['test_results']
                all_results_list.append(log_entry)

            # --- 6. Train ANN (Flatten) [Commented out in original] ---
            # ... (Code remains commented out if it was in original)

        print(f"\n{'=' * 25} RUN {run} COMPLETE {'=' * 25}\n")

    # ================== End Loop ==================
    print("\n" + "=" * 60)
    print(f"All {N_RUNS} runs completed. Aggregating results...")
    print("=" * 60)

    # ================== 1. Save CSV ==================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f'output/full_experiment_results_all_runs_{timestamp}.csv'

    if all_results_list:
        try:
            fieldnames = pd.DataFrame(all_results_list).columns.fillna('unknown')
            fieldnames = [str(f) for f in fieldnames]

            cleaned_results = []
            for row in all_results_list:
                cleaned_row = {str(k): v for k, v in row.items()}
                cleaned_results.append(cleaned_row)

            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(cleaned_results)
            print(f"\nSaved {len(all_results_list)} records to: {csv_file}")
        except Exception as e:
            print(f"\nCSV Save Failed: {e}")
    else:
        print("\nNo results to save.")

    # ================== 2. Aggregate Data ==================
    df = pd.DataFrame(all_results_list)
    df.replace('N/A', np.nan, inplace=True)

    mean_ann_raw_acc = df[df['model_type'] == 'ANN_Raw']['accuracy'].mean()

    df_snn = df[df['model_type'] == 'SNN'].copy()
    df_ann = df[df['model_type'] == 'ANN'].copy()
    df_rnn = df[df['model_type'] == 'RNN'].copy()
    df_lstm = df[df['model_type'] == 'LSTM'].copy()
    df_ann_avg = df[df['model_type'] == 'ANN_Avg'].copy()
    df_ann_flatten = df[df['model_type'] == 'ANN_Flatten'].copy()

    for d in [df_snn, df_ann, df_rnn, df_lstm, df_ann_avg, df_ann_flatten]:
        d['train_delta'] = pd.to_numeric(d['train_delta'])
        d['test_delta'] = pd.to_numeric(d['test_delta'])

    agg_cols = ['train_delta', 'train_encoding', 'test_delta', 'test_encoding', 'is_same_encoding']
    cv_cols = [col for col in df_snn.columns if col.startswith('cv_')]
    
    snn_agg_ops = {'accuracy': ('accuracy', 'mean'), 'avg_spikes': ('avg_spikes', 'mean'),
                   'accuracy_drop': ('accuracy_drop', 'mean')}
    for col in cv_cols: snn_agg_ops[col] = (col, 'mean')
    snn_means = df_snn.groupby(agg_cols).agg(**snn_agg_ops).reset_index()

    base_agg_ops = {'accuracy': ('accuracy', 'mean'), 'avg_spikes': ('avg_spikes', 'mean'),
                    'accuracy_drop': ('accuracy_drop', 'mean')}
    
    ann_means = df_ann.groupby(agg_cols).agg(**base_agg_ops).reset_index()
    rnn_means = df_rnn.groupby(agg_cols).agg(**base_agg_ops).reset_index()
    lstm_means = df_lstm.groupby(agg_cols).agg(**base_agg_ops).reset_index()
    ann_avg_means = df_ann_avg.groupby(agg_cols).agg(**base_agg_ops).reset_index()
    ann_flatten_means = df_ann_flatten.groupby(agg_cols).agg(**base_agg_ops).reset_index()

    # ================== 3. Reconstruct for Plotting ==================
    print("Reconstructing aggregated data for plotting...")
    
    def build_agg_dict(mean_df, train_deltas):
        agg_dict = {}
        for train_delta in train_deltas:
            train_delta_float = float(train_delta)
            model_type = mean_df['model_type'].iloc[0] if not mean_df.empty else 'Unknown'
            subset = mean_df[mean_df['train_delta'] == train_delta_float]
            if not subset.empty:
                agg_dict[train_delta_float] = {
                    'train_delta': train_delta_float,
                    'train_encoding': subset.iloc[0]['train_encoding'],
                    'model_type': model_type,
                    'test_results': subset.to_dict('records')
                }
        return agg_dict

    snn_results_agg = build_agg_dict(snn_means, TRAIN_DELTAS)
    ann_encoded_results_agg = build_agg_dict(ann_means, TRAIN_DELTAS)
    rnn_results_agg = build_agg_dict(rnn_means, TRAIN_DELTAS)
    lstm_results_agg = build_agg_dict(lstm_means, TRAIN_DELTAS)
    ann_avg_results_agg = build_agg_dict(ann_avg_means, TRAIN_DELTAS)
    ann_flatten_results_agg = build_agg_dict(ann_flatten_means, TRAIN_DELTAS)

    # ================== 4. Visualization ==================
    print("Generating visualization...")
    if snn_results_agg or ann_encoded_results_agg:
        plot_comprehensive_comparison(
            snn_results_agg, ann_encoded_results_agg, rnn_results_agg, lstm_results_agg,
            ann_avg_results_agg, ann_flatten_results_agg,
            mean_ann_raw_acc, N_RUNS,
            save_path=f'output/comprehensive_comparison_avg_{N_RUNS}_runs_{timestamp}.png'
        )
    else:
        print("No aggregated results to plot.")

    # ================== 5. Print Summary ==================
    print("\n" + "=" * 60)
    print(f"Experiment Summary (Avg over {N_RUNS} runs)")
    print("=" * 60)

    print(f"\n[Baseline] ANN (Raw): {mean_ann_raw_acc:.2f}%")
    print("\n[Encoded Performance (ID Test Avg)]")

    snn_id = snn_means[snn_means['is_same_encoding'] == True].set_index('train_delta')
    ann_id = ann_means[ann_means['is_same_encoding'] == True].set_index('train_delta')
    rnn_id = rnn_means[rnn_means['is_same_encoding'] == True].set_index('train_delta')
    lstm_id = lstm_means[lstm_means['is_same_encoding'] == True].set_index('train_delta')
    ann_avg_id = ann_avg_means[ann_avg_means['is_same_encoding'] == True].set_index('train_delta')
    ann_flat_id = ann_flatten_means[ann_flatten_means['is_same_encoding'] == True].set_index('train_delta')

    def get_mean_stat(df, delta, col, default=np.nan):
        try: return df.loc[delta][col]
        except KeyError: return default

    for train_delta in TRAIN_DELTAS:
        d = float(train_delta)
        print(f"\n--- Train Delta = {d} ({encoder.get_encoding_name(d)}) ---")

        try:
            snn_acc = get_mean_stat(snn_id, d, 'accuracy')
            snn_spikes = get_mean_stat(snn_id, d, 'avg_spikes')
            ann_acc = get_mean_stat(ann_id, d, 'accuracy')
            rnn_acc = get_mean_stat(rnn_id, d, 'accuracy')
            lstm_acc = get_mean_stat(lstm_id, d, 'accuracy')
            ann_avg_acc = get_mean_stat(ann_avg_id, d, 'accuracy')
            ann_flat_acc = get_mean_stat(ann_flat_id, d, 'accuracy')

            print(f"  SNN:        {snn_acc:6.2f}% (Spikes: {snn_spikes:6.0f})")
            print(f"  LSTM:       {lstm_acc:6.2f}%")
            print(f"  RNN:        {rnn_acc:6.2f}%")
            print(f"  ANN (Avg):  {ann_avg_acc:6.2f}%")
            print(f"  ANN (Flat): {ann_flat_acc:6.2f}%")
            print(f"  ANN (T_last): {ann_acc:6.2f}%")

            # CV Summary
            cv_str = "  SNN Layer CV: "
            found_cv = []
            for col in snn_id.columns:
                if col.startswith('cv_'):
                    val = get_mean_stat(snn_id, d, col)
                    cv_str += f"{col.split('_')[1]}: {val:.4f} | "
                    found_cv.append(col)
            if found_cv: print(cv_str.rstrip(' | '))

        except Exception as e:
            print(f"  Error processing Delta = {d}: {e}")

    print("\n" + "=" * 60)
    print("Experiment Completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()