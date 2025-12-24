"""
LIF-SNN Model
"""
import torch
import torch.nn as nn

# ----------------------------------------------------
# Reconstruct SimpleSNN using snnTorch
# ----------------------------------------------------
import snntorch as snn
from snntorch import surrogate

# 1. Select a surrogate gradient
# fast_sigmoid or atan are commonly used alternatives to boxcar
spike_grad = surrogate.fast_sigmoid(slope=25)


class SimpleSNN(nn.Module):
    """
    3-layer Fully Connected SNN implemented using snnTorch
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_steps=5, beta=0.95):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # LIF Neurons (using snnTorch's Leaky layer)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        """
        Args:
            x: (batch, num_steps, features)

        Returns:
            output_spikes: (batch, output_dim) - Accumulated output spikes
            spike_records: dict - Activity records for each layer
        """
        # 1. Manually initialize membrane potential (state) before loop
        #    Avoid init_hidden=True for clarity and fewer errors
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record spike outputs for each layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []

        # Standard time loop pattern in snnTorch
        for t in range(self.num_steps):
            x_t = x[:, t, :]

            # 2. Use local variables to pass and update state in loop
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)  

            #    Do this for all layers
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)  

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3) 

            # Record spikes for this time step
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)

        # Stack records into a tensor (batch, time, neurons)
        spike_records = {
            'layer1': torch.stack(spk1_rec, dim=1),
            'layer2': torch.stack(spk2_rec, dim=1),
            'layer3': torch.stack(spk3_rec, dim=1)
        }

        # Accumulate output layer spikes as final classification basis
        output_spike_sum = torch.stack(spk3_rec, dim=1).sum(dim=1)

        return output_spike_sum, spike_records

    # count_total_spikes function remains the same
    def count_total_spikes(self, spike_records):
        """Calculate total spikes across all layers"""
        total = 0
        for key in spike_records:
            total += spike_records[key].sum().item()
        return total


class SimpleANN(nn.Module):
    """Baseline ANN model for comparison"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, num_steps, features) - Take last time step or mean
        """
        # Simple approach: Take last time step
        if len(x.shape) == 3:
            x = x[:, -1, :]  # (batch, features)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class TemporalANN_Avg(nn.Module):
    """
    Temporal ANN (Average Pooling)
    - Averages features across time dimension before feeding into MLP
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, time, features)
        # Average pooling over time dimension
        x = x.mean(dim=1)  # (batch, features)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TemporalANN_Flatten(nn.Module):
    """
    Temporal ANN (Flatten)
    - Flattens (time * features) into a single vector
    """
    def __init__(self, input_dim, output_dim, n_steps=30, hidden_dim=128): 
        super().__init__()
        # Input dimension becomes time_steps * features per step
        flattened_dim = input_dim * n_steps
        
        self.fc1 = nn.Linear(flattened_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, time, features)
        # Flatten: (batch, time * features)
        x = x.reshape(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) 
        x = self.fc3(x)
        return x

# ----------------------------------------------------
# RNN/LSTM Comparison Models
# ----------------------------------------------------
class RecurrentNet(nn.Module):
    """
    RNN/LSTM Baseline Model for Comparison
    - RNN/LSTM layers handle temporal information
    - Last FC layer for classification
    """

    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type='RNN', num_layers=2):
        """
        Args:
            input_dim: Input feature dimension (F_encoded)
            hidden_dim: RNN/LSTM hidden dimension (H)
            output_dim: Number of output classes
            rnn_type: 'RNN' or 'LSTM'
            num_layers: Number of recurrent layers (default 2, to match SNN/ANN depth)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        # Create RNN or LSTM layer based on rnn_type
        # batch_first=True makes input/output tensors (batch, seq_len, features)
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type must be 'RNN' or 'LSTM'")

        # Fully connected layer for classification
        # Maps the hidden state of the last time step to output classes
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, num_steps, features)
        """
        # Initialize hidden state (and cell state for LSTM)
        # Shape: (num_layers, batch, hidden_dim)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
            hidden = (h0, c0)
        else:
            hidden = h0

        # RNN/LSTM forward pass
        # out: (batch, num_steps, hidden_dim) - output for all time steps
        # hidden: (num_layers, batch, hidden_dim) - hidden state of last time step
        out, hidden = self.rnn(x, hidden)

        # Use the output of the last time step for classification
        # out[:, -1, :] shape is (batch, hidden_dim)
        last_step_out = out[:, -1, :]

        # Classification via FC layer
        output = self.fc(last_step_out)

        return output


if __name__ == "__main__":
    # Test code
    batch_size = 4
    input_dim = 10 
    hidden_dim = 32
    output_dim = 3
    num_steps = 5

    # Create random input
    x = torch.randn(batch_size, num_steps, input_dim)

    print(f"Input shape: {x.shape}")

    # --- Test SNN ---
    snn = SimpleSNN(input_dim, hidden_dim, output_dim, num_steps)
    snn_output, spike_records = snn(x)
    print(f"\nSNN output shape: {snn_output.shape}")
    print(f"  Total spikes: {snn.count_total_spikes(spike_records)}")

    # --- Test ANN ---
    ann = SimpleANN(input_dim, hidden_dim, output_dim)
    ann_output = ann(x)
    print(f"\nANN (Last Frame) output shape: {ann_output.shape}")

    # --- Test Temporal ANN (Avg) ---
    ann_avg = TemporalANN_Avg(input_dim, hidden_dim, output_dim)
    ann_avg_output = ann_avg(x)
    print(f"\nANN (Avg-Pool) output shape: {ann_avg_output.shape}")

    # --- Test RNN ---
    rnn = RecurrentNet(input_dim, hidden_dim, output_dim, rnn_type='RNN')
    rnn_output = rnn(x)
    print(f"\nRNN (Full Sequence) output shape: {rnn_output.shape}")

    # --- Test LSTM ---
    lstm = RecurrentNet(input_dim, hidden_dim, output_dim, rnn_type='LSTM')
    lstm_output = lstm(x)
    print(f"\nLSTM (Full Sequence) output shape: {lstm_output.shape}")