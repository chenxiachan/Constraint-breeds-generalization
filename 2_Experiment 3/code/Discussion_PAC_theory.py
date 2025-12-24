import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Normal, kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
import snntorch as snn
from snntorch import surrogate
import os
import sys

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.encoding_wrapper import DynamicEncoder, EncodingConfig

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spike_grad = surrogate.fast_sigmoid(slope=25)

class BayesianSNN(nn.Module):
    """
    Bayesian SNN with variational inference for PAC-Bayes verification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_steps=5, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = 1.0
        
        # Variational parameters (Mean and LogVar)
        # Layer 1
        self.w1_mu = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.w1_logvar = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1 - 5) # Init with low variance
        
        # Layer 2
        self.w2_mu = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        self.w2_logvar = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1 - 5)
        
        # LIF Neurons
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (batch, time, input_dim)
        batch_size = x.shape[0]
        
        # Sample weights
        w1 = self.reparameterize(self.w1_mu, self.w1_logvar)
        w2 = self.reparameterize(self.w2_mu, self.w2_logvar)
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        
        for t in range(self.num_steps):
            cur1 = torch.matmul(x[:, t, :], w1)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = torch.matmul(spk1, w2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            
        return torch.stack(spk2_rec, dim=1).sum(dim=1)

    def kl_divergence(self, delta, T, sigma0=1.0):
        """
        Compute KL(Q || P_Lambda)
        
        Correct Formula: sigma_prior^2 = sigma0^2 * exp(Sigma_lambda * T)
        
        Where Sigma_lambda is the Lyapunov Sum.
        - Expansive (delta < 0): Sigma > 0 -> Large Variance (Diffuse Prior)
        - Dissipative (delta > 0): Sigma < 0 -> Small Variance (Concentrated Prior)
        """
        # Get correct Lyapunov Sum
        lyapunov_sum = get_lyapunov_sum(delta)
        
        # Exponent = Sigma * T
        exponent = lyapunov_sum * T
        
        # Clip for numerical stability (but allow enough range to see differences)
        # We clip to [-5, 5] to prevent prior variance from becoming too extreme (e.g. 1e-9 or 1e9)
        # which causes optimization difficulties.
        exponent = np.clip(exponent, -5, 5)
        
        prior_var = sigma0**2 * np.exp(exponent)
        prior_std = np.sqrt(prior_var)
        
        # Posterior
        all_mu = torch.cat([self.w1_mu.flatten(), self.w2_mu.flatten()])
        all_logvar = torch.cat([self.w1_logvar.flatten(), self.w2_logvar.flatten()])
        
        posterior = Normal(all_mu, torch.exp(0.5 * all_logvar))
        prior = Normal(torch.zeros_like(all_mu), torch.tensor(prior_std).to(all_mu.device))
        
        return kl_divergence(posterior, prior).sum()

    def pac_bayes_bound(self, train_error, lambda_contraction, T_enc, m, delta=0.05, sigma0=1.0):
        kl = self.kl_divergence(lambda_contraction, T_enc, sigma0)
        
        # McAllester's Bound
        numerator = kl + np.log(2 * np.sqrt(m) / delta)
        denominator = 2 * (m - 1)
        
        # Ensure non-negative inside sqrt
        if numerator < 0:
            numerator = 0
            
        gen_bound = train_error + torch.sqrt(numerator / denominator)
        return gen_bound.item(), kl.item()

def get_lyapunov_sum(delta):
    """
    Get Lyapunov Sum based on paper Table 3.
    """
    lookup = {
        -1.5: 3.0,    # Expansive
        -1.0: 2.0,
        -0.3: 0.6,
         0.0: 0.0,    # Critical
         0.3: -0.3,
         1.0: -2.0,
         1.5: -3.0,
         2.0: -4.0,   # Transition
         2.5: -5.0,
         5.0: -10.0,
        10.0: -20.0   # Dissipative
    }
    # Fallback to linear approximation if not in table
    return lookup.get(delta, -2.0 * delta)

def verify_pac_bayes(num_runs=3):
    print("\n" + "="*50)
    print("VERIFICATION 1: PAC-Bayes Bound")
    print("="*50)
    
    # Data
    digits = load_digits()
    X = digits.data
    y = digits.target
    X = (X - X.min()) / (X.max() - X.min() + 1e-8) # Normalize
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to tensor
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.LongTensor(y_test).to(DEVICE)
    
    results = []
    deltas = [-1.5, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]
    
    encoder_config = DynamicEncoder(num_steps=30, tmax=4.0)
    
    for delta in deltas:
        print(f"\nTesting delta={delta}...")
        lyapunov_sum = get_lyapunov_sum(delta)
        
        # Encode Data
        with torch.no_grad():
            X_train_enc = encoder_config.encode(X_train_t, delta, device=DEVICE)
            X_test_enc = encoder_config.encode(X_test_t, delta, device=DEVICE)
            
            # Check for NaNs/Infs in encoding
            if torch.isnan(X_train_enc).any() or torch.isinf(X_train_enc).any():
                print(f"  WARNING: Encoding contains NaNs/Infs for delta={delta}!")
                # Replace NaNs with 0 and Infs with max value
                X_train_enc = torch.nan_to_num(X_train_enc, nan=0.0, posinf=10.0, neginf=-10.0)
                X_test_enc = torch.nan_to_num(X_test_enc, nan=0.0, posinf=10.0, neginf=-10.0)
            
        train_ds = TensorDataset(X_train_enc, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # Model
        input_dim = X_train_enc.shape[2]
        hidden_dim = 64
        output_dim = 10
        
        model = BayesianSNN(input_dim, hidden_dim, output_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # Lower LR
        criterion = nn.CrossEntropyLoss()
        
        # Train
        model.train()
        for epoch in range(100): # Increase epochs
            for xb, yb in train_loader:
                optimizer.zero_grad()
                outputs = model(xb)
                # outputs = torch.stack([model(xb) for _ in range(5)]).mean(0)
                nll = criterion(outputs, yb)
                
                # KL weighting (beta)
                # Use a very small beta to ensure learning happens even with tight priors
                kl = model.kl_divergence(delta, encoder_config.tmax)
                # loss = nll + 1e-7 * kl / len(X_train)
                
                beta = 1e-2  # 增大10000倍！
                loss = nll + beta * kl / len(X_train)
                
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        with torch.no_grad():
            # Monte Carlo averaging
            train_outs = torch.stack([model(X_train_enc) for _ in range(5)]).mean(0)
            test_outs = torch.stack([model(X_test_enc) for _ in range(5)]).mean(0)
            
            train_pred = train_outs.argmax(dim=1)
            test_pred = test_outs.argmax(dim=1)
            
            train_err = (train_pred != y_train_t).float().mean().item()
            test_err = (test_pred != y_test_t).float().mean().item()
            actual_gap = test_err - train_err
            
            bound, kl_val = model.pac_bayes_bound(
                train_err, delta, encoder_config.tmax, len(X_train)
            )
            
            print(f"  Train Err: {train_err:.4f}, Test Err: {test_err:.4f}")
            print(f"  Gap: {actual_gap:.4f}, Bound: {bound:.4f}, KL: {kl_val:.2f}")
            
            results.append({
                'delta': delta,
                'lyapunov_sum': lyapunov_sum,
                'train_err': train_err,
                'test_err': test_err,
                'actual_gap': actual_gap,
                'bound': bound,
                'kl': kl_val
            })
            
    return pd.DataFrame(results)

def verify_convergence():
    print("\n" + "="*50)
    print("VERIFICATION 2: Convergence & Gradient Stability")
    print("="*50)
    
    digits = load_digits()
    X = digits.data
    y = digits.target
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    
    deltas = [-1.5, -1.0, 0.0, 1.0, 2.0, 10.0]
    encoder_config = DynamicEncoder(num_steps=30, tmax=4.0)
    
    results = []
    
    for delta in deltas:
        print(f"\nTesting delta={delta}...")
        
        with torch.no_grad():
            X_enc = encoder_config.encode(X_train_t, delta, device=DEVICE)
            # Handle NaNs
            X_enc = torch.nan_to_num(X_enc, nan=0.0, posinf=10.0, neginf=-10.0)
            
        train_ds = TensorDataset(X_enc, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # Standard SNN (non-Bayesian) for gradient analysis
        class StandardSNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                self.lif1 = snn.Leaky(beta=0.95, spike_grad=spike_grad)
                self.lif2 = snn.Leaky(beta=0.95, spike_grad=spike_grad, output=True)
                
            def forward(self, x):
                mem1 = self.lif1.init_leaky()
                mem2 = self.lif2.init_leaky()
                spk2_rec = []
                for t in range(x.shape[1]):
                    cur1 = self.fc1(x[:, t, :])
                    spk1, mem1 = self.lif1(cur1, mem1)
                    cur2 = self.fc2(spk1)
                    spk2, mem2 = self.lif2(cur2, mem2)
                    spk2_rec.append(spk2)
                return torch.stack(spk2_rec, dim=1).sum(dim=1)
                
        model = StandardSNN(X_enc.shape[2], 64, 10).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Lower LR for stability
        criterion = nn.CrossEntropyLoss()
        
        grad_norms = []
        losses = []
        
        model.train()
        for epoch in range(200): # Increase epochs
            epoch_grads = []
            epoch_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                
                # Measure gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item()**2
                total_norm = total_norm**0.5
                epoch_grads.append(total_norm)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            grad_norms.extend(epoch_grads)
            losses.append(epoch_loss / len(train_loader))
            
        # Calculate convergence rate (slope of log loss)
        # We take the last 10 epochs
        log_losses = np.log(np.array(losses) + 1e-8)
        slope, _ = np.polyfit(np.arange(len(losses)), log_losses, 1)

        late_losses = np.array(losses[-20:])
        late_slope, _ = np.polyfit(np.arange(len(late_losses)), 
                           np.log(late_losses + 1e-8), 1)
        
        grad_mean = np.mean(grad_norms)
        grad_std = np.std(grad_norms)
        grad_cv = grad_std / (grad_mean + 1e-8)
        
        print(f"  Grad Mean: {grad_mean:.4f}, CV: {grad_cv:.4f}")
        print(f"  Convergence Rate (slope): {slope:.4f}, Late Slope: {late_slope:.4f}")
        
        results.append({
            'delta': delta,
            'grad_mean': grad_mean,
            'grad_cv': grad_cv,
            'convergence_rate': slope
        })
        
    return pd.DataFrame(results)

def main():
    # 1. PAC-Bayes Verification
    df_pac = verify_pac_bayes()
    
    # 2. Convergence Verification
    df_conv = verify_convergence()
    
    # 3. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: KL vs Delta
    axes[0, 0].plot(df_pac['delta'], df_pac['kl'], 'o-')
    axes[0, 0].set_xlabel('Delta (Dissipation)')
    axes[0, 0].set_ylabel('KL Divergence')
    axes[0, 0].set_title('KL Divergence vs Dissipation')
    axes[0, 0].grid(True)
    
    # Plot 2: Bound vs Actual Gap
    axes[0, 1].scatter(df_pac['actual_gap'], df_pac['bound'])
    max_val = max(df_pac['actual_gap'].max(), df_pac['bound'].max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--')
    axes[0, 1].set_xlabel('Actual Generalization Gap')
    axes[0, 1].set_ylabel('PAC-Bayes Bound')
    axes[0, 1].set_title('Bound Tightness')
    axes[0, 1].grid(True)
    
    # Plot 3: Gradient Norm vs Delta
    axes[1, 0].plot(df_conv['delta'], df_conv['grad_mean'], 'o-')
    axes[1, 0].set_xlabel('Delta')
    axes[1, 0].set_ylabel('Mean Gradient Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Gradient Stability')
    axes[1, 0].grid(True)
    
    # Plot 4: Convergence Rate vs Delta
    axes[1, 1].plot(df_conv['delta'], df_conv['convergence_rate'], 'o-')
    axes[1, 1].set_xlabel('Delta')
    axes[1, 1].set_ylabel('Convergence Rate (Log Slope)')
    axes[1, 1].set_title('Convergence Speed')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('theory_verification_results.png')
    print("\nResults saved to theory_verification_results.png")
    
    # Statistical Tests
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    
    # Test 1: KL decreases with Delta (Dissipation)
    # We expect negative correlation between KL and Delta
    # Higher Delta -> More Dissipative -> Larger Prior Var -> Smaller KL
    corr_kl, p_kl = spearmanr(df_pac['delta'], df_pac['kl'])
    print(f"KL vs Delta Correlation: rho={corr_kl:.3f}, p={p_kl:.3e}")
    if corr_kl < -0.5:
        print(">> PASS: Dissipation reduces KL divergence.")
    else:
        print(">> FAIL: No strong negative correlation found.")
        
    # Test 2: Bound validity
    valid_rate = (df_pac['bound'] >= df_pac['actual_gap']).mean()
    print(f"Bound Validity Rate: {valid_rate:.1%}")
    if valid_rate > 0.8:
        print(">> PASS: PAC-Bayes bound is valid.")
    else:
        print(">> FAIL: Bound is often violated.")
        
    # Test 3: Gradient Stability (Transition Optimality)
    # Check if Transition (delta=2.0) has lower CV than Expansive (-1.5) and Dissipative (10.0)
    try:
        cv_trans = df_conv[df_conv['delta'] == 2.0]['grad_cv'].values[0]
        cv_exp = df_conv[df_conv['delta'] == -1.5]['grad_cv'].values[0]
        cv_diss = df_conv[df_conv['delta'] == 10.0]['grad_cv'].values[0]
        
        print(f"Grad CV - Trans: {cv_trans:.3f}, Exp: {cv_exp:.3f}, Diss: {cv_diss:.3f}")
        
        if cv_trans < cv_exp and cv_trans < cv_diss:
             print(">> PASS: Transition regime has optimal gradient stability (lowest CV).")
        else:
             print(">> FAIL: Transition regime is not strictly optimal.")
    except:
        print(">> SKIP: Could not find specific delta values for comparison.")
    
if __name__ == "__main__":
    main()