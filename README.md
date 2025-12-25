# Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias"**.

## 📖 Overview

This work investigates how **dissipative temporal dynamics** act as an inductive bias to enable neural network generalization. We demonstrate that physical constraints embedded in dynamical systems promote:

1. **Cross-encoding invariance** - Networks trained with transition dynamics generalize across unseen encodings
2. **Structured feature emergence** - Self-organized receptive fields resembling biological V1 cells
3. **Zero-shot transfer** - Robust performance on unseen physical regimes in reinforcement learning

## 📁 Repository Structure

```text
.
├── 0_Experiment 1/          # Cross-encoding generalization (Figure 1)
├── 1_Experiment 2/          # Receptive field emergence (Figure 2)
├── 2_Experiment 3/          # Zero-shot RL transfer (Tables 2-3)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chenxiachan/Constraint-breeds-generalization.git
cd Constraint-breeds-generalization

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Running Experiments

### Experiment 1: Cross-Encoding Generalization

Tests SNN vs ANN generalization across different Duffing encodings (δ values).

```bash
cd "0_Experiment 1/code"
# Run the main experiment (configuration is defined at the top of the script)
python 0_main_Fig1.py
```

**Key outputs:**

- Cross-encoding accuracy matrices
- SNN spike activity analysis
- Comprehensive comparison plots (SNN vs ANN vs RNN vs LSTM)

### Experiment 2: Receptive Field Emergence

Demonstrates spontaneous V1-like receptive field structure emergence under transition dynamics.

```bash
cd "1_Experiment 2/code"

# Main receptive field learning (V1 emergence)
python 0_main_Fig2.py

# Information bottleneck analysis (Appendix)
python 0_Appendix_Fig4.py
```

**Key outputs:**

- Learned receptive field visualizations
- Gabor-like filter emergence across δ regimes
- Temporal information bottleneck metrics

### Experiment 3: Zero-Shot RL Transfer

Evaluates generalization to unseen physical regimes in reinforcement learning.

#### 1. Encoding-Level Experiments

```bash
cd "2_Experiment 3/code/0_Encoding"

# Train with fixed episode length
python 0_rl_train_fixed_episodes.py

# Train with early stopping
python 0_rl_train_earlystop.py
```

#### 2. Architecture-Level Experiments (CLI Support)

The RL scripts support various command-line arguments for flexibility.

##### CartPole (REINFORCE)

```bash
cd "../1_Architecture"

# Run full Beta Sweep (default)
python 00_rl_REIN_Cartpole.py --mode beta_sweep --runs 5

# Run Single Agent Training
python 00_rl_REIN_Cartpole.py --mode single_agent --agent_type ann --runs 5
```

##### LunarLander (PPO) - Supports Parallel Execution

```bash
# Run full Beta Sweep in Parallel (Recommended)
# Automatically uses optimal worker count based on CPU cores
python 00_rl_PPO_Lunarlander_parallel.py --mode beta_sweep --parallel --runs 5

# Run Single Agent Training (e.g., for debugging or specific analysis)
python 00_rl_PPO_Lunarlander_parallel.py --mode single_agent --agent_type lstm --runs 5
```

**Common Arguments for Exp 3:**

- `--mode`: `beta_sweep` (default), `quick_test`, `single_agent`
- `--agent_type`: `ann`, `lstm`, `leaky_fixed`, `rleaky_fixed`
- `--beta`: Leaky parameter (default: 0.9)
- `--runs`: Number of repeats per configuration
- `--parallel`: (LunarLander only) Enable parallel execution
- `--output_dir`: Custom output directory

**Key outputs:**

- Training curves across difficulty levels
- Generalization gap analysis (Easy → Very Hard)
- β-sweep results for SNN membrane dynamics

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{xia2026constraint,
  title={Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias},
  author={Xia Chen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- [snnTorch](https://github.com/jeshraghian/snntorch) for SNN implementations
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for RL environments

## 📬 Contact

For questions or collaboration, please open an issue or contact [x.c.chen@tum.de](mailto:x.c.chen@tum.de).
