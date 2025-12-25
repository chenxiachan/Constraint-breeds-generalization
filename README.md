# Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias"**.

## 📖 Overview

Conventional deep learning often equates generalization with unconstrained scaling. In contrast, biological systems operate under strict metabolic constraints. This work demonstrates that these **physical constraints act as a fundamental temporal inductive bias**, compelling systems to abstract robust invariants from noisy sensory streams.

Through a phase-space analysis of signal propagation, we reveal that a critical **"transition" regime**—poised between expansive noise amplification and excessive dissipative compression—maximizes generalization capabilities. We validate this principle across three scales:

| Scale | Experiment | Key Finding |
| :--- | :--- | :--- |
| **Representational** | [Exp 1] Cross-Encoding | Networks trained with transition dynamics generalize to unseen encodings. |
| **Structural** | [Exp 2] Receptive Fields | Spontaneous emergence of biological V1-like spatial filters. |
| **Behavioral** | [Exp 3] Zero-Shot RL | Robust policy transfer to unseen physical regimes in robotic control. |

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/chenxiachan/Constraint-breeds-generalization.git
cd Constraint-breeds-generalization
```

### 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🧪 Experiments

### 1. Cross-Encoding Generalization

> Investigating invariance across Duffing oscillator encodings (δ)

```bash
cd "0_Experiment 1/code"
python 0_main_Fig1.py
```

### 2. Receptive Field Emergence

> Spontaneous V1-like structure formation under transition dynamics

```bash
cd "1_Experiment 2/code"
python 0_main_Fig2.py        # Main Learning
python 0_Appendix_Fig4.py    # Information Bottleneck Analysis
```

### 3. Zero-Shot RL Transfer

> Generalization to unseen physical dynamics in CartPole & LunarLander

#### Encoding-Level Tests

```bash
cd "2_Experiment 3/code/0_Encoding"
python 0_rl_train_fixed_episodes.py # Standard training
python 0_rl_train_earlystop.py      # Early stopping
```

#### Architecture-Level Tests (CLI)

```bash
cd "2_Experiment 3/code/1_Architecture"

# CartPole (REINFORCE)
python 00_rl_REIN_Cartpole.py --mode beta_sweep --runs 5         # Full Beta Sweep
python 00_rl_REIN_Cartpole.py --mode single_agent --agent_type ann --runs 5 # Single Agent

# LunarLander (PPO) - Parallel
python 00_rl_PPO_Lunarlander_parallel.py --mode beta_sweep --parallel --runs 5
python 00_rl_PPO_Lunarlander_parallel.py --mode single_agent --agent_type lstm --runs 5
```

## 📚 Citation

```bibtex
@article{xia2026constraint,
  title={Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias},
  author={Xia Chen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📁 Structure

<details>
<summary>Click to expand file tree</summary>

```text
.
├── 0_Experiment 1/          # Cross-encoding generalization
├── 1_Experiment 2/          # Receptive field emergence
├── 2_Experiment 3/          # Zero-shot RL transfer
├── requirements.txt
└── README.md
```

</details>

## 📬 Contact

For collaboration or questions, please contact [x.c.chen@tum.de](mailto:x.c.chen@tum.de).
Licensed under the [MIT License](LICENSE).
