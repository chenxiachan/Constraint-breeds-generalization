# Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias

[![arXiv](https://img.shields.io/badge/arXiv-2512.23916-b31b1b.svg)](https://arxiv.org/abs/2512.23916)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper **"Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias"**.
<img width="890" height="235" alt="image" src="https://github.com/user-attachments/assets/6ac41b82-deec-4937-8eea-50d67eace111" />

## 📖 Overview

Conventional deep learning often equates generalization with unconstrained scaling. In contrast, biological systems operate under strict metabolic constraints. This work demonstrates that these **dissipative constraints act as a fundamental temporal inductive bias**, compelling systems to abstract robust invariants from noisy sensory streams.

Through a phase-space analysis of signal propagation, we reveal that a critical **"transition" regime** poised between expansive noise amplification and excessive dissipative compression, maximizes generalization capabilities. We validate this principle across three scales:

| Scale | Task | Key Finding |
| :--- | :--- | :--- |
| **Exp 1: Representational** | Cross-Encoding Classification | Networks trained with transition dynamics generalize to unseen encodings. |
| **Exp 2: Structural** | Unsupervised Learning | Spontaneous emergence of structured receptive fields. |
| **Exp 3: Behavioral** | Zero-Shot RL | Robust policy transfer to unseen physical regimes in robotic control. |

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

> Investigate how SNNs trained on specific dynamics generalize to others.

```bash
cd "0_Experiment 1/code"
python 0_main_Fig1.py
```

<img width="1951" height="1127" alt="Figure1" src="https://github.com/user-attachments/assets/0d5f2a35-88c7-4534-85bd-06f407a7d779" />

### 2. Receptive Field Emergence

> Observe the spontaneous formation of structured filters under transition dynamics.

<!-- ![alt text](<1_Experiment 2/output/rf_evolution_combined.gif>) -->
<!-- ![alt text](<1_Experiment 2/output/fine_scan_evolution.gif>) -->

```bash
cd "1_Experiment 2/code"
python 0_main_Fig2.py        # Main Learning
python 0_Appendix_Fig4.py    # Information Bottleneck Analysis
```

![rf_evolution_combined](https://github.com/user-attachments/assets/6a9bf386-3855-496d-ad89-b299da53ee32)

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

![cartpole_comparison_grid](https://github.com/user-attachments/assets/47258bcd-0aa3-4eea-a790-9ed13799c70a)
![lunarlander_transfer_grid_clean](https://github.com/user-attachments/assets/386eae11-6f8b-4a80-b647-2794e87d4d60)

## 📚 Citation

```bibtex
@article{xia2025constraint,
  title={Constraint Breeds Generalization: Temporal Dynamics as an Inductive Bias},
  author={Xia Chen},
  journal={arXiv preprint arXiv:2512.23916},
  year={2025}
}
```


## 📬 Contact

For collaboration or questions, please contact [x.c.chen@tum.de](mailto:x.c.chen@tum.de).
Licensed under the [MIT License](LICENSE).
