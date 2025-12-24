import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
from torch.distributions import Categorical
import pandas as pd
from scipy import stats
from datetime import datetime
import os
import json
import warnings

# SNN dependencies
import snntorch as snn
from snntorch import surrogate
from numba import jit

# Settings
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'sans-serif'] # Changed to standard font
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================
# Environment Definition
# ============================================================
class FixedCartPole(gym.Env):
    """Modified CartPole Environment."""
    def __init__(self, difficulty='easy'):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8, -np.inf, -0.418, -np.inf]),
            high=np.array([4.8, np.inf, 0.418, np.inf]),
            dtype=np.float32
        )
        self.max_steps = 200

        if difficulty == 'easy':
            self.pole_length = 0.5
            self.pole_mass = 0.1
            self.noise = 0.0
            self.init_range = 0.05
        elif difficulty == 'medium':
            self.pole_length = 0.8
            self.pole_mass = 0.3
            self.noise = 0.005
            self.init_range = 0.08
        elif difficulty == 'hard':
            self.pole_length = 1.2
            self.pole_mass = 0.5
            self.noise = 0.01
            self.init_range = 0.10
        else:
            self.pole_length = 1.5
            self.pole_mass = 0.7
            self.noise = 0.015
            self.init_range = 0.12

        self.difficulty = difficulty
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.force = 10.0
        self.tau = 0.02

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(
            -self.init_range, self.init_range, 4
        ).astype(np.float32)
        self.steps = 0
        return self.state, {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force if action == 1 else -self.force
        if self.noise > 0:
            force += np.random.normal(0, self.noise)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.pole_mass * self.pole_length * theta_dot ** 2 * sintheta) / \
               (self.cart_mass + self.pole_mass)
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.pole_length * (4 / 3 - self.pole_mass * costheta ** 2 /
                                        (self.cart_mass + self.pole_mass)))
        xacc = temp - self.pole_mass * self.pole_length * thetaacc * costheta / \
               (self.cart_mass + self.pole_mass)
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1
        terminated = bool(
            x < -2.4 or x > 2.4 or
            theta < -0.209 or theta > 0.209
        )
        truncated = self.steps >= self.max_steps
        reward = 1.0 if not terminated else 0.0
        return self.state, reward, terminated, truncated, {}


# ============================================================
# Dynamic Encoding
# ============================================================
@jit(nopython=True)
def mixed_oscillator_transformer_vectorized(x0, y0, z0, alpha=1.0, beta=0.2, delta=0.1,
                                            gamma=0.1, omega=1.0, drive=0.0, tmax=20.0, h=0.01):
    nsteps = round((tmax - 0) / h)
    t = np.linspace(0, tmax, nsteps + 1)
    x = np.zeros(nsteps + 1)
    y = np.zeros(nsteps + 1)
    z = np.zeros(nsteps + 1)
    x[0] = x0
    y[0] = y0
    z[0] = z0
    for i in range(nsteps):
        xi, yi, zi = x[i], y[i], z[i]
        ti = i * h
        k1x = h * yi
        k1y = h * (-alpha * xi - beta * xi ** 3 - delta * yi + gamma * zi + drive * np.cos(omega * ti))
        k1z = h * (-omega * xi - delta * zi + gamma * xi * yi)
        x2 = xi + k1x / 2
        y2 = yi + k1y / 2
        z2 = zi + k1z / 2
        t2 = ti + h / 2
        k2x = h * y2
        k2y = h * (-alpha * x2 - beta * x2 ** 3 - delta * y2 + gamma * z2 + drive * np.cos(omega * t2))
        k2z = h * (-omega * x2 - delta * z2 + gamma * x2 * y2)
        x3 = xi + k2x / 2
        y3 = yi + k2y / 2
        z3 = zi + k2z / 2
        k3x = h * y3
        k3y = h * (-alpha * x3 - beta * x3 ** 3 - delta * y3 + gamma * z3 + drive * np.cos(omega * t2))
        k3z = h * (-omega * x3 - delta * z3 + gamma * x3 * y3)
        x4 = xi + k3x
        y4 = yi + k3y
        z4 = zi + k3z
        t4 = ti + h
        k4x = h * y4
        k4y = h * (-alpha * x4 - beta * x4 ** 3 - delta * y4 + gamma * z4 + drive * np.cos(omega * t4))
        k4z = h * (-omega * x4 - delta * z4 + gamma * x4 * y4)
        x[i + 1] = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        y[i + 1] = yi + (k1y + 2 * k2y + 2 * k3y + k4y) / 6
        z[i + 1] = zi + (k1z + 2 * k2z + 2 * k3z + k4z) / 6
    return np.column_stack((x, y, z))


def mixed_oscillator_encode(data, num_steps, tmax=8, params=None):
    if params is None:
        params = {
            'alpha': 1.0, 'beta': 0.2, 'delta': 0.1,
            'gamma': 0.1, 'omega': 1.0, 'drive': 0.0
        }
    batch_size = data.shape[0]
    num_features = data.shape[1]
    encoded_data = np.zeros((batch_size, num_steps, num_features * 3))
    data_numpy = data.cpu().numpy()
    data_max = np.max(np.abs(data_numpy))
    if data_max > 0:
        data_numpy = data_numpy / data_max
    feature_batch_size = min(50, num_features)
    for b in range(batch_size):
        for j in range(0, num_features, feature_batch_size):
            end_idx = min(j + feature_batch_size, num_features)
            current_features = data_numpy[b, j:end_idx]
            for k, value in enumerate(current_features):
                oscillator_output = mixed_oscillator_transformer_vectorized(
                    value, value * 0.2, -value,
                    alpha=params['alpha'],
                    beta=params['beta'],
                    delta=params['delta'],
                    gamma=params['gamma'],
                    omega=params['omega'],
                    drive=params['drive'],
                    tmax=tmax
                )
                feature_idx = j + k
                for dim in range(3):
                    encoded_data[b, :, feature_idx * 3 + dim] = np.interp(
                        np.linspace(0, 1, num_steps),
                        np.linspace(0, 1, oscillator_output.shape[0]),
                        oscillator_output[:, dim]
                    )
    return torch.from_numpy(encoded_data).float()


class DynamicEncoder:
    """Dynamic Encoder using mixed oscillators."""
    def __init__(self, num_steps=5, tmax=8.0):
        self.num_steps = num_steps
        self.tmax = tmax

    def encode(self, data, delta, device='cpu'):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        data_cpu = data.cpu()
        params = {
            'alpha': 1.0,
            'beta': 0.2,
            'delta': delta,
            'gamma': 0.1,
            'omega': 1.0,
            'drive': 0.0
        }
        encoded = mixed_oscillator_encode(
            data_cpu,
            num_steps=self.num_steps,
            tmax=self.tmax,
            params=params
        )
        return encoded.to(device)


# ============================================================
# Networks
# ============================================================
spike_grad = surrogate.fast_sigmoid(slope=25)


class SimpleSNN(nn.Module):
    """Spiking Neural Network."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_steps=5, beta=0.95):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_steps = num_steps

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for t in range(self.num_steps):
            x_t = x[:, t, :]
            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        output_spike_sum = torch.stack(spk3_rec, dim=1).sum(dim=1)
        return output_spike_sum


class SimpleANN(nn.Module):
    """Standard 3-layer MLP."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # If input is temporal, take mean
        if len(x.shape) == 3:
            x = x.mean(dim=1) 

        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


# ============================================================
# Agents
# ============================================================
class SNNReinforceAgent:
    """SNN + REINFORCE Agent."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_steps,
                 encoder, delta, lr=1e-3, gamma=0.99, grad_clip=1.0):
        self.encoder = encoder
        self.delta = delta
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_encoder = (encoder is not None)

        snn_input_dim = input_dim
        self.policy_net = SimpleSNN(
            input_dim=snn_input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_steps=num_steps
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs_buffer = []
        self.rewards_buffer = []

    def select_action(self, state, training=True):
        if self.use_encoder:
            encoded_state = self.encoder.encode(state, self.delta, device=self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            encoded_state = state_tensor.unsqueeze(1).repeat(1, self.policy_net.num_steps, 1)

        if training:
            self.policy_net.train()
            output_spike_sum = self.policy_net(encoded_state)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                output_spike_sum = self.policy_net(encoded_state)

        action_probs = F.softmax(output_spike_sum, dim=-1)
        m = Categorical(action_probs)
        action = m.sample()
        if training:
            self.log_probs_buffer.append(m.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards_buffer.append(reward)

    def update_at_episode_end(self):
        if not self.log_probs_buffer:
            return 0.0

        discounted_returns = []
        cumulative_return = 0
        for r in reversed(self.rewards_buffer):
            cumulative_return = r + self.gamma * cumulative_return
            discounted_returns.insert(0, cumulative_return)

        returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32).to(self.device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        policy_loss_terms = []
        for log_prob, G_t in zip(self.log_probs_buffer, returns_tensor):
            policy_loss_terms.append(-log_prob * G_t)

        self.optimizer.zero_grad()
        total_loss = torch.stack(policy_loss_terms).sum()
        total_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()

        loss_value = total_loss.item()
        self.log_probs_buffer = []
        self.rewards_buffer = []
        return loss_value


class ANNReinforceAgent:
    """ANN + REINFORCE Agent."""
    def __init__(self, input_dim, hidden_dim, action_dim,
                 encoder=None, delta=None, lr=1e-3, gamma=0.99, grad_clip=1.0):
        self.encoder = encoder
        self.delta = delta
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_encoder = (encoder is not None)

        if self.use_encoder:
            ann_input_dim = input_dim 
        else:
            ann_input_dim = 4

        self.policy_net = SimpleANN(
            input_dim=ann_input_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs_buffer = []
        self.rewards_buffer = []

    def select_action(self, state, training=True):
        if self.use_encoder:
            encoded_state = self.encoder.encode(state, self.delta, device=self.device)
        else:
            encoded_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if training:
            self.policy_net.train()
            output = self.policy_net(encoded_state)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                output = self.policy_net(encoded_state)

        action_probs = F.softmax(output, dim=-1)
        m = Categorical(action_probs)
        action = m.sample()
        if training:
            self.log_probs_buffer.append(m.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards_buffer.append(reward)

    def update_at_episode_end(self):
        if not self.log_probs_buffer:
            return 0.0

        discounted_returns = []
        cumulative_return = 0
        for r in reversed(self.rewards_buffer):
            cumulative_return = r + self.gamma * cumulative_return
            discounted_returns.insert(0, cumulative_return)

        returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32).to(self.device)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        policy_loss_terms = []
        for log_prob, G_t in zip(self.log_probs_buffer, returns_tensor):
            policy_loss_terms.append(-log_prob * G_t)

        self.optimizer.zero_grad()
        total_loss = torch.stack(policy_loss_terms).sum()
        total_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.grad_clip)

        self.optimizer.step()

        loss_value = total_loss.item()
        self.log_probs_buffer = []
        self.rewards_buffer = []
        return loss_value


# ============================================================
# Training and Evaluation
# ============================================================
def train(env, agent,
          episodes=500,
          verbose=False,
          early_stopping=True,
          eval_interval=20,
          eval_episodes=20,
          solved_threshold=195.0,
          solved_k=5):
    """
    Train agent with early stopping based on evaluation threshold.
    """
    rewards = []
    total_steps = 0
    solved_counter = 0
    solved_at_episode = None
    solved_at_steps = None

    solve_info = {
        "solved": False,
        "solved_at_episode": None,
        "solved_at_steps": None,
        "eval_history": []
    }

    train_difficulty = env.difficulty if hasattr(env, "difficulty") else "easy"

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        ep_steps = 0

        # 1. Standard REINFORCE Training Episode
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            agent.store_reward(reward)
            state = next_state
            ep_reward += reward

            ep_steps += 1
            total_steps += 1

        agent.update_at_episode_end()
        rewards.append(ep_reward)

        if verbose and (ep + 1) % 50 == 0:
            avg_recent = np.mean(rewards[-50:])
            print(f"    Ep {ep + 1}/{episodes}, Train Avg(50): {avg_recent:.1f}")

        # 2. Evaluation-based Early Stopping
        if early_stopping and (ep + 1) % eval_interval == 0:
            eval_env = FixedCartPole(train_difficulty)

            eval_result = evaluate(
                eval_env,
                agent,
                episodes=eval_episodes,
                seed=1234
            )
            mean_eval_return = eval_result['mean']

            solve_info["eval_history"].append({
                "train_episode": ep + 1,
                "mean_return": float(mean_eval_return)
            })

            if verbose:
                print(f"    [Eval] after Ep {ep + 1}: mean_return={mean_eval_return:.1f}")

            if mean_eval_return >= solved_threshold:
                solved_counter += 1
            else:
                solved_counter = 0

            if solved_counter >= solved_k:
                solved_at_episode = ep + 1
                solved_at_steps = total_steps
                solve_info["solved"] = True
                solve_info["solved_at_episode"] = int(solved_at_episode)
                solve_info["solved_at_steps"] = int(solved_at_steps)

                if verbose:
                    print(f"    >>> Solved! "
                          f"episode={solved_at_episode}, "
                          f"total_steps={solved_at_steps}, "
                          f"mean_eval_return={mean_eval_return:.1f}")
                break

    if not solve_info["solved"]:
        solve_info["solved_at_episode"] = len(rewards)
        solve_info["solved_at_steps"] = int(total_steps)

    agent.solve_info = solve_info

    return rewards


def evaluate(env, agent, episodes=100, seed=None):
    """Evaluate agent."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    rewards = []
    success_count = 0

    for ep_idx in range(episodes):
        ep_seed = seed + ep_idx if seed is not None else None
        state, _ = env.reset(seed=ep_seed)
        ep_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += reward

        rewards.append(ep_reward)
        if ep_reward >= 195:
            success_count += 1

    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'median': np.median(rewards),
        'success_rate': success_count / episodes,
        'rewards': rewards
    }


def run_single_experiment(agent_type, delta=None, run_id=0, train_episodes=500,
                          eval_seed=None, verbose=False, early_stopping=True):
    """Run a single experiment."""
    # Seed setup
    if delta is None:
        seed = 2025 + run_id * 1000 + 9999 
    else:
        seed = 2025 + run_id * 1000 + int(abs(delta) * 100)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    test_difficulties = ['easy', 'medium', 'hard', 'very_hard']

    # Hyperparameters
    SNN_INPUT_DIM = 12
    ANN_RAW_INPUT_DIM = 4
    HIDDEN_DIM = 128
    ACTION_DIM = 2
    NUM_STEPS = 5
    ENC_TMAX = 8.0
    LR = 1e-3
    GRAD_CLIP = 1.0

    encoder = None if delta is None else DynamicEncoder(num_steps=NUM_STEPS, tmax=ENC_TMAX)
    env_train = FixedCartPole('easy')

    if verbose:
        agent_desc = f"{agent_type.upper()}"
        if delta is None:
            agent_desc += "-Raw"
        elif delta > 0:
            agent_desc += f"-Diss(δ={delta})"
        else:
            agent_desc += f"-Exp(δ={delta})"
        print(f"  [Run {run_id + 1}] Training {agent_desc}, seed={seed}...")

    # Create Agent
    if agent_type == 'snn':
        agent = SNNReinforceAgent(
            input_dim=SNN_INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            action_dim=ACTION_DIM,
            num_steps=NUM_STEPS,
            encoder=encoder,
            delta=delta if delta is not None else 0.0,
            lr=LR,
            grad_clip=GRAD_CLIP
        )
    else:  # ann
        agent = ANNReinforceAgent(
            input_dim=SNN_INPUT_DIM if delta is not None else ANN_RAW_INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            action_dim=ACTION_DIM,
            encoder=encoder,
            delta=delta,
            lr=LR,
            grad_clip=GRAD_CLIP
        )

    # Train
    train_curve = train(env_train, agent, episodes=train_episodes,
                        verbose=verbose, early_stopping=early_stopping)

    # Evaluate
    eval_results = {}
    for diff in test_difficulties:
        env_test = FixedCartPole(diff)
        eval_s = (seed + hash(diff)) % 100000 if eval_seed is not None else None
        eval_result = evaluate(env_test, agent, 100, seed=eval_s)
        eval_results[diff] = eval_result

    return {
        'run_id': run_id,
        'agent_type': agent_type,
        'delta': delta,
        'seed': seed,
        'train_curve': train_curve,
        'eval': eval_results
    }


def run_comprehensive_experiments(num_runs=5, train_episodes=500, use_eval_seed=True):
    """
    Run comprehensive comparison experiments (7 Groups).
    """
    print("=" * 70)
    print(f"Comprehensive Experiment (N={num_runs})")
    print("=" * 70)
    print(f"Config: train_episodes={train_episodes}, eval_seed={'fixed' if use_eval_seed else 'random'}")
    print("\nGroups:")
    print("  1. Diss-SNN: SNN + Dissipative Encoding (δ=10.0)")
    print("  2. Tran-SNN: SNN + Transition Encoding (δ=2.0)")
    print("  3. Exp-SNN: SNN + Expansive Encoding (δ=-1.5)")
    print("  4. ANN-Raw: ANN + Raw Input")
    print("  5. ANN-Diss: ANN + Dissipative Encoding (δ=10.0)")
    print("  6. ANN-Tran: ANN + Transition Encoding (δ=2.0)")
    print("  7. ANN-Exp: ANN + Expansive Encoding (δ=-1.5)")

    all_results = {
        'diss_snn': [],
        'tran_snn': [],
        'exp_snn': [],
        'ann_raw': [],
        'ann_diss': [],
        'ann_tran': [],
        'ann_exp': []
    }

    for run_id in range(num_runs):
        print(f"\n[Run {run_id + 1}/{num_runs}]")
        print("-" * 70)

        # 1. Diss-SNN
        print("1. Diss-SNN...")
        result = run_single_experiment(
            agent_type='snn',
            delta=10.0,
            run_id=run_id,
            train_episodes=train_episodes,
            eval_seed=1 if use_eval_seed else None,
            verbose=(run_id == 0),
            early_stopping=True
        )
        all_results['diss_snn'].append(result)
        diss_snn_gap = result['eval']['easy']['mean'] - result['eval']['very_hard']['mean']

        # 2. Tran-SNN
        print("2. Tran-SNN...")
        result = run_single_experiment(
            agent_type='snn',
            delta=2,
            run_id=run_id,
            train_episodes=train_episodes,
            eval_seed=1 if use_eval_seed else None,
            verbose=(run_id == 0),
            early_stopping=True
        )
        all_results['tran_snn'].append(result)
        tran_snn_gap = result['eval']['easy']['mean'] - result['eval']['very_hard']['mean']

        # 3. Exp-SNN
        print("3. Exp-SNN...")
        result = run_single_experiment(
            agent_type='snn',
            delta=-1.5,
            run_id=run_id,
            train_episodes=train_episodes,
            eval_seed=1 if use_eval_seed else None,
            verbose=(run_id == 0),
            early_stopping=True
        )
        all_results['exp_snn'].append(result)
        exp_snn_gap = result['eval']['easy']['mean'] - result['eval']['very_hard']['mean']

        # 4. ANN-Raw
        print("4. ANN-Raw...")
        result = run_single_experiment(
            agent_type='ann',
            delta=None,
            run_id=run_id,
            train_episodes=train_episodes,
            eval_seed=1 if use_eval_seed else None,
            verbose=(run_id == 0),
            early_stopping=True
        )
        all_results['ann_raw'].append(result)
        ann_raw_gap = result['eval']['easy']['mean'] - result['eval']['very_hard']['mean']

        # 5. ANN-Diss
        print("5. ANN-Diss...")
        result = run_single_experiment(
            agent_type='ann',
            delta=10.0,
            run_id=run_id,
            train_episodes=train_episodes,
            eval_seed=1 if use_eval_seed else None,
            verbose=(run_id == 0),
            early_stopping=True
        )
        all_results['ann_diss'].append(result)
        ann_diss_gap = result['eval']['easy']['mean'] - result['eval']['very_hard']['mean']

        # 6. ANN-Tran
        print("6. ANN-Tran...")
        result = run_single_experiment(
            agent_type='ann',
            delta=2,
            run_id=run_id,
            train_episodes=train_episodes,
            eval_seed=1 if use_eval_seed else None,
            verbose=(run_id == 0),
            early_stopping=True
        )
        all_results['ann_tran'].append(result)
        ann_tran_gap = result['eval']['easy']['mean'] - result['eval']['very_hard']['mean']

        # 7. ANN-Exp
        print("7. ANN-Exp...")
        result = run_single_experiment(
            agent_type='ann',
            delta=-1.5,
            run_id=run_id,
            train_episodes=train_episodes,
            eval_seed=1 if use_eval_seed else None,
            verbose=(run_id == 0),
            early_stopping=True
        )
        all_results['ann_exp'].append(result)
        ann_exp_gap = result['eval']['easy']['mean'] - result['eval']['very_hard']['mean']

        # Print quick comparison
        print(f"\n  Generalization Gap (Easy - VH):")
        print(f"    Diss-SNN:  {diss_snn_gap:.1f}")
        print(f"    Exp-SNN:   {exp_snn_gap:.1f}")
        print(f"    Tran-SNN:  {tran_snn_gap:.1f}")
        print(f"    ANN-Raw:   {ann_raw_gap:.1f}")
        print(f"    ANN-Diss:  {ann_diss_gap:.1f}")
        print(f"    ANN-Tran:  {ann_tran_gap:.1f}")
        print(f"    ANN-Exp:   {ann_exp_gap:.1f}")

    return all_results


# ============================================================
# Statistical Analysis
# ============================================================
def compute_comprehensive_statistics(all_results):
    """Compute comprehensive statistics."""
    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    groups = ['diss_snn', 'tran_snn', 'exp_snn', 'ann_raw', 'ann_diss', 'ann_tran', 'ann_exp']

    print("\n" + "=" * 70)
    print("Comprehensive Statistical Analysis")
    print("=" * 70)

    # 1. Performance Summary
    print("\n1. Mean Performance per Difficulty:")
    print(f"{'Group':<15} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Very_hard':>8}")
    print("-" * 70)

    perf_data = {}
    for group in groups:
        perfs = []
        for diff in difficulties:
            rewards_all = []
            for result in all_results[group]:
                rewards_all.extend(result['eval'][diff]['rewards'])
            perfs.append(np.mean(rewards_all))
        perf_data[group] = perfs
        print(f"{group:<15} {perfs[0]:>8.1f} {perfs[1]:>8.1f} {perfs[2]:>8.1f} {perfs[3]:>8.1f}")

    # 2. Gap Analysis
    print("\n2. Generalization Gap (Easy - X):")
    print(f"{'Group':<15} {'Medium':>8} {'Hard':>8} {'Very_hard':>8} {'Avg Gap':>10}")
    print("-" * 70)

    gap_data = {}
    for group in groups:
        gaps = []
        for i, diff in enumerate(['medium', 'hard', 'very_hard']):
            if len(perf_data[group]) > 0:
                gap = perf_data[group][0] - perf_data[group][i + 1]
                gaps.append(gap)
            else:
                gaps.append(np.nan)

        avg_gap = np.nanmean(gaps)
        gap_data[group] = gaps + [avg_gap]
        print(f"{group:<15} {gaps[0]:>8.1f} {gaps[1]:>8.1f} {gaps[2]:>8.1f} {avg_gap:>10.1f}")

    # 3. Key Comparisons
    print("\n3. Key Comparisons:")
    print("\n   3.1 Encoding Effect (Dissipative vs Expansive):")

    snn_encoding_effect = gap_data['exp_snn'][-1] - gap_data['diss_snn'][-1]
    print(f"      SNN: Exp_gap - Diss_gap = {snn_encoding_effect:+.1f}")

    ann_encoding_effect = gap_data['ann_exp'][-1] - gap_data['ann_diss'][-1]
    print(f"      ANN: Exp_gap - Diss_gap = {ann_encoding_effect:+.1f}")

    print(f"\n      -> Encoding effect is stronger in {'SNN' if abs(snn_encoding_effect) > abs(ann_encoding_effect) else 'ANN'}")

    print("\n   3.2 Network Type Effect (SNN vs ANN):")

    diss_network_effect = gap_data['diss_snn'][-1] - gap_data['ann_diss'][-1]
    print(f"      Dissipative: SNN_gap - ANN_gap = {diss_network_effect:+.1f}")

    exp_network_effect = gap_data['exp_snn'][-1] - gap_data['ann_exp'][-1]
    print(f"      Expansive: SNN_gap - ANN_gap = {exp_network_effect:+.1f}")

    tran_network_effect = gap_data['tran_snn'][-1] - gap_data['ann_tran'][-1]
    print(f"      Transition:  SNN_gap - ANN_gap = {tran_network_effect:+.1f}")

    raw_vs_diss = gap_data['ann_raw'][-1] - gap_data['ann_diss'][-1]
    raw_vs_exp = gap_data['ann_raw'][-1] - gap_data['ann_exp'][-1]
    raw_vs_tran = gap_data['ann_raw'][-1] - gap_data['ann_tran'][-1]
    print(f"\n      Raw vs Encoded (ANN):")
    print(f"        Raw - Diss = {raw_vs_diss:+.1f}")
    print(f"        Raw - Tran = {raw_vs_tran:+.1f}")
    print(f"        Raw - Exp  = {raw_vs_exp:+.1f}")

    # 4. Ranking
    print("\n4. Generalization Ranking (Lower Gap is Better):")
    ranking = sorted(gap_data.items(), key=lambda x: x[1][-1])
    for rank, (group, gaps) in enumerate(ranking, 1):
        print(f"   {rank}. {group:<15} Avg Gap = {gaps[-1]:.1f}")

    return {
        'performance': perf_data,
        'gaps': gap_data,
        'ranking': ranking,
        'encoding_effect_snn': snn_encoding_effect,
        'encoding_effect_ann': ann_encoding_effect,
        'network_effect_diss': diss_network_effect,
        'network_effect_exp': exp_network_effect,
        'network_effect_tran': tran_network_effect
    }


# ============================================================
# Visualization
# ============================================================
def visualize_comprehensive_results(all_results, stats, output_dir='./experiment_outputs'):
    """Visualize comprehensive results."""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(24, 16))
    difficulties = ['easy', 'medium', 'hard', 'very_hard']

    groups = ['diss_snn', 'tran_snn', 'exp_snn', 'ann_raw', 'ann_diss', 'ann_tran', 'ann_exp']
    group_labels = ['Diss-SNN', 'Tran-SNN', 'Exp-SNN', 'ANN-Raw', 'ANN-Diss', 'ANN-Tran', 'ANN-Exp']
    colors = ['blue', 'orange', 'red', 'green', 'cyan', 'purple', 'magenta']

    perf_data = stats['performance']
    gap_data = stats['gaps']

    # Subplot 1: Performance Comparison
    ax1 = plt.subplot(3, 4, 1)
    x = np.arange(len(difficulties))
    num_groups = len(groups)
    width = 0.8 / num_groups

    for i, (group, color, label) in enumerate(zip(groups, colors, group_labels)):
        perfs = perf_data[group]
        ax1.bar(x + (i - num_groups / 2 + 0.5) * width, perfs, width, label=label, alpha=0.8, color=color)

    ax1.set_ylabel('Mean Reward')
    ax1.set_title(f'Performance Comparison ({num_groups} Groups)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in difficulties], rotation=15)
    ax1.axhline(y=200, color='black', linestyle='--', alpha=0.3)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Gap Comparison
    ax2 = plt.subplot(3, 4, 2)
    x2 = np.arange(3)
    for i, (group, color, label) in enumerate(zip(groups, colors, group_labels)):
        gaps = gap_data[group][:3]
        ax2.plot(x2, gaps, marker='o', label=label, color=color, linewidth=2)

    ax2.set_ylabel('Generalization Gap')
    ax2.set_title('Gap Curve (Lower is Better)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(['Medium', 'Hard', 'Very_hard'], rotation=15)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Average Gap Ranking
    ax3 = plt.subplot(3, 4, 3)
    avg_gaps_dict = {g: gap_data[g][-1] for g in groups}
    sorted_groups = sorted(groups, key=lambda g: avg_gaps_dict[g])
    sorted_labels = [group_labels[groups.index(g)] for g in sorted_groups]
    sorted_colors = [colors[groups.index(g)] for g in sorted_groups]
    sorted_gaps = [avg_gaps_dict[g] for g in sorted_groups]

    bars = ax3.barh(sorted_labels, sorted_gaps, color=sorted_colors, alpha=0.7)
    ax3.set_xlabel('Average Gap')
    ax3.set_title('Generalization Ranking')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    for bar, val in zip(bars, sorted_gaps):
        width = bar.get_width()
        ax3.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}', ha='left', va='center', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')

    # Subplot 4: Encoding Effect
    ax4 = plt.subplot(3, 4, 4)
    encoding_effects = [
        stats['encoding_effect_snn'],
        stats['encoding_effect_ann']
    ]
    x4 = np.arange(2)
    bars = ax4.bar(x4, encoding_effects, color=['blue', 'green'], alpha=0.7)
    ax4.set_ylabel('Encoding Effect (Exp - Diss)')
    ax4.set_title('Encoding Effect: SNN vs ANN')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(['SNN', 'ANN'])
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, encoding_effects):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{val:+.1f}', ha='center', va='bottom' if val > 0 else 'top',
                 fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Subplots 5-8: Distribution per Difficulty
    for idx, diff in enumerate(difficulties):
        ax = plt.subplot(3, 4, 5 + idx)
        for group, color, label in zip(groups, colors, group_labels):
            rewards_all = []
            for result in all_results[group]:
                rewards_all.extend(result['eval'][diff]['rewards'])
            ax.hist(rewards_all, bins=20, alpha=0.4, label=label, color=color)

        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{diff.capitalize()} Environment')
        ax.axvline(x=195, color='black', linestyle='--', alpha=0.3)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')

    # Subplot 9: Success Rate
    ax9 = plt.subplot(3, 4, 9)
    for group, color, label in zip(groups, colors, group_labels):
        srs = []
        for diff in difficulties:
            sr_list = [r['eval'][diff]['success_rate'] for r in all_results[group]]
            srs.append(np.mean(sr_list))
        ax9.plot(difficulties, srs, marker='o', label=label, color=color, linewidth=2)

    ax9.set_ylabel('Success Rate')
    ax9.set_xlabel('Difficulty')
    ax9.set_title('Success Rate Curve')
    ax9.set_ylim([0, 1.05])
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3)

    # Subplot 10: Performance Retention
    ax10 = plt.subplot(3, 4, 10)
    for group, color, label in zip(groups, colors, group_labels):
        perfs = perf_data[group]
        baseline = perfs[0]
        retention = [p / (baseline + 1e-8) * 100 for p in perfs]
        ax10.plot(difficulties, retention, marker='s', label=label, color=color, linewidth=2)

    ax10.set_ylabel('Performance Retention (%)')
    ax10.set_xlabel('Difficulty')
    ax10.set_title('Retention (vs Easy)')
    ax10.axhline(y=100, color='black', linestyle='--', alpha=0.3)
    ax10.legend(fontsize=7)
    ax10.grid(True, alpha=0.3)

    # Subplot 11: Network Type Effect
    ax11 = plt.subplot(3, 4, 11)
    network_effects = [
        stats['network_effect_diss'],
        stats['network_effect_tran'],
        stats['network_effect_exp']
    ]
    x11 = np.arange(3)
    bars = ax11.bar(x11, network_effects, color=['blue', 'orange', 'red'], alpha=0.7)
    ax11.set_ylabel('Network Effect (SNN_gap - ANN_gap)')
    ax11.set_title('Network Type Effect')
    ax11.set_xticks(x11)
    ax11.set_xticklabels(['Dissipative', 'Transition', 'Expansive'])
    ax11.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, network_effects):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width() / 2, height,
                  f'{val:+.1f}', ha='center', va='bottom' if val > 0 else 'top',
                  fontsize=10, fontweight='bold')
    ax11.grid(True, alpha=0.3, axis='y')

    # Subplot 12: Gap Heatmap
    ax12 = plt.subplot(3, 4, 12)
    heatmap_data = []
    for group in groups:
        heatmap_data.append(gap_data[group][:3])

    heatmap_data = np.array(heatmap_data)
    im = ax12.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax12.set_yticks(range(len(groups)))
    ax12.set_yticklabels(group_labels)
    ax12.set_xticks(range(3))
    ax12.set_xticklabels(['Medium', 'Hard', 'V_hard'])
    ax12.set_title('Gap Heatmap (Cooler is Better)')

    for i in range(len(groups)):
        for j in range(3):
            text = ax12.text(j, i, f'{heatmap_data[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax12, label='Gap')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_file = os.path.join(output_dir, f'comprehensive_viz_{timestamp}.png')
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {fig_file}")

    plt.show()


# ============================================================
# Save Results
# ============================================================
def save_comprehensive_results(all_results, stats, output_dir='./experiment_outputs'):
    """Save comprehensive results."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*70)
    print("Saving Experiment Data...")
    print("="*70)

    # 1. Training Curves
    print("\n[1/8] Saving training curves...")
    training_curves = []
    for group in all_results.keys():
        for result in all_results[group]:
            run_id = result['run_id']
            agent_type = result['agent_type']
            delta = result.get('delta', None)
            seed = result['seed']
            train_curve = result['train_curve']
            for episode, reward in enumerate(train_curve):
                training_curves.append({
                    'group': group, 'run_id': run_id, 'agent_type': agent_type,
                    'delta': delta, 'seed': seed, 'episode': episode + 1, 'reward': reward
                })
    training_file = os.path.join(output_dir, f'training_curves_{timestamp}.csv')
    pd.DataFrame(training_curves).to_csv(training_file, index=False)
    print(f"  ✓ Training curves ({len(training_curves)} rows): {training_file}")

    # 2. Training Stats
    print("\n[2/8] Saving training stats...")
    training_stats = []
    for group in all_results.keys():
        all_curves = [result['train_curve'] for result in all_results[group]]
        min_length = min(len(curve) for curve in all_curves)
        aligned_curves = np.array([curve[:min_length] for curve in all_curves])
        
        mean_curve = np.mean(aligned_curves, axis=0)
        std_curve = np.std(aligned_curves, axis=0)
        min_curve = np.min(aligned_curves, axis=0)
        max_curve = np.max(aligned_curves, axis=0)
        median_curve = np.median(aligned_curves, axis=0)

        for episode in range(min_length):
            training_stats.append({
                'group': group, 'episode': episode + 1,
                'mean_reward': mean_curve[episode], 'std_reward': std_curve[episode],
                'min_reward': min_curve[episode], 'max_reward': max_curve[episode],
                'median_reward': median_curve[episode], 'num_runs': len(all_curves)
            })
    training_stats_file = os.path.join(output_dir, f'training_stats_{timestamp}.csv')
    pd.DataFrame(training_stats).to_csv(training_stats_file, index=False)
    print(f"  ✓ Training stats ({len(training_stats)} rows): {training_stats_file}")

    # 3. Evaluation Detailed
    print("\n[3/8] Saving evaluation details...")
    eval_detailed = []
    for group in all_results.keys():
        for result in all_results[group]:
            run_id = result['run_id']
            agent_type = result['agent_type']
            delta = result.get('delta', None)
            seed = result['seed']
            for diff, eval_result in result['eval'].items():
                for ep_idx, reward in enumerate(eval_result['rewards']):
                    eval_detailed.append({
                        'group': group, 'run_id': run_id, 'agent_type': agent_type,
                        'delta': delta, 'seed': seed, 'difficulty': diff,
                        'eval_episode': ep_idx + 1, 'reward': reward,
                        'success': 1 if reward >= 195 else 0
                    })
    eval_detail_file = os.path.join(output_dir, f'eval_detailed_{timestamp}.csv')
    pd.DataFrame(eval_detailed).to_csv(eval_detail_file, index=False)
    print(f"  ✓ Eval details ({len(eval_detailed)} rows): {eval_detail_file}")

    # 4. Evaluation Summary
    print("\n[4/8] Saving evaluation summary...")
    eval_summary = []
    for group in all_results.keys():
        for result in all_results[group]:
            run_id = result['run_id']
            agent_type = result['agent_type']
            delta = result.get('delta', None)
            seed = result['seed']
            for diff, eval_result in result['eval'].items():
                eval_summary.append({
                    'group': group, 'run_id': run_id, 'agent_type': agent_type,
                    'delta': delta, 'seed': seed, 'difficulty': diff,
                    'mean_reward': eval_result['mean'], 'std_reward': eval_result['std'],
                    'median_reward': eval_result['median'], 'success_rate': eval_result['success_rate'],
                    'min_reward': np.min(eval_result['rewards']), 'max_reward': np.max(eval_result['rewards'])
                })
    eval_summary_file = os.path.join(output_dir, f'eval_summary_{timestamp}.csv')
    pd.DataFrame(eval_summary).to_csv(eval_summary_file, index=False)
    print(f"  ✓ Eval summary ({len(eval_summary)} rows): {eval_summary_file}")

    # 5. Gap Analysis
    print("\n[5/8] Saving gap analysis...")
    gap_analysis = []
    for group in all_results.keys():
        for result in all_results[group]:
            run_id = result['run_id']
            agent_type = result['agent_type']
            delta = result.get('delta', None)
            seed = result['seed']
            easy_reward = result['eval']['easy']['mean']
            medium_reward = result['eval']['medium']['mean']
            hard_reward = result['eval']['hard']['mean']
            vh_reward = result['eval']['very_hard']['mean']
            gap_analysis.append({
                'group': group, 'run_id': run_id, 'agent_type': agent_type,
                'delta': delta, 'seed': seed,
                'easy_reward': easy_reward, 'medium_reward': medium_reward,
                'hard_reward': hard_reward, 'very_hard_reward': vh_reward,
                'gap_easy_medium': easy_reward - medium_reward,
                'gap_easy_hard': easy_reward - hard_reward,
                'gap_easy_vh': easy_reward - vh_reward,
                'gap_medium_hard': medium_reward - hard_reward,
                'gap_hard_vh': hard_reward - vh_reward,
                'avg_gap': (easy_reward - medium_reward + easy_reward - hard_reward + easy_reward - vh_reward) / 3,
                'performance_retention': vh_reward / (easy_reward + 1e-8) * 100
            })
    gap_file = os.path.join(output_dir, f'gap_analysis_{timestamp}.csv')
    pd.DataFrame(gap_analysis).to_csv(gap_file, index=False)
    print(f"  ✓ Gap analysis ({len(gap_analysis)} rows): {gap_file}")

    # 6. Stats Summary
    print("\n[6/8] Saving stats summary...")
    summary_data = []
    for group in stats['performance'].keys():
        for i, diff in enumerate(['easy', 'medium', 'hard', 'very_hard']):
            summary_data.append({
                'group': group, 'difficulty': diff,
                'mean_performance': stats['performance'][group][i],
                'gap_from_easy': stats['gaps'][group][i-1] if i > 0 else 0
            })
    summary_file = os.path.join(output_dir, f'stats_summary_{timestamp}.csv')
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"  ✓ Stats summary ({len(summary_data)} rows): {summary_file}")

    # 7. Variance Analysis
    print("\n[7/8] Saving variance analysis...")
    variance_analysis = []
    for group in all_results.keys():
        for diff in ['easy', 'medium', 'hard', 'very_hard']:
            rewards_across_runs = [result['eval'][diff]['mean'] for result in all_results[group]]
            success_rates = [result['eval'][diff]['success_rate'] for result in all_results[group]]
            variance_analysis.append({
                'group': group, 'difficulty': diff,
                'mean_across_runs': np.mean(rewards_across_runs),
                'std_across_runs': np.std(rewards_across_runs),
                'min_across_runs': np.min(rewards_across_runs),
                'max_across_runs': np.max(rewards_across_runs),
                'median_across_runs': np.median(rewards_across_runs),
                'cv_across_runs': np.std(rewards_across_runs) / (np.mean(rewards_across_runs) + 1e-8),
                'mean_success_rate': np.mean(success_rates),
                'std_success_rate': np.std(success_rates)
            })
    variance_file = os.path.join(output_dir, f'variance_analysis_{timestamp}.csv')
    pd.DataFrame(variance_analysis).to_csv(variance_file, index=False)
    print(f"  ✓ Variance analysis ({len(variance_analysis)} rows): {variance_file}")

    # 8. Metadata
    print("\n[8/8] Saving metadata...")
    training_episodes_stats = {}
    for group in all_results.keys():
        episodes_list = [len(r['train_curve']) for r in all_results[group]]
        training_episodes_stats[group] = {
            'mean': float(np.mean(episodes_list)),
            'min': int(np.min(episodes_list)),
            'max': int(np.max(episodes_list))
        }

    metadata = {
        'timestamp': timestamp,
        'version': 'Enhanced',
        'groups': list(all_results.keys()),
        'num_runs_per_group': len(all_results['diss_snn']),
        'total_experiments': len(all_results.keys()) * len(all_results['diss_snn']),
        'effects': {
            'encoding_effect_snn': float(stats['encoding_effect_snn']),
            'encoding_effect_ann': float(stats['encoding_effect_ann']),
            'network_effect_diss': float(stats['network_effect_diss']),
            'network_effect_tran': float(stats.get('network_effect_tran', 0)),
            'network_effect_exp': float(stats['network_effect_exp'])
        },
        'ranking': [{'group': g, 'avg_gap': float(gaps[-1])} for g, gaps in stats['ranking']],
        'training_episodes': training_episodes_stats,
        'files_generated': {
            'training_curves': f'training_curves_{timestamp}.csv',
            'training_stats': f'training_stats_{timestamp}.csv',
            'eval_detailed': f'eval_detailed_{timestamp}.csv',
            'eval_summary': f'eval_summary_{timestamp}.csv',
            'gap_analysis': f'gap_analysis_{timestamp}.csv',
            'stats_summary': f'stats_summary_{timestamp}.csv',
            'variance_analysis': f'variance_analysis_{timestamp}.csv',
            'metadata': f'metadata_{timestamp}.json'
        },
        'data_stats': {
            'total_training_data_points': len(training_curves),
            'total_eval_data_points': len(eval_detailed),
            'total_gap_analyses': len(gap_analysis)
        }
    }
    metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata: {metadata_file}")

    print("\n" + "="*70)
    print(f"✓ All data saved to: {os.path.abspath(output_dir)}")
    print("="*70)
    return metadata


# ============================================================
# Main
# ============================================================
def main():
    """Main experiment flow."""
    print("=" * 70)
    print("Enhanced SNN vs ANN Comprehensive Comparison")
    print("=" * 70)

    output_dir = './experiment_outputs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results directory: {os.path.abspath(output_dir)}")

    num_runs = 10
    train_episodes = 5000

    all_results = run_comprehensive_experiments(
        num_runs=num_runs,
        train_episodes=train_episodes,
        use_eval_seed=True
    )

    stats = compute_comprehensive_statistics(all_results)
    save_comprehensive_results(all_results, stats, output_dir=output_dir)
    visualize_comprehensive_results(all_results, stats, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("✓ Experiment Completed!")
    print("=" * 70)

    return all_results, stats


if __name__ == "__main__":
    results, stats = main()