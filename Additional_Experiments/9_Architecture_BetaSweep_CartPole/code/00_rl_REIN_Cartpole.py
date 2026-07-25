"""CartPole REINFORCE Experiment: Beta Sweep Analysis for Leaky SNN Agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
import numpy as np
import gymnasium as gym
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# ============================================================
# Configuration
# ============================================================
@dataclass
class REINFORCEConfig:
    lr: float
    grad_clip: float
    hidden_dim: int
    max_episodes: int
    convergence_threshold: float = 195.0
    patience: int = 1000
    eval_interval: int = 50


# REINFORCE configurations - aligned with LunarLander structure
REINFORCE_CONFIGS = {
    'ann': REINFORCEConfig(
        lr=5e-4, grad_clip=1.0, hidden_dim=256, max_episodes=3000
    ),
    'lstm': REINFORCEConfig(
        lr=5e-4, grad_clip=1.0, hidden_dim=256, max_episodes=3000
    ),
    'rleaky_fixed': REINFORCEConfig(
        lr=5e-4, grad_clip=1.0, hidden_dim=256, max_episodes=6000
    ),
    'leaky_fixed': REINFORCEConfig(
        lr=5e-4, grad_clip=1.0, hidden_dim=256, max_episodes=6000
    ),
}


# ============================================================
# Custom CartPole with Difficulty Levels
# ============================================================
class FixedCartPole(gym.Env):
    """CartPole wrapper with difficulty levels - aligned with LunarLander style."""

    def __init__(self, difficulty='easy'):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-4.8, -np.inf, -0.418, -np.inf]),
            high=np.array([4.8, np.inf, 0.418, np.inf]),
            dtype=np.float32
        )
        self.max_steps = 200
        self.difficulty = difficulty

        # Difficulty configs: (pole_length, pole_mass, noise, init_range)
        configs = {
            'easy': (0.5, 0.1, 0.0, 0.05),
            'medium': (0.8, 0.3, 0.005, 0.08),
            'hard': (1.2, 0.5, 0.01, 0.10),
            'very_hard': (1.5, 0.7, 0.015, 0.12)
        }


        self.pole_length, self.pole_mass, self.noise, self.init_range = configs[difficulty]
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

        terminated = bool(x < -2.4 or x > 2.4 or theta < -0.209 or theta > 0.209)
        truncated = self.steps >= self.max_steps
        reward = 1.0 if not terminated else 0.0

        return self.state, reward, terminated, truncated, {}

    def close(self):
        pass


# ============================================================
# Network Architectures (Actor Networks - aligned with LunarLander)
# ============================================================

class SimpleANN(nn.Module):
    """ANN Actor for CartPole"""

    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x, state=None):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x, None


class ImprovedLSTM(nn.Module):
    """LSTM Actor for CartPole (Aligned to 2 Layers)"""

    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.orthogonal_(self.fc.weight, gain=0.01)

    def forward(self, x, hidden=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        out, hidden = self.lstm(x, hidden)

        out = self.ln(out[:, -1, :])
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))


class RLeakyFixedSNN(nn.Module):
    """RLeaky SNN Actor (Aligned to 2 Hidden Layers)"""

    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2, beta=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.snn_type = 'RLeaky'

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.RLeaky(
            beta=beta,
            linear_features=hidden_dim,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            learn_beta=False,
            learn_recurrent=True,
        )

        # Layer 2 (New)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.RLeaky(
            beta=beta,
            linear_features=hidden_dim,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            learn_beta=False,
            learn_recurrent=True,
        )

        # Output Layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

        # Initialize recurrent weights
        if hasattr(self.lif1, 'recurrent'):
            nn.init.orthogonal_(self.lif1.recurrent.weight, gain=0.5)
        if hasattr(self.lif2, 'recurrent'):
            nn.init.orthogonal_(self.lif2.recurrent.weight, gain=0.5)

    def forward(self, x, state=None):
        # State handling for 2 layers: state is ((spk1, mem1), (spk2, mem2))
        if state is None:
            spk1, mem1 = self.lif1.init_rleaky()
            spk2, mem2 = self.lif2.init_rleaky()

            # Move to device
            spk1, mem1 = spk1.to(x.device), mem1.to(x.device)
            spk2, mem2 = spk2.to(x.device), mem2.to(x.device)
        else:
            (spk1, mem1), (spk2, mem2) = state

        # Layer 1
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, spk1, mem1)

        # Layer 2
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, spk2, mem2)

        # Output
        output = self.fc3(spk2)

        return output, ((spk1.detach(), mem1.detach()), (spk2.detach(), mem2.detach()))

    def get_beta_value(self):
        return self.beta


class LeakyFixedSNN(nn.Module):
    """Leaky SNN Actor (Aligned to 2 Hidden Layers)"""

    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2, beta=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.snn_type = 'Leaky'

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            learn_beta=False,
            learn_threshold=False,
        )

        # Layer 2 (New)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(slope=25),
            learn_beta=False,
            learn_threshold=False,
        )

        # Output Layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)

    def forward(self, x, state=None):
        # State handling for 2 layers: state is (mem1, mem2)
        if state is None:
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem1, mem2 = mem1.to(x.device), mem2.to(x.device)
        else:
            mem1, mem2 = state

        # Layer 1
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)

        # Layer 2
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        # Output
        output = self.fc3(spk2)

        return output, (mem1.detach(), mem2.detach())

    def get_beta_value(self):
        return self.beta


# ============================================================
# REINFORCE Agent Classes
# ============================================================

class REINFORCEAgent:
    """Base REINFORCE Agent - aligned with LunarLander PPO structure"""

    def __init__(self, config: REINFORCEConfig, actor, device='cuda'):
        self.config = config
        self.device = device
        self.gamma = 0.99

        self.actor = actor.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr
        )

        self.actor_state = None
        self.log_probs_buffer = []
        self.rewards_buffer = []
        self.training_rewards = []
        self.eval_history = []

    def reset_episode_state(self):
        """Reset actor state at episode start"""
        self.actor_state = None

    def select_action(self, state, training=True):
        """Select action - aligned with LunarLander interface"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if training:
            self.actor.train()

            # Get action from actor
            if hasattr(self.actor, 'init_hidden'):  # LSTM
                if self.actor_state is None:
                    self.actor_state = self.actor.init_hidden(1, self.device)
                output, self.actor_state = self.actor(state_tensor, self.actor_state)
                self.actor_state = tuple(h.detach() for h in self.actor_state)
            else:  # ANN or SNN
                output, self.actor_state = self.actor(state_tensor, self.actor_state)

            probs = F.softmax(output, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            self.log_probs_buffer.append(log_prob)

            return action.item()

        else:  # Evaluation
            self.actor.eval()
            with torch.no_grad():
                if hasattr(self.actor, 'init_hidden'):
                    if self.actor_state is None:
                        self.actor_state = self.actor.init_hidden(1, self.device)
                    output, self.actor_state = self.actor(state_tensor, self.actor_state)
                else:
                    output, self.actor_state = self.actor(state_tensor, self.actor_state)

                probs = F.softmax(output, dim=-1)
                action = torch.argmax(probs, dim=-1)
                return action.item()

    def store_reward(self, reward):
        """Store reward for episode"""
        self.rewards_buffer.append(reward)

    def update_at_episode_end(self):
        """REINFORCE update at episode end"""
        if not self.log_probs_buffer:
            return 0.0, {}

        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.rewards_buffer):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        loss_terms = [-log_prob * G for log_prob, G in zip(self.log_probs_buffer, returns)]

        self.actor_optimizer.zero_grad()
        loss = torch.stack(loss_terms).sum()
        loss.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           max_norm=self.config.grad_clip)

        self.actor_optimizer.step()

        loss_val = loss.item()

        # Clear buffers
        self.log_probs_buffer = []
        self.rewards_buffer = []

        return loss_val, {}

    def get_learned_dynamics(self):
        """Get beta value for SNNs"""
        if hasattr(self.actor, 'get_beta_value'):
            return {
                'beta': self.actor.get_beta_value(),
                'snn_type': getattr(self.actor, 'snn_type', None)
            }
        return {'beta': None, 'snn_type': None}


# ============================================================
# Agent Factory
# ============================================================

def create_reinforce_agent(agent_type: str, config: REINFORCEConfig, beta=0.9, device='cuda'):
    """
    Factory function to create REINFORCE agents
    Aligned with LunarLander's create_ppo_agent structure
    """
    if agent_type == 'ann':
        actor = SimpleANN(4, config.hidden_dim, 2)
        return REINFORCEAgent(config, actor, device)

    elif agent_type == 'lstm':
        actor = ImprovedLSTM(4, config.hidden_dim, 2)
        return REINFORCEAgent(config, actor, device)

    elif agent_type == 'rleaky_fixed':
        actor = RLeakyFixedSNN(4, config.hidden_dim, 2, beta=beta)
        return REINFORCEAgent(config, actor, device)

    elif agent_type == 'leaky_fixed':
        actor = LeakyFixedSNN(4, config.hidden_dim, 2, beta=beta)
        return REINFORCEAgent(config, actor, device)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================
# Training & Evaluation (aligned with LunarLander)
# ============================================================

def train_reinforce_until_convergence(agent, config, env_difficulty='easy', verbose=False, run_id=0):
    """Train REINFORCE agent until convergence - aligned with LunarLander structure"""
    env = FixedCartPole(env_difficulty)

    episode = 0
    converged = False
    converged_at = None
    no_improvement_count = 0
    best_eval_mean = 0

    while episode < config.max_episodes and not converged:
        state, _ = env.reset()
        agent.reset_episode_state()
        ep_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, term, trunc, _ = env.step(action)
            agent.store_reward(reward)
            ep_reward += reward
            state = next_state
            done = term or trunc

        # Update at episode end (REINFORCE)
        loss, info = agent.update_at_episode_end()
        agent.training_rewards.append(ep_reward)
        episode += 1

        # Evaluation
        if episode % config.eval_interval == 0:
            eval_result = evaluate_agent(agent, env_difficulty, num_episodes=20, seed=1234)
            eval_mean = eval_result['mean']

            agent.eval_history.append({
                'episode': episode,
                'eval_mean': eval_mean,
                'train_mean': np.mean(agent.training_rewards[-config.eval_interval:])
            })

            if verbose and episode % 200 == 0:
                info_str = ""
                dyn = agent.get_learned_dynamics()
                if dyn.get('beta') is not None:
                    info_str += f", β={dyn['beta']:.4f}"
                if dyn.get('snn_type') is not None:
                    info_str = f" [{dyn['snn_type']}]" + info_str

                print(f"    Run {run_id + 1}, Ep {episode}/{config.max_episodes}: "
                      f"Train={agent.eval_history[-1]['train_mean']:.1f}, "
                      f"Eval={eval_mean:.1f}{info_str}")

            # Check convergence
            if eval_mean >= config.convergence_threshold:
                if converged_at is None:
                    converged_at = episode
                    if verbose:
                        print(f"    >>> Converged at episode {episode}!")
                converged = True
                break

            # Early stopping
            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                no_improvement_count = 0
            else:
                no_improvement_count += config.eval_interval

            if no_improvement_count >= config.patience:
                if verbose:
                    print(f"    Stopping early at episode {episode}")
                break

    env.close()

    return {
        'converged': converged,
        'converged_at': converged_at if converged_at else episode,
        'final_episode': episode,
        'training_rewards': agent.training_rewards,
        'eval_history': agent.eval_history,
        'best_eval_mean': best_eval_mean
    }


def evaluate_agent(agent, difficulty, num_episodes=100, seed=None):
    """Evaluate agent - aligned with LunarLander"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = FixedCartPole(difficulty)
    all_rewards = []

    for ep_idx in range(num_episodes):
        ep_seed = seed + ep_idx if seed is not None else None
        state, _ = env.reset(seed=ep_seed)
        agent.reset_episode_state()

        ep_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        all_rewards.append(ep_reward)

    env.close()

    all_rewards = np.array(all_rewards)
    success_count = np.sum(all_rewards >= 195)

    return {
        'mean': float(np.mean(all_rewards)),
        'std': float(np.std(all_rewards)),
        'median': float(np.median(all_rewards)),
        'min': float(np.min(all_rewards)),
        'max': float(np.max(all_rewards)),
        'success_rate': float(success_count / num_episodes),
        'rewards': all_rewards.tolist()
    }


def run_single_reinforce_trial(agent_type: str, config: REINFORCEConfig, run_id: int,
                               verbose=False, beta=0.9):
    """Run single REINFORCE trial - aligned with LunarLander"""
    seed = 2025 + run_id * 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if verbose:
        print(f"\n  [{agent_type.upper()}] Run {run_id + 1}")
        print("  " + "-" * 60)

    device = 'cpu'
    agent = create_reinforce_agent(agent_type, config, beta=beta, device=device)

    # Train
    train_result = train_reinforce_until_convergence(
        agent, config,
        env_difficulty='easy',
        verbose=verbose,
        run_id=run_id
    )

    if verbose:
        status = "CONVERGED" if train_result['converged'] else "MAX EPISODES"
        print(f"  Training: {status} at episode {train_result['converged_at']}")
        print(f"  Best eval on Easy: {train_result['best_eval_mean']:.1f}")

    # Evaluate on all difficulties
    difficulties = ['easy', 'medium', 'hard', 'very_hard']
    eval_results = {}

    for diff in difficulties:
        eval_result = evaluate_agent(agent, diff, num_episodes=100, seed=seed)
        eval_results[diff] = eval_result

        if verbose:
            print(f"  {diff:12s}: {eval_result['mean']:6.1f} ± {eval_result['std']:5.1f} "
                  f"(success: {eval_result['success_rate']:.1%})")

    # Gap
    gap = eval_results['easy']['mean'] - eval_results['very_hard']['mean']
    if verbose:
        print(f"  {'Gap':12s}: {gap:6.1f}")

    # Get learned dynamics
    learned_dynamics = agent.get_learned_dynamics()
    if verbose and learned_dynamics.get('beta') is not None:
        print(f"  {'β':12s}: {learned_dynamics['beta']:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'agent_type': agent_type,
        'run_id': run_id,
        'seed': seed,
        'config': asdict(config),
        'training': train_result,
        'evaluation': eval_results,
        'gap': gap,
        'learned_dynamics': learned_dynamics
    }


# ============================================================
# Beta Sweep (aligned with LunarLander)
# ============================================================

BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def run_reinforce_beta_sweep(num_runs=3, verbose=True, output_dir='results_cartpole_reinforce'):
    """Run REINFORCE beta sweep - aligned with LunarLander"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        CARTPOLE REINFORCE BETA SWEEP EXPERIMENT              ║
    ╚══════════════════════════════════════════════════════════════╝

    Testing dissipation-recurrence with REINFORCE (simple algorithm):
    - Beta values: """ + str(BETA_VALUES) + """
    - Architectures: Leaky, RLeaky (Fixed threshold)
    - Algorithm: REINFORCE (Policy Gradient)
    - Runs: """ + str(num_runs) + """
    """)

    all_results = []

    for arch_type in ['leaky', 'rleaky']:
        for beta in BETA_VALUES:
            config_key = f'{arch_type}_fixed'
            config = REINFORCE_CONFIGS[config_key]

            for run_id in range(num_runs):
                print(f"\n{'=' * 70}")
                print(f"{arch_type.upper()} | β={beta:.2f} | Run {run_id + 1}/{num_runs}")
                print("=" * 70)

                result = run_single_reinforce_trial(
                    config_key, config, run_id,
                    verbose=True,
                    beta=beta
                )

                all_results.append(result)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / 'cartpole_reinforce_beta_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path / 'cartpole_reinforce_beta_sweep.json'}")

    return all_results


def visualize_reinforce_results(results, output_dir='results_cartpole_reinforce'):
    """Visualize REINFORCE results - aligned with LunarLander style"""

    # Organize data
    data = {}
    for r in results:
        arch = r['agent_type'].split('_')[0]
        beta = r['learned_dynamics']['beta']

        key = f"{arch}_{beta}"
        if key not in data:
            data[key] = []
        data[key].append(r)

    # Aggregate
    stats = {}
    for arch in ['leaky', 'rleaky']:
        stats[arch] = {}
        for beta in BETA_VALUES:
            key = f"{arch}_{beta}"
            if key in data:
                gaps = [r['gap'] for r in data[key]]
                episodes = [r['training']['converged_at'] for r in data[key]]
                success_rates = [r['evaluation']['easy']['success_rate'] for r in data[key]]

                stats[arch][beta] = {
                    'gap_mean': np.mean(gaps),
                    'gap_std': np.std(gaps),
                    'episodes_mean': np.mean(episodes),
                    'success_rate_mean': np.mean(success_rates)
                }

    # Plot
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3, figure=fig)

    colors = {'leaky': 'green', 'rleaky': 'blue'}

    # Gap plot
    ax1 = fig.add_subplot(gs[0, 0])
    for arch in ['leaky', 'rleaky']:
        betas = sorted(stats[arch].keys())
        gaps = [stats[arch][b]['gap_mean'] for b in betas]
        stds = [stats[arch][b]['gap_std'] for b in betas]

        ax1.plot(betas, gaps, 'o-', color=colors[arch],
                 label=arch.upper(), linewidth=2, markersize=8)
        ax1.fill_between(betas,
                         np.array(gaps) - np.array(stds),
                         np.array(gaps) + np.array(stds),
                         color=colors[arch], alpha=0.2)

    ax1.set_xlabel('β (Leak parameter)', fontsize=12)
    ax1.set_ylabel('Generalization Gap', fontsize=12)
    ax1.set_title('CartPole REINFORCE: β vs Gap', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episodes plot
    ax2 = fig.add_subplot(gs[0, 1])
    for arch in ['leaky', 'rleaky']:
        betas = sorted(stats[arch].keys())
        episodes = [stats[arch][b]['episodes_mean'] for b in betas]

        ax2.plot(betas, episodes, 'o-', color=colors[arch],
                 label=arch.upper(), linewidth=2, markersize=8)

    ax2.set_xlabel('β (Leak parameter)', fontsize=12)
    ax2.set_ylabel('Episodes to Convergence', fontsize=12)
    ax2.set_title('CartPole REINFORCE: β vs Learning Speed', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Success rate plot
    ax3 = fig.add_subplot(gs[0, 2])
    for arch in ['leaky', 'rleaky']:
        betas = sorted(stats[arch].keys())
        success_rates = [stats[arch][b]['success_rate_mean'] * 100 for b in betas]

        ax3.plot(betas, success_rates, 'o-', color=colors[arch],
                 label=arch.upper(), linewidth=2, markersize=8)

    ax3.set_xlabel('β (Leak parameter)', fontsize=12)
    ax3.set_ylabel('Success Rate (%)', fontsize=12)
    ax3.set_title('CartPole REINFORCE: β vs Success Rate', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])

    plt.tight_layout()

    output_path = Path(output_dir)
    save_path = output_path / 'cartpole_reinforce_beta_sweep.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {save_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='beta_sweep',
                        choices=['beta_sweep', 'quick_test', 'single_agent'],
                        help='Experiment mode')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per configuration')
    parser.add_argument('--output_dir', type=str, default='results_cartpole_reinforce',
                        help='Output directory')
    parser.add_argument('--agent_type', type=str, default='ann',
                        choices=['ann', 'lstm', 'leaky_fixed', 'rleaky_fixed'],
                        help='Agent type for quick_test or single_agent mode')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Beta value for SNN agents')
    args = parser.parse_args()

    if args.mode == 'quick_test':
        print(f"REINFORCE Quick test: Running {args.agent_type.upper()} with β={args.beta}")
        print("=" * 70)

        config = REINFORCE_CONFIGS.get(args.agent_type, REINFORCE_CONFIGS['ann'])

        result = run_single_reinforce_trial(
            args.agent_type,
            config,
            0,
            verbose=True,
            beta=args.beta
        )

        print("\n" + "=" * 70)
        print("✓ REINFORCE Quick test complete!")
        print("=" * 70)
        print(f"\nResults summary:")
        print(f"  Agent: {args.agent_type.upper()}")
        print(f"  Beta: {args.beta}")
        print(f"  Converged: {result['training']['converged']}")
        print(f"  Episodes: {result['training']['converged_at']}")
        print(f"  Easy score: {result['evaluation']['easy']['mean']:.1f}")
        print(f"  Success rate: {result['evaluation']['easy']['success_rate']:.1%}")
        print(f"  Gap: {result['gap']:.1f}")

    elif args.mode == 'single_agent':
        print(f"REINFORCE Single agent: {args.agent_type.upper()} | β={args.beta} | {args.runs} runs")
        print("=" * 70)

        config = REINFORCE_CONFIGS.get(args.agent_type, REINFORCE_CONFIGS['ann'])
        results = []

        for run_id in range(args.runs):
            print(f"\nRun {run_id + 1}/{args.runs}")
            print("-" * 70)

            result = run_single_reinforce_trial(
                args.agent_type,
                config,
                run_id,
                verbose=True,
                beta=args.beta
            )
            results.append(result)
        
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        json_filename = f'cartpole_reinforce_{args.agent_type}_results.json'
        if 'leaky' in args.agent_type:
            json_filename = f'cartpole_reinforce_{args.agent_type}_beta{args.beta}_results.json'

        with open(output_path / json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path / json_filename}")

        # Aggregate
        gaps = [r['gap'] for r in results]
        easy_scores = [r['evaluation']['easy']['mean'] for r in results]
        success_rates = [r['evaluation']['easy']['success_rate'] for r in results]
        episodes = [r['training']['converged_at'] for r in results]

        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS")
        print("=" * 70)
        print(f"Agent: {args.agent_type.upper()}")
        print(f"Beta: {args.beta}")
        print(f"Gap: {np.mean(gaps):.1f} ± {np.std(gaps):.1f}")
        print(f"Easy score: {np.mean(easy_scores):.1f} ± {np.std(easy_scores):.1f}")
        print(f"Success rate: {np.mean(success_rates):.1%} ± {np.std(success_rates):.1%}")
        print(f"Episodes: {np.mean(episodes):.0f} ± {np.std(episodes):.0f}")

    else:  # beta_sweep
        results = run_reinforce_beta_sweep(
            num_runs=args.runs,
            verbose=True,
            output_dir=args.output_dir
        )

        print("\nGenerating visualizations...")
        visualize_reinforce_results(results, args.output_dir)

        print("\n" + "=" * 70)
        print("✓ CARTPOLE REINFORCE EXPERIMENT COMPLETE")
        print("=" * 70)