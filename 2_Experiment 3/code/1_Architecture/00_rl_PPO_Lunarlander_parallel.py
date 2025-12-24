"""LunarLander PPO Experiment: Beta Sweep Analysis with Parallel Execution."""

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

# Import parallel utilities
from parallel_utils import (
    get_best_device,
    get_optimal_workers,
    run_experiments_parallel,
    create_experiment_configs,
    get_system_info
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# ============================================================
# Configuration
# ============================================================
@dataclass
class PPOConfig:
    lr_actor: float
    lr_critic: float
    grad_clip: float
    hidden_dim: int
    max_episodes: int

    # PPO-specific
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    convergence_threshold: float = 200.0
    patience: int = 2000
    eval_interval: int = 50

    # === TBPTT parameter ===
    tbptt_chunk_size: int = 64  # Truncate BPTT every N steps


# PPO configurations - tuned for LunarLander
PPO_CONFIGS = {
    'ann': PPOConfig(
        lr_actor=3e-4, lr_critic=1e-3, grad_clip=0.5,
        hidden_dim=256, max_episodes=15000,
        batch_size=2048,
        eval_interval=200
    ),
    'lstm': PPOConfig(
        lr_actor=3e-4, lr_critic=1e-3, grad_clip=0.5,
        hidden_dim=256, max_episodes=15000,
        batch_size=1024,
        eval_interval=200,
        tbptt_chunk_size=64
    ),
    'rleaky_fixed': PPOConfig(
        lr_actor=3e-4, lr_critic=1e-3, grad_clip=0.5,
        hidden_dim=256, max_episodes=15000,
        batch_size=1024,
        eval_interval=200,
        tbptt_chunk_size=64
    ),
    'leaky_fixed': PPOConfig(
        lr_actor=3e-4, lr_critic=1e-3, grad_clip=0.5,
        hidden_dim=256, max_episodes=15000,
        batch_size=2048,
        eval_interval=200
    ),
}


# ============================================================
# Custom LunarLander with Difficulty Levels
# ============================================================
class FixedLunarLander:
    """LunarLander wrapper with difficulty levels."""
    def __init__(self, difficulty='easy', render_mode=None):
        self.env = gym.make('LunarLander-v3', render_mode=render_mode)
        self.difficulty = difficulty

        # Difficulty configurations
        configs = {
            'easy': (0.8, 0.0, 0.0, 0.3, 2.0),
            'medium': (0.9, 3.0, 0.3, 0.6, 1.5),
            'hard': (1.0, 6.0, 0.7, 0.9, 1.2),
            'very_hard': (1.1, 8.0, 1.0, 1.2, 0.9)
        }

        self.gravity_scale, self.wind_power, self.turbulence, \
        self.init_vel_scale, self.landing_zone_scale = configs[difficulty]

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)

        if hasattr(self.env.unwrapped, 'lander'):
            lander = self.env.unwrapped.lander
            if self.init_vel_scale != 1.0:
                current_vel = lander.linearVelocity
                lander.linearVelocity = (
                    current_vel[0] * self.init_vel_scale,
                    current_vel[1] * self.init_vel_scale
                )

        if hasattr(self.env.unwrapped, 'world'):
            world = self.env.unwrapped.world
            world.gravity = (0, -10.0 * self.gravity_scale)

        return obs, info

    def step(self, action):
        if self.wind_power > 0 or self.turbulence > 0:
            if hasattr(self.env.unwrapped, 'lander'):
                lander = self.env.unwrapped.lander

                if self.wind_power > 0:
                    wind_force = np.random.uniform(-self.wind_power, self.wind_power)
                    lander.ApplyForceToCenter((wind_force, 0), True)

                if self.turbulence > 0:
                    turb_x = np.random.normal(0, self.turbulence)
                    turb_y = np.random.normal(0, self.turbulence)
                    lander.ApplyForceToCenter((turb_x, turb_y), True)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.landing_zone_scale != 1.0 and terminated:
            x_pos = obs[0]
            target_zone = 0.3 * self.landing_zone_scale
            if abs(x_pos) > target_zone:
                reward -= 50

        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


# ============================================================
# Network Architectures (Actor Networks)
# ============================================================

class SimpleANN(nn.Module):
    """ANN Actor for LunarLander"""
    def __init__(self, input_dim=8, hidden_dim=256, output_dim=4):
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
    """LSTM Actor for LunarLander (2 Layers)"""
    def __init__(self, input_dim=8, hidden_dim=256, output_dim=4, num_layers=2):
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
    """RLeaky SNN Actor (2 Hidden Layers)"""

    def __init__(self, input_dim=8, hidden_dim=256, output_dim=4, beta=0.9):
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

        # Layer 2
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

        if hasattr(self.lif1, 'recurrent'):
            nn.init.orthogonal_(self.lif1.recurrent.weight, gain=0.5)
        if hasattr(self.lif2, 'recurrent'):
            nn.init.orthogonal_(self.lif2.recurrent.weight, gain=0.5)

    def forward(self, x, state=None):
        if state is None:
            spk1, mem1 = self.lif1.init_rleaky()
            spk2, mem2 = self.lif2.init_rleaky()
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
    """Leaky SNN Actor (2 Hidden Layers)"""

    def __init__(self, input_dim=8, hidden_dim=256, output_dim=4, beta=0.9):
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

        # Layer 2
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
# Critic Networks (Value Networks for PPO)
# ============================================================

class ANNCritic(nn.Module):
    """ANN Critic (Value network)"""
    def __init__(self, input_dim=8, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)

    def forward(self, x, state=None):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x, None


# ============================================================
# PPO Agent Classes
# ============================================================

class PPOMemory:
    """Memory buffer for PPO"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get_batch(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )


class PPOAgent:
    """Base PPO Agent"""
    def __init__(self, config: PPOConfig, actor, critic, device='cpu'):
        self.config = config
        self.device = device

        self.actor = actor.to(device)
        self.critic = critic.to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )

        self.memory = PPOMemory()
        self.actor_state = None
        self.critic_state = None

        self.training_rewards = []
        self.eval_history = []

    def reset_episode_state(self):
        self.actor_state = None
        self.critic_state = None

    def _detach_state(self, state):
        """Recursively detach nested tuple states (for SNNs with multiple layers)"""
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.detach()
        if isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        return state

    def select_action(self, state, training=True):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if training:
            self.actor.train()
            self.critic.train()

            # Get action from actor
            if hasattr(self.actor, 'init_hidden'):  # LSTM
                if self.actor_state is None:
                    self.actor_state = self.actor.init_hidden(1, self.device)
                output, self.actor_state = self.actor(state_tensor, self.actor_state)
                self.actor_state = tuple(h.detach() for h in self.actor_state)
            else:  # ANN or SNN
                output, self.actor_state = self.actor(state_tensor, self.actor_state)

            # Get value from critic (always ANN now, so no state tracking needed)
            value, _ = self.critic(state_tensor, None)

            probs = F.softmax(output, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            return action.item(), value.item(), log_prob.item()

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
                return action.item(), 0.0, 0.0

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self):
        """PPO update - FIXED VERSION"""
        if len(self.memory.states) < self.config.batch_size:
            return 0.0, {}

        states, actions, rewards, values, old_log_probs, dones = self.memory.get_batch()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO epochs
        total_loss = 0
        for _ in range(self.config.n_epochs):
            # Reset states at the beginning of each epoch
            actor_state = None

            # Forward pass
            actor_outputs = []
            critic_outputs = []

            for i in range(len(states)):
                # Reset state at episode boundaries
                if i == 0 or dones[i-1]:
                    actor_state = None

                state = states_t[i:i+1]

                # Actor forward
                if hasattr(self.actor, 'init_hidden'):  # LSTM
                    if actor_state is None:
                        actor_state = self.actor.init_hidden(1, self.device)
                    output, actor_state = self.actor(state, actor_state)
                    # Detach states periodically (TBPTT)
                    if i % self.config.tbptt_chunk_size == 0 and i > 0:
                        actor_state = tuple(h.detach() for h in actor_state)
                else:  # ANN or SNN
                    output, actor_state = self.actor(state, actor_state)
                    # Detach states periodically for SNNs
                    if actor_state is not None and i % self.config.tbptt_chunk_size == 0 and i > 0:
                        actor_state = self._detach_state(actor_state)

                actor_outputs.append(output)

                # Critic is always ANN, no state management
                value, _ = self.critic(state, None)
                critic_outputs.append(value)

            actor_outputs = torch.cat(actor_outputs, dim=0)
            critic_outputs = torch.cat(critic_outputs, dim=0).squeeze()

            # Compute new log probs and entropy
            probs = F.softmax(actor_outputs, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                               1 + self.config.clip_epsilon) * advantages_t
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(critic_outputs, returns_t)

            # Total loss
            loss = actor_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            # Update
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_loss += loss.item()

        self.memory.clear()

        avg_loss = total_loss / self.config.n_epochs
        return avg_loss, {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

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

def create_ppo_agent(agent_type: str, config: PPOConfig, beta=0.9, device='cpu'):
    """Factory function to create PPO agents"""

    # All agents use ANN Critic
    critic = ANNCritic(8, config.hidden_dim)

    if agent_type == 'ann':
        actor = SimpleANN(8, config.hidden_dim, 4)
        return PPOAgent(config, actor, critic, device)

    elif agent_type == 'lstm':
        actor = ImprovedLSTM(8, config.hidden_dim, 4)
        return PPOAgent(config, actor, critic, device)

    elif agent_type == 'rleaky_fixed':
        actor = RLeakyFixedSNN(8, config.hidden_dim, 4, beta=beta)
        return PPOAgent(config, actor, critic, device)

    elif agent_type == 'leaky_fixed':
        actor = LeakyFixedSNN(8, config.hidden_dim, 4, beta=beta)
        return PPOAgent(config, actor, critic, device)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================
# Training & Evaluation
# ============================================================

def train_ppo_until_convergence(agent, config, env_difficulty='easy', verbose=False, run_id=0):
    """Train PPO agent until convergence"""
    env = FixedLunarLander(env_difficulty)

    episode = 0
    step_count = 0
    converged = False
    converged_at = None
    no_improvement_count = 0
    best_eval_mean = -1000.0

    while episode < config.max_episodes and not converged:
        state, _ = env.reset()
        agent.reset_episode_state()
        ep_reward = 0
        done = False

        while not done:
            action, value, log_prob = agent.select_action(state, training=True)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            agent.memory.add(state, action, reward, value, log_prob, done)
            ep_reward += reward
            state = next_state
            step_count += 1

            # Update when batch is full
            if step_count % config.batch_size == 0:
                loss, info = agent.update()

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

                print(f"    Run {run_id+1}, Ep {episode}/{config.max_episodes}: "
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
    """Evaluate agent"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = FixedLunarLander(difficulty)
    all_rewards = []

    for ep_idx in range(num_episodes):
        ep_seed = seed + ep_idx if seed is not None else None
        state, _ = env.reset(seed=ep_seed)
        agent.reset_episode_state()

        ep_reward = 0
        done = False

        while not done:
            action, _, _ = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        all_rewards.append(ep_reward)

    env.close()

    all_rewards = np.array(all_rewards)
    success_count = np.sum(all_rewards >= 200)

    return {
        'mean': float(np.mean(all_rewards)),
        'std': float(np.std(all_rewards)),
        'median': float(np.median(all_rewards)),
        'min': float(np.min(all_rewards)),
        'max': float(np.max(all_rewards)),
        'success_rate': float(success_count / num_episodes),
        'rewards': all_rewards.tolist()
    }


def run_single_ppo_trial(agent_type: str, config: PPOConfig, run_id: int,
                         verbose=False, beta=0.9, device='cpu'):
    """Run single PPO trial"""
    seed = 2025 + run_id * 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if verbose:
        print(f"\n  [{agent_type.upper()}] Run {run_id+1}")
        print("  " + "-" * 60)

    agent = create_ppo_agent(agent_type, config, beta=beta, device=device)

    # Train
    train_result = train_ppo_until_convergence(
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


def run_single_trial_from_config(config_dict: Dict) -> Dict:
    """
    Wrapper for run_single_ppo_trial that takes a single config dict.
    This is needed for parallel execution with multiprocessing.
    
    Args:
        config_dict: Dictionary containing all parameters
            - agent_type: str
            - beta: float
            - run_id: int
            - device: str
            - verbose: bool
            - ppo_config_key: str (key in PPO_CONFIGS)
    
    Returns:
        Result dictionary
    """
    agent_type = config_dict['agent_type']
    beta = config_dict['beta']
    run_id = config_dict['run_id']
    device = config_dict.get('device', 'cpu')
    verbose = config_dict.get('verbose', False)
    
    # Get PPO config
    config_key = config_dict.get('ppo_config_key', agent_type)
    ppo_config = PPO_CONFIGS.get(config_key, PPO_CONFIGS['ann'])
    
    return run_single_ppo_trial(
        agent_type=agent_type,
        config=ppo_config,
        run_id=run_id,
        verbose=verbose,
        beta=beta,
        device=device
    )


# ============================================================
# Beta Sweep (PARALLEL VERSION)
# ============================================================

BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def run_ppo_beta_sweep(num_runs=3, verbose=True, output_dir='results_lunarlander_ppo',
                       parallel=True, max_workers=None):
    """
    Run PPO beta sweep experiment (PARALLEL VERSION)
    
    Args:
        num_runs: Number of runs per configuration
        verbose: Whether to print progress
        output_dir: Output directory for results
        parallel: Whether to run experiments in parallel
        max_workers: Maximum number of parallel workers (None = auto)
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     LUNARLANDER PPO BETA SWEEP EXPERIMENT (PARALLEL)         ║
    ╚══════════════════════════════════════════════════════════════╝
    
    CRITICAL FIXES:
    - Fixed recurrent state reset at episode boundaries
    - All agents now use ANN Critic for stability
    - Added TBPTT to prevent gradient explosion
    
    Testing dissipation-recurrence with PPO (stable algorithm):
    - Beta values: """ + str(BETA_VALUES) + """
    - Architectures: Leaky, RLeaky (Fixed threshold)
    - Algorithm: PPO (Proximal Policy Optimization)
    - Runs: """ + str(num_runs) + """
    - Parallel: """ + str(parallel) + """
    """)

    # Get system info and best device
    get_system_info(verbose=True)
    # device = get_best_device(verbose=True)
    device = 'cpu'
    
    if parallel:
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = get_optimal_workers(device, reserve_cores=1)
        
        print(f"\nUsing {max_workers} parallel workers")
        print(f"Device: {device.upper()}")
        
        # Calculate total experiments
        total_experiments = 2 * len(BETA_VALUES) * num_runs  # 2 archs * 6 betas * runs
        print(f"\nTotal experiments: {total_experiments}")
        print(f"Estimated speedup: ~{max_workers}x (ideal)")
        
        # Create all experiment configurations
        experiment_configs = []
        
        for arch_type in ['leaky', 'rleaky']:
            for beta in BETA_VALUES:
                for run_id in range(num_runs):
                    config_dict = {
                        'agent_type': f'{arch_type}_fixed',
                        'ppo_config_key': f'{arch_type}_fixed',
                        'beta': beta,
                        'run_id': run_id,
                        'device': device,
                        'verbose': False,  # Disable per-run verbose in parallel mode
                        'arch_name': arch_type,  # For display
                    }
                    experiment_configs.append(config_dict)
        
        print("\nStarting parallel execution...\n")
        
        # Run in parallel
        all_results = run_experiments_parallel(
            experiment_configs=experiment_configs,
            experiment_fn=run_single_trial_from_config,
            max_workers=max_workers,
            verbose=True
        )
        
    else:
        # Sequential execution (original behavior)
        print("\nRunning sequentially (use --parallel for faster execution)\n")
        
        all_results = []
        
        for arch_type in ['leaky', 'rleaky']:
            for beta in BETA_VALUES:
                config_key = f'{arch_type}_fixed'
                config = PPO_CONFIGS[config_key]

                for run_id in range(num_runs):
                    print(f"\n{'='*70}")
                    print(f"{arch_type.upper()} | β={beta:.2f} | Run {run_id+1}/{num_runs}")
                    print("=" * 70)

                    result = run_single_ppo_trial(
                        config_key, config, run_id,
                        verbose=True,
                        beta=beta,
                        device=device
                    )

                    all_results.append(result)

    # Filter out failed experiments before saving
    valid_results = [r for r in all_results if r is not None and 'agent_type' in r]
    
    if len(valid_results) < len(all_results):
        print(f"\n⚠ {len(all_results) - len(valid_results)} experiments failed")
    
    print(f"✓ {len(valid_results)} experiments completed successfully")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    with open(output_path / 'lunarlander_ppo_beta_sweep_fixed.json', 'w') as f:
        json.dump(valid_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path / 'lunarlander_ppo_beta_sweep_fixed.json'}")

    return valid_results


def visualize_ppo_results(results, output_dir='results_lunarlander_ppo'):
    """Visualize PPO results"""

    # Filter out failed experiments (None or missing keys)
    valid_results = [r for r in results if r is not None and 'agent_type' in r]
    
    if len(valid_results) == 0:
        print("⚠ No valid results to visualize!")
        return
    
    if len(valid_results) < len(results):
        print(f"⚠ Filtered out {len(results) - len(valid_results)} failed experiments")

    # Organize data
    data = {}
    for r in valid_results:
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
    ax1.set_title('LunarLander PPO: β vs Gap', fontsize=14, fontweight='bold')
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
    ax2.set_title('LunarLander PPO: β vs Learning Speed', fontsize=14, fontweight='bold')
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
    ax3.set_title('LunarLander PPO: β vs Success Rate', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 100])

    plt.tight_layout()

    output_path = Path(output_dir)
    save_path = output_path / 'lunarlander_ppo_beta_sweep_fixed.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {save_path}")


def visualize_single_agent_performance(results, output_dir, agent_name):
    """
    Visualize single agent performance across multiple runs
    """
    import pandas as pd

    # Organize data
    data_rows = []
    gap_rows = []

    for r in results:
        run_id = r['run_id']
        for diff in ['easy', 'medium', 'hard', 'very_hard']:
            data_rows.append({
                'Run': f"Run {run_id + 1}",
                'Difficulty': diff.capitalize(),
                'Score': r['evaluation'][diff]['mean']
            })
        gap_rows.append({
            'Run': f"Run {run_id + 1}",
            'Gap': r['gap'],
            'Episodes': r['training']['converged_at']
        })

    df_scores = pd.DataFrame(data_rows)
    df_stats = pd.DataFrame(gap_rows)

    # Create figure
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 3, figure=fig)

    # Subplot 1: Score distribution by difficulty
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=df_scores, x='Difficulty', y='Score', hue='Difficulty', 
                legend=False, ax=ax1, palette="viridis")
    sns.stripplot(data=df_scores, x='Difficulty', y='Score', ax=ax1,
                  color='black', alpha=0.6, jitter=True)

    ax1.set_title(f'{agent_name.upper()} Performance across Difficulties', fontweight='bold')
    ax1.set_ylabel('Mean Reward')
    ax1.axhline(200, color='r', linestyle='--', alpha=0.5, label='Solved (200)')
    ax1.legend(loc='upper right')

    # Subplot 2: Gap distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=df_stats, y='Gap', ax=ax2, color='skyblue', capsize=.2)
    sns.stripplot(data=df_stats, y='Gap', ax=ax2, color='red', s=8)
    ax2.set_title('Generalization Gap (Lower is Better)', fontweight='bold')
    ax2.set_ylabel('Gap (Easy - Very Hard)')
    mean_gap = df_stats['Gap'].mean()
    ax2.text(0, mean_gap + 5, f"Mean: {mean_gap:.1f}", ha='center', fontweight='bold')

    # Subplot 3: Convergence episodes
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(data=df_stats, x='Episodes', kde=True, ax=ax3, color='orange', bins=5)
    ax3.set_title('Convergence Speed (Episodes)', fontweight='bold')
    ax3.set_xlabel('Episodes to Converge')

    plt.suptitle(f"Performance Analysis: {agent_name.upper()} ({len(results)} runs)", 
                 fontsize=16, y=1.05)
    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    save_path = output_path / f'lunarlander_ppo_{agent_name}_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved: {save_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LunarLander PPO Beta Sweep (Parallel Version)')
    parser.add_argument('--mode', type=str, default='beta_sweep',
                       choices=['beta_sweep', 'quick_test', 'single_agent'],
                       help='Experiment mode')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration')
    parser.add_argument('--output_dir', type=str, default='results_lunarlander_ppo',
                       help='Output directory')
    parser.add_argument('--agent_type', type=str, default='ann',
                       choices=['ann', 'lstm', 'leaky_fixed', 'rleaky_fixed'],
                       help='Agent type for quick_test or single_agent mode')
    parser.add_argument('--beta', type=float, default=0.9,
                       help='Beta value for SNN agents')
    parser.add_argument('--parallel', action='store_true',
                       help='Run experiments in parallel (much faster!)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (auto = best available)')
    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = get_best_device(verbose=True)
    else:
        device = args.device
        print(f"Using specified device: {device}")

    if args.mode == 'quick_test':
        print(f"\nPPO Quick test: Running {args.agent_type.upper()} with β={args.beta}")
        print("="*70)

        config = PPO_CONFIGS.get(args.agent_type, PPO_CONFIGS['ann'])

        result = run_single_ppo_trial(
            args.agent_type,
            config,
            0,
            verbose=True,
            beta=args.beta,
            device=device
        )

        print("\n" + "="*70)
        print("✓ PPO Quick test complete!")
        print("="*70)
        print(f"\nResults summary:")
        print(f"  Agent: {args.agent_type.upper()}")
        print(f"  Beta: {args.beta}")
        print(f"  Converged: {result['training']['converged']}")
        print(f"  Episodes: {result['training']['converged_at']}")
        print(f"  Easy score: {result['evaluation']['easy']['mean']:.1f}")
        print(f"  Success rate: {result['evaluation']['easy']['success_rate']:.1%}")
        print(f"  Gap: {result['gap']:.1f}")

    elif args.mode == 'single_agent':
        print(f"\nPPO Single agent: {args.agent_type.upper()} | β={args.beta} | {args.runs} runs")
        print("=" * 70)

        config = PPO_CONFIGS.get(args.agent_type, PPO_CONFIGS['ann'])
        results = []

        for run_id in range(args.runs):
            print(f"\nRun {run_id + 1}/{args.runs}")
            print("-" * 70)

            result = run_single_ppo_trial(
                args.agent_type,
                config,
                run_id,
                verbose=True,
                beta=args.beta,
                device=device
            )
            results.append(result)

        # Save JSON
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        json_filename = f'lunarlander_ppo_{args.agent_type}_results.json'
        if 'leaky' in args.agent_type:
            json_filename = f'lunarlander_ppo_{args.agent_type}_beta{args.beta}_results.json'

        with open(output_path / json_filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path / json_filename}")

        # Generate visualization
        print("\nGenerating performance visualization...")
        visualize_single_agent_performance(results, args.output_dir, args.agent_type)

        # Aggregate Stats
        gaps = [r['gap'] for r in results]
        easy_scores = [r['evaluation']['easy']['mean'] for r in results]
        success_rates = [r['evaluation']['easy']['success_rate'] for r in results]
        episodes = [r['training']['converged_at'] for r in results]

        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS SUMMARY")
        print("=" * 70)
        print(f"Agent: {args.agent_type.upper()}")
        print(f"Runs: {args.runs}")
        print(f"Gap: {np.mean(gaps):.1f} ± {np.std(gaps):.1f}")
        print(f"Easy score: {np.mean(easy_scores):.1f} ± {np.std(easy_scores):.1f}")
        print(f"Success rate: {np.mean(success_rates):.1%} ± {np.std(success_rates):.1%}")
        print(f"Episodes: {np.mean(episodes):.0f} ± {np.std(episodes):.0f}")

    else:  # beta_sweep
        results = run_ppo_beta_sweep(
            num_runs=args.runs,
            verbose=True,
            output_dir=args.output_dir,
            parallel=args.parallel,
            max_workers=args.workers
        )

        print("\nGenerating visualizations...")
        visualize_ppo_results(results, args.output_dir)

        print("\n" + "="*70)
        print("✓ LUNARLANDER PPO EXPERIMENT COMPLETE")
        print("="*70)
