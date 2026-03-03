"""
Soft Actor-Critic (SAC) Agent for SDN Routing Optimization.

Implements:
    - Dual Q-Networks (twin critics) for stable value estimation
    - Entropy-regularized policy for balanced exploration/exploitation
    - Automatic temperature (alpha) tuning
    - Experience replay buffer
    - Target network with soft updates

Architecture:
    Actor:  Input (state_dim) -> [128 ReLU] -> [128 ReLU] -> Output (action_dim softmax)
    Critic: Input (state_dim) -> [128 ReLU] -> [128 ReLU] -> Output (action_dim Q-values)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque, namedtuple
from typing import List, Optional, Tuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class SACPolicy(nn.Module):
    """
    Stochastic policy network for SAC.

    Outputs a probability distribution over discrete actions
    using softmax, enabling entropy-based exploration.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = None):
        super(SACPolicy, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 128]

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(0.1))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return logits

    def get_action_probs(self, state: torch.Tensor):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        # Clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-8)
        log_probs = torch.log(probs)
        return probs, log_probs


class SACQNetwork(nn.Module):
    """
    Q-Network for SAC (twin critic architecture).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = None):
        super(SACQNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 128]

        # Q1
        q1_layers = []
        prev_dim = state_dim
        for h_dim in hidden_layers:
            q1_layers.append(nn.Linear(prev_dim, h_dim))
            q1_layers.append(nn.ReLU())
            prev_dim = h_dim
        q1_layers.append(nn.Linear(prev_dim, action_dim))
        self.q1 = nn.Sequential(*q1_layers)

        # Q2
        q2_layers = []
        prev_dim = state_dim
        for h_dim in hidden_layers:
            q2_layers.append(nn.Linear(prev_dim, h_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = h_dim
        q2_layers.append(nn.Linear(prev_dim, action_dim))
        self.q2 = nn.Sequential(*q2_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        return self.q1(state), self.q2(state)


class ReplayBuffer:
    """
    Experience replay buffer for SAC training.
    """

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic agent for SDN routing optimization.

    Features:
        - Entropy-regularized RL for robust exploration
        - Twin Q-networks to mitigate overestimation
        - Automatic temperature tuning
        - Experience replay
        - Soft target updates (Polyak averaging)
        - Model checkpointing (save/load)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = None,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_alpha: bool = True,
                 buffer_size: int = 50000,
                 batch_size: int = 64,
                 min_replay_size: int = 1000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 device: str = None):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: Hidden layer sizes
            learning_rate: Learning rate for all optimizers
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Initial entropy temperature
            auto_alpha: Whether to auto-tune alpha
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            min_replay_size: Minimum buffer before training
            epsilon_start: Initial exploration rate (for compatibility)
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay factor per episode
            device: 'cuda' or 'cpu'
        """
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size

        # Exploration (epsilon-greedy fallback for early training)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Policy network
        self.policy = SACPolicy(state_dim, action_dim, hidden_layers).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Twin Q-networks
        self.critic = SACQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.critic_target = SACQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Entropy temperature
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training counters
        self.train_steps = 0
        self.episodes_completed = 0

        # Statistics
        self.losses = []
        self.rewards_history = []

        print(f"[SACAgent] Initialized on {self.device}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Policy params: "
              f"{sum(p.numel() for p in self.policy.parameters()):,}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using the stochastic policy.
        """
        # Epsilon-greedy for early exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy.eval()
            probs, _ = self.policy.get_action_probs(state_tensor)
            self.policy.train()

            if training:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            else:
                action = probs.argmax(dim=1)

            return action.item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one SAC training step.
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).unsqueeze(1).to(self.device)

        # --- Critic Update ---
        with torch.no_grad():
            next_probs, next_log_probs = self.policy.get_action_probs(next_states)
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            # V(s') = E_a[Q(s', a) - alpha * log pi(a|s')]
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + self.gamma * next_v * (1 - dones)

        q1, q2 = self.critic(states)
        q1_a = q1.gather(1, actions)
        q2_a = q2.gather(1, actions)

        critic_loss = F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Policy Update ---
        probs, log_probs = self.policy.get_action_probs(states)
        q1_pi, q2_pi = self.critic(states)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        # --- Alpha Update ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (
                (probs * (log_probs + self.target_entropy)).sum(dim=1).detach()
            )).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Soft Target Update ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.train_steps += 1
        loss_val = critic_loss.item()
        self.losses.append(loss_val)

        return loss_val

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_completed += 1

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'train_steps': self.train_steps,
            'episodes': self.episodes_completed,
            'losses': self.losses[-1000:],
            'rewards': self.rewards_history[-1000:],
        }, path)
        print(f"[SACAgent] Saved checkpoint to {path}")

    def load(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.alpha = checkpoint.get('alpha', 0.2)
            self.train_steps = checkpoint.get('train_steps', 0)
            self.episodes_completed = checkpoint.get('episodes', 0)
            self.losses = checkpoint.get('losses', [])
            self.rewards_history = checkpoint.get('rewards', [])
            print(f"[SACAgent] Loaded checkpoint from {path}")
            print(f"  Episodes: {self.episodes_completed}, "
                  f"Epsilon: {self.epsilon:.4f}")

    def get_statistics(self) -> dict:
        return {
            'episodes': self.episodes_completed,
            'train_steps': self.train_steps,
            'epsilon': round(self.epsilon, 4),
            'alpha': round(self.alpha, 4),
            'avg_loss': round(np.mean(self.losses[-100:]), 6) if self.losses else 0.0,
            'avg_reward': round(np.mean(self.rewards_history[-100:]), 3)
                if self.rewards_history else 0.0,
            'buffer_size': len(self.replay_buffer),
        }


if __name__ == '__main__':
    """Test SAC agent standalone."""
    print("Testing SAC Agent...\n")

    state_dim = 40
    action_dim = 10

    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)

    for i in range(100):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.select_action(state, training=True)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = (i == 99)
        agent.store_experience(state, action, reward, next_state, done)

    print(f"Buffer size: {len(agent.replay_buffer)}")
    print(f"Epsilon: {agent.epsilon:.4f}")

    for i in range(1000):
        state = np.random.randn(state_dim).astype(np.float32)
        agent.store_experience(state, 0, 1.0, state, False)

    for i in range(10):
        loss = agent.train_step()
        if loss is not None:
            print(f"  Train step {i + 1}: loss={loss:.6f}")

    stats = agent.get_statistics()
    print(f"\nAgent stats: {stats}")
    print("\nSAC Agent test completed!")
