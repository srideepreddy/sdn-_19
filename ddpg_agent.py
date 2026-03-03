"""
Deep Deterministic Policy Gradient (DDPG) Agent for SDN Routing.

Implements:
    - Actor-Critic architecture with target networks
    - Ornstein-Uhlenbeck noise for exploration
    - Soft target updates (Polyak averaging)
    - Experience replay buffer
    - Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Optional, Tuple


class DDPGActor(nn.Module):
    """Actor network that maps states to action probabilities (discrete)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class DDPGCritic(nn.Module):
    """Critic network that estimates Q-values for state-action pairs."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for SDN routing.

    Features:
        - Deterministic actor with softmax for discrete actions
        - Critic for Q-value estimation
        - Ornstein-Uhlenbeck exploration noise
        - Soft target network updates
        - Model checkpointing
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float = 0.0001, lr_critic: float = 0.001,
                 gamma: float = 0.99, tau: float = 0.005,
                 device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor
        self.actor = DDPGActor(state_dim, action_dim).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # OU Noise parameters
        self.ou_mu = np.zeros(action_dim)
        self.ou_theta = 0.15
        self.ou_sigma = 0.2
        self.ou_state = np.zeros(action_dim)

    def _ou_noise(self):
        """Ornstein-Uhlenbeck noise for exploration."""
        dx = self.ou_theta * (self.ou_mu - self.ou_state) + \
             self.ou_sigma * np.random.randn(self.action_dim)
        self.ou_state += dx
        return self.ou_state

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state = torch.FloatTensor(state).to(self.device)
        probs = self.actor(state)

        if training:
            noise = torch.FloatTensor(self._ou_noise() * 0.1).to(self.device)
            probs = probs + noise
            probs = F.softmax(probs, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            return action.item()
        else:
            return torch.argmax(probs).item()

    def update(self, states, actions, rewards, next_states, dones):
        """Update actor and critic networks."""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Ensure batch dimensions
        if states.dim() == 1:
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)

        # --- Critic Update ---
        with torch.no_grad():
            next_probs = self.actor_target(next_states)
            next_q = self.critic_target(next_states)
            next_v = (next_probs * next_q).sum(dim=-1)
            td_target = rewards + self.gamma * next_v * (1 - dones)

        current_q = self.critic(states)
        # Gather Q-values for taken actions
        if actions.dim() == 1:
            q_values = current_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        else:
            q_values = current_q.gather(1, actions).squeeze(-1)

        critic_loss = F.mse_loss(q_values, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        probs = self.actor(states)
        q_vals = self.critic(states)
        actor_loss = -(probs * q_vals).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Target Updates ---
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            print(f"[DDPGAgent] Loaded model from {path}")
