"""
Asynchronous Advantage Actor-Critic (A3C) Agent for SDN Routing.

Implements:
    - Actor-Critic architecture with shared or separate backbones
    - Advantage estimation (V(s) - Q(s,a))
    - Entropy regularization for exploration
    - Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Optional, Tuple

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCritic, self).__init__()
        self.common = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.common(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

class A3CAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001, gamma: float = 0.99, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        state = torch.FloatTensor(state).to(self.device)
        probs, _ = self.model(state)
        if training:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            return action.item()
        else:
            return torch.argmax(probs).item()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current probs and values
        probs, values = self.model(states)
        _, next_values = self.model(next_states)

        # Calculate TD Target and Advantage
        td_target = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantage = td_target - values.squeeze()

        # Actor Loss (Policy Gradient)
        m = torch.distributions.Categorical(probs)
        log_probs = m.log_prob(actions)
        actor_loss = -(log_probs * advantage.detach()).mean()

        # Critic Loss (Value Function)
        critic_loss = F.mse_loss(values.squeeze(), td_target.detach())

        # Entropy Loss (Exploration)
        entropy_loss = -m.entropy().mean()

        # Total Loss
        total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[A3CAgent] Loaded model from {path}")
