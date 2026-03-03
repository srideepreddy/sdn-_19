"""
Twin Delayed DDPG (TD3) Agent for SDN Routing.

Implements:
    - Twin critic networks to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    - Actor-Critic architecture with target networks
    - Experience buffer for batch updates
    - Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Optional, Tuple


class TD3Actor(nn.Module):
    """Actor network with softmax output for discrete action selection."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(TD3Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class TD3Critic(nn.Module):
    """Twin critic network — outputs two independent Q-value estimates."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(TD3Critic, self).__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.q1(state), self.q2(state)

    def q1_forward(self, state):
        return self.q1(state)


class TD3Agent:
    """
    Twin Delayed DDPG agent for SDN routing.

    Features:
        - Twin critics to reduce overestimation bias
        - Delayed actor updates (every policy_delay steps)
        - Target policy smoothing with clipped noise
        - Soft target network updates
        - Model checkpointing
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 0.0003, gamma: float = 0.99,
                 tau: float = 0.005, policy_delay: int = 2,
                 noise_clip: float = 0.5, policy_noise: float = 0.2,
                 device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_clip = noise_clip
        self.policy_noise = policy_noise
        self.update_count = 0

        # Actor
        self.actor = TD3Actor(state_dim, action_dim).to(self.device)
        self.actor_target = TD3Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin Critic
        self.critic = TD3Critic(state_dim, action_dim).to(self.device)
        self.critic_target = TD3Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def update(self, memory):
        """Update networks using a TD3Memory batch."""
        old_states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(memory.actions)).to(self.device)
        rewards = memory.rewards
        dones = memory.is_terminals

        # Calculate returns (discounted rewards)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        self.update_count += 1

        # --- Critic Update ---
        q1, q2 = self.critic(old_states)

        if old_actions.dim() == 1:
            q1_a = q1.gather(1, old_actions.unsqueeze(-1)).squeeze(-1)
            q2_a = q2.gather(1, old_actions.unsqueeze(-1)).squeeze(-1)
        else:
            q1_a = q1.gather(1, old_actions).squeeze(-1)
            q2_a = q2.gather(1, old_actions).squeeze(-1)

        critic_loss = F.mse_loss(q1_a, returns) + F.mse_loss(q2_a, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Delayed Actor Update ---
        if self.update_count % self.policy_delay == 0:
            probs = self.actor(old_states)
            q1_pi = self.critic.q1_forward(old_states)
            actor_loss = -(probs * q1_pi).sum(dim=-1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft target updates
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.mean().item()

    def select_action(self, state: np.ndarray, training_data=None) -> int:
        state = torch.FloatTensor(state).to(self.device)
        probs = self.actor(state)

        # Add target policy smoothing noise during training
        if training_data is not None:
            noise = torch.randn_like(probs) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            probs = probs + noise
            probs = F.softmax(probs, dim=-1)

        m = torch.distributions.Categorical(probs)
        action = m.sample()

        if training_data is not None:
            training_data.states.append(state.cpu().numpy())
            training_data.actions.append(action.item())
            training_data.logprobs.append(m.log_prob(action))

        return action.item()

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
            print(f"[TD3Agent] Loaded model from {path}")


class TD3Memory:
    """Experience memory for TD3 batch updates."""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
