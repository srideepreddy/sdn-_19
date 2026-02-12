"""
Proximal Policy Optimization (PPO) Agent for SDN Routing.

Implements:
    - PPO with clipped objective
    - Actor-Critic architecture
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

class PPOPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPOPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.0003, gamma: float = 0.99, eps_clip: float = 0.2, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.state_dim = state_dim
        self.action_dim = action_dim

    def update(self, memory):
        # Flatten memory
        old_states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(memory.actions)).to(self.device)
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        rewards = memory.rewards
        dones = memory.is_terminals

        # Calculate Returns (Discounted Rewards)
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(5):
            probs, state_values = self.policy(old_states)
            m = torch.distributions.Categorical(probs)
            logprobs = m.log_prob(old_actions)
            dist_entropy = m.entropy()
            
            # PPO Ratio
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogates
            advantages = returns - state_values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Total Loss
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values.squeeze(), returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        return loss.mean().item()

    def select_action(self, state: np.ndarray, training_data=None) -> int:
        state = torch.FloatTensor(state).to(self.device)
        probs, _ = self.policy(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        if training_data is not None:
            training_data.states.append(state.cpu().numpy())
            training_data.actions.append(action.item())
            training_data.logprobs.append(m.log_prob(action))

        return action.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[PPOAgent] Loaded model from {path}")

class PPOMemory:
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
