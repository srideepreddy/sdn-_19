"""
Deep Q-Network (DQN) Agent for SDN Routing Optimization.

Implements:
    - Q-Network with configurable hidden layers
    - Experience replay buffer
    - Target network with periodic hard updates
    - Epsilon-greedy exploration with exponential decay
    - Batch training with MSE loss

Architecture:
    Input (state_dim) -> [128 ReLU] -> [128 ReLU] -> Output (action_dim)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque, namedtuple
from typing import List, Optional, Tuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """
    Q-Network for estimating action-value function Q(s, a).

    Architecture:
        - Input layer: state_dim
        - Hidden layers: configurable (default [128, 128]) with ReLU + BatchNorm
        - Output layer: action_dim (Q-values for each action)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = None):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_layers: List of hidden layer sizes
        """
        super(QNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 128]

        # Build network layers
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-Network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Q-values tensor [batch_size, action_dim]
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Stores transitions (s, a, r, s', done) and provides random
    batch sampling for decorrelated training.
    """

    def __init__(self, capacity: int = 50000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of samples

        Returns:
            List of Experience tuples
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for SDN routing optimization.

    Features:
        - Double DQN with separate policy and target networks
        - Experience replay with configurable buffer size
        - Epsilon-greedy exploration with exponential decay
        - Periodic target network synchronization
        - Model checkpointing (save/load)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = None,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 50000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 min_replay_size: int = 1000,
                 device: str = None):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of candidate paths)
            hidden_layers: Q-Network hidden layer sizes
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Multiplicative decay per episode
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            min_replay_size: Minimum buffer size before training starts
            device: 'cuda' or 'cpu' (auto-detected if None)
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
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training counters
        self.train_steps = 0
        self.episodes_completed = 0

        # Statistics
        self.losses = []
        self.rewards_history = []

        print(f"[DQNAgent] Initialized on {self.device}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Policy net params: "
              f"{sum(p.numel() for p in self.policy_net.parameters()):,}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation
            training: If True, uses epsilon-greedy. If False, greedy only.

        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Greedy action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)
            self.policy_net.train()
            return q_values.argmax(dim=1).item()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step with experience replay.

        Samples a batch from the replay buffer, computes TD targets
        using the target network, and updates the policy network.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(
            np.array([e.state for e in batch])
        ).to(self.device)
        actions = torch.LongTensor(
            [e.action for e in batch]
        ).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(
            [e.reward for e in batch]
        ).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(
            np.array([e.next_state for e in batch])
        ).to(self.device)
        dones = torch.FloatTensor(
            [float(e.done) for e in batch]
        ).unsqueeze(1).to(self.device)

        # Current Q-values: Q(s, a) for the taken actions
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q-values using target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        loss_val = loss.item()
        self.losses.append(loss_val)

        # Update target network periodically
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss_val

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)
        self.episodes_completed += 1

    def save(self, path: str):
        """
        Save agent checkpoint.

        Args:
            path: File path for saving the checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'episodes': self.episodes_completed,
            'losses': self.losses[-1000:],  # Keep last 1000
            'rewards': self.rewards_history[-1000:],
        }, path)
        print(f"[DQNAgent] Saved checkpoint to {path}")

    def load(self, path: str):
        """
        Load agent checkpoint.

        Args:
            path: File path of the checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.train_steps = checkpoint.get('train_steps', 0)
        self.episodes_completed = checkpoint.get('episodes', 0)
        self.losses = checkpoint.get('losses', [])
        self.rewards_history = checkpoint.get('rewards', [])
        print(f"[DQNAgent] Loaded checkpoint from {path}")
        print(f"  Episodes: {self.episodes_completed}, "
              f"Epsilon: {self.epsilon:.4f}")

    def get_statistics(self) -> dict:
        """
        Get training statistics.

        Returns:
            Dictionary with training metrics
        """
        return {
            'episodes': self.episodes_completed,
            'train_steps': self.train_steps,
            'epsilon': round(self.epsilon, 4),
            'avg_loss': round(np.mean(self.losses[-100:]), 6) if self.losses else 0.0,
            'avg_reward': round(np.mean(self.rewards_history[-100:]), 3)
                if self.rewards_history else 0.0,
            'buffer_size': len(self.replay_buffer),
        }


if __name__ == '__main__':
    """Test DQN agent standalone."""
    print("Testing DQN Agent...\n")

    state_dim = 40  # 10 links * 4 features
    action_dim = 10  # 10 candidate paths

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    # Simulate some experiences
    for i in range(100):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.select_action(state, training=True)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = (i == 99)

        agent.store_experience(state, action, reward, next_state, done)

    print(f"Buffer size: {len(agent.replay_buffer)}")
    print(f"Epsilon: {agent.epsilon:.4f}")

    # Fill buffer to min_replay_size
    for i in range(1000):
        state = np.random.randn(state_dim).astype(np.float32)
        agent.store_experience(state, 0, 1.0, state, False)

    # Train for a few steps
    for i in range(10):
        loss = agent.train_step()
        if loss is not None:
            print(f"  Train step {i + 1}: loss={loss:.6f}")

    stats = agent.get_statistics()
    print(f"\nAgent stats: {stats}")
    print("\nDQN Agent test completed!")
