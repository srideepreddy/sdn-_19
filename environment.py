"""
DRL Environment for SDN Routing.

Provides a Gymnasium-compatible environment where the DRL agent observes
network state (link utilization, delay, packet count, queue size) and
selects the best routing path (action = next-hop path index).

State Space:
    - Per-link utilization (normalized 0-1)
    - Per-link delay estimate (normalized)
    - Per-link packet count (normalized)
    - Per-link bandwidth capacity (normalized)

Action Space:
    - Discrete: index into candidate paths between src and dst

Reward:
    - Positive for low delay and high throughput
    - Negative for congestion and packet loss
    - R = w1 * throughput - w2 * delay - w3 * packet_loss
"""

import os
import json
import time
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any

# Path to shared stats file
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
STATS_FILE = os.path.join(DATA_DIR, 'net_stats.json')
DRL_DECISION_FILE = os.path.join(DATA_DIR, 'drl_decision.json')


class SDNRoutingEnv(gym.Env):
    """
    Gymnasium environment for SDN DRL routing optimization.

    The environment reads live network stats from data/net_stats.json,
    computes candidate paths, and lets the DRL agent choose the best path.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, mode: str = 'simulation', max_steps: int = 500):
        """
        Initialize the SDN routing environment.

        Args:
            mode: 'simulation' for synthetic training or 'live' for real network
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.mode = mode
        self.max_steps = max_steps
        self.current_step = 0

        # Build network graph matching custom_topology.py
        self.graph = self._build_graph()
        self.num_links = self.graph.number_of_edges()
        self.link_list = list(self.graph.edges())

        # State: 4 features per link (utilization, delay, packets, bandwidth)
        self.features_per_link = 4
        state_dim = self.num_links * self.features_per_link
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

        # Action: select one of K candidate paths (max 10 paths)
        self.max_paths = 10
        self.action_space = spaces.Discrete(self.max_paths)

        # Current flow to route
        self.current_src = None
        self.current_dst = None
        self.candidate_paths = []

        # Link utilizations (simulated or read from stats)
        self.link_utilization = {e: 0.0 for e in self.link_list}

        # Performance tracking
        self.episode_rewards = []
        self.total_reward = 0.0

        # Reward weights
        self.w_throughput = 1.0
        self.w_delay = 0.5
        self.w_loss = 0.3

    def _build_graph(self) -> nx.Graph:
        """
        Build NetworkX graph matching the custom topology.

        Returns:
            NetworkX graph with 4 switches and weighted edges
        """
        G = nx.Graph()

        # Add switches (using integer IDs 1-4)
        for i in range(1, 5):
            G.add_node(i, type='switch', name=f's{i}')

        # Add host nodes (IDs 5-10 for h1-h6)
        host_map = {5: 1, 6: 1, 7: 2, 8: 3, 9: 4, 10: 4}  # host -> switch
        for h_id, sw_id in host_map.items():
            G.add_node(h_id, type='host', name=f'h{h_id - 4}')
            G.add_edge(h_id, sw_id, bw=1000, delay=1, weight=1)

        # Switch-to-switch links with variable bandwidth
        G.add_edge(1, 2, bw=100, delay=2, weight=1)   # s1-s2: 100 Mbps
        G.add_edge(3, 4, bw=100, delay=2, weight=1)   # s3-s4: 100 Mbps
        G.add_edge(1, 4, bw=50, delay=5, weight=2)    # s1-s4: 50 Mbps
        G.add_edge(2, 3, bw=50, delay=5, weight=2)    # s2-s3: 50 Mbps
        G.add_edge(1, 3, bw=10, delay=10, weight=5)   # s1-s3: 10 Mbps
        G.add_edge(2, 4, bw=10, delay=10, weight=5)   # s2-s4: 10 Mbps

        return G

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0.0

        # Reset link utilizations
        self.link_utilization = {e: 0.0 for e in self.link_list}

        # Generate a random flow to route
        self._generate_random_flow()

        state = self._get_state()
        info = {
            'src': self.current_src,
            'dst': self.current_dst,
            'num_paths': len(self.candidate_paths),
        }

        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step — agent selects a path.

        Args:
            action: Index of the candidate path to use

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Clamp action to valid range
        action = min(action, len(self.candidate_paths) - 1)
        action = max(action, 0)

        # Get selected path
        if self.candidate_paths:
            selected_path = self.candidate_paths[action]
        else:
            selected_path = []

        # Execute the routing (update link utilization)
        self._execute_routing(selected_path)

        # Calculate reward and metrics
        reward, metrics = self._calculate_reward_with_metrics(selected_path)
        self.total_reward += reward

        # Write decision for controller to pick up (live mode)
        if self.mode == 'live' and selected_path:
            self._write_decision(selected_path)

        # Generate next flow
        self._generate_random_flow()

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        state = self._get_state()
        info = {
            'selected_path': selected_path,
            'reward': reward,
            'total_reward': self.total_reward,
            'step': self.current_step,
            'throughput': metrics['throughput'],
            'delay': metrics['delay'],
            'loss': metrics['loss']
        }

        return state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        Build the observation vector from current network state.

        In 'live' mode, reads from data/net_stats.json.
        In 'simulation' mode, uses internal tracked utilizations.

        Returns:
            Normalized state vector as numpy array
        """
        if self.mode == 'live':
            return self._get_live_state()

        # Simulation mode
        state = []
        for edge in self.link_list:
            u, v = edge
            edge_data = self.graph.edges[u, v]

            util = self.link_utilization.get(edge, 0.0)
            bw = edge_data.get('bw', 100) / 1000.0  # Normalize to Gbps
            delay = edge_data.get('delay', 1) / 20.0  # Normalize: max 20ms
            packets = util * 100  # Simulated packet count (normalized)

            state.extend([
                min(1.0, util),           # Utilization (0-1)
                min(1.0, delay),          # Delay (normalized)
                min(1.0, packets / 100),  # Packet count (normalized)
                min(1.0, bw),             # Bandwidth (normalized)
            ])

        return np.array(state, dtype=np.float32)

    def _get_live_state(self) -> np.ndarray:
        """Read live state from net_stats.json."""
        try:
            with open(STATS_FILE, 'r') as f:
                stats = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        state = []
        link_stats = stats.get('link_stats', {})

        for edge in self.link_list:
            u, v = edge
            link_key = f"s{u}-s{v}"
            alt_key = f"s{v}-s{u}"
            ls = link_stats.get(link_key, link_stats.get(alt_key, {}))

            util = ls.get('utilization', 0.0)
            bw = ls.get('bandwidth_mbps', 100) / 1000.0
            tx = ls.get('tx_bps', 0) / 1e9  # Normalize to Gbps

            state.extend([
                min(1.0, util),
                min(1.0, tx),
                0.0,  # Packet count placeholder
                min(1.0, bw),
            ])

        return np.array(state, dtype=np.float32)

    def _generate_random_flow(self):
        """Generate a random source-destination pair and compute candidate paths."""
        # Get switch nodes only (IDs 1-4)
        switches = [n for n, d in self.graph.nodes(data=True)
                     if d.get('type') == 'switch']

        if len(switches) < 2:
            self.current_src = 1
            self.current_dst = 2
        else:
            src, dst = np.random.choice(switches, 2, replace=False)
            self.current_src = int(src)
            self.current_dst = int(dst)

        # Compute candidate paths
        try:
            all_paths = list(nx.all_simple_paths(
                self.graph, self.current_src, self.current_dst,
                cutoff=6  # Max path length
            ))
            # Filter to switch-only paths and sort by hop count
            switch_paths = []
            for p in all_paths:
                sp = [n for n in p if self.graph.nodes[n].get('type') == 'switch']
                if len(sp) >= 2 and sp not in switch_paths:
                    switch_paths.append(sp)

            switch_paths.sort(key=len)
            self.candidate_paths = switch_paths[:self.max_paths]
        except nx.NetworkXNoPath:
            self.candidate_paths = []

        # Pad to max_paths
        while len(self.candidate_paths) < self.max_paths:
            if self.candidate_paths:
                self.candidate_paths.append(self.candidate_paths[0])
            else:
                self.candidate_paths.append([self.current_src, self.current_dst])

    def _execute_routing(self, path: List[int]):
        """
        Simulate routing along the selected path.

        Updates link utilization for all links in the path.

        Args:
            path: List of switch IDs along the routing path
        """
        # Add utilization to each link in the path
        traffic_load = np.random.uniform(0.05, 0.2)  # Random traffic load

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = (u, v) if (u, v) in self.link_utilization else (v, u)
            if edge in self.link_utilization:
                self.link_utilization[edge] = min(
                    1.0, self.link_utilization[edge] + traffic_load
                )

        # Natural decay on all links (traffic completion)
        for edge in self.link_utilization:
            self.link_utilization[edge] *= 0.95  # 5% decay per step

    def _calculate_reward_with_metrics(self, path: List[int]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward and return individual performance metrics.
        """
        if not path or len(path) < 2:
            return -10.0, {'throughput': 0.0, 'delay': 1.0, 'loss': 1.0}

        total_delay = 0.0
        max_utilization = 0.0
        min_bandwidth = float('inf')

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                edge_data = self.graph.edges[u, v]
                delay = edge_data.get('delay', 1)
                bw = edge_data.get('bw', 100)

                edge_key = (u, v) if (u, v) in self.link_utilization else (v, u)
                util = self.link_utilization.get(edge_key, 0.0)

                total_delay += delay * (1 + util * 3)
                max_utilization = max(max_utilization, util)
                min_bandwidth = min(min_bandwidth, bw)

        throughput_score = min(1.0, min_bandwidth / 100.0)
        delay_penalty = min(1.0, total_delay / 50.0)
        loss_penalty = max_utilization ** 2

        reward = (self.w_throughput * throughput_score
                  - self.w_delay * delay_penalty
                  - self.w_loss * loss_penalty)
        
        hop_bonus = 0.1 * (1.0 / len(path))
        
        metrics = {
            'throughput': throughput_score * 100.0, # Mbps equivalent
            'delay': total_delay, # ms equivalent
            'loss': loss_penalty * 10.0 # Percentage proxy
        }

        return reward + hop_bonus, metrics

    def _write_decision(self, path: List[int]):
        """
        Write routing decision for the Ryu controller to read.

        Args:
            path: List of switch DPIDs
        """
        decision = {
            'src_dpid': self.current_src,
            'dst_dpid': self.current_dst,
            'path': path,
            'path_names': [f's{p}' for p in path],
            'timestamp': time.time(),
        }

        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(DRL_DECISION_FILE, 'w') as f:
                json.dump(decision, f)
        except Exception as e:
            print(f"[Environment] Error writing decision: {e}")

    def render(self, mode='human'):
        """Display environment state."""
        print(f"\n--- Step {self.current_step} ---")
        print(f"Flow: s{self.current_src} -> s{self.current_dst}")
        print(f"Candidate paths: {len(self.candidate_paths)}")
        print(f"Link utilizations:")
        for edge, util in sorted(self.link_utilization.items()):
            bar = '█' * int(util * 20)
            print(f"  s{edge[0]}-s{edge[1]}: {util:.2f} |{bar}|")

    def close(self):
        """Clean up resources."""
        pass


if __name__ == '__main__':
    """Test environment standalone."""
    print("Testing SDN Routing Environment...\n")

    env = SDNRoutingEnv(mode='simulation', max_steps=20)
    obs, info = env.reset()
    print(f"State shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial flow: s{info['src']} -> s{info['dst']}")
    print(f"Candidate paths: {info['num_paths']}")

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  Step {step + 1}: action={action}, reward={reward:.3f}, "
              f"path={info['selected_path']}")

        if terminated or truncated:
            break

    env.render()
    print(f"\nTotal reward: {total_reward:.3f}")
    print("Environment test completed!")
