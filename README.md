# Collaborative Routing in SDN using Deep Reinforcement Learning

A research project implementing intelligent routing in Software-Defined Networks using a Deep Q-Network (DQN) agent, with real-time visualization.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Mininet    │◄───►│     Ryu      │◄───►│  DRL Agent  │
│  Topology    │     │  Controller  │     │   (DQN)     │
│  4sw + 6host │     │  OpenFlow1.3 │     │  PyTorch    │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
                     net_stats.json         drl_decision.json
                           │                     │
                    ┌──────▼─────────────────────▼──────┐
                    │      Flask Dashboard (port 5000)   │
                    │  NetworkX Graph + Chart.js Charts   │
                    └────────────────────────────────────┘
```

## Project Structure

```
sdn_drl_routing/
├── topology/
│   └── custom_topology.py    # 4 switches, 6 hosts, variable bandwidths
├── controller/
│   ├── ryu_controller.py     # OpenFlow 1.3 controller with DRL integration
│   └── stats_collector.py    # Collects stats → data/net_stats.json
├── drl/
│   ├── environment.py        # Gym-compatible RL environment
│   ├── dqn_agent.py          # DQN with replay buffer + target network
│   └── train.py              # Training script (simulation or live)
├── visualization/
│   ├── dashboard.py          # Flask web dashboard (auto-refresh)
│   └── net_graph.py          # NetworkX + Matplotlib graph renderer
├── data/
│   ├── net_stats.json        # Shared state file
│   ├── checkpoints/          # Model checkpoints
│   └── results/              # Training results
├── run.sh                    # Start all components
├── requirements.txt
└── README.md
```

## Network Topology

```
      h1  h2          h3
       \  /            |
       s1 ----100M---- s2
       |  \___50M___/  |
      10M            10M
       |  /---50M---\  |
       s3 ----100M---- s4
       |               / \
       h4            h5   h6
```

- **s1-s2, s3-s4**: 100 Mbps backbone links
- **s1-s4, s2-s3**: 50 Mbps cross-links
- **s1-s3, s2-s4**: 10 Mbps alternate paths (congestion-prone)

## Prerequisites

- **Ubuntu 20.04+** (Mininet requires Linux)
- Python 3.10+
- Mininet: `sudo apt install mininet`
- Open vSwitch: `sudo apt install openvswitch-switch`

## Setup

```bash
# Clone/download the project
cd sdn_drl_routing

# Install Python dependencies
pip install -r requirements.txt

# Install Ryu (if not available via pip)
pip install ryu

# Make run script executable
chmod +x run.sh
```

## Usage

### Full System (requires root)
```bash
sudo ./run.sh
```
This starts: Ryu Controller → Mininet Topology → DRL Agent → Dashboard

Open **http://localhost:5000** to see the live dashboard.

### Simulation Training Only (no Mininet needed)
```bash
./run.sh --train-only
```

### Dashboard Only
```bash
./run.sh --dashboard-only
```

### Manual Component Start
```bash
# Terminal 1: Ryu Controller
ryu-manager controller/ryu_controller.py --ofp-tcp-listen-port 6633 --observe-links

# Terminal 2: Mininet
sudo python3 -c "from topology.custom_topology import run_topology; run_topology()"

# Terminal 3: DRL Training
python3 -m drl.train --mode live --episodes 1000

# Terminal 4: Dashboard
python3 -m visualization.dashboard --port 5000
```

## DRL Formulation

| Component | Description |
|-----------|-------------|
| **State** | Per-link: utilization, delay, packet count, bandwidth (normalized) |
| **Action** | Select path index from K candidate paths |
| **Reward** | `R = throughput - 0.5×delay - 0.3×packet_loss` |
| **Algorithm** | DQN with target network, experience replay (50K buffer) |
| **Exploration** | ε-greedy: 1.0 → 0.01 (decay 0.995/episode) |

## Testing

```bash
# In Mininet CLI:
pingall                           # Test connectivity
h1 ping h6                       # Ping across topology
iperf h1 h6                      # Bandwidth test

# Generate congestion:
h1 iperf -s &                    # Start iperf server on h1
h6 iperf -c 10.0.0.1 -t 60      # Generate 60s traffic flow
```

## Expected Behavior

1. **Without DRL**: Traffic follows shortest path, congested links degrade
2. **With DRL**: Agent learns to route around congestion, selecting alternate paths
3. **Dashboard**: Links change color (green→yellow→red) as utilization increases; DRL path shown in blue

## Target Improvements (vs Shortest Path)

- **Latency**: ~25% reduction
- **Throughput**: ~55% increase
- **Packet Loss**: ~55% reduction
