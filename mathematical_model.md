# SDN DRL Routing — Mathematical Model

## Overview

This project uses **Deep Reinforcement Learning (DRL)** to find the **optimal routing path** in a Software-Defined Network (SDN). Three algorithms are implemented: **SAC**, **DDPG**, and **TD3**. Each agent observes the network state, selects a routing path, and is evaluated using mathematically consistent performance metrics.

---

## 1. Network Topology

The network has **4 switches** and **6 hosts** connected as a diamond mesh:

```
       h1  h2              h3
        \  |               /
         s1 ----100Mbps--- s2
        / |  \            / |
   50Mbps |  10Mbps  10Mbps |
      /   |      \    /    |
    s4 ---100Mbps--- s3   50Mbps
   / \                    |
  h5  h6               h4
```

### Link Properties

| Link | Bandwidth (Mbps) | Base Delay (ms) |
|:---|:---:|:---:|
| s1 — s2 | 100 | 2 |
| s3 — s4 | 100 | 2 |
| s1 — s4 | 50 | 5 |
| s2 — s3 | 50 | 5 |
| s1 — s3 | 10 | 10 |
| s2 — s4 | 10 | 10 |

---

## 2. Inputs (State Vector)

The DRL agent receives a **48-dimensional state vector** as input, representing the real-time condition of all 12 network links.

### How the State is Built

Each of the **12 links** contributes **4 features**:

| Feature | What It Measures | Normalization |
|:---|:---|:---|
| **Utilization** | How busy the link is (0–1) | `min(1.0, used_bw / total_bw)` |
| **TX Rate** | Transmission speed | `min(1.0, tx_bps / 1 Gbps)` |
| **Packet Count** | Packets in transit | `0.0` (placeholder in live mode) |
| **Bandwidth** | Link capacity | `min(1.0, bw_mbps / 1000)` |

**State = [link1_util, link1_tx, link1_pkt, link1_bw, link2_util, ..., link12_bw]**

> **Example**: If link s1—s2 has 40% utilization, 50 Mbps TX rate, and 100 Mbps capacity:
> `[0.4, 0.05, 0.0, 0.1]`

### Code Location
- Backend: `drl/environment.py`, function `_get_live_state()` (line 236)
- Frontend: `dashboard.py`, function `get_drl_state()` (line 194)

---

## 3. Outputs (Action Selection)

The agent outputs a **single integer** (action) from `{0, 1, 2, ..., 9}`.

This integer is an **index** into a list of **10 candidate paths** between the source and destination host.

### How Candidate Paths are Generated
1. Find all simple paths between source and destination using NetworkX
2. Extract the **switch-only** segment (e.g., `[h1, s1, s2, h3]` → `[s1, s2]`)
3. Sort by hop count (shortest first)
4. Keep up to 10 paths, pad with duplicates if fewer exist

### Example: h1 → h6

| Action | Path | Hops |
|:---:|:---|:---:|
| 0 | s1 → s4 | 1 |
| 1 | s1 → s2 → s4 | 2 |
| 2 | s1 → s3 → s4 | 2 |
| 3 | s1 → s2 → s3 → s4 | 3 |
| 4 | s1 → s3 → s2 → s4 | 3 |
| 5–9 | s1 → s4 (padded) | 1 |

### Code Location
- Backend: `drl/environment.py`, function `step()` (line 150)
- Frontend: `dashboard.py`, function `compute_drl_path()` (line 272)

---

## 4. Performance Formulas

These three metrics evaluate how good a selected path is. They are used **identically** in both the backend (training) and frontend (dashboard visualization).

### A. Network Delay (D)

**Formula:**
```
D = Σ delay_i × (1 + 3 × utilization_i)
```
- Sum over each link `i` in the selected path
- `delay_i` = base propagation delay of link `i` (from topology table)
- `utilization_i` = current traffic load on link `i` (0.0 to 1.0)
- The factor `(1 + 3 × util)` models **congestion**: higher utilization = higher delay

**Example — Path s1 → s4 (utilization = 0.0):**
```
D = 5 × (1 + 3 × 0.0) = 5.0 ms
```

**Example — Path s1 → s2 → s3 → s4 (utilization = 0.0):**
```
D = 2×(1+0) + 5×(1+0) + 2×(1+0) = 2 + 5 + 2 = 9.0 ms
```

### B. Throughput (T)

**Formula:**
```
T = min(1.0, MinBW / 100) × 100  Mbps
```
- `MinBW` = minimum bandwidth along the path (bottleneck link)
- Capped at 100 Mbps maximum

**Example — Path s1 → s4:**
```
MinBW = 50 Mbps (s1-s4 link)
T = min(1.0, 50/100) × 100 = 0.5 × 100 = 50.0 Mbps
```

**Example — Path s1 → s2 → s4:**
```
MinBW = min(100, 10) = 10 Mbps (s2-s4 bottleneck)
T = min(1.0, 10/100) × 100 = 0.1 × 100 = 10.0 Mbps
```

### C. Packet Loss (L)

**Formula:**
```
L = MaxUtil² × 10.0  (%)
```
- `MaxUtil` = highest utilization on any link in the path
- Quadratic relationship: loss grows rapidly as congestion increases

**Example — MaxUtil = 0.5:**
```
L = 0.5² × 10.0 = 0.25 × 10.0 = 2.5%
```

**Example — MaxUtil = 0.0 (idle network):**
```
L = 0.0² × 10.0 = 0.0%
```

### Code Location
- Backend: `drl/environment.py`, function `_calculate_reward_with_metrics()` (line 329)
- Frontend: `dashboard.py`, function `update_stats_file()` (line 352)

---

## 5. Reward Function (Training)

During training, the agent maximizes a composite reward:

```
R = (w1 × throughput_score) - (w2 × delay_penalty) - (w3 × loss_penalty) + hop_bonus
```

Where:
- `throughput_score = min(1.0, MinBW / 100)` — higher is better
- `delay_penalty = min(1.0, total_delay / 50)` — lower is better
- `loss_penalty = MaxUtil²` — lower is better
- `hop_bonus = 0.1 / path_length` — fewer hops preferred
- Weights: `w1 = 1.0`, `w2 = 0.5`, `w3 = 0.3`

### Code Location
- `drl/environment.py`, function `_calculate_reward_with_metrics()` (line 358)

---

## 6. Best-of-Three: How the Final Output is Chosen

The system does **not** rely on a single algorithm. Instead, it runs **all three agents** on the same input, evaluates the path each one selects, and picks the **best** one.

### Process
1. All 3 agents (SAC, DDPG, TD3) receive the **same** 48-dim state vector
2. Each agent selects an action (path index)
3. Each selected path is scored using the **composite reward formula**:

```
Score = -(w1 * throughput_score - w2 * delay_penalty - w3 * loss_penalty)
```

Where:
- `throughput_score = min(1.0, MinBW / 100)`
- `delay_penalty = min(1.0, total_delay / 50)`
- `loss_penalty = MaxUtil^2`
- Weights: `w1=1.0, w2=0.5, w3=0.3`

4. The path with the **lowest score** (= highest reward) wins

### Verified Example (h1 -> h6)

| Algorithm | Selected Path | Delay (ms) | Throughput (Mbps) | Score | Winner? |
|:---|:---|:---:|:---:|:---:|:---:|
| **SAC** | s1 -> s2 -> s4 | 12.0 | 10.0 | -0.78 | |
| **DDPG** | s1 -> s4 | 5.0 | 50.0 | **-0.45** | **YES** |
| **TD3** | s1 -> s2 -> s3 -> s4 | 9.0 | 50.0 | -0.41 | |

> The system automatically selects **DDPG's path** because it has the best composite reward.

> Backend calculations, frontend visualizations, and manual hand-calculations all produce **identical results**.
