"""
Live Visualization Dashboard for SDN DRL Routing.

Layout: Sidebar (controls + views + packet send) | Center (topology / charts / packet log).
Features: Animated yellow dot on topology showing packet movement along path.
"""

import os
import sys
import json
import base64
import time
import random
import threading
import networkx as nx
import numpy as np
from flask import Flask, render_template_string, jsonify, request, redirect
from flasgger import Swagger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.net_graph import render_topology
from controller.stats_collector import read_stats, STATS_FILE
try:
    from drl.sac_agent import SACAgent
    from drl.ddpg_agent import DDPGAgent
    from drl.td3_agent import TD3Agent
    HAS_DRL = True
except ImportError:
    print("[Dashboard] Warning: DRL dependencies (like torch) not found. Intelligent routing will be disabled.")
    HAS_DRL = False

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

app_state = {
    'active_topology': 'diamond',
    'disabled_links': set(),
    'selected_src': None,
    'selected_dst': None,
    'current_path': [],
    'routing_mode': 'drl',        # Start with DRL
    'drl_algorithm': 'sac',       # Default DRL algorithm
    'packet_log': [],
    'packet_counter': 0,
    'continuous_send': False,
    'agents': {} # Store initialized agents here
}

# Global var to share port with Swagger redirect
DASHBOARD_PORT = 9000

# Swagger Configuration for Main App (Port 9000)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}
swagger = Swagger(app, config=swagger_config)

# Separate app for Swagger UI on port 8000
swagger_ui_app = Flask("SwaggerUI")

@swagger_ui_app.route('/')
def swagger_ui_index():
    # Placeholder, will be patched at runtime or we can use a relative link if possible
    # But since it's a separate app, we need the dashboard port.
    # We'll use a global or pass it in.
    return redirect(f"http://localhost:{DASHBOARD_PORT}/apidocs/")

# Initialize and Load Trained Agents
def init_agents():
    if not HAS_DRL:
        print("[Dashboard] Skipping agent initialization (no DRL dependencies).")
        return
    
    state_dim = 48 # Matches training env
    action_dim = 10
    
    try:
        sac = SACAgent(state_dim, action_dim)
        sac.load(os.path.join(CHECKPOINT_DIR, 'sac_trained.pt'))
        
        ddpg = DDPGAgent(state_dim, action_dim)
        ddpg.load(os.path.join(CHECKPOINT_DIR, 'ddpg_trained.pt'))
        
        td3 = TD3Agent(state_dim, action_dim)
        td3.load(os.path.join(CHECKPOINT_DIR, 'td3_trained.pt'))
        
        app_state['agents'] = {'sac': sac, 'ddpg': ddpg, 'td3': td3}
    except Exception as e:
        print(f"[Dashboard] Error loading agents: {e}")

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'checkpoints')
try:
    init_agents()
except Exception as e:
    print(f"[Dashboard] Error loading agents: {e}")


# ========================================================
# Routing Logic
# ========================================================

# ========================================================
# Topology Configurations
# ========================================================

TOPOLOGIES = {
    'diamond': {
        'name': 'Diamond Mesh',
        'switches': ['s1', 's2', 's3', 's4'],
        'hosts': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
        'links': [
            ('s1', 's2', 100, 2), ('s3', 's4', 100, 2),
            ('s1', 's4', 50, 5), ('s2', 's3', 50, 5),
            ('s1', 's3', 10, 10), ('s2', 's4', 10, 10),
        ],
        'host_map': {'h1': 's1', 'h2': 's1', 'h3': 's2', 'h4': 's3', 'h5': 's4', 'h6': 's4'},
        'pos': {
            's1': (-1.2, 1.0), 's2': (1.2, 1.0), 's3': (-1.2, -1.0), 's4': (1.2, -1.0),
            'h1': (-2.3, 1.8), 'h2': (-0.1, 1.8), 'h3': (2.3, 1.8),
            'h4': (-2.3, -1.8), 'h5': (0.1, -1.8), 'h6': (2.3, -1.8),
        }
    },
    'ring': {
        'name': 'Ring Topology',
        'switches': ['s1', 's2', 's3', 's4'],
        'hosts': ['h1', 'h2', 'h3', 'h4'],
        'links': [
            ('s1', 's2', 100, 2), ('s2', 's3', 100, 2),
            ('s3', 's4', 100, 2), ('s4', 's1', 100, 2),
        ],
        'host_map': {'h1': 's1', 'h2': 's2', 'h3': 's3', 'h4': 's4'},
        'pos': {
            's1': (-1.5, 1.5), 's2': (1.5, 1.5), 's3': (1.5, -1.5), 's4': (-1.5, -1.5),
            'h1': (-2.5, 2.0), 'h2': (2.5, 2.0), 'h3': (2.5, -2.0), 'h4': (-2.5, -2.0),
        }
    },
    'mesh': {
        'name': 'Full Mesh',
        'switches': ['s1', 's2', 's3', 's4'],
        'hosts': ['h1', 'h2', 'h3', 'h4'],
        'links': [
            ('s1', 's2', 100, 2), ('s1', 's3', 100, 2), ('s1', 's4', 100, 2),
            ('s2', 's3', 100, 2), ('s2', 's4', 100, 2), ('s3', 's4', 100, 2),
        ],
        'host_map': {'h1': 's1', 'h2': 's2', 'h3': 's3', 'h4': 's4'},
        'pos': {
            's1': (0.0, 1.5), 's2': (1.5, -0.5), 's3': (-1.5, -0.5), 's4': (0.0, -1.8),
            'h1': (0.0, 2.2), 'h2': (2.2, -0.8), 'h3': (-2.2, -0.8), 'h4': (0.0, -2.3),
        }
    },
    'star': {
        'name': 'Star Topology',
        'switches': ['s1', 's2', 's3', 's4'],
        'hosts': ['h1', 'h2', 'h3', 'h4', 'h5'],
        'links': [
            ('s1', 's2', 100, 2), ('s1', 's3', 100, 2), ('s1', 's4', 100, 2),
        ],
        'host_map': {'h1': 's1', 'h2': 's2', 'h3': 's3', 'h4': 's4', 'h5': 's4'},
        'pos': {
            's1': (0.0, 0.0), 's2': (-1.5, 1.5), 's3': (1.5, 1.5), 's4': (0.0, -1.8),
            'h1': (-0.5, 0.5), 'h2': (-2.2, 2.0), 'h3': (2.2, 2.0), 'h4': (-0.8, -2.2), 'h5': (0.8, -2.2),
        }
    }
}

def build_graph(disabled_links=None):
    topo_key = app_state.get('active_topology', 'diamond')
    topo = TOPOLOGIES[topo_key]
    G = nx.Graph()
    for sw in topo['switches']:
        G.add_node(sw, type='switch')
    for h in topo['hosts']:
        G.add_node(h, type='host')
    for src, dst, bw, delay in topo['links']:
        if tuple(sorted((src, dst))) not in (disabled_links or set()):
            G.add_edge(src, dst, bw=bw, delay=delay, weight=delay)
    for h, sw in topo['host_map'].items():
        G.add_edge(h, sw, bw=1000, delay=1, weight=1)
    return G

def get_drl_state():
    """Construct state vector exactly like drl/environment.py."""
    stats = read_stats()
    link_stats = stats.get('link_stats', {})
    G = build_graph()
    
    # Environment uses switch IDs 1-4 and host IDs 5-10
    # Dashboard uses s1-s4 and h1-h6. We need to map them to match the model.
    sw_map = {'s1': 1, 's2': 2, 's3': 3, 's4': 4}
    host_map = {'h1': 5, 'h2': 6, 'h3': 7, 'h4': 8, 'h5': 9, 'h6': 10}
    
    # Get edge list as expected by environment (12 edges)
    # environment.py link_list: [(5, 1), (6, 1), (7, 2), (8, 3), (9, 4), (10, 4), (1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)]
    edges = [
        (5, 1), (6, 1), (7, 2), (8, 3), (9, 4), (10, 4), # Hosts
        (1, 2), (3, 4), (1, 4), (2, 3), (1, 3), (2, 4)  # Switches
    ]
    
    mapping = {**sw_map, **host_map}
    rev_mapping = {v: k for k, v in mapping.items()}
    
    state = []
    for u_id, v_id in edges:
        u_name, v_name = rev_mapping[u_id], rev_mapping[v_id]
        link_key = f"{u_name}-{v_name}"
        alt_key = f"{v_name}-{u_name}"
        ls = link_stats.get(link_key, link_stats.get(alt_key, {}))
        
        util = ls.get('utilization', 0.0)
        bw_kbps = ls.get('bandwidth_mbps', 100 if u_id < 5 and v_id < 5 else 1000)
        # Normalize as in environment.py
        # Environment uses 100/50/10 for switches, 1000 for hosts.
        # It normalizes by dividing by 1000.
        bw_norm = bw_kbps / 1000.0
        
        # tx_bps normalization: environment uses min(1.0, tx_bps / 1e9)
        tx_bps = ls.get('tx_bps', 0)
        tx_norm = min(1.0, tx_bps / 1e9)
        
        # Delay: env uses 0.0 in live mode? (checked environment.py:260) 
        # Actually it uses min(1.0, tx), 0.0, min(1.0, bw)
        state.extend([
            min(1.0, util),
            min(1.0, tx_norm),
            0.0, # Packet count placeholder in live mode
            min(1.0, bw_norm),
        ])
        
    return np.array(state, dtype=np.float32)

def get_candidate_paths(src, dst):
    """Generate candidate paths exactly like drl/environment.py."""
    G = build_graph()
    try:
        all_paths = list(nx.all_simple_paths(G, src, dst, cutoff=6))
        # Env filters to switch-only paths (if src/dst are switches)
        # But here src/dst are hosts. We need the switch segment.
        switch_paths = []
        for p in all_paths:
            # Extract switch sequence: e.g., ['h1', 's1', 's2', 'h3'] -> ['s1', 's2']
            sp = [n for n in p if n.startswith('s')]
            if len(sp) >= 2 and sp not in switch_paths:
                switch_paths.append(sp)
        
        switch_paths.sort(key=len)
        candidates = switch_paths[:10]
        
        # Pad to 10
        while len(candidates) < 10:
            if candidates:
                candidates.append(candidates[0])
            else:
                # Fallback if no paths
                candidates.append([])
        return candidates
    except nx.NetworkXNoPath:
        return [[]]*10

def evaluate_path(path_switches, link_stats):
    """Score a switch path using the same formulas as environment.py.
    Returns (delay, throughput, loss, composite_score).
    Lower composite_score = better path.
    """
    if not path_switches or len(path_switches) < 2:
        return 999.0, 0.0, 100.0, 999.0
    
    G = build_graph()
    total_delay = 0.0
    max_utilization = 0.0
    min_bandwidth = float('inf')
    
    for i in range(len(path_switches) - 1):
        u, v = path_switches[i], path_switches[i+1]
        if G.has_edge(u, v):
            edge_data = G.edges[u, v]
            d = edge_data.get('delay', 1)
            b = edge_data.get('bw', 100)
            
            lk, ak = f"{u}-{v}", f"{v}-{u}"
            ls = link_stats.get(lk, link_stats.get(ak, {}))
            util = ls.get('utilization', 0.0)
            
            total_delay += d * (1 + util * 3)
            max_utilization = max(max_utilization, util)
            min_bandwidth = min(min_bandwidth, b)
    
    throughput = min(1.0, min_bandwidth / 100.0) * 100.0
    loss = (max_utilization ** 2) * 10.0
    
    # Composite score (same weights as environment.py reward, lower = better)
    # R = w1*throughput - w2*delay - w3*loss  =>  score = -R (minimize)
    score = -(1.0 * min(1.0, min_bandwidth / 100.0)
              - 0.5 * min(1.0, total_delay / 50.0)
              - 0.3 * (max_utilization ** 2))
    
    return total_delay, throughput, loss, score


def compute_drl_path(src, dst, disabled):
    """Run ALL 3 DRL agents, compare their outputs, and select the best path."""
    if not HAS_DRL:
        return [], "DRL dependencies not found"
    
    agents = app_state.get('agents', {})
    if not agents:
        return [], "No agents initialized"
    
    state = get_drl_state()
    candidates = get_candidate_paths(src, dst)
    stats = read_stats()
    link_stats = stats.get('link_stats', {})
    
    # Run all agents and evaluate each selected path
    results = []
    for algo_name in ['sac', 'ddpg', 'td3']:
        agent = agents.get(algo_name)
        if not agent:
            continue
        
        # Handle inconsistent select_action signatures
        if algo_name == 'td3':
            action = agent.select_action(state)
        else:
            action = agent.select_action(state, training=False)
        
        path_sw = candidates[action]
        delay, throughput, loss, score = evaluate_path(path_sw, link_stats)
        
        results.append({
            'algo': algo_name,
            'action': action,
            'path': path_sw,
            'delay': delay,
            'throughput': throughput,
            'loss': loss,
            'score': score,
        })
    
    if not results:
        return [], "No agents available"
    
    # Pick the best path (lowest composite score)
    best = min(results, key=lambda r: r['score'])
    
    # Store comparison info for the UI
    app_state['drl_comparison'] = results
    app_state['drl_winner'] = best['algo']
    app_state['drl_algorithm'] = best['algo']  # Update active algorithm to winner
    
    if not best['path']:
        return [], "No path found by any agent"
    
    full_path = [src] + best['path'] + [dst]
    return full_path, ""


def compute_shortest_path(src, dst, disabled):
    G = build_graph(disabled)
    try:
        return nx.shortest_path(G, src, dst, weight='delay'), ""
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], "No path exists"

def update_stats_file(path, disabled_links):
    stats = read_stats()
    topo_key = app_state.get('active_topology', 'diamond')
    topo = TOPOLOGIES[topo_key]
    
    # Update topology in stats
    stats['topology'] = {
        'switches': topo['switches'],
        'hosts': topo['hosts'],
        'links': [{'src': l[0], 'dst': l[1], 'bw': l[2]} for l in topo['links']],
        'host_switch_map': topo['host_map'],
        'node_pos': topo['pos']
    }

    switch_path = [n for n in path if n.startswith('s')]
    stats['selected_path'] = switch_path
    
    link_stats = {}
    for src, dst, bw, delay in topo['links']:
        key = f"{src}-{dst}"
        sk = tuple(sorted((src, dst)))
        is_disabled = sk in disabled_links
        is_on_path = any(
            tuple(sorted((switch_path[i], switch_path[i+1]))) == sk
            for i in range(len(switch_path) - 1)
        ) if len(switch_path) >= 2 else False
        
        if is_disabled:
            util = 0.0
        elif is_on_path:
            util = random.uniform(0.3, 0.6)
        else:
            util = random.uniform(0.05, 0.25)
            
        link_stats[key] = {
            'utilization': round(util, 3),
            'bandwidth_mbps': bw,
            'tx_bps': int(util * bw * 1e6),
            'rx_bps': int(util * bw * 0.95e6),
            'disabled': is_disabled,
        }
    stats['link_stats'] = link_stats
    # ... rest of the function ...
    # Calculate performance metrics using formulas from drl/environment.py
    total_delay = 0.0
    max_utilization = 0.0
    min_bandwidth = float('inf')
    
    if path:
        G = build_graph()
        # Filter switches from path
        sw_path = [n for n in path if n.startswith('s')]
        for i in range(len(sw_path) - 1):
            u, v = sw_path[i], sw_path[i+1]
            if G.has_edge(u, v):
                edge_data = G.edges[u, v]
                d = edge_data.get('delay', 1)
                b = edge_data.get('bw', 100)
                
                lk, ak = f"{u}-{v}", f"{v}-{u}"
                ls = link_stats.get(lk, link_stats.get(ak, {}))
                util = ls.get('utilization', 0.0)
                
                total_delay += d * (1 + util * 3)
                max_utilization = max(max_utilization, util)
                min_bandwidth = min(min_bandwidth, b)
        
        # Matching drl/environment.py:354-368
        t_val = min(1.0, min_bandwidth / 100.0) * 100.0
        d_val = total_delay
        l_val = (max_utilization ** 2) * 10.0
    else:
        d_val, l_val, t_val = 50.0, 10.0, 0.0

    perf = stats.get('performance', {'delay': [], 'packet_loss': [], 'throughput': []})
    perf['delay'] = (perf.get('delay', []) + [round(d_val, 2)])[-50:]
    perf['packet_loss'] = (perf.get('packet_loss', []) + [round(l_val, 4)])[-50:]
    perf['throughput'] = (perf.get('throughput', []) + [round(t_val, 2)])[-50:]
    
    stats['performance'] = perf
    stats['timestamp'] = time.time()
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)


# ========================================================
# Packet Simulation
# ========================================================

def simulate_single_packet(src, dst, mode, disabled):
    if mode == 'drl':
        path, err = compute_drl_path(src, dst, disabled)
    else:
        path, err = compute_shortest_path(src, dst, disabled)

    app_state['packet_counter'] += 1
    seq = app_state['packet_counter']
    ts = time.time()

    if not path:
        result = {
            'seq': seq, 'src': src, 'dst': dst,
            'path': [], 'hops': 0, 'rtt': None,
            'status': 'DROPPED', 'timestamp': ts,
        }
    else:
        G = build_graph(disabled)
        total_delay = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                total_delay += G.edges[u, v].get('delay', 1)
        rtt = round(total_delay * 2 + random.gauss(0, total_delay * 0.15), 2)
        rtt = max(0.5, rtt)
        stats = read_stats()
        link_stats = stats.get('link_stats', {})
        max_util = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            lk, ak = f"{u}-{v}", f"{v}-{u}"
            ls = link_stats.get(lk, link_stats.get(ak, {}))
            max_util = max(max_util, ls.get('utilization', 0))
        drop_prob = min(0.3, max_util * 0.15)
        delivered = random.random() > drop_prob
        result = {
            'seq': seq, 'src': src, 'dst': dst,
            'path': path, 'hops': len(path) - 1,
            'rtt': rtt if delivered else None,
            'status': 'DELIVERED' if delivered else 'DROPPED',
            'timestamp': ts,
        }

    app_state['packet_log'].append(result)
    if len(app_state['packet_log']) > 200:
        app_state['packet_log'] = app_state['packet_log'][-200:]
    if path:
        update_stats_file(path, disabled)
    return result


# ========================================================
# Dashboard HTML
# ========================================================

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SDN DRL Routing</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-0:#060918;--bg-1:#0d1025;--bg-2:#111432;--bg-3:#161a3a;--bg-input:#0a0d1e;
            --border:#1e2248;--border-l:#2a2f5c;
            --t1:#eef0ff;--t2:#8b8fb5;--t3:#5c5f82;
            --blue:#4f6ef7;--blue-l:#7b93ff;--cyan:#06d6d6;
            --emerald:#10b981;--amber:#f59e0b;--rose:#ef4444;--violet:#8b5cf6;
            --r:12px;--rs:8px;
        }
        *{margin:0;padding:0;box-sizing:border-box;}
        body{font-family:'Inter',sans-serif;background:var(--bg-0);color:var(--t1);display:flex;height:100vh;overflow:hidden;}
        ::-webkit-scrollbar{width:4px;}
        ::-webkit-scrollbar-track{background:transparent;}
        ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}

        .sb{width:300px;min-width:300px;background:var(--bg-1);border-right:1px solid var(--border);display:flex;flex-direction:column;height:100vh;overflow-y:auto;}
        .sb-brand{padding:20px;border-bottom:1px solid var(--border);}
        .sb-brand h1{font-size:16px;font-weight:800;background:linear-gradient(135deg,var(--blue-l),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
        .sb-brand p{font-size:10px;color:var(--t3);margin-top:2px;letter-spacing:1px;}
        .sec{padding:14px 20px;border-bottom:1px solid var(--border);}
        .sec:last-child{border-bottom:none;}
        .sec-title{font-size:10px;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;}
        .fg{margin-bottom:10px;}
        .fg label{display:block;font-size:11px;font-weight:600;color:var(--t2);margin-bottom:4px;}
        .fg select{width:100%;padding:8px 10px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--rs);color:var(--t1);font-family:'Inter',sans-serif;font-size:12px;font-weight:500;outline:none;cursor:pointer;transition:border .2s;-webkit-appearance:none;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='%235c5f82'%3E%3Cpath d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;}
        .fg select:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(79,110,247,.15);}
        .mode-sw{display:flex;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--rs);padding:3px;gap:3px;margin-bottom:10px;}
        .mode-b{flex:1;padding:6px 0;border:none;border-radius:6px;font-family:'Inter';font-size:11px;font-weight:600;cursor:pointer;background:transparent;color:var(--t3);transition:all .25s;}
        .mode-b.active{background:var(--blue);color:#fff;box-shadow:0 2px 8px rgba(79,110,247,.3);}
        .algo-picker{margin-bottom:10px;animation:fadeSlide .3s ease;}
        @keyframes fadeSlide{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
        .algo-btns{display:flex;gap:5px;}
        .algo-b{flex:1;padding:8px 4px;border:1px solid var(--border);border-radius:var(--rs);font-family:'Inter';font-size:11px;font-weight:600;cursor:pointer;background:var(--bg-input);color:var(--t3);transition:all .25s;display:flex;flex-direction:column;align-items:center;gap:3px;}
        .algo-b:hover{border-color:var(--blue);color:var(--t1);}
        .algo-b.active{background:linear-gradient(135deg,rgba(79,110,247,.15),rgba(34,211,238,.1));border-color:var(--blue);color:var(--blue-l);box-shadow:0 2px 10px rgba(79,110,247,.2);}
        .algo-icon{font-size:14px;}
        .btn{width:100%;padding:9px;border:none;border-radius:var(--rs);font-family:'Inter';font-size:12px;font-weight:700;cursor:pointer;transition:all .2s;}
        .btn-p{background:linear-gradient(135deg,var(--blue),#3b52d4);color:#fff;box-shadow:0 4px 14px rgba(79,110,247,.2);}
        .btn-p:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(79,110,247,.3);}
        .btn-p:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none;}
        .btn-g{background:transparent;border:1px solid var(--border);color:var(--t2);margin-top:6px;}
        .btn-g:hover{border-color:var(--border-l);color:var(--t1);}
        .btn-emerald{background:linear-gradient(135deg,var(--emerald),#059669);color:#fff;box-shadow:0 4px 14px rgba(16,185,129,.2);}
        .btn-emerald:hover{transform:translateY(-1px);}
        .btn-emerald:disabled{opacity:.4;cursor:not-allowed;transform:none;}
        .btn-amber{background:linear-gradient(135deg,var(--amber),#d97706);color:#fff;}
        .btn-amber:hover{transform:translateY(-1px);}
        .btn-rose{background:linear-gradient(135deg,var(--rose),#dc2626);color:#fff;}
        .btn-rose:hover{transform:translateY(-1px);}
        .btn-row{display:flex;gap:6px;}
        .btn-row .btn{flex:1;}
        .path-box{background:var(--bg-input);border:1px solid var(--border);border-radius:var(--rs);padding:10px;text-align:center;min-height:36px;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:4px;}
        .pc{display:inline-block;padding:3px 9px;background:linear-gradient(135deg,var(--blue),#3b52d4);border-radius:4px;font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace;}
        .ps{color:var(--t3);font-size:9px;}
        .lr{display:flex;align-items:center;justify-content:space-between;padding:7px 10px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--rs);margin-bottom:5px;}
        .lr:hover{border-color:var(--border-l);}
        .lm{display:flex;align-items:center;gap:7px;}
        .ld{width:7px;height:7px;border-radius:50%;}
        .ld.up{background:var(--emerald);box-shadow:0 0 6px rgba(16,185,129,.4);}
        .ld.dn{background:var(--rose);box-shadow:0 0 6px rgba(239,68,68,.4);}
        .ln{font-size:12px;font-weight:600;}
        .lb{font-size:10px;color:var(--t3);font-family:'JetBrains Mono',monospace;}
        .tg{width:36px;height:19px;border-radius:10px;border:none;cursor:pointer;position:relative;transition:background .3s;flex-shrink:0;}
        .tg.on{background:var(--emerald);}
        .tg.off{background:var(--border);}
        .tg::after{content:'';position:absolute;width:15px;height:15px;border-radius:50%;background:#fff;top:2px;transition:left .25s;box-shadow:0 1px 3px rgba(0,0,0,.3);}
        .tg.on::after{left:19px;}
        .tg.off::after{left:2px;}
        .badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace;}
        .badge-ok{background:rgba(16,185,129,.12);color:var(--emerald);}
        .badge-err{background:rgba(239,68,68,.12);color:var(--rose);}
        .badge-info{background:rgba(79,110,247,.12);color:var(--blue-l);}
        .badge-warn{background:rgba(245,158,11,.12);color:var(--amber);}
        .sr{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border);font-size:12px;}
        .sr:last-child{border-bottom:none;}
        .sr .l{color:var(--t3);}
        .sr .v{font-weight:600;font-family:'JetBrains Mono',monospace;font-size:12px;}
        .view-tabs{display:flex;flex-direction:column;gap:4px;}
        .view-tab{padding:8px 12px;background:var(--bg-input);border:1px solid var(--border);border-radius:var(--rs);font-size:12px;font-weight:600;color:var(--t2);cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:8px;font-family:'Inter',sans-serif;text-align:left;}
        .view-tab:hover{border-color:var(--border-l);color:var(--t1);}
        .view-tab.active{border-color:var(--blue);background:rgba(79,110,247,.08);color:var(--blue-l);}
        .view-tab .icon{font-size:13px;width:18px;text-align:center;}
        .pkt-stats{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;}
        .pkt-stat{background:var(--bg-input);border:1px solid var(--border);border-radius:var(--rs);padding:8px;text-align:center;}
        .pkt-stat .val{font-size:16px;font-weight:800;font-family:'JetBrains Mono',monospace;}
        .pkt-stat .lbl{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;}

        .center{flex:1;display:flex;flex-direction:column;height:100vh;overflow:hidden;}
        .topbar{display:flex;justify-content:space-between;align-items:center;padding:12px 24px;border-bottom:1px solid var(--border);background:var(--bg-1);flex-shrink:0;}
        .tb-left{display:flex;align-items:center;gap:10px;}
        .live-dot{width:8px;height:8px;border-radius:50%;background:var(--emerald);box-shadow:0 0 8px rgba(16,185,129,.5);animation:blink 2s ease-in-out infinite;}
        @keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}
        .tb-time{font-size:11px;color:var(--t3);font-family:'JetBrains Mono',monospace;}
        .tb-metrics{display:flex;gap:24px;}
        .tm{text-align:center;}
        .tm-v{font-size:18px;font-weight:800;font-family:'JetBrains Mono',monospace;letter-spacing:-.5px;}
        .tm-v.delay{color:var(--amber);}
        .tm-v.loss{color:var(--rose);}
        .tm-v.tp{color:var(--emerald);}
        .tm-l{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-top:1px;}
        .view-area{flex:1;display:flex;align-items:center;justify-content:center;padding:20px;overflow:hidden;position:relative;}
        .view-panel{display:none;width:100%;height:100%;align-items:center;justify-content:center;}
        .view-panel.active{display:flex;}

        /* Topology container with animation overlay */
        .topo-container{position:relative;width:100%;height:100%;display:flex;align-items:center;justify-content:center;}
        .topo-container img{max-width:100%;max-height:100%;object-fit:contain;border-radius:var(--r);border:1px solid var(--border);}
        .topo-overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10;}

        /* Animated packet dot */
        .packet-dot{
            position:absolute;
            width:14px;height:14px;
            border-radius:50%;
            background:radial-gradient(circle,#ffe066,#f59e0b);
            box-shadow:0 0 12px 4px rgba(245,158,11,.6),0 0 24px 8px rgba(245,158,11,.25);
            transform:translate(-50%,-50%);
            z-index:20;
            pointer-events:none;
            transition:none;
        }
        .packet-dot.moving{
            transition:left 0.35s cubic-bezier(.4,0,.2,1), top 0.35s cubic-bezier(.4,0,.2,1);
        }
        .packet-dot.delivered{
            background:radial-gradient(circle,#6ee7b7,#10b981);
            box-shadow:0 0 12px 4px rgba(16,185,129,.6),0 0 24px 8px rgba(16,185,129,.25);
        }
        .packet-dot.dropped{
            background:radial-gradient(circle,#fca5a5,#ef4444);
            box-shadow:0 0 12px 4px rgba(239,68,68,.6),0 0 24px 8px rgba(239,68,68,.25);
        }
        .packet-dot.fadeout{
            opacity:0;
            transform:translate(-50%,-50%) scale(2);
            transition:opacity .5s, transform .5s;
        }
        /* Trailing particles */
        .trail-dot{
            position:absolute;
            width:6px;height:6px;
            border-radius:50%;
            background:#f59e0b;
            opacity:.5;
            transform:translate(-50%,-50%);
            pointer-events:none;
            z-index:15;
            transition:opacity .4s;
        }

        #view-delay,#view-throughput{flex-direction:column;padding:40px;}
        #view-delay .chart-wrap,#view-throughput .chart-wrap{width:100%;max-width:900px;height:350px;background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r);padding:24px;}
        .chart-title{font-size:14px;font-weight:700;color:var(--t1);margin-bottom:16px;font-family:'JetBrains Mono',monospace;}
        #view-utilization,#view-packets{flex-direction:column;padding:20px;align-items:center;overflow-y:auto;}
        .util-card,.pkt-card{width:100%;max-width:900px;background:var(--bg-2);border:1px solid var(--border);border-radius:var(--r);padding:24px;}
        .ltable,.pkt-table{width:100%;border-collapse:collapse;font-size:12px;}
        .ltable th,.pkt-table th{text-align:left;padding:9px 10px;color:var(--t3);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--bg-2);}
        .ltable td,.pkt-table td{padding:8px 10px;border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-size:11px;}
        .ltable tr:last-child td,.pkt-table tr:last-child td{border-bottom:none;}
        .ut{height:6px;border-radius:3px;background:var(--border);overflow:hidden;min-width:120px;}
        .uf{height:100%;border-radius:3px;transition:width .5s;}
        .pkt-card{max-height:calc(100vh - 140px);overflow-y:auto;}
        .pkt-table td.seq{color:var(--t3);font-size:10px;}
        .pkt-table td.path-cell{font-size:10px;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
        .pkt-table td.rtt{font-weight:600;}
        .empty-msg{text-align:center;padding:40px;color:var(--t3);font-size:13px;}

        /* Toast notification */
        .toast{position:fixed;top:20px;right:20px;z-index:1000;padding:14px 20px;border-radius:var(--r);font-size:12px;font-weight:600;font-family:'Inter',sans-serif;color:#fff;box-shadow:0 8px 32px rgba(0,0,0,.4);transform:translateX(120%);transition:transform .4s cubic-bezier(.4,0,.2,1),opacity .4s;opacity:0;max-width:360px;}
        .toast.show{transform:translateX(0);opacity:1;}
        .toast-drl{background:linear-gradient(135deg,#f59e0b,#d97706);border:1px solid rgba(255,255,255,.15);}
        .toast-drl .toast-icon{font-size:16px;margin-right:8px;}
        .toast-noroute{background:linear-gradient(135deg,#ef4444,#dc2626);border:1px solid rgba(255,255,255,.15);}
        .toast-reset{background:linear-gradient(135deg,var(--emerald),#059669);border:1px solid rgba(255,255,255,.15);}
        .toast-sub{font-size:10px;font-weight:400;opacity:.8;margin-top:4px;}
    </style>
</head>
<body>
    <aside class="sb">
        <div class="sb-brand"><h1>SDN DRL Routing</h1><p>DEEP REINFORCEMENT LEARNING</p></div>

        <div class="sec">
            <div class="sec-title">⬡ Topology Selection</div>
            <div class="fg"><label>Network Topology</label>
                <select id="topo-select" onchange="switchTopology(this.value)">
                    <option value="diamond">Diamond Mesh (Default)</option>
                    <option value="ring">Ring Topology</option>
                    <option value="mesh">Full Mesh</option>
                    <option value="star">Star Topology</option>
                </select>
            </div>
        </div>

        <div class="sec">
            <div class="sec-title">⬡ Route Selection</div>
            <div class="fg"><label>Source Host</label>
                <select id="src-host">
                    <option value="">Select source...</option>
                </select>
            </div>
            <div class="fg"><label>Destination Host</label>
                <select id="dst-host">
                    <option value="">Select destination...</option>
                </select>
            </div>
            <div class="mode-sw">
                <button class="mode-b active" id="mode-drl" onclick="setMode('drl')">DRL Agent</button>
                <button class="mode-b" id="mode-sp" onclick="setMode('shortest')">Shortest Path</button>
            </div>
            <div class="fg algo-picker" id="algo-picker">
                <label>DRL Algorithm</label>
                <div class="algo-btns">
                    <button class="algo-b active" id="algo-sac" onclick="pickAlgo('sac')">
                        <span class="algo-icon">🔥</span> SAC
                    </button>
                    <button class="algo-b" id="algo-ddpg" onclick="pickAlgo('ddpg')">
                        <span class="algo-icon">⚡</span> DDPG
                    </button>
                    <button class="algo-b" id="algo-td3" onclick="pickAlgo('td3')">
                        <span class="algo-icon">🎯</span> TD3
                    </button>
                </div>
            </div>
            <button class="btn btn-p" id="route-btn" onclick="computeRoute()" disabled>Compute Route</button>
        </div>

        <div class="sec">
            <div class="sec-title">📦 Send Packets</div>
            <div class="btn-row">
                <button class="btn btn-emerald" id="send-one" onclick="sendPacket()" disabled>Send 1</button>
                <button class="btn btn-amber" id="send-cont" onclick="startContinuous()" disabled>▶ Stream</button>
            </div>
            <button class="btn btn-rose" id="stop-cont" onclick="stopContinuous()" style="display:none;margin-top:6px;">■ Stop Stream</button>
            <div class="pkt-stats">
                <div class="pkt-stat"><div class="val" id="pkt-sent" style="color:var(--blue-l);">0</div><div class="lbl">Sent</div></div>
                <div class="pkt-stat"><div class="val" id="pkt-delivered" style="color:var(--emerald);">0</div><div class="lbl">Delivered</div></div>
                <div class="pkt-stat"><div class="val" id="pkt-dropped" style="color:var(--rose);">0</div><div class="lbl">Dropped</div></div>
                <div class="pkt-stat"><div class="val" id="pkt-avg-rtt" style="color:var(--amber);">—</div><div class="lbl">Avg RTT</div></div>
            </div>
            <button class="btn btn-g" onclick="clearPackets()">Clear Log</button>
        </div>

        <div class="sec">
            <div class="sec-title">◈ Active Path</div>
            <div class="path-box" id="sidebar-path"><span style="color:var(--t3);font-size:11px;">No route computed</span></div>
        </div>

        <div class="sec">
            <div class="sec-title">◉ Link Control</div>
            <div id="link-controls">
                <!-- Dynamically populated -->
            </div>
            <button class="btn btn-g" onclick="resetLinks()">Reset All Links</button>
        </div>

        <div class="sec">
            <div class="sec-title">◧ Views</div>
            <div class="view-tabs">
                <button class="view-tab active" onclick="switchView('topology',this)"><span class="icon">◉</span>Topology</button>
                <button class="view-tab" onclick="switchView('packets',this)"><span class="icon">📦</span>Packet Log</button>
                <button class="view-tab" onclick="switchView('delay',this)"><span class="icon">◔</span>Delay Chart</button>
                <button class="view-tab" onclick="switchView('throughput',this)"><span class="icon">◑</span>Throughput</button>
                <button class="view-tab" onclick="switchView('utilization',this)"><span class="icon">◧</span>Utilization</button>
            </div>
        </div>

        <div class="sec">
            <div class="sec-title">◎ Status</div>
            <div class="sr"><span class="l">Algorithm</span><span id="status-mode" class="badge badge-info">DRL</span></div>
            <div class="sr" id="status-algo-row"><span class="l">DRL Agent</span><span id="status-algo" class="badge badge-warn">SAC</span></div>
            <div class="sr"><span class="l">Links</span><span class="v" id="status-links">6 / 6</span></div>
            <div class="sr"><span class="l">Hops</span><span class="v" id="status-hops">—</span></div>
        </div>
    </aside>

    <main class="center">
        <div class="topbar">
            <div class="tb-left"><div class="live-dot"></div><span class="tb-time" id="update-time">—</span></div>
        </div>
        <div class="view-area">
            <div class="view-panel active" id="view-topology">
                <div class="topo-container" id="topo-container">
                    <img id="topo-img" src="" alt="Loading...">
                    <div class="topo-overlay" id="topo-overlay"></div>
                </div>
            </div>
            <div class="view-panel" id="view-packets">
                <div class="pkt-card">
                    <div class="chart-title" style="color:var(--cyan);">📦 Packet Transmission Log</div>
                    <table class="pkt-table" id="pkt-table">
                        <thead><tr><th>#</th><th>Source</th><th>Dest</th><th>Path</th><th>Hops</th><th>RTT (ms)</th><th>Status</th><th>Time</th></tr></thead>
                        <tbody id="pkt-tbody"></tbody>
                    </table>
                    <div class="empty-msg" id="pkt-empty">No packets sent yet. Select hosts and click "Send 1" or "▶ Stream".</div>
                </div>
            </div>
            <div class="view-panel" id="view-delay"><div class="chart-wrap"><div class="chart-title" style="color:var(--amber);">⏱ Delay Over Time (ms)</div><canvas id="delay-chart" style="width:100%;height:280px;"></canvas></div></div>
            <div class="view-panel" id="view-throughput"><div class="chart-wrap"><div class="chart-title" style="color:var(--emerald);">⚡ Throughput (Mbps)</div><canvas id="throughput-chart" style="width:100%;height:280px;"></canvas></div></div>
            <div class="view-panel" id="view-utilization"><div class="util-card"><div class="chart-title" style="color:var(--violet);">◧ Link Utilization</div><table class="ltable" id="link-table"><thead><tr><th>Link</th><th>BW</th><th>Status</th><th>Usage</th><th style="min-width:120px">Util</th></tr></thead><tbody></tbody></table></div></div>
        </div>
    </main>

    <!-- Toast container -->
    <div id="toast-container"></div>

    <script>
        let routingMode='shortest', continuousTimer=null, packetData=[];
        let currentPath = [];  // Track current active path

        let activeNodesPos = {}; // Loaded dynamically per topology

        function updateNodePosMap(dataPos) {
            activeNodesPos = {};
            for (let id in dataPos) {
                const [x, y] = dataPos[id];
                activeNodesPos[id] = {
                    x: 6 + (x + 3) / 6 * 88,
                    y: 6 + (2.5 - y) / 5 * 88
                };
            }
        }

        // Initialize topology
        function initTopologies() {
            fetch('/api/topologies').then(r=>r.json()).then(data=>{
                const sel = document.getElementById('topo-select');
                if (sel) sel.value = data.current;
                switchTopology(data.current, true);
            });
        }

        function switchTopology(topoId, silent) {
            fetch('/api/set_topology', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({topology: topoId})
            }).then(r=>r.json()).then(data => {
                updateNodePosMap(data.pos);
                updateUIForTopology(data);
                if (!silent) {
                    showToast('<span class="toast-icon">🌐</span> Topology Switched', 'Active: ' + data.name, 'toast-reset');
                    currentPath = [];
                    document.getElementById('sidebar-path').innerHTML='<span style="color:var(--t3);font-size:11px;">No route computed</span>';
                    document.getElementById('status-hops').textContent = '\u2014';
                    packetData = [];
                    renderPacketTable();
                    updatePktStats();
                }
                refresh();
            });
        }

        function switchAlgorithm(algoId) {
            fetch('/api/set_algorithm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({algorithm: algoId})
            }).then(r=>r.json()).then(data => {
                if (data.status === 'ok') {
                    showToast('<span class="toast-icon">🤖</span> Algorithm Switched', 'Active: ' + algoId.toUpperCase(), 'toast-drl');
                    if (routingMode === 'drl') {
                        computeRoute();
                    }
                }
            });
        }

        function updateUIForTopology(data) {
            // Populate host selects
            const src = document.getElementById('src-host');
            const dst = document.getElementById('dst-host');
            if (src && dst) {
                const options = '<option value="">Select host...</option>' + 
                    data.hosts.map(h => `<option value="${h}">${h.toUpperCase()}</option>`).join('');
                src.innerHTML = options;
                dst.innerHTML = options;
            }

            // Populate links (we get these from stats refresh usually, but let's clear for now)
            const linkWrap = document.getElementById('link-controls');
            if (linkWrap) linkWrap.innerHTML = '<div class="empty-msg" style="padding:10px;">Refreshing links...</div>';
        }

        function getImageRect() {
            const img = document.getElementById('topo-img');
            const overlay = document.getElementById('topo-overlay');
            if (!img || !overlay || !img.complete || img.naturalWidth === 0) return null;

            const rect = img.getBoundingClientRect();
            const ovRect = overlay.getBoundingClientRect();

            return {
                left: rect.left - ovRect.left,
                top: rect.top - ovRect.top,
                width: rect.width,
                height: rect.height
            };
        }

        function nodeToPixel(nodeName) {
            const pos = activeNodesPos[nodeName];
            if (!pos) return null;
            const r = getImageRect();
            if (!r) return null;
            return {
                x: r.left + (pos.x / 100) * r.width,
                y: r.top + (pos.y / 100) * r.height,
            };
        }

        function animatePacket(path, status) {
            if (!path || path.length < 2) return;
            const overlay = document.getElementById('topo-overlay');
            if (!overlay) return;

            // Ensure topology view is visible
            const topoPanel = document.getElementById('view-topology');
            if (!topoPanel.classList.contains('active')) return;

            // Create dot
            const dot = document.createElement('div');
            dot.className = 'packet-dot';
            overlay.appendChild(dot);

            const startPos = nodeToPixel(path[0]);
            if (!startPos) { dot.remove(); return; }

            // Position at start
            dot.style.left = startPos.x + 'px';
            dot.style.top = startPos.y + 'px';

            let step = 0;
            const trails = [];

            function moveToNext() {
                step++;
                if (step >= path.length) {
                    // Reached destination
                    dot.classList.add(status === 'DELIVERED' ? 'delivered' : 'dropped');
                    setTimeout(() => {
                        dot.classList.add('fadeout');
                        trails.forEach(t => { t.style.opacity = '0'; });
                        setTimeout(() => {
                            dot.remove();
                            trails.forEach(t => t.remove());
                        }, 500);
                    }, 300);
                    return;
                }

                const nextPos = nodeToPixel(path[step]);
                if (!nextPos) { dot.remove(); trails.forEach(t=>t.remove()); return; }

                // Leave a trail dot at current position
                const trail = document.createElement('div');
                trail.className = 'trail-dot';
                trail.style.left = dot.style.left;
                trail.style.top = dot.style.top;
                overlay.appendChild(trail);
                trails.push(trail);
                setTimeout(() => { trail.style.opacity = '0'; }, 200);

                // Move packet dot
                dot.classList.add('moving');
                dot.style.left = nextPos.x + 'px';
                dot.style.top = nextPos.y + 'px';

                // Sync JS delay with CSS transition (0.35s)
                setTimeout(moveToNext, 400);
            }

            // Start animation after a tiny delay so initial position renders
            setTimeout(moveToNext, 50);
        }

        // ===== CHARTS =====
        const chartOpts={responsive:true,maintainAspectRatio:false,animation:{duration:250},plugins:{legend:{display:false}},elements:{point:{radius:0},line:{borderWidth:2}},scales:{x:{display:false},y:{grid:{color:'rgba(30,34,72,.6)',drawBorder:false},ticks:{color:'#5c5f82',font:{family:"'JetBrains Mono'",size:10},padding:8},border:{display:false}}},interaction:{intersect:false,mode:'index'}};
        const delayChart=new Chart(document.getElementById('delay-chart'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,.08)',fill:true,tension:.4}]},options:chartOpts});
        const tpChart=new Chart(document.getElementById('throughput-chart'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#10b981',backgroundColor:'rgba(16,185,129,.08)',fill:true,tension:.4}]},options:chartOpts});

        function switchView(n,btn){
            document.querySelectorAll('.view-panel').forEach(p=>p.classList.remove('active'));
            document.querySelectorAll('.view-tab').forEach(t=>t.classList.remove('active'));
            document.getElementById('view-'+n).classList.add('active');
            if(btn)btn.classList.add('active');
            if(n==='delay'||n==='throughput')setTimeout(()=>{delayChart.resize();tpChart.resize();},50);
        }

        function utilColor(u){if(u<.3)return'#10b981';if(u<.5)return'#22c55e';if(u<.7)return'#eab308';if(u<.85)return'#f97316';return'#ef4444';}

        // ===== TOAST NOTIFICATIONS =====
        function showToast(msg, sub, cls, duration=4000){
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast ' + cls;
            toast.innerHTML = msg + (sub ? '<div class="toast-sub">' + sub + '</div>' : '');
            container.appendChild(toast);
            requestAnimationFrame(()=>{ requestAnimationFrame(()=>{ toast.classList.add('show'); }); });
            setTimeout(()=>{
                toast.classList.remove('show');
                setTimeout(()=>toast.remove(), 400);
            }, duration);
        }

        function setMode(m, silent){
            routingMode=m;
            document.getElementById('mode-drl').classList.toggle('active',m==='drl');
            document.getElementById('mode-sp').classList.toggle('active',m==='shortest');
            document.getElementById('status-mode').textContent=m==='drl'?'DRL':'SPF';
            document.getElementById('status-mode').className=m==='drl'?'badge badge-warn':'badge badge-info';
            // Show/hide algorithm picker and status row
            const picker = document.getElementById('algo-picker');
            const algoRow = document.getElementById('status-algo-row');
            if(picker) picker.style.display = m==='drl' ? 'block' : 'none';
            if(algoRow) algoRow.style.display = m==='drl' ? 'flex' : 'none';
            if(!silent){
                const s=document.getElementById('src-host').value,d=document.getElementById('dst-host').value;
                if(s&&d&&s!==d)computeRoute();
            }
        }

        function pickAlgo(algoId) {
            // Update button states
            document.querySelectorAll('.algo-b').forEach(b => b.classList.remove('active'));
            document.getElementById('algo-' + algoId).classList.add('active');
            // Update status badge
            const badge = document.getElementById('status-algo');
            if(badge) badge.textContent = algoId.toUpperCase();
            // Call backend
            switchAlgorithm(algoId);
        }

        document.getElementById('src-host').addEventListener('change',chk);
        document.getElementById('dst-host').addEventListener('change',chk);
        function chk(){
            const s=document.getElementById('src-host').value,d=document.getElementById('dst-host').value;
            const ok=s&&d&&s!==d;
            document.getElementById('route-btn').disabled=!ok;
            document.getElementById('send-one').disabled=!ok;
            document.getElementById('send-cont').disabled=!ok;
        }

        function computeRoute(){
            const s=document.getElementById('src-host').value,d=document.getElementById('dst-host').value;
            if(!s||!d||s===d)return;
            const btn=document.getElementById('route-btn');btn.textContent='Computing...';btn.disabled=true;
            fetch('/api/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({src:s,dst:d,mode:routingMode})})
            .then(r=>r.json()).then(data=>{
                btn.textContent='Compute Route';btn.disabled=false;
                if(data.path?.length){
                    currentPath=data.path;
                    showPath(data.path);
                    document.getElementById('status-hops').textContent=data.path.length-1;
                    // Update mode display to match what backend actually used
                    if(data.mode) setMode(data.mode, true);
                } else {
                    currentPath=[];
                    document.getElementById('sidebar-path').innerHTML='<span class="badge badge-err">'+(data.error||'No path')+'</span>';
                    document.getElementById('status-hops').textContent='\u2014';
                }
                refresh();
            }).catch(()=>{btn.textContent='Compute Route';btn.disabled=false;});
        }

        function showPath(p){document.getElementById('sidebar-path').innerHTML=p.map(n=>'<span class="pc">'+n.toUpperCase()+'</span>').join('<span class="ps">→</span>');}

        // ===== PACKET SENDING =====
        function sendPacket(){
            const s=document.getElementById('src-host').value,d=document.getElementById('dst-host').value;
            if(!s||!d||s===d)return;
            fetch('/api/send_packet',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({src:s,dst:d,mode:routingMode})})
            .then(r=>r.json()).then(pkt=>{
                packetData.unshift(pkt);
                if(packetData.length>200)packetData.length=200;
                renderPacketTable();
                updatePktStats();
                if(pkt.path?.length){
                    showPath(pkt.path);
                    // Trigger animation on topology
                    animatePacket(pkt.path, pkt.status);
                }
                refresh();
            });
        }

        function startContinuous(){
            document.getElementById('send-cont').style.display='none';
            document.getElementById('stop-cont').style.display='block';
            // Stay on topology to see animations
            switchView('topology',document.querySelector('.view-tab:nth-child(1)'));
            continuousTimer=setInterval(sendPacket,800);
        }

        function stopContinuous(){
            clearInterval(continuousTimer);continuousTimer=null;
            document.getElementById('send-cont').style.display='block';
            document.getElementById('stop-cont').style.display='none';
        }

        function clearPackets(){
            packetData=[];
            fetch('/api/clear_packets',{method:'POST'});
            renderPacketTable();
            updatePktStats();
        }

        function renderPacketTable(){
            const tbody=document.getElementById('pkt-tbody');
            const empty=document.getElementById('pkt-empty');
            if(!packetData.length){tbody.innerHTML='';empty.style.display='block';return;}
            empty.style.display='none';
            tbody.innerHTML=packetData.map(p=>{
                const statusBadge=p.status==='DELIVERED'?'<span class="badge badge-ok">DELIVERED</span>':'<span class="badge badge-err">DROPPED</span>';
                const rttStr=p.rtt!==null?p.rtt.toFixed(2):'—';
                const rttColor=p.rtt!==null?(p.rtt<10?'var(--emerald)':p.rtt<20?'var(--amber)':'var(--rose)'):'var(--t3)';
                const pathStr=p.path?.length?p.path.map(n=>n.toUpperCase()).join('→'):'—';
                const timeStr=new Date(p.timestamp*1000).toLocaleTimeString();
                return `<tr><td class="seq">${p.seq}</td><td>${p.src.toUpperCase()}</td><td>${p.dst.toUpperCase()}</td><td class="path-cell" title="${pathStr}">${pathStr}</td><td>${p.hops}</td><td class="rtt" style="color:${rttColor}">${rttStr}</td><td>${statusBadge}</td><td style="color:var(--t3);font-size:10px;">${timeStr}</td></tr>`;
            }).join('');
        }

        function updatePktStats(){
            const sent=packetData.length;
            const delivered=packetData.filter(p=>p.status==='DELIVERED').length;
            const dropped=packetData.filter(p=>p.status==='DROPPED').length;
            const rtts=packetData.filter(p=>p.rtt!==null).map(p=>p.rtt);
            const avg=rtts.length?rtts.reduce((a,b)=>a+b,0)/rtts.length:0;
            document.getElementById('pkt-sent').textContent=sent;
            document.getElementById('pkt-delivered').textContent=delivered;
            document.getElementById('pkt-dropped').textContent=dropped;
            document.getElementById('pkt-avg-rtt').textContent=rtts.length?avg.toFixed(1):'—';
        }

        // ===== LINK CONTROLS =====
        function toggleLink(s,d,btn){
            const enable=btn.classList.contains('off');
            fetch('/api/toggle_link',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({src:s,dst:d,enable})})
            .then(r=>r.json()).then(data=>{
                btn.classList.toggle('on',enable);btn.classList.toggle('off',!enable);
                const dot=document.getElementById('dot-'+[s,d].sort().join('-'));
                if(dot){dot.classList.toggle('up',enable);dot.classList.toggle('dn',!enable);}
                document.getElementById('status-links').textContent=data.active_links+' / 6';

                // Handle auto-reroute from backend
                if(data.rerouted && data.new_path?.length){
                    currentPath = data.new_path;
                    showPath(data.new_path);
                    document.getElementById('status-hops').textContent = data.new_path.length - 1;
                    setMode('drl', true);  // Switch mode indicator to DRL
                    showToast(
                        '<span class="toast-icon">🔄</span> DRL Auto-Reroute',
                        'Link ' + s.toUpperCase() + '↔' + d.toUpperCase() + ' broken — rerouted via ' + data.new_path.map(n=>n.toUpperCase()).join('→'),
                        'toast-drl'
                    );
                } else if(data.rerouted && !data.new_path?.length){
                    currentPath = [];
                    document.getElementById('sidebar-path').innerHTML='<span class="badge badge-err">DROPPED</span>';
                    document.getElementById('status-hops').textContent = '\u2014';
                    showToast(
                        '<span class="toast-icon">❌</span> No Alternative Route',
                        'All paths to destination are broken — packet DROPPED.',
                        'toast-noroute'
                    );
                }
                refresh();
            });
        }
        function resetLinks(){
            fetch('/api/reset_links',{method:'POST'}).then(r=>r.json()).then(()=>{
                document.querySelectorAll('.tg').forEach(b=>{b.classList.remove('off');b.classList.add('on');});
                document.querySelectorAll('.ld').forEach(d=>{d.classList.remove('dn');d.classList.add('up');});
                document.getElementById('status-links').textContent='6 / 6';
                // Reset to shortest path mode
                setMode('shortest', true);
                const s=document.getElementById('src-host').value,d=document.getElementById('dst-host').value;
                if(s&&d&&s!==d){
                    computeRoute();
                    showToast('<span class="toast-icon">✅</span> Links Restored', 'Reverted to shortest path routing.', 'toast-reset');
                }
                refresh();
            });
        }

        // ===== REFRESH =====
        function refresh(){
            fetch('/api/stats').then(r=>r.json()).then(st=>{
                // Update time
                document.getElementById('update-time').textContent = new Date(st.timestamp*1000).toLocaleTimeString();
                
                // Update topology metadata if changed (handles first load)
                if (st.topology && st.topology.node_pos) {
                    updateNodePosMap(st.topology.node_pos);
                    // Update host lists if they are empty
                    if (document.getElementById('src-host').options.length <= 1) {
                        updateUIForTopology(st.topology);
                    }
                }

                // Update metrics (defensive check in case they were removed from HTML)
                const p=st.performance||{},dl=p.delay||[],lo=p.packet_loss||[],tp=p.throughput||[];
                const mDelay = document.getElementById('metric-delay');
                const mLoss = document.getElementById('metric-loss');
                const mTput = document.getElementById('metric-throughput');
                
                if(mDelay && dl.length) mDelay.textContent = dl.at(-1).toFixed(1) + 'ms';
                if(mLoss && lo.length) mLoss.textContent = (lo.at(-1)*100).toFixed(2) + '%';
                if(mTput && tp.length) mTput.textContent = tp.at(-1).toFixed(1);
                
                if (delayChart && dl.length) {
                    delayChart.data.labels=dl.map((_,i)=>i);
                    delayChart.data.datasets[0].data=dl;
                    delayChart.update('none');
                }
                if (tpChart && tp.length) {
                    tpChart.data.labels=tp.map((_,i)=>i);
                    tpChart.data.datasets[0].data=tp;
                    tpChart.update('none');
                }
                
                const path=st.selected_path||[];
                if(path.length){showPath(path);document.getElementById('status-hops').textContent=path.length-1;}
                
                // Update Link Controls dynamically
                const linkWrap = document.getElementById('link-controls');
                if (linkWrap) {
                    const ls = st.link_stats || {};
                    let linkHtml = '';
                    const sortedKeys = Object.keys(ls).sort();
                    for (let key of sortedKeys) {
                        const v = ls[key];
                        const [s,d] = key.split('-');
                        const on = !v.disabled;
                        linkHtml += `
                            <div class="lr">
                                <div class="lm">
                                    <div class="ld ${on?'up':'dn'}" id="dot-${s}-${d}"></div>
                                    <span class="ln">${s.toUpperCase()} ↔ ${d.toUpperCase()}</span>
                                    <span class="lb">${v.bandwidth_mbps}M</span>
                                </div>
                                <button class="tg ${on?'on':'off'}" onclick="toggleLink('${s}','${d}',this)"></button>
                            </div>`;
                    }
                    if (linkHtml) linkWrap.innerHTML = linkHtml;
                }

                const ls=st.link_stats||{},tbody=document.querySelector('#link-table tbody');
                tbody.innerHTML='';
                for(const[k,v]of Object.entries(ls)){
                    const u=v.utilization||0,bw=v.bandwidth_mbps||0,dis=v.disabled||false;
                    const c=dis?'#2a2f5c':utilColor(u);
                    const status=dis?'<span class="badge badge-err">DOWN</span>':'<span class="badge badge-ok">ACTIVE</span>';
                    const tr=document.createElement('tr');
                    tr.innerHTML='<td>'+k.toUpperCase()+'</td><td>'+bw+'M</td><td>'+status+'</td><td>'+(dis?'—':(u*100).toFixed(1)+'%')+'</td><td><div class="ut"><div class="uf" style="width:'+(dis?0:u*100)+'%;background:'+c+'"></div></div></td>';
                    tbody.appendChild(tr);
                }
            });
            fetch('/api/graph').then(r=>r.json()).then(d=>{
                if(d.image)document.getElementById('topo-img').src='data:image/png;base64,'+d.image;
            });
        }

        // Call init on load
        window.addEventListener('load', initTopologies);
        setInterval(refresh, 2000);
    </script>
</body>
</html>
"""


# ========================================================
# Flask Routes
# ========================================================

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/topologies')
def api_topologies():
    """
    Get available network topologies.
    ---
    responses:
      200:
        description: List of available topologies and the current active one.
    """
    return jsonify({
        'current': app_state['active_topology'],
        'available': {k: v['name'] for k, v in TOPOLOGIES.items()}
    })

@app.route('/api/set_topology', methods=['POST'])
def api_set_topology():
    topo_key = request.get_json().get('topology')
    if topo_key not in TOPOLOGIES:
        return jsonify({'error': 'Invalid topology'}), 400
    
    app_state['active_topology'] = topo_key
    app_state['disabled_links'].clear()
    app_state['current_path'] = []
    app_state['selected_src'] = None
    app_state['selected_dst'] = None
    
    # Reset stats file for new topology
    update_stats_file([], set())
    
    topo = TOPOLOGIES[topo_key]
    return jsonify({
        'status': 'ok',
        'name': topo['name'],
        'hosts': topo['hosts'],
        'pos': topo['pos']
    })

@app.route('/api/set_algorithm', methods=['POST'])
def api_set_algorithm():
    algo = request.get_json().get('algorithm')
    if algo not in ['sac', 'ddpg', 'td3']:
        return jsonify({'error': 'Invalid algorithm'}), 400
    app_state['drl_algorithm'] = algo
    return jsonify({'status': 'ok', 'algorithm': algo})

@app.route('/api/stats')
def api_stats():
    """
    Get current network statistics.
    ---
    responses:
      200:
        description: Current network statistics including topology and link performance.
    """
    return jsonify(read_stats())

@app.route('/api/graph')
def api_graph():
    """
    Get the network topology graph image (base64).
    ---
    responses:
      200:
        description: Base64 encoded PNG image of the network graph.
    """
    try:
        stats = read_stats()
        for ls in stats.get('link_stats', {}).values():
            if ls.get('disabled', False):
                ls['utilization'] = -1
        return jsonify({'image': base64.b64encode(render_topology(stats)).decode('utf-8')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/route', methods=['POST'])
def api_route():
    data = request.get_json()
    src, dst, mode = data.get('src'), data.get('dst'), data.get('mode', 'shortest')
    if not src or not dst:
        return jsonify({'error': 'Select both hosts'}), 400
    if src == dst:
        return jsonify({'error': 'Same host selected'}), 400
    app_state['selected_src'] = src
    app_state['selected_dst'] = dst
    app_state['routing_mode'] = mode
    disabled = app_state['disabled_links']
    # Use the requested algorithm
    if mode == 'drl':
        path, error = compute_drl_path(src, dst, disabled)
    else:
        path, error = compute_shortest_path(src, dst, disabled)
    if path:
        app_state['current_path'] = path
        update_stats_file(path, disabled)
        return jsonify({'path': path, 'mode': mode, 'hops': len(path) - 1})
    return jsonify({'path': [], 'error': error or 'No path'})

@app.route('/api/send_packet', methods=['POST'])
def api_send_packet():
    data = request.get_json()
    src, dst = data.get('src'), data.get('dst')
    mode = data.get('mode', 'drl')
    if not src or not dst or src == dst:
        return jsonify({'error': 'Invalid hosts'}), 400
    result = simulate_single_packet(src, dst, mode, app_state['disabled_links'])
    return jsonify(result)

@app.route('/api/clear_packets', methods=['POST'])
def api_clear_packets():
    app_state['packet_log'].clear()
    app_state['packet_counter'] = 0
    return jsonify({'status': 'ok'})

@app.route('/api/toggle_link', methods=['POST'])
def api_toggle_link():
    data = request.get_json()
    link_key = tuple(sorted((data['src'], data['dst'])))
    if data.get('enable', True):
        app_state['disabled_links'].discard(link_key)
    else:
        app_state['disabled_links'].add(link_key)

    active_links = 6 - len(app_state['disabled_links'])
    result = {'status': 'ok', 'active_links': active_links, 'rerouted': False}

    # Auto-reroute: check if the toggled link is on the current path
    cur_path = app_state['current_path']
    src = app_state.get('selected_src')
    dst = app_state.get('selected_dst')
    if not data.get('enable', True) and cur_path and len(cur_path) >= 2 and src and dst:
        # Check if this broken link is on the current path
        path_affected = False
        for i in range(len(cur_path) - 1):
            edge = tuple(sorted((cur_path[i], cur_path[i + 1])))
            if edge == link_key:
                path_affected = True
                break
        if path_affected:
            # Auto-reroute using DRL
            new_path, err = compute_drl_path(src, dst, app_state['disabled_links'])
            if new_path:
                app_state['current_path'] = new_path
                app_state['routing_mode'] = 'drl'
                update_stats_file(new_path, app_state['disabled_links'])
                result['rerouted'] = True
                result['new_path'] = new_path
                result['mode'] = 'drl'
            else:
                app_state['current_path'] = []
                result['rerouted'] = True
                result['new_path'] = []
                result['error'] = err or 'No alternative path'
    return jsonify(result)

@app.route('/api/reset_links', methods=['POST'])
def api_reset_links():
    app_state['disabled_links'].clear()
    # Re-route using shortest path if hosts are selected
    src = app_state.get('selected_src')
    dst = app_state.get('selected_dst')
    if src and dst:
        path, _ = compute_shortest_path(src, dst, set())
        if path:
            app_state['current_path'] = path
            app_state['routing_mode'] = 'shortest'
            update_stats_file(path, set())
    return jsonify({'status': 'ok', 'active_links': 6})


def start_dashboard(host='127.0.0.1', port=9000, debug=False):
    global DASHBOARD_PORT
    DASHBOARD_PORT = port
    # Start Swagger UI redirector on port 8000
    def run_swagger_ui():
        print(f"[Dashboard] Swagger UI bridge starting on http://{host}:8000")
        swagger_ui_app.run(host=host, port=8000, debug=False, threaded=True)
    
    swagger_thread = threading.Thread(target=run_swagger_ui, daemon=True)
    swagger_thread.start()
    
    print(f"\n{'='*50}")
    print(f"  SDN DRL Routing Dashboard: http://{host}:{port}")
    print(f"  Swagger API Documentation: http://{host}:8000")
    print(f"{'='*50}\n")
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    start_dashboard(args.host, args.port, args.debug)
