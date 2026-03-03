"""
Network Graph Renderer — Premium dark theme with glow effects.

Renders SDN topology as a high-quality PNG:
    - Dark gradient background
    - Switches: rounded squares with gradient fills and glow
    - Hosts: soft circles with subtle shadows
    - Links: congestion gradient (emerald → amber → rose)
    - DRL path: cyan glow with animated-style thickness
    - Disabled links: dashed red with X markers
    - Clean typography with bandwidth + utilization labels
"""

import os
import io
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
STATS_FILE = os.path.join(DATA_DIR, 'net_stats.json')

# Premium color palette
COLORS = {
    'bg_dark':      '#0a0e1a',
    'bg_card':      '#131829',
    'grid':         '#1a2035',
    'switch':       '#6366f1',   # Indigo
    'switch_path':  '#06b6d4',   # Cyan
    'switch_edge':  '#818cf8',   # Light indigo
    'host':         '#10b981',   # Emerald
    'host_edge':    '#34d399',   # Light emerald
    'host_link':    '#334155',   # Slate
    'path_glow':    '#06b6d4',   # Cyan
    'path_color':   '#22d3ee',   # Light cyan
    'text':         '#f1f5f9',   # Slate 100
    'text_muted':   '#94a3b8',   # Slate 400
    'label_bg':     '#1e293b',   # Slate 800
    'disabled':     '#ef4444',   # Red
    'border':       '#1e2248',   # Border color
}

# Congestion colors (5-stop gradient)
CONGESTION = [
    (0.00, '#10b981'),  # Emerald
    (0.30, '#22c55e'),  # Green
    (0.50, '#eab308'),  # Yellow
    (0.70, '#f97316'),  # Orange
    (0.85, '#ef4444'),  # Red
]

# Default node positions (will be overridden by dynamic stats if available)
DEFAULT_NODE_POS = {
    's1': (-1.2,  1.0),  's2': ( 1.2,  1.0),
    's3': (-1.2, -1.0),  's4': ( 1.2, -1.0),
    'h1': (-2.3,  1.8),  'h2': (-0.1,  1.8),
    'h3': ( 2.3,  1.8),
    'h4': (-2.3, -1.8),  'h5': ( 0.1, -1.8),
    'h6': ( 2.3, -1.8),
}


def congestion_color(util: float) -> str:
    """Map utilization 0-1 to congestion color."""
    if util < 0:
        return COLORS['disabled']
    for threshold, color in reversed(CONGESTION):
        if util >= threshold:
            return color
    return CONGESTION[0][1]


def read_stats() -> Dict:
    try:
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_graph(stats: Dict) -> nx.Graph:
    G = nx.Graph()
    topo = stats.get('topology', {})
    
    switches = topo.get('switches', ['s1', 's2', 's3', 's4'])
    hosts = topo.get('hosts', ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    for sw in switches:
        G.add_node(sw, type='switch')
    for h in hosts:
        G.add_node(h, type='host')

    links = topo.get('links', [
        {"src": "s1", "dst": "s2", "bw": 100},
        {"src": "s3", "dst": "s4", "bw": 100},
        {"src": "s1", "dst": "s4", "bw": 50},
        {"src": "s2", "dst": "s3", "bw": 50},
        {"src": "s1", "dst": "s3", "bw": 10},
        {"src": "s2", "dst": "s4", "bw": 10},
    ])
    for link in links:
        G.add_edge(link['src'], link['dst'], bw=link.get('bw', 100), type='switch_link')

    host_map = topo.get('host_switch_map', {
        "h1": "s1", "h2": "s1", "h3": "s2",
        "h4": "s3", "h5": "s4", "h6": "s4"
    })
    for h, sw in host_map.items():
        if h in G and sw in G:
            G.add_edge(h, sw, bw=1000, type='host_link')

    return G


def render_topology(stats: Dict = None, figsize: Tuple = (14, 9), dpi: int = 110) -> bytes:
    """Render premium topology graph. Returns PNG bytes."""
    if stats is None:
        stats = read_stats()

    G = build_graph(stats)
    link_stats = stats.get('link_stats', {})
    selected_path = stats.get('selected_path', [])

    # ── Figure Setup ──
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(COLORS['bg_dark'])
    ax.set_facecolor(COLORS['bg_dark'])

    # Get positions from stats or defaults
    node_pos_data = stats.get('topology', {}).get('node_pos', DEFAULT_NODE_POS)
    # Ensure keys are correct and convert to tuple if needed
    pos = {}
    for n in G.nodes():
        if n in node_pos_data:
            p = node_pos_data[n]
            # Ensure it's a tuple of floats
            pos[n] = (float(p[0]), float(p[1]))
        else:
            pos[n] = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))

    # Subtle grid dots
    for x in np.arange(-3, 3.5, 0.5):
        for y in np.arange(-2.5, 2.5, 0.5):
            ax.plot(x, y, '.', color=COLORS['grid'], markersize=1, alpha=0.3)

    # Build path edges set
    path_edges = set()
    if selected_path and len(selected_path) >= 2:
        for i in range(len(selected_path) - 1):
            path_edges.add(tuple(sorted((selected_path[i], selected_path[i + 1]))))

    # ── Draw Host-Switch Links ──
    host_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'host_link']
    for u, v in host_edges:
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=COLORS['host_link'], linewidth=1, alpha=0.4,
                linestyle=':', zorder=1)

    # ── Draw Switch-Switch Links ──
    switch_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'switch_link']

    for u, v in switch_edges:
        link_key = f"{u}-{v}"
        alt_key = f"{v}-{u}"
        ls = link_stats.get(link_key, link_stats.get(alt_key, {}))
        util = ls.get('utilization', 0.0)
        is_disabled = ls.get('disabled', False)
        bw = G.edges[u, v].get('bw', 100)
        edge_key = tuple(sorted((u, v)))
        is_on_path = edge_key in path_edges

        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]

        if is_disabled:
            # Dashed red line with X marker at midpoint
            ax.plot(x, y, color=COLORS['disabled'], linewidth=2, alpha=0.5,
                    linestyle='--', dash_capstyle='round', zorder=2)
            mid_x, mid_y = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            ax.text(mid_x, mid_y, '✕', fontsize=14, color=COLORS['disabled'],
                    ha='center', va='center', fontweight='bold', zorder=5,
                    path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['bg_dark'])])
        elif is_on_path:
            # Glowing cyan path
            ax.plot(x, y, color=COLORS['path_glow'], linewidth=12, alpha=0.15, zorder=2,
                    solid_capstyle='round')
            ax.plot(x, y, color=COLORS['path_glow'], linewidth=6, alpha=0.3, zorder=3,
                    solid_capstyle='round')
            ax.plot(x, y, color=COLORS['path_color'], linewidth=3, alpha=1.0, zorder=4,
                    solid_capstyle='round')
        else:
            # Normal link with congestion color
            color = congestion_color(util)
            ax.plot(x, y, color=color, linewidth=2.5, alpha=0.7, zorder=2,
                    solid_capstyle='round')

        if not is_disabled:
            # Draw bandwidth/utilization labels
            mid_x, mid_y = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            dx, dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
            length = max(0.01, np.sqrt(dx**2 + dy**2))
            nx, ny = -dy/length, dx/length
            
            label = f"{int(bw)} Mbps\n{int(util*100)}% Util"
            ax.text(mid_x + nx*0.16, mid_y + ny*0.16, label, 
                    fontsize=9, color='white', ha='center', va='center',
                    fontfamily='monospace', fontweight='bold', zorder=12,
                    bbox=dict(facecolor=COLORS['bg_card'], alpha=0.8, edgecolor=COLORS['border'], 
                              boxstyle='round,pad=0.3', lw=1),
                    path_effects=[pe.withStroke(linewidth=2, foreground=COLORS['bg_dark'])])

    # ── Draw Switch Nodes ──
    switches = [n for n, d in G.nodes(data=True) if d.get('type') == 'switch']
    for sw in switches:
        x, y = pos[sw]
        on_path = sw in selected_path

        # Glow effect for path nodes
        if on_path:
            glow = plt.Circle((x, y), 0.22, color=COLORS['path_glow'],
                              alpha=0.12, zorder=5)
            ax.add_patch(glow)
            glow2 = plt.Circle((x, y), 0.17, color=COLORS['path_glow'],
                               alpha=0.2, zorder=5)
            ax.add_patch(glow2)

        # Switch body (rounded rectangle effect via circle)
        face_color = COLORS['switch_path'] if on_path else COLORS['switch']
        edge_color = COLORS['path_color'] if on_path else COLORS['switch_edge']

        node_circle = plt.Circle((x, y), 0.13, facecolor=face_color,
                                 edgecolor=edge_color, linewidth=2.5,
                                 zorder=8)
        ax.add_patch(node_circle)

        # Inner highlight dot
        inner = plt.Circle((x, y - 0.02), 0.04, facecolor='white', alpha=0.25, zorder=9)
        ax.add_patch(inner)

        # Label
        ax.text(x, y, sw.upper(), fontsize=10, color='white',
                ha='center', va='center', fontweight='bold', zorder=10,
                path_effects=[pe.withStroke(linewidth=2, foreground=face_color)])

    # ── Draw Host Nodes ──
    hosts = [n for n, d in G.nodes(data=True) if d.get('type') == 'host']
    for h in hosts:
        x, y = pos[h]

        # Soft shadow
        shadow = plt.Circle((x + 0.02, y - 0.02), 0.09, facecolor='black',
                            alpha=0.15, zorder=6)
        ax.add_patch(shadow)

        # Host circle
        host_circle = plt.Circle((x, y), 0.09, facecolor=COLORS['host'],
                                 edgecolor=COLORS['host_edge'], linewidth=1.5,
                                 zorder=8)
        ax.add_patch(host_circle)

        # Label below
        ax.text(x, y - 0.18, h.upper(), fontsize=8, color=COLORS['text_muted'],
                ha='center', va='top', fontweight='bold', zorder=10)

    # Legend removed as requested

    # ── Title ──
    path_str = ' → '.join(s.upper() for s in selected_path) if selected_path else 'No Active Route'
    ax.set_title(f'Active Path:  {path_str}',
                 fontsize=13, color=COLORS['text'], fontweight='bold',
                 pad=18, fontfamily='monospace',
                 path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['bg_dark'])])

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


if __name__ == '__main__':
    print("Rendering test topology...")
    png = render_topology()
    out = os.path.join(DATA_DIR, 'topology_test.png')
    with open(out, 'wb') as f:
        f.write(png)
    print(f"  Saved {len(png)} bytes → {out}")
