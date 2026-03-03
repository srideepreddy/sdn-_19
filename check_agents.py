import os
import sys
import json
import torch
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.curdir))

from dashboard import get_drl_state, get_candidate_paths, init_agents, app_state

def diagnostic_check():
    print("="*60)
    print(" SDN DRL ALGORITHM DIAGNOSTIC CHECK")
    print("="*60)
    
    # 1. Initialize Agents
    print("\n[1/4] Loading Agents (SAC, DDPG, TD3)...")
    init_agents()
    if not app_state['agents']:
        print("Error: No agents loaded. Please ensure checkpoints exist.")
        return

    # 2. Capture Current Input (State Vector)
    print("\n[2/4] Capturing Current Network State (Input)...")
    state = get_drl_state()
    print(f"Full 48-dim State Vector captured.")
    
    # Structure it for readability
    # 12 edges, 4 features each: Utilization, TX Rate, Packet Placeholder, BW
    edges = [
        "h1-s1", "h2-s1", "h3-s2", "h4-s3", "h5-s4", "h6-s4", # Hosts
        "s1-s2", "s3-s4", "s1-s4", "s2-s3", "s1-s3", "s2-s4"  # Switches
    ]
    
    print("\nInput Breakdown (per link):")
    print(f"{'Link':<10} | {'Util':<6} | {'TX Norm':<8} | {'Pkt':<4} | {'BW Norm':<8}")
    print("-" * 50)
    for i in range(12):
        feat = state[i*4 : (i+1)*4]
        print(f"{edges[i]:<10} | {feat[0]:.3f}  | {feat[1]:.3f}    | {feat[2]:.1f} | {feat[3]:.3f}")

    # 3. Candidate Paths (Potential Outputs)
    # Testing for h1-h6 as a benchmark
    src, dst = 'h1', 'h6'
    print(f"\n[3/4] Candidate Paths for {src} -> {dst} (Output Space):")
    candidates = get_candidate_paths(src, dst)
    for i, p in enumerate(candidates):
        print(f"  Action {i}: {p}")

    # 4. Agent Outputs (Inference)
    print("\n[4/4] Running Inference for all Algorithms...")
    print(f"{'Algorithm':<10} | {'Action':<6} | {'Selected Path'}")
    print("-" * 50)
    
    for algo in ['sac', 'ddpg', 'td3']:
        agent = app_state['agents'].get(algo)
        if not agent:
            print(f"{algo.upper():<10} | {'N/A':<6} | Not Loaded")
            continue
            
        # Handle inconsistent select_action signatures across agents
        if algo == 'td3':
            action = agent.select_action(state)
        else:
            action = agent.select_action(state, training=False)
        
        path = candidates[action]
        print(f"{algo.upper():<10} | {action:<6} | {path}")

    print("\n" + "="*60)
    print(" DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == "__main__":
    diagnostic_check()
