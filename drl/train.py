"""
DRL Training Script for SDN Routing.

Trains a DQN agent to learn optimal routing decisions in the SDN
environment. Supports two modes:

    - simulation: Synthetic training without real network (default)
    - live: Reads live stats from Ryu controller via data/net_stats.json

Usage:
    python drl/train.py                    # Simulation mode
    python drl/train.py --mode live        # Live network mode
    python drl/train.py --episodes 2000    # Custom episode count
    python drl/train.py --load checkpoint  # Resume from checkpoint
"""

import argparse
import os
import sys
import time
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.environment import SDNRoutingEnv
from drl.dqn_agent import DQNAgent


# Directories
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'data', 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'data', 'results')


def train(args):
    """
    Main training loop.

    1. Creates SDN environment and DQN agent
    2. Runs episodes with epsilon-greedy exploration
    3. Trains agent with experience replay after each step
    4. Logs progress and saves checkpoints periodically
    5. Saves final training results
    """
    print("=" * 60)
    print("  DRL-SDN Routing Training")
    print("=" * 60)
    print(f"  Mode:          {args.mode}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Max Steps:     {args.max_steps}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Gamma:         {args.gamma}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Buffer Size:   {args.buffer_size}")
    print("=" * 60)

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize environment
    env = SDNRoutingEnv(mode=args.mode, max_steps=args.max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"\n[Environment] State dim: {state_dim}, Action dim: {action_dim}")

    # Initialize DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        min_replay_size=args.min_replay,
    )

    # Load checkpoint if specified
    if args.load:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, args.load)
        if os.path.exists(checkpoint_path):
            agent.load(checkpoint_path)
        else:
            print(f"[Warning] Checkpoint not found: {checkpoint_path}")

    # Training history
    episode_rewards = []
    episode_lengths = []
    episode_losses = []

    best_avg_reward = float('-inf')
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"{'Episode':>8} | {'Reward':>10} | {'Avg(100)':>10} | "
          f"{'Epsilon':>8} | {'Loss':>10} | {'Steps':>6}")
    print(f"{'='*60}")

    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_loss_sum = 0.0
        loss_count = 0

        for step in range(args.max_steps):
            # Select action
            action = agent.select_action(state, training=True)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store experience
            done = terminated or truncated
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss_sum += loss
                loss_count += 1

            episode_reward += reward
            state = next_state

            if done:
                break

        # Decay exploration
        agent.decay_epsilon()
        agent.rewards_history.append(episode_reward)

        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        avg_loss = episode_loss_sum / max(loss_count, 1)
        episode_losses.append(avg_loss)

        # Calculate running averages
        avg_reward_100 = np.mean(episode_rewards[-100:])

        # Log progress
        if episode % args.log_interval == 0 or episode == 1:
            print(f"{episode:>8} | {episode_reward:>10.3f} | "
                  f"{avg_reward_100:>10.3f} | {agent.epsilon:>8.4f} | "
                  f"{avg_loss:>10.6f} | {step + 1:>6}")

        # Save checkpoint
        if episode % args.save_interval == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f'dqn_ep{episode}.pt'
            )
            agent.save(checkpoint_path)

            # Save best model
            if avg_reward_100 > best_avg_reward:
                best_avg_reward = avg_reward_100
                best_path = os.path.join(CHECKPOINT_DIR, 'dqn_best.pt')
                agent.save(best_path)
                print(f"  ★ New best avg reward: {best_avg_reward:.3f}")

    # Training complete
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"  Total time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Final epsilon:  {agent.epsilon:.4f}")
    print(f"  Best avg(100):  {best_avg_reward:.3f}")
    print(f"  Train steps:    {agent.train_steps}")
    print(f"  Buffer size:    {len(agent.replay_buffer)}")

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, 'dqn_final.pt')
    agent.save(final_path)

    # Save training results
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'best_avg_reward': best_avg_reward,
        'total_time': elapsed,
        'agent_stats': agent.get_statistics(),
        'config': vars(args),
    }

    results_path = os.path.join(RESULTS_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    env.close()
    return results


def evaluate(args):
    """
    Evaluate a trained DQN agent.

    Loads a checkpoint and runs evaluation episodes (no exploration).
    """
    print("=" * 60)
    print("  DRL-SDN Routing Evaluation")
    print("=" * 60)

    env = SDNRoutingEnv(mode=args.mode, max_steps=args.max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, args.load or 'dqn_best.pt')
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    else:
        print(f"[Error] No checkpoint found at {checkpoint_path}")
        return

    eval_rewards = []
    for ep in range(args.eval_episodes):
        state, info = env.reset()
        episode_reward = 0.0

        for step in range(args.max_steps):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        eval_rewards.append(episode_reward)
        print(f"  Eval episode {ep + 1}: reward = {episode_reward:.3f}")

    avg = np.mean(eval_rewards)
    std = np.std(eval_rewards)
    print(f"\n  Average reward: {avg:.3f} ± {std:.3f}")
    print(f"  Min: {min(eval_rewards):.3f}, Max: {max(eval_rewards):.3f}")

    env.close()


def main():
    """Parse arguments and run training or evaluation."""
    parser = argparse.ArgumentParser(
        description='DRL-SDN Routing Training'
    )

    # Mode
    parser.add_argument('--mode', type=str, default='simulation',
                        choices=['simulation', 'live'],
                        help='Training mode')

    # Training params
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--buffer_size', type=int, default=50000,
                        help='Replay buffer size')
    parser.add_argument('--min_replay', type=int, default=1000,
                        help='Min buffer size before training')
    parser.add_argument('--target_update', type=int, default=100,
                        help='Target network update frequency')

    # Exploration
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Epsilon decay rate')

    # Logging & saving
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Episodes between log prints')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Episodes between checkpoints')

    # Load/eval
    parser.add_argument('--load', type=str, default=None,
                        help='Checkpoint filename to load')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation instead of training')
    parser.add_argument('--eval_episodes', type=int, default=20,
                        help='Number of evaluation episodes')

    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
