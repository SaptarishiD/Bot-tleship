from functools import partial
import os
import yaml
import json
from datetime import datetime

import numpy as np

from .ppo import PPOBuffer
from .vec_env import DummyVecEnv


def get_save_paths(args):
    """Generate save directory and agent path."""
    dir_name = os.path.join(args["agent"]["save_dir"], args["agent"]["model_name"])
    agent_path = f"{os.path.join(dir_name, args['agent']['algo'])}.pt"
    return dir_name, agent_path


def save_agent(args, agent):
    """Save the agent and configuration."""
    dir_name, agent_path = get_save_paths(args)
    os.makedirs(dir_name, exist_ok=True)  # Ensure directory exists
    
    # Save agent
    agent.save(agent_path)
    print(f"Agent saved to {agent_path}")

    # Save configuration as JSON
    args_path = os.path.join(dir_name, "config.json")
    with open(args_path, "w") as f:
        json.dump(args, f, indent=4)
    print(f"Configuration saved to {args_path}")

def load_agent(args, agent):
    """Load the agent if specified."""
    _, agent_path = get_save_paths(args)

    file_exists = os.path.isfile(agent_path)
    resume = args.get("resume", False)

    if resume and file_exists:
        agent.load(agent_path)
        print(f"Agent loaded from {agent_path}")
    elif file_exists and not resume:
        raise Exception("A model exists at the save path. Use 'resume' flag to continue training.")
    elif resume and not file_exists:
        raise Exception("Resume flag specified, but no model found.")
    else:
        print("No pre-existing model found. Starting fresh training.")

def collect_trajectories_vec_env(vec_env, n_samples, device, policy, value_fn, max_steps=500,
                                 policy_accepts_batch=False):
    """Collect trajectories from a vectorized environment."""
    states = vec_env.reset()
    vec_dim = states.shape[0]

    buf = PPOBuffer()
    traj_ids = [buf.create_traj() for _ in range(vec_dim)]
    steps = np.zeros((vec_dim,))

    # Ensure the policy can accept batches
    if not policy_accepts_batch:
        def batch_policy(p, sts):
            acts, lps = [], []
            for s in sts:
                a, lp = p(s)
                acts.append(a)
                lps.append(lp)
            return np.array(acts), np.array(lps)
        policy = partial(batch_policy, policy)

    # Initialize trackers
    sum_rew_tracker = np.zeros(vec_dim)
    sum_rews = []
    traj_lens = []
    last_rews = []

    while buf.size() < n_samples:
        # Get actions and step in environment
        actions, log_probs = policy(states)
        next_states, rewards, dones, _ = vec_env.step(actions)
        sum_rew_tracker += rewards.flatten()

        # Store data in buffer
        for i, traj_id in enumerate(traj_ids):
            buf.put_single_data(traj_id, states[i], actions[i], log_probs[i], rewards[i])

        states = next_states
        steps += 1

        # Check buffer size and finalize trajectories
        if buf.size() >= n_samples:
            assert buf.size() == n_samples, "Number of samples should already have been checked??"
            for i, traj_id in enumerate(traj_ids):
                buf.finish_traj(traj_id, 0 if dones[i] else value_fn(states[i]))
        else:
            for i, traj_id in enumerate(traj_ids):
                if dones[i] or steps[i] == max_steps:
                    sum_rews.append(sum_rew_tracker[i])
                    sum_rew_tracker[i] = 0
                    traj_lens.append(steps[i])
                    last_rews.append(rewards[i])
                    buf.finish_traj(traj_id, 0 if dones[i] else value_fn(states[i]))
                    steps[i] = 0
                    traj_ids[i] = buf.create_traj()

    # Gather rollout statistics
    rollout_info = {
        "sum_rew_avg": np.mean(sum_rews) if sum_rews else 0,
        "traj_len_avg": np.mean(traj_lens) if traj_lens else 0,
        "last_rew_avg": np.mean(last_rews) if last_rews else 0
    }
    return buf.get(device), rollout_info
