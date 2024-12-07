from functools import partial
import os
import yaml
import json
from datetime import datetime

import numpy as np

from .ppo import PPOBuffer
from .vec_env import DummyVecEnv


def get_save_paths(args):
    dir_name = os.path.join(args["agent"]["save_dir"], args["agent"]["model_name"])
    return dir_name, f"{os.path.join(dir_name, args['agent']['algo'])}.pt"

def save_agent(dir_name, args, agent):
    if dir_name is None:
        dir_name, agent_path = get_save_paths(args)
    else:
        agent_path = f"{os.path.join(dir_name, args['agent']['algo'])}.pt"
    os.makedirs(dir_name, exist_ok=True)
    args_path = os.path.join(dir_name, "config.yaml")
    agent.save(agent_path)
    with open(args_path, "w") as f:
        f.write(yaml.dump(args, default_flow_style=False))

def load_agent(dir_name, args, agent):
    if dir_name is None:
        _, agent_path = get_save_paths(args)
    else:
        agent_path = os.path.join(dir_name, args["agent"]["algo"] + ".pt")
    file_exists = os.path.isfile(agent_path)
    resume = ("resume" in args) and args["resume"]
    if resume and file_exists:
        agent.load(agent_path)
    elif file_exists and not resume:
        raise Exception("A model exists at the save path. Use -r to resume training.")
    elif resume and not file_exists:
        raise Exception("Resume flag specified, but no model found.")





def collect_trajectories_vec_env(vec_env, n_samples, device, policy, value_fn, max_steps=500,
                                 policy_accepts_batch=False):
    states = vec_env.reset()
    vec_dim = states.shape[0]

    buf = PPOBuffer()
    traj_ids = [buf.create_traj() for _ in range(vec_dim)]
    steps = np.zeros((vec_dim,))
    # ensure that the policy can accept batches
    if not policy_accepts_batch:
        def batch_policy(p, sts):
            acts, lps = [], []
            for s in sts:
                a, lp = p(s)
                acts.append(a)
                lps.append(lp)
            return np.array(acts), np.array(lps)
        policy = partial(batch_policy, policy)
    # do parallel rollout
    sum_rew_tracker = np.zeros(vec_dim)
    sum_rews = []
    traj_lens = []
    last_rews = []
    while buf.size() < n_samples:
        # get actions and step in environment
        actions, log_probs = policy(states)
        next_states, rewards, dones, _ = vec_env.step(actions)
        sum_rew_tracker += rewards.flatten()
        # put data into buffer
        for i, traj_id in enumerate(traj_ids):
            buf.put_single_data(traj_id, states[i], actions[i], log_probs[i], rewards[i])
        states = next_states
        steps += 1
        # if the buffer is full, finish all trajectories
        if buf.size() >= n_samples:
            assert buf.size() == n_samples, "Number of samples should already have been checked??"
            for i, traj_id in enumerate(traj_ids):
                buf.finish_traj(traj_id, 0 if dones[i] else value_fn(states[i]))
        else:
            # otherwise, finish the trajectories that are done or have hit the max number of steps
            for i, traj_id in enumerate(traj_ids):
                if dones[i] or steps[i] == max_steps:
                    sum_rews.append(sum_rew_tracker[i])
                    sum_rew_tracker[i] = 0
                    traj_lens.append(steps[i])
                    last_rews.append(rewards[i])
                    buf.finish_traj(traj_id, 0 if dones[i] else value_fn(states[i]))
                    steps[i] = 0
                    traj_ids[i] = buf.create_traj()
    rollout_info = {
        "sum_rew_avg": np.mean(sum_rews),
        "traj_len_avg": np.mean(traj_lens),
        "last_rew_avg": np.mean(last_rews)
    }
    return buf.get(device), rollout_info

