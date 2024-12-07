from functools import partial
import os
import yaml
import json
from datetime import datetime

import numpy as np

from .ppo import PPOBuffer
from .vec_env import DummyVecEnv


"""

# Citation: we also referred to: https://github.com/abhaybd/Fleet-AI but made various modifications to fit our cause and to make it easier for us to understand. We had to understand the code and delete quite a bit but still make it work to play the battleship game. Initially we had looked at just the openai baselines library implementation but due to some architecture and hyperparameter issues it wasn't learning as well, so we found this other implementation and modified it

"""



def get_save_paths(args):
    dir_name = os.path.join(args["agent"]["save_dir"], args["agent"]["model_name"])
    agent_path = f"{os.path.join(dir_name, args['agent']['algo'])}.pt"
    return dir_name, agent_path


def save_agent(args, agent):
    dir_name, agent_path = get_save_paths(args)
    os.makedirs(dir_name, exist_ok=True)  
    
    
    agent.save(agent_path)
    print(f"Agent saved to {agent_path}")

    
    args_path = os.path.join(dir_name, "config.yaml")
    with open(args_path, "w") as f:
        yaml.dump(args, f, default_flow_style=False)
    print(f"Configuration saved to {args_path}")

def load_agent(args, agent):
    _, agent_path = get_save_paths(args)

    file_exists = os.path.isfile(agent_path)
    resume = args.get("resume", False)

    if resume and file_exists:
        agent.load(agent_path)
        print(f"Agent loaded from {agent_path}")
    elif file_exists and not resume:
        raise Exception("A model exists at the save path. Use 'resume' flag to continue training.")

def collect_trajectories_vec_env(vec_env, n_samples, device, policy, value_fn, max_steps=500,
                                 policy_accepts_batch=False):
    states = vec_env.reset()
    vec_dim = states.shape[0]

    buf = PPOBuffer()
    traj_ids = [buf.create_traj() for _ in range(vec_dim)]
    steps = np.zeros((vec_dim,))


    if not policy_accepts_batch:
        def batch_policy(p, sts):
            acts, lps = [], []
            for s in sts:
                a, lp = p(s)
                acts.append(a)
                lps.append(lp)
            return np.array(acts), np.array(lps)
        policy = partial(batch_policy, policy)

    
    sum_rew_tracker = np.zeros(vec_dim)
    sum_rews = []
    traj_lens = []
    last_rews = []

    while buf.size() < n_samples:
        
        actions, log_probs = policy(states)
        next_states, rewards, dones, _ = vec_env.step(actions)
        sum_rew_tracker += rewards.flatten()

        
        for i, traj_id in enumerate(traj_ids):
            buf.put_single_data(traj_id, states[i], actions[i], log_probs[i], rewards[i])

        states = next_states
        steps += 1

        
        if buf.size() >= n_samples:
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

    rollout_info = {
        "sum_rew_avg": np.mean(sum_rews) if sum_rews else 0,
        "traj_len_avg": np.mean(traj_lens) if traj_lens else 0,
        "last_rew_avg": np.mean(last_rews) if last_rews else 0
    }
    return buf.get(device), rollout_info
