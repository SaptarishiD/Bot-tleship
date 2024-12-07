

import sys
import argparse
from itertools import chain

import numpy as np

from .vec_env import DummyVecEnv


from .util import collect_trajectories_vec_env, save_agent, load_agent
from .battleship_util import create_agent_from_args, create_env_fn


import torch


from functools import partial
import yaml


CONFIG = {
    "agent": {
        "algo": "ppo",
        "model_name": "PPO_flat-ships-latent_flat_discount-0.7-latest",
        "actor_type": "disc",
        "save_dir": "models",
        "layers": [256, 256],
    },
    "env": {
        "max_steps": 200,
        "state_space": "flat-ships-latent",
        "action_space": "flat",
        "board_width": 10,
        "board_height": 10,
        "latent_var_precision": 16,
        "ship_sizes": [2, 3, 3, 4, 5],
    },
    "training": {
        "discount": 0.7,
        "gpu_idx": 0,
        "num_procs": 1,
        "ppo": {
            "actor_learning_rate": 0.0005,
            "actor_steps": 70,
            "clip_ratio": 0.3,
            "critic_learning_rate": 0.002,
            "critic_steps": 70,
            "entropy_coeff": 0.002,
            "gae_lam": 0.95,
            "target_kl": 0.012,
        },
        "save_interval": 5,
        "seed": 42,
        "total_steps": 4000,
        "train_samples": 3900,
        "use_gpu": False,
    },
}

def train(config):
    # Set seeds
    np.random.seed(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])

    env_fn = create_env_fn(config)
    env = DummyVecEnv([env_fn])
    device = torch.device("cpu")

    agent = create_agent_from_args(device, config, env)
    load_agent(config, agent)

    save_interval = config["training"]["save_interval"]
    save_agent(config, agent)

    while agent.total_it < config["training"]["total_steps"]:
        print(f"Training iteration {agent.total_it}...")
        print(f"Total steps: {config['training']['total_steps']}")

        sample, _ = collect_trajectories_vec_env(
            env, config["training"]["train_samples"], device,
            agent.select_action, agent.get_value, 
            max_steps=config["env"]["max_steps"]
        )

        agent.train(sample, 
                    actor_steps=config["training"]["ppo"]["actor_steps"],
                    critic_steps=config["training"]["ppo"]["critic_steps"])

        if save_interval != -1 and agent.total_it % save_interval == 0:
            save_agent(config, agent)

    save_agent(config, agent)
    print("Training Done!!!")

if __name__ == "__main__":
    train(CONFIG)
