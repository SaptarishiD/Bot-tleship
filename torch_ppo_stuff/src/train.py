


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





def read_config(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def parse_args(load_config):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to configuration YAML file")
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    return config


def train(save_agent, load_agent, load_config):
    args = parse_args(load_config)

    np.random.seed(args["training"]["seed"])
    torch.manual_seed(args["training"]["seed"])

    env_fn = create_env_fn(args)


    device = torch.device("cpu")

    env = DummyVecEnv([env_fn])
   

    agent = create_agent_from_args(device, args, env)
    load_agent(args, agent)


    save_interval = args["training"]["save_interval"]

    save_agent(args, agent)
    while agent.total_it < args["training"]["total_steps"]:
        print(f"========\n Training iteration {agent.total_it}... ========\n")
        print(f"Total steps: {args['training']['total_steps']}")
        sample, rollout_info = collect_trajectories_vec_env(env, args["training"]["train_samples"], device,
                                                            agent.select_action, agent.get_value,
                                                            max_steps=args["env"]["max_steps"],
                                                            policy_accepts_batch=False)
        train_info = agent.train(sample, actor_steps=args["training"]["ppo"]["actor_steps"],
                                    critic_steps=args["training"]["ppo"]["critic_steps"])



        if save_interval != -1 and agent.total_it % save_interval == 0:
            save_agent(args, agent)


    save_agent(args, agent)


    env.close()

    print("Training complete!")



if __name__ == "__main__":
    train(partial(save_agent, None), partial(load_agent, None), read_config)


