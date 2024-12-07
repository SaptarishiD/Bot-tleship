from .ppo import PPO

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from actor_critic import MultiDiscActor, Critic

from .BattleshipActor import BattleshipActor
from .env import BattleshipEnv


"""

# Citation: we also referred to: https://github.com/abhaybd/Fleet-AI but made various modifications to fit our cause and to make it easier for us to understand. We had to understand the code and delete quite a bit but still make it work to play the battleship game. Initially we had looked at just the openai baselines library implementation but due to some architecture and hyperparameter issues it wasn't learning as well, so we found this other implementation and modified it

"""



def create_env_fn(args):
    sizes = args["env"]["ship_sizes"] if "ship_sizes" in args["env"] else (1,2,3,4,5)
    board_width = 10
    board_height = 10

    latent_var_precision = 16
    return lambda: BattleshipEnv(observation_space=args["env"]["state_space"],
                                 action_space=args["env"]["action_space"],
                                 latent_var_precision=latent_var_precision,
                                 ships=sizes, board_height=board_height, board_width=board_width)

def create_agent_from_args(device, args, env):
    args_training = args["training"]
    args_ppo = args_training["ppo"]
    act_space = args["env"]["action_space"]
    actor_type = args["agent"]["actor_type"]
    layers = (256, 256)
   
    actor_fn = lambda dev: BattleshipActor(dev, env.observation_space.shape[0], env.action_space.n)

    critic_fn = lambda dev: Critic(dev, env.observation_space.shape[0], layers=layers)
    ppo = PPO(device=device, actor_fn=actor_fn, critic_fn=critic_fn,
              discount=args_training["discount"],
              gae_lam=args_ppo["gae_lam"],
              clip_ratio=args_ppo["clip_ratio"],
              actor_learning_rate=args_ppo["actor_learning_rate"],
              critic_learning_rate=args_ppo["critic_learning_rate"],
              entropy_coeff=args_ppo["entropy_coeff"],
              target_kl=args_ppo["target_kl"])
    return ppo

def load_agent_from_args(device, model_dir, args):
    env_fn = create_env_fn(args)
    env = env_fn()
    agent = create_agent_from_args(device, args, env)

    model_path = os.path.join(model_dir, f"{args['agent']['algo']}.pt")
    if os.path.isfile(model_path):
        agent.load(model_path)
   
    return agent
