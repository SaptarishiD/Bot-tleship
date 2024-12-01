import gymnasium as gym
from gymnasium.envs.registration import register
from ppo_env import BattleshipEnv

register(
    id="BattleshipEnvSD-v0",
    entry_point="BattleshipEnv",  # Replace with the actual module path
    kwargs={"board_height": 10, "board_width" : 10, "ships": [2, 3, 3, 4, 5]},
)
