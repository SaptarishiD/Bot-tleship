
import gymnasium as gym
from gymnasium.envs.registration import register
from ppo_env.ppo import BattleshipEnv


register(
    id="BattleshipEnvSD-v0",
    entry_point="ppo_env.ppo:BattleshipEnv",  # Replace with the actual module path
    kwargs={"board_size": 10, "ships": [2, 3, 3, 4, 5]},
)


env = gym.make("BattleshipEnvSD-v0")