
import gymnasium as gym
from gymnasium.envs.registration import register
from ppo_env.ppo import BattleshipEnv


register(
    id="BattleshipEnvSD-v0",
    entry_point="ppo_env.ppo:BattleshipEnv",  # Replace with the actual module path
    kwargs={"board_size": 10, "ships": [2, 3, 3, 4, 5]},
)


env = gym.make("BattleshipEnvSD-v0")
obs, info = env.reset()
print("Initial Observation:")
print(obs)

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
