import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('/Users/user/Documents_stuff/Ashoka/monsoon24-courses/AI/project/final-project-bot-tleship/'))
from board import Board
from stable_baselines3 import PPO
from helper_functions import test_bot

class PPOBot:
    def __init__(self, board: Board, model_path: str):
        self.board = board
        print("Model path: ", model_path)
        # check if the file exists

        
        self.model = PPO.load(model_path)  # Load the trained PPO model
        self.board_size = board.get_size()

    def attack(self):
        obs = self.board.get_hidden_board()  # Get the current board state
        # print("Observation shape before flattening: ", obs.shape)
        obs = obs.flatten()  # Flatten the observation to match the model input shape
        # print(f"Observation: {obs}")
        obs = obs.reshape(10,10)
        action, _ = self.model.predict(obs)  # Get the action from the PPO model
        x, y = divmod(action, self.board_size)
        return self.board.attack(x, y)
    

if __name__ == "__main__":

    # Evaluate PPOBot
    ppo_model_path = "/Users/user/Documents_stuff/Ashoka/monsoon24-courses/AI/project/final-project-bot-tleship/ppo_stuff/models/20241202-014536-ppo-battleship"  # Replace with the actual model path
    moves_ppo, mean_ppo, median_ppo, max_ppo, min_ppo, std_ppo, avg_moves_ppo = test_bot(
        100, 10, Bot=lambda board: PPOBot(board, ppo_model_path)
    )
    print("PPOBot Performance:")
    print(f"Mean: {mean_ppo}, Median: {median_ppo}, Max: {max_ppo}, Min: {min_ppo}, Std: {std_ppo}")
    print(f"Average moves per game: {avg_moves_ppo}")
