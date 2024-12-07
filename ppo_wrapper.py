
import torch
import numpy as np
from board import Board
from torch_ppo_stuff.src.battleship_util import load_agent_from_args
import yaml
import os

import matplotlib.pyplot as plt





SHIPS = [2, 3, 3, 4, 5]

class PPOBot:
    def __init__(self, board: Board):
        self.board = board
        self.board_size = board.get_size()
        
        
        config_path = "torch_ppo_stuff/models/PPO_flat-ships-latent_flat_discount-0.7-latest/config.yaml"
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.device = torch.device("cpu")
        self.agent = load_agent_from_args(self.device, os.path.dirname(config_path), self.config)
        
    def _board_to_state(self):
        """Convert board state to PPO agent's expected observation format"""
        board = self.board.get_hidden_board()
        hits = np.zeros_like(board, dtype=np.int8)
        misses = np.zeros_like(board, dtype=np.int8)
        
        hits[board == 2] = 1  
        misses[board == -1] = 1  
        
        
        state = np.hstack((misses.flatten(), hits.flatten()))
        
        
        if "ships" in self.config["env"]["state_space"]:
            ships_sunk = np.zeros(len(SHIPS), dtype=np.int8)
            state = np.hstack((state, ships_sunk))
            
        
        if "latent" in self.config["env"]["state_space"]:
            latent_dim = self.config["env"]["latent_var_precision"]
            latent = np.random.binomial(1, 0.5, latent_dim)
            state = np.hstack((state, latent))
            
        return state

    def attack(self):
        """Get action from PPO agent and execute attack"""
        
        state = self._board_to_state()
        
        
        action = self.agent.select_action_greedy(state)
        
        
        if self.config["env"]["action_space"] == "flat":
            x = action // self.board_size
            y = action % self.board_size
        else:
            x, y = action
            
        
        while self.board.get_hidden_board()[x, y] in [-1, 2]: 
            
            x = np.random.randint(0, self.board_size)
            y = np.random.randint(0, self.board_size)
            
        return self.board.attack(x, y)
    

def plot_board(board, iteration):
    plt.imshow(board, cmap='viridis', interpolation='none')
    plt.title(f'Iteration {iteration}')
    plt.colorbar()
    plt.savefig(f'plots/bots/latest_ppo_board_iteration_{iteration}.png')
    plt.close()

    
if __name__ == "__main__":

    print(1)
    board = Board(10)
    bot = PPOBot(board)
    print(board.get_board())
    
    moves = 0
    while board.check_if_game_over() == False:
        bot.attack()
        plot_board(board.get_board(), moves)
        moves += 1
    print(moves)