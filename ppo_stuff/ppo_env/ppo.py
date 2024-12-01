import sys
import os
sys.path.insert(0, os.path.abspath('/Users/user/Documents_stuff/Ashoka/monsoon24-courses/AI/project/final-project-bot-tleship/'))
import board
import gymnasium as gym
import numpy as np

from typing import Optional

from gymnasium import spaces

BOARD_DIMS = 10
SHIPS = [2, 3, 3, 4, 5]



class BattleshipEnv(gym.Env):


    def __init__(self, board_size: int = BOARD_DIMS, ships: list[int] = SHIPS):
        self.board_size = board_size
        self.ships = ships
        self.board = board.Board(BOARD_DIMS, ships)
        self.action_space = spaces.Discrete(BOARD_DIMS * BOARD_DIMS)
        # this is what we can observe of the enemy's board. 0 is unknown, -1 is miss, 1 means a ship exists there and 2 means it has been hit
        self.observation_space = spaces.Box(low=-1, high=1, shape=(BOARD_DIMS, BOARD_DIMS), dtype=np.int32)


    def step(self, action):
        x, y = divmod(action, self.board_size)
        game_over = self.board.attack(x, y)
        hidden_board = self.board.get_hidden_board()

        # Calculate reward
        if self.board.board[x, y] == 2:  # Hit
            reward = 5.0
        elif self.board.board[x, y] == -1:  # Miss
            reward = -1
        else:
            reward = -0.5

        return hidden_board, reward, game_over, {} 
    
    def _get_info(self):
        return {"Remaining ship cells": np.sum(self.board == 1)}

    # mostly a copy of __init__ from board.py
    def _init_board(self):
        self.board = np.zeros((self.board_size, self.board_size))
        for size in self.ships:
            placed = False
            while not placed:
                x = np.random.randint(0, self.board_size)
                y = np.random.randint(0, self.board_size)
                orientation = np.random.randint(0, 2)  # 0: horizontal, 1: vertical
                if orientation == 0:
                    if x + size <= self.board_size and all(self.board[x + i, y] == 0 for i in range(size)):
                        for i in range(size):
                            self.board[x + i, y] = 1
                        placed = True
                else:
                    if y + size <= self.board_size and all(self.board[x, y + i] == 0 for i in range(size)):
                        for i in range(size):
                            self.board[x, y + i] = 1
                        placed = True

    def _get_obs(self):
        return np.where(self.board==1, 0, self.board)
    

    # TODO: Test the RNG stuff here
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._init_board()
        return self._get_obs(), self._get_info()
    

    def step(self, action: int):
        x, y = divmod(action, self.board_size)
        if self.board[x, y] == 1:  # Hit
            self.board[x, y] = 2
            reward = 1
        elif self.board[x, y] == 0:  # Miss
            self.board[x, y] = -1
            reward = -0.1
        else:  # Invalid action (cell already hit)
            reward = -1

        terminated = np.all(self.board != 1)
        truncated = False  # We don't apply a time limit wrapper here
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    



