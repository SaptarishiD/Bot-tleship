import numpy as np
from board import Board
from helper_functions import test_bot

class RandomBot():
    def __init__(self, board: Board):
        self.board = board
        self.board_size = board.get_size()
    
    def attack(self):
        x = np.random.randint(0, self.board_size)
        y = np.random.randint(0, self.board_size)
        while self.board.get_hidden_board()[x, y] == 2 or self.board.get_hidden_board()[x, y] == -1:
            x = np.random.randint(0, self.board_size)
            y = np.random.randint(0, self.board_size)
        return self.board.attack(x, y)
    
if __name__ == "__main__":
    moves, mean, median, max, min, std = test_bot(100, 10, RandomBot)

    print(f"Mean: {mean}, Median: {median}, Max: {max}, Min: {min}, Std: {std}")
    