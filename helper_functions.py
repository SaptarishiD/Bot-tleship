import numpy as np
from board import Board

def test_bot(n=100, board_size:int = 10, ships:list[int] = [2, 3, 3, 4, 5], Bot=None):
    if Bot is None:
        raise ValueError("Bot is not defined")
    
    total_moves = []
    for i in range(n):
        board = Board(board_size, ships)
        bot = Bot(board)
        num_moves = 0
        
        while not board.check_if_game_over():
            bot.attack()
            num_moves += 1
            # print(board.get_hidden_board())
        total_moves.append(num_moves)
    
    mean, median = np.mean(total_moves), np.median(total_moves)
    max, min = np.max(total_moves), np.min(total_moves)
    std = np.std(total_moves)
    
    return total_moves, mean, median, max, min, std