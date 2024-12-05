import numpy as np
from board import Board
from tqdm import tqdm

def test_bot(n=100, board_size:int = 10, ships:list[int] = [2, 3, 3, 4, 5], Bot=None):
    if Bot is None:
        raise ValueError("Bot is not defined")
    
    total_moves = []
    avg_hits_at_move = [0] * board_size**2
    avg_moves_for_hit = [0] * np.sum(ships)
    for i in tqdm(range(n)):
        board = Board(board_size, ships)
        bot = Bot(board)
        num_moves = 0
        prev_hits = 0
        while not board.check_if_game_over():
            bot.attack()
            num_moves += 1
            hits = np.sum(board.get_board() == 2)
            try:
                avg_hits_at_move[num_moves-1] += hits
                if hits != prev_hits:
                    avg_moves_for_hit[hits-1] += num_moves
                    prev_hits = hits
            except:
                print(board.get_board())
                quit()
            # print(board.get_hidden_board())
        total_moves.append(num_moves)
    
    mean, median = np.mean(total_moves), np.median(total_moves)
    max, min = np.max(total_moves), np.min(total_moves)
    std = np.std(total_moves)
    avg_hits_at_move = [hits/n for hits in avg_hits_at_move]
    #print(avg_moves_for_hit)
    avg_moves_for_hit = [moves/n for moves in avg_moves_for_hit]
    
    return total_moves, mean, median, max, min, std, avg_hits_at_move, avg_moves_for_hit