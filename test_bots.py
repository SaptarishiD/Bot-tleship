from board import Board
from bot_random import RandomBot
from bot_humanlike import HumanLikeBot
import numpy as np


def play_bots(bot1_class, bot2_class, board_size=10, n=100):
    bot1_wins = 0
    bot2_wins = 0
    bot1_win_moves = []
    bot2_win_moves = []
    bot1_loss_moves = []
    bot2_loss_moves = []
    
    for _ in range(n):
        
        board1 = Board(board_size)
        board2 = Board(board_size)
        
        
        bot1 = bot1_class(board2)  
        bot2 = bot2_class(board1)  
        
        bot1_move_count = 0
        bot2_move_count = 0
        
        
        while True:
            
            bot1.attack()
            bot1_move_count += 1
            if board2.check_if_game_over():
                bot1_wins += 1
                bot1_win_moves.append(bot1_move_count)
                bot2_loss_moves.append(bot2_move_count)
                break
            
            
            bot2.attack()
            bot2_move_count += 1
            if board1.check_if_game_over():
                bot2_wins += 1
                bot1_loss_moves.append(bot1_move_count)
                bot2_win_moves.append(bot2_move_count)
                break
            
            print(f"Game {_} | Bot1 moves: {bot1_move_count} | Bot2 moves: {bot2_move_count}")
            
        if _ % 50 == 0:
            print(f"Game {_} completed")
    
    bot1_win_avg_moves = sum(bot1_win_moves) / len(bot1_win_moves) if bot1_win_moves else 0
    bot2_win_avg_moves = sum(bot2_win_moves) / len(bot2_win_moves) if bot2_win_moves else 0
    bot1_loss_avg_moves = sum(bot1_loss_moves) / len(bot1_loss_moves) if bot1_loss_moves else 0
    bot2_loss_avg_moves = sum(bot2_loss_moves) / len(bot2_loss_moves) if bot2_loss_moves else 0
    return {
        "bot1_wins": bot1_wins,
        "bot2_wins": bot2_wins,
        "bot1_win_avg_moves": bot1_win_avg_moves,
        "bot2_win_avg_moves": bot2_win_avg_moves,
        "bot1_loss_avg_moves": bot1_loss_avg_moves,
        "bot2_loss_avg_moves": bot2_loss_avg_moves,
        "total_games": n,
    }
    
if __name__ == "__main__":
    
    results = play_bots(RandomBot, HumanLikeBot, board_size=10, n=1000)
    
    
    print("Bot1 wins:", results["bot1_wins"], "Win rate:", np.round(results["bot1_wins"] / results["total_games"], 3))
    print("Bot2 wins:", results["bot2_wins"], "Win rate:", np.round(results["bot2_wins"] / results["total_games"], 3))
    print("Bot1 win average moves:", np.round(results["bot1_win_avg_moves"], 3))
    print("Bot2 win average moves:", np.round(results["bot2_win_avg_moves"], 3))
    print("Total games:", results["total_games"])