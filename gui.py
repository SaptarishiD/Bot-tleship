import tkinter as tk
from tkinter import messagebox
from bot_random import RandomBot
from mcts_bot import MCTSBot

from board import Board

SHIPS = [2, 3, 3, 4, 5]
N = 10

class HumanPlayer:
    def __init__(self, board):
        self.board = board
    
    def attack(self, x, y):
        return self.board.attack(x, y)

class BattleshipGUI:
    def __init__(self, root, player1, player2):
        if player1 is None or player2 is None:
            raise ValueError("Both players must be defined")
        
        self.root = root
        self.moves = 0
        
        self.player1board = Board(N, SHIPS)
        self.player2board = Board(N, SHIPS)
        
        self.player1 = player1(self.player2board)
        self.player2 = player2(self.player1board)

        self.create_widgets()

    def create_widgets(self):
        self.root.title("Battleship Game")
        
        # Player 1 Board
        self.player1_frame = tk.LabelFrame(self.root, text="Your Board | Moves: " + str(self.moves))
        self.player1_frame.grid(row=0, column=0, padx=10, pady=10)
        self.player1_buttons = self.create_board(self.player1_frame, self.player1board.get_board(), interactive=False)

        # Player 2 Board (Hidden)
        self.player2_frame = tk.LabelFrame(self.root, text="Opponent's Board | Bot Type: " + str(self.player2.__class__.__name__))
        self.player2_frame.grid(row=0, column=1, padx=10, pady=10)
        self.player2_buttons = self.create_board(self.player2_frame, self.player2board.get_hidden_board(), interactive=True)
        
        self.moves += 1

    def create_board(self, parent, board, interactive):
        print(board)
        buttons = []
        for r in range(10):
            row = []
            for c in range(10):
                button = tk.Button(
                    parent,
                    text="",
                    width=2,
                    height=1,
                    command=lambda x=r, y=c: self.handle_click(x, y) if interactive else None,
                    highlightbackground=self.get_tile_color(board[r][c]),
                    bg=self.get_tile_color(board[r][c])
                )
                button.grid(row=r, column=c, padx=1, pady=1)
                row.append(button)
            buttons.append(row)
        return buttons

    def get_tile_color(self, value):
        if value == 0:
            return "blue"  # Water
        elif value == -1:
            return "white"  # Miss
        elif value == 1:
            return "black"  # Ship
        elif value == 2:
            return "red"  # Hit
        return "blue"

    def handle_click(self, x, y):
        if self.player2board.get_hidden_board()[x][y] != 0:
            messagebox.showinfo("Invalid Move", "You've already tried this spot!")
            return
        
        self.player1.attack(x, y)
        if self.player2board.check_if_game_over():
            messagebox.showinfo("Victory", "You win!")
            self.root.quit()

        # Bot's turn
        self.player2.attack()
        self.player1_board = self.player1board.get_board()
        if self.player1board.check_if_game_over():
            messagebox.showinfo("Defeat", "You lose!")
            self.root.quit()    
        
        self.create_widgets()


# Main
if __name__ == "__main__":

    root = tk.Tk()
    game = BattleshipGUI(root, HumanPlayer, MCTSBot)
    root.mainloop()
