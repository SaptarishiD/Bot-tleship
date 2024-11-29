import random
import math
from collections import defaultdict
from board import Board
import numpy as np
from helper_functions import test_bot 
import time

# Board constants
UNKNOWN = 0
MISS = -1
HIT = 2

# Ship configuration (example)

class Node:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

    def is_fully_expanded(self):
        return len(self.children) == sum(1 for row in self.board for cell in row if cell == UNKNOWN)

    def best_child(self, exploration_weight=1.41):
        return max(
            self.children,
            key=lambda child: (child.reward / child.visits) +
                              exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
        )

class MCTSBot:
    def __init__(self, board: Board):
        self.board = board
        
    def validate_partial_ship_placement(self, board, ships):
        """
        Validate the current hits on the board and return partial validity.
        """
        visited = set()
        ship_lengths = []

        def dfs(x, y, direction=None):
            """
            Depth-first search to find the length of a ship.
            """
            if (x, y) in visited or board[x][y] != HIT:
                return 0

            visited.add((x, y))
            length = 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(board) and 0 <= ny < len(board[0]):
                    # Ensure the ship does not bend
                    if direction is None or direction == (dx, dy):
                        length += dfs(nx, ny, (dx, dy))
            return length

        # Find all ship lengths
        for x in range(len(board[0])):
            for y in range(len(board[0])):
                if board[x][y] == HIT and (x, y) not in visited:
                    ship_lengths.append(dfs(x, y))

        # Compare found ship lengths to the expected ship sizes
        valid_ships = 0
        total_valid_squares = 0
        ships_remaining = ships[:]

        for length in ship_lengths:
            if length in ships_remaining:
                valid_ships += 1
                total_valid_squares += length
                ships_remaining.remove(length)

        # Calculate the reward
        total_ship_squares = sum(ships)
        penalty = abs(total_ship_squares - total_valid_squares)
        return valid_ships, total_valid_squares, penalty

    def simulate(self, board, ships):
        """
        Simulate random hits until all ship squares are accounted for.
        """
        temp_board = np.copy(board)
        unknown_squares = [(x, y) for x in range(len(board)) for y in range(len(board[0])) if board[x][y] == UNKNOWN]
        random.shuffle(unknown_squares)

        ship_squares = sum(ships)  # Total ship squares to find
        hits = (board == HIT).sum()  # Current hits on the board

        while hits < ship_squares and unknown_squares:
            x, y = unknown_squares.pop()
            temp_board[x][y] = HIT
            hits += 1

        # Add misses for the rest of the unknowns
        # for x, y in unknown_squares:
        #     temp_board[x][y] = MISS

        # Reward based on validity
        valid_ships, valid_squares, penalty = self.validate_partial_ship_placement(temp_board, ships)
        reward = (valid_squares / ship_squares) * valid_ships / len(ships)
        return reward
    
    def weighted_heatmap(self, board):
        """Calculate a heatmap for hit adjacent squares."""
        heatmap = np.ones_like(board)
        for x in range(len(board)):
            for y in range(len(board[0])):
                if board[x][y] == HIT:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(board) and 0 <= ny < len(board[0]) and board[nx][ny] == UNKNOWN:
                            heatmap[nx][ny] += 2
        return heatmap/np.sum(heatmap)

    def mcts(self, root, ships, iteration_time=1, max_iterations=1000):
        """
        Monte Carlo Tree Search for Battleship.
        """
        start_time = time.time()
        iterations = 0
        while time.time() - start_time < iteration_time and iterations < max_iterations:
            node = root

            # Selection
            while node.children and node.is_fully_expanded():
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded():
                # Choose a random unexplored square to hit
                heatmap = self.weighted_heatmap(node.board)
                unexplored = [(x, y) for x in range(len(node.board)) for y in range(len(node.board[0])) if node.board[x][y] == UNKNOWN]
                unexplored_probs = [heatmap[x][y] for x, y in unexplored]
                    
                if unexplored:
                    x, y = random.choices(unexplored, weights=unexplored_probs, k=1)[0]
                    new_board = np.copy(node.board)
                    new_board[x][y] = HIT  # Assume it's a hit
                    child = Node(new_board, node)
                    node.children.append(child)
                    node = child

            # Simulation
            reward = self.simulate(node.board, ships)

            # Backpropagation
            while node:
                node.visits += 1
                node.reward += reward
                node = node.parent
            
            iterations += 1

        return root.best_child(exploration_weight=0)  # Return the best move after all iterations
    
    def attack(self):
        #Check for 0 sandwhiched between 2s
        for i in range(1, self.board.size-1):
            for j in range(1, self.board.size-1):
                if self.board.get_hidden_board()[i, j] == 0:
                    if self.board.get_hidden_board()[i-1, j] == 2 and self.board.get_hidden_board()[i+1, j] == 2:
                        self.board.attack(i, j)
                        return
                    if self.board.get_hidden_board()[i, j-1] == 2 and self.board.get_hidden_board()[i, j+1] == 2:
                        self.board.attack(i, j)
                        return
        
        root = Node(self.board.get_hidden_board())
        best_child = self.mcts(root, self.board.get_ships(), iteration_time=1, max_iterations=1000)
        
        x, y = np.where((best_child.board - root.board) == 2)
        
        self.board.attack(x[0], y[0])
        

if __name__ == "__main__":
    # Example board
    board = Board(10)
    bot = MCTSBot(board)
    print(board.get_board())
    
    while not board.check_if_game_over():
        bot.attack()
        print(board.get_hidden_board(), "\r")
    
    moves, mean, median, max_, min_, std = test_bot(100, 10, Bot=MCTSBot)
    print(f"Mean: {mean}, Median: {median}, Max: {max_}, Min: {min_}, Std: {std}")