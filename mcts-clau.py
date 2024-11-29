import numpy as np
from board import Board
import time
from collections import defaultdict
from bot_random import RandomBot
import math

class MCTSNode:
    def __init__(self, board_state, parent=None, move=None):
        self.board_state = board_state.copy()
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children = {}  # Dictionary of move: MCTSNode
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_valid_moves()
        
    def get_valid_moves(self):
        """Get all valid moves (unhit positions) on the board"""
        valid = []
        for i in range(len(self.board_state)):
            for j in range(len(self.board_state)):
                if self.board_state[i, j] in [0, 1]:  # Unhit position
                    valid.append((i, j))
        return valid
    
    def UCB1(self, exploration_weight=1.414):
        """Upper Confidence Bound 1 formula for node selection"""
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits)

class MCTSBot:
    def __init__(self, board: Board, simulation_time=1.0):
        self.board = board
        self.board_size = board.get_size()
        self.simulation_time = simulation_time
        self.probability_map = np.ones((self.board_size, self.board_size)) / (self.board_size * self.board_size)
        self.ships = [5, 4, 3, 3, 2]  # Ship lengths
        self.moves_made = []
        
    def simulate_attack(self, board_state, move):
        """Simulate an attack on a board state"""
        new_state = board_state.copy()
        hit = new_state[move] == 1
        new_state[move] = 2 if hit else -1
        return new_state, hit
    
    def simulate_random_game(self, board_state):
        """Simulate a random game from current state to end"""
        current_state = board_state.copy()
        moves_remaining = [(i, j) for i in range(self.board_size) 
                          for j in range(self.board_size) 
                          if current_state[i, j] in [0, 1]]
        
        ships_remaining = np.sum(current_state == 1)
        hits = 0
        
        while ships_remaining > 0 and moves_remaining:
            move_idx = np.random.randint(len(moves_remaining))
            move = moves_remaining.pop(move_idx)
            
            if current_state[move] == 1:
                hits += 1
                ships_remaining -= 1
            
            current_state[move] = 2 if current_state[move] == 1 else -1
            
        return hits, len(moves_remaining)
    
    def MCTS(self, root_state):
        """Monte Carlo Tree Search main function"""
        root = MCTSNode(root_state)
        end_time = time.time() + self.simulation_time
        
        while time.time() < end_time:
            node = root
            current_state = root_state.copy()
            
            # Selection
            while node.untried_moves == [] and node.children:
                node = max(node.children.values(), key=lambda n: n.UCB1())
                current_state, _ = self.simulate_attack(current_state, node.move)
            
            # Expansion
            if node.untried_moves:
                move = node.untried_moves.pop()
                new_state, hit = self.simulate_attack(current_state, move)
                node.children[move] = MCTSNode(new_state, parent=node, move=move)
                node = node.children[move]
                current_state = new_state
            
            # Simulation
            hits, remaining = self.simulate_random_game(current_state)
            
            # Backpropagation
            while node:
                node.visits += 1
                node.wins += hits
                node = node.parent
        
        # Return best move based on visit count
        return max(root.children.items(), key=lambda x: x[1].visits)[0]
    
    def update_probability_map(self, last_move, was_hit):
        """Update probability map based on hit/miss information"""
        x, y = last_move
        self.probability_map[x, y] = 0  # Mark as tried
        
        if was_hit:
            # Increase probability of adjacent cells
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.board_size and 0 <= new_y < self.board_size:
                    self.probability_map[new_x, new_y] *= 2
        
        # Normalize probability map
        total = np.sum(self.probability_map)
        if total > 0:
            self.probability_map /= total
    
    def attack(self):
        """Make an attack move using MCTS"""
        current_state = self.board.get_hidden_board()
        
        # Use MCTS to find the best move
        move = self.MCTS(current_state)
        was_hit = self.board.attack(move[0], move[1])
        
        # Update probability map
        self.update_probability_map(move, was_hit)
        self.moves_made.append((move, was_hit))
        
        return was_hit

def test_bots(n_games=100, board_size=10):
    """Compare MCTS bot against Random bot"""
    results = {
        'mcts': {
            'moves': [],
            'wins': 0,
            'time': []
        },
        'random': {
            'moves': [],
            'wins': 0,
            'time': []
        }
    }
    
    for i in range(n_games):
        # Test MCTS bot
        board = Board(board_size)
        bot = MCTSBot(board, simulation_time=0.1)
        print(f"Game {i + 1}")
        start_time = time.time()
        moves = 0
        
        while not board.check_if_game_over():
            bot.attack()
            moves += 1
        
        end_time = time.time()
        results['mcts']['moves'].append(moves)
        results['mcts']['time'].append(end_time - start_time)
        results['mcts']['wins'] += 1
        
        # Test Random bot
        board = Board(board_size)
        bot = RandomBot(board)
        start_time = time.time()
        moves = 0
        
        while not board.check_if_game_over():
            bot.attack()
            moves += 1
        
        end_time = time.time()
        results['random']['moves'].append(moves)
        results['random']['time'].append(end_time - start_time)
        results['random']['wins'] += 1
    
    # Calculate statistics
    stats = {}
    for bot_type in ['mcts', 'random']:
        moves = results[bot_type]['moves']
        times = results[bot_type]['time']
        stats[bot_type] = {
            'avg_moves': np.mean(moves),
            'median_moves': np.median(moves),
            'std_moves': np.std(moves),
            'min_moves': np.min(moves),
            'max_moves': np.max(moves),
            'avg_time': np.mean(times),
            'total_time': sum(times)
        }
    
    return stats

if __name__ == "__main__":
    # Run comparison tests
    stats = test_bots(n_games=10)
    
    print("\nMCTS Bot Statistics:")
    print(f"Average moves to win: {stats['mcts']['avg_moves']:.2f}")
    print(f"Median moves to win: {stats['mcts']['median_moves']:.2f}")
    print(f"Standard deviation: {stats['mcts']['std_moves']:.2f}")
    print(f"Best game (min moves): {stats['mcts']['min_moves']}")
    print(f"Worst game (max moves): {stats['mcts']['max_moves']}")
    print(f"Average time per game: {stats['mcts']['avg_time']:.2f} seconds")
    print(f"Total time for all games: {stats['mcts']['total_time']:.2f} seconds")
    
    print("\nRandom Bot Statistics:")
    print(f"Average moves to win: {stats['random']['avg_moves']:.2f}")
    print(f"Median moves to win: {stats['random']['median_moves']:.2f}")
    print(f"Standard deviation: {stats['random']['std_moves']:.2f}")
    print(f"Best game (min moves): {stats['random']['min_moves']}")
    print(f"Worst game (max moves): {stats['random']['max_moves']}")
    print(f"Average time per game: {stats['random']['avg_time']:.2f} seconds")
    print(f"Total time for all games: {stats['random']['total_time']:.2f} seconds")