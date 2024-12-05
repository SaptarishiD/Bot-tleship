import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

class BattleshipEnv(gym.Env):
    """
    Custom Gymnasium environment for Battleship
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        # Two-layer observation space: 
        # Layer 1: Ships placement
        # Layer 2: Hit/miss information
        self.observation_space = spaces.Box(
            low=0, high=10, 
            shape=(2, grid_size, grid_size)
        )

        # Action space is all possible grid locations
        self.action_space = spaces.Discrete(grid_size * grid_size)

        # Game state variables
        self.ships = [
            {'size': 5, 'name': 'Carrier'},
            {'size': 4, 'name': 'Battleship'},
            {'size': 3, 'name': 'Cruiser'},
            {'size': 3, 'name': 'Submarine'},
            {'size': 2, 'name': 'Destroyer'}
        ]
        
        # Reset variables
        self.ship_grid = None
        self.hit_grid = None
        self.remaining_ships = None

    def _place_ships(self):
        """
        Randomly place ships on the grid
        """
        self.ship_grid = np.zeros((self.grid_size, self.grid_size))
        placed_ships = []

        for ship in self.ships:
            while True:
                orientation = random.choice(['horizontal', 'vertical'])
                if orientation == 'horizontal':
                    x = random.randint(0, self.grid_size - ship['size'])
                    y = random.randint(0, self.grid_size - 1)
                    if np.all(self.ship_grid[y, x:x+ship['size']] == 0):
                        self.ship_grid[y, x:x+ship['size']] = 1
                        placed_ships.append({
                            'name': ship['name'], 
                            'coords': [(y, x+i) for i in range(ship['size'])]
                        })
                        break
                else:  # vertical
                    x = random.randint(0, self.grid_size - 1)
                    y = random.randint(0, self.grid_size - ship['size'])
                    if np.all(self.ship_grid[y:y+ship['size'], x] == 0):
                        self.ship_grid[y:y+ship['size'], x] = 1
                        placed_ships.append({
                            'name': ship['name'], 
                            'coords': [(y+i, x) for i in range(ship['size'])]
                        })
                        break

        return placed_ships
    
    def step(self, action):
        """
        Take an action in the environment
        """
        # Convert 1D action to 2D coordinates
        y = action // self.grid_size
        x = action % self.grid_size

        # Track if this is a hit or miss
        terminated = False
        truncated = False
        reward = 0
        
        # Check if action is valid (not already shot)
        if self.hit_grid[y, x] == 1:
            reward = -1  # Penalty for repeating a shot
            truncated = True
        else:
            self.hit_grid[y, x] = 1

            # Check if hit
            if self.ship_grid[y, x] == 1:
                reward = 1 # Reward for hitting a ship
                
                # Remove coordinate from remaining ships
                if (y, x) in self.remaining_ships:
                    self.remaining_ships.remove((y, x))
                
                # Check if all ships are sunk
                if len(self.remaining_ships) == 0:
                    reward = 10  # Major reward for sinking all ships
                    terminated = True
            else:
                reward = -1  # Small penalty for missing

        # Create observation
        observation = np.stack([self.ship_grid, self.hit_grid], dtype=np.float32)

        return observation, reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        """
        super().reset(seed=seed)

        # Reset game state
        self.hit_grid = np.zeros((self.grid_size, self.grid_size))
        self._place_ships()
        
        # Track remaining ship coordinates
        self.remaining_ships = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.ship_grid[y, x] == 1:
                    self.remaining_ships.append((y, x))
                    
        observation = np.stack([self.ship_grid, self.hit_grid], dtype=np.float32)
        

        return observation, {}

    

    def render(self):
        """
        Render the current game state
        """
        if self.render_mode == 'human':
            print("Ship Grid:")
            print(self.ship_grid)
            print("\nHit Grid:")
            print(self.hit_grid)

    def close(self):
        """
        Close the environment
        """
        pass