import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

class BattleshipEnv(gym.Env):
    """
    Custom Gymnasium environment for Battleship
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, render_mode=None, grid_size=10, max_moves=100):
        super().__init__()
        self.grid_size = grid_size
        self.max_moves = max_moves  # Maximum number of moves allowed
        self.render_mode = render_mode

        # Observation space: Ships placement + hit/miss information
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, grid_size, grid_size), dtype=np.float32
        )

        # Action space: All possible grid locations (1D flattened)
        self.action_space = spaces.Discrete(grid_size * grid_size)

        # Internal state variables
        self.ship_grid = None
        self.hit_grid = None
        self.remaining_ships = None
        self.current_moves = 0  # Track moves

        # Define the ship types
        self.ships = [
            {'size': 5, 'name': 'Carrier'},
            {'size': 4, 'name': 'Battleship'},
            {'size': 3, 'name': 'Cruiser'},
            {'size': 3, 'name': 'Submarine'},
            {'size': 2, 'name': 'Destroyer'}
        ]

    def _place_ships(self):
        """
        Randomly place ships on the grid
        """
        self.ship_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for ship in self.ships:
            while True:
                orientation = random.choice(['horizontal', 'vertical'])
                if orientation == 'horizontal':
                    x = random.randint(0, self.grid_size - ship['size'])
                    y = random.randint(0, self.grid_size - 1)
                    if np.all(self.ship_grid[y, x:x + ship['size']] == 0):
                        self.ship_grid[y, x:x + ship['size']] = 1
                        break
                else:
                    x = random.randint(0, self.grid_size - 1)
                    y = random.randint(0, self.grid_size - ship['size'])
                    if np.all(self.ship_grid[y:y + ship['size'], x] == 0):
                        self.ship_grid[y:y + ship['size'], x] = 1
                        break

    def step(self, action):
        """
        Perform an action in the environment.
        Args:
            action (int): The index of the action to take.
        Returns:
            observation, reward, terminated, truncated, info
        """
        y = action // self.grid_size
        x = action % self.grid_size

        reward = 0
        terminated = False
        truncated = False

        # Check if the action is valid
        if self.hit_grid[y, x] == 1:
            reward -= 2*self.grid_size # Penalty for trying an invalid move
            #choose random legal move:
            # while self.hit_grid[y, x] == 1:
            #     y = random.randint(0, self.grid_size - 1)
            #     x = random.randint(0, self.grid_size - 1)
            truncated = True
        else:
            self.hit_grid[y, x] = 1
            if self.ship_grid[y, x] == 1:
                reward = self.grid_size  # Reward for hitting a ship
                self.remaining_ships.remove((y, x))
                if len(self.remaining_ships) == 0:
                    reward = self.grid_size * self.grid_size  # Bonus for sinking all ships
                    terminated = True
            else:
                reward = -1  # Penalty for missing

        # Increment move count
        self.current_moves += 1
        if self.current_moves >= self.max_moves:
            truncated = True

        observation = np.stack([self.ship_grid, self.hit_grid], axis=0)
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)

        # Reset internal state variables
        self.current_moves = 0
        self.hit_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._place_ships()

        # Create a list of all ship coordinates
        self.remaining_ships = [
            (y, x) for y in range(self.grid_size) for x in range(self.grid_size)
            if self.ship_grid[y, x] == 1
        ]

        # Generate initial observation and action mask
        observation = np.stack([self.ship_grid, self.hit_grid], axis=0)
        info = {}

        return observation, info

    def render(self):
        """
        Render the current game state.
        """
        if self.render_mode == 'human':
            print("Ship Grid:")
            print(self.ship_grid)
            print("\nHit Grid:")
            print(self.hit_grid)

    def close(self):
        """
        Clean up any resources.
        """
        pass
