from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import gym

class _Ship(object):
    def __init__(self, size, row, col, dr, dc):
        self.size = size
        self.row = row
        self.col = col
        self.dr = dr
        self.dc = dc


class BattleshipEnv(gym.Env):
    def __init__(self, observation_space="flat-ships", action_space="coords", board_width=10, board_height=10,
                 ships=(1, 2, 3, 4, 5), latent_var_precision=8):
        self.board = np.zeros((board_height, board_width), dtype=bool)
        self.ship_lens = sorted(ships)
        self.shots = np.zeros_like(self.board, dtype=bool)
        self.observation_space_type = observation_space
        self.action_space_type = action_space
        self.latent_var_precision = latent_var_precision
        self.window = None
        self.ships = None

        if action_space == "coords":
            self.action_space = gym.spaces.MultiDiscrete((board_height, board_width))
        elif action_space == "flat":
            self.action_space = gym.spaces.Discrete(board_height * board_width)
        else:
            raise Exception(f"Unrecognized action space {action_space}")

        allowed_parts = {"flat", "ships", "latent"}
        parts = observation_space.split("-")
        assert len(set(parts)) == len(parts), "No repeated parts in observation space!"
        assert all(p in allowed_parts for p in parts), "Unrecognized part in " + observation_space
        parts = set(parts)
        state_dim = 0
        if "flat" in parts:
            state_dim += 2 * board_height * board_width
        if "ships" in parts:
            state_dim += len(ships)
        if "latent" in parts:
            state_dim += latent_var_precision
        self.observation_space = gym.spaces.MultiBinary(state_dim)
        self.reset()

    def _is_sunk(self, ship: _Ship):
        row, col = ship.row, ship.col
        return all(self.shots[row + i * ship.dr, col + i * ship.dc] for i in range(ship.size))

    def _observe(self):
        hits = (self.board & self.shots).astype(np.int8)
        misses = (~self.board & self.shots).astype(np.int8)
        obs = np.empty(0, dtype=np.int8)
        if "flat" in self.observation_space_type:
            misses_hits = np.hstack((misses.flatten(), hits.flatten()))
            obs = np.hstack((obs, misses_hits))
        if "ships" in self.observation_space_type:
            ship_obs = np.array([self._is_sunk(ship) for ship in self.ships], dtype=np.int8)
            obs = np.hstack((obs, ship_obs))
        if "latent" in self.observation_space_type:
            bits = (np.random.random(self.latent_var_precision) >= 0.5).astype(np.int8)
            obs = np.hstack((obs, bits))
        return obs

    def _done(self):
        return (self.board == (self.board & self.shots)).all()

    def step(self, action: Any):
        if self.action_space_type == "coords":
            row, col = action
        elif self.action_space_type == "flat":
            row = action // self.board.shape[0]
            col = action % self.board.shape[1]
        else:
            raise AssertionError
        if self.shots[row, col]:
            reward = -1
        else:
            reward = 1 if self.board[row, col] else -0.15
            self.shots[row, col] = True
        # print("Board: ")

        # # Plot the board as a heatmap and save to file
        # plt.subplot(1, 2, 1)
        # plt.imshow(self.board, cmap='hot', interpolation='nearest')
        # plt.title('Board Heatmap')
        # plt.colorbar(label='Value')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')


        # plt.subplot(1, 2, 2)
        # plt.imshow(self.shots, cmap='hot', interpolation='nearest')
        # plt.title('Shots Heatmap')
        # plt.colorbar(label='Value')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')

        # plt.tight_layout()
        # # plt.savefig(f'src/plots/board_heatmap_{row}_{col}.png')
        # plt.show()
        # plt.close()

        return self._observe(), reward, self._done(), {}

    def can_move(self, action: Any):
        if self.action_space_type == "coords":
            row, col = action
        elif self.action_space_type == "flat":
            row = action // self.board.shape[0]
            col = action % self.board.shape[1]
        else:
            raise AssertionError
        return not self.shots[row, col]

    def _can_place(self, size, row, col, dr, dc):
        for _ in range(size):
            if row < 0 or row >= self.board.shape[0] or col < 0 or col >= self.board.shape[1]:
                return False
            if self.board[row, col]:
                return False
            row += dr
            col += dc
        return True

    def reset(self):
        self.board[:] = False
        self.shots[:] = False
        self.ships = []
        for ship in reversed(self.ship_lens):
            placed = False
            while not placed:
                row = np.random.randint(0, self.board.shape[0])
                col = np.random.randint(0, self.board.shape[1])
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dr, dc = dirs[np.random.randint(0, len(dirs))]
                if self._can_place(ship, row, col, dr, dc):
                    placed = True
                    self.ships.append(_Ship(ship, row, col, dr, dc))
                    r, c = row, col
                    for _ in range(ship):
                        self.board[r, c] = True
                        r += dr
                        c += dc
        self.ships = self.ships[::-1]
        return self._observe()

    def close(self):
        if self.window is not None:
            self.window.close()
        self.window = None

if __name__ == "__main__":
    be = BattleshipEnv("flat-ships")
    be.shots = be.board.astype(bool)
