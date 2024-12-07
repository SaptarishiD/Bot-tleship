import numpy as np
from collections import deque
from board import Board
from helper_functions import test_bot


class HumanLikeBot:
    def __init__(self, board: Board):
        self.board = board
        self.board_size = board.get_size()
        self.state = {
            "board": board.get_hidden_board(),
            "moves": [(x, y) for x in range(self.board_size) for y in range(self.board_size)],
            "mode": "hunt",
            "targets": deque(),
            "destroy_targets": deque(),
            "last_hit": None,
            "first_hit": None,  # Initialize first_hit
            "direction": None,
        }

    def is_valid_attack(self, x, y):
        print("Value at", x, y, "is", self.board.get_hidden_board()[x, y])
        return self.board.get_hidden_board()[x, y] == 0

    def remove_move(self, x, y):
        self.state["moves"] = [(mx, my) for mx, my in self.state["moves"] if (mx, my) != (x, y)]
        self.state["targets"] = deque([(tx, ty) for tx, ty in self.state["targets"] if (tx, ty) != (x, y)])
        self.state["destroy_targets"] = deque([(dx, dy) for dx, dy in self.state["destroy_targets"] if (dx, dy) != (x, y)])

    def hunt_mode(self):
        while self.state["moves"]:
            x, y = self.state["moves"].pop(np.random.randint(len(self.state["moves"])))
            if not self.is_valid_attack(x, y):
                continue  # Skip invalid moves
            self.board.attack(x, y)
            self.remove_move(x, y)

            if self.board.get_hidden_board()[x, y] == 2:  # Hit
                self.state["last_hit"] = (x, y)
                self.state["first_hit"] = (x, y)  # Set first_hit
                self.update_targets()
                self.state["mode"] = "target"  # Switch to target mode
            break

    def target_mode(self):
        while self.state["targets"]:
            x, y = self.state["targets"].pop()
            if not self.is_valid_attack(x, y):
                continue  # Skip invalid moves
            self.board.attack(x, y)
            self.remove_move(x, y)

            if self.board.get_hidden_board()[x, y] == 2:  # Another hit
                if self.state["last_hit"]:
                    last_x, last_y = self.state["last_hit"]
                    if y == last_y:  # Same column, move vertically (down/up)
                        self.state["direction"] = "vertical"
                    elif x == last_x:  # Same row, move horizontally (left/right)
                        self.state["direction"] = "horizontal"
                self.state["last_hit"] = (x, y)

                if self.state["direction"]:  # Switch to destroy mode if direction is determined
                    self.update_destroy_targets()
                    self.state["mode"] = "destroy"
            break
        else:
            self.state["mode"] = "hunt"  # Return to hunt mode if no targets remain
            self.attack()

    def destroy_mode(self):
        while self.state["destroy_targets"]:
            x, y = self.state["destroy_targets"].popleft()
            if not self.is_valid_attack(x, y):
                continue  # Skip invalid moves
            self.board.attack(x, y)
            self.remove_move(x, y)

            if self.board.get_hidden_board()[x, y] == 2:  # Another hit
                self.state["last_hit"] = (x, y)
                self.update_destroy_targets()

            # Check if the ship has been fully uncovered
            if self.check_if_ship_sunk(self.state["last_hit"], self.state["first_hit"]):
                self.state["mode"] = "hunt"  # Return to hunt mode for the next ship
                self.state["direction"] = None  # Reset direction
                self.state["last_hit"] = None  # Reset the last hit
                self.state["first_hit"] = None  # Reset the first hit
            break
        else:
            self.state["mode"] = "hunt"  # Return to hunt mode if no destroy targets remain
            self.attack()

    def update_targets(self):
        last_hit_x, last_hit_y = self.state["last_hit"]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x, y = last_hit_x + dx, last_hit_y + dy
            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.is_valid_attack(x, y):
                self.state["targets"].append((x, y))

    def update_destroy_targets(self):
        last_hit_x, last_hit_y = self.state["last_hit"]

        if self.state["direction"] == "horizontal":
            # Check left and right along the row (y-axis)
            for dy in [-1, 1]:
                y = last_hit_y + dy
                if 0 <= y < self.board_size and self.is_valid_attack(last_hit_x, y):
                    self.state["destroy_targets"].append((last_hit_x, y))
        elif self.state["direction"] == "vertical":
            # Check up and down along the column (x-axis)
            for dx in [-1, 1]:
                x = last_hit_x + dx
                print("last_hit_x", last_hit_x, "dx", dx, "x", x)
                print("Checking", x, last_hit_y)
                if 0 <= x < self.board_size and self.is_valid_attack(x, last_hit_y):
                    print("Appending", x, last_hit_y)
                    self.state["destroy_targets"].append((x, last_hit_y))
                    print(self.state["destroy_targets"])
        print(self.state["destroy_targets"])

    def check_if_ship_sunk(self, last_hit, first_hit):
        def check_adjacent(x, y, dx, dy):
            for d in [-1, 1]:
                nx, ny = x + d * dx, y + d * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board.get_hidden_board()[nx, ny] == 2:
                        return False
            return True

        last_hit_x, last_hit_y = last_hit
        first_hit_x, first_hit_y = first_hit

        if self.state["direction"] == "horizontal":
            return check_adjacent(last_hit_x, last_hit_y, 0, 1) and check_adjacent(first_hit_x, first_hit_y, 0, 1)
        elif self.state["direction"] == "vertical":
            return check_adjacent(last_hit_x, last_hit_y, 1, 0) and check_adjacent(first_hit_x, first_hit_y, 1, 0)

        return True

    def attack(self):
        if self.state["mode"] == "hunt":
            self.hunt_mode()
        elif self.state["mode"] == "target":
            self.target_mode()
        elif self.state["mode"] == "destroy":
            self.destroy_mode()


if __name__ == "__main__":
    # board = Board(10)
    # bot = HumanLikeBot(board)
    # while not board.check_if_game_over():
    #     bot.attack()
    #     print(board.get_board())
    moves, mean, median, max_, min_, std, avg_hits_at_move, avg_moves_for_hit = test_bot(1000, 10, Bot=HumanLikeBot)
    print(f"Mean: {mean}, Median: {median}, Max: {max_}, Min: {min_}, Std: {std}, avg_hits_at_move: {avg_hits_at_move}, avg_moves_for_hit: {avg_moves_for_hit}")
