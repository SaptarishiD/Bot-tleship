import numpy as np

class Board():
    def __init__(self, size: int, ships: list[int] = [2, 3, 3, 4, 5]):
        self.board = np.zeros((size, size))
        self.size = size
        
        for size in ships:
            placed = False
            while not placed:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                orientation = np.random.randint(0, 2)
                if orientation == 0:
                    if x + size < self.size:
                        placed = True
                        for i in range(size):
                            if self.board[x + i, y] == 1:
                                placed = False
                                continue
                            self.board[x + i, y] = 1
                else:
                    if y + size < self.size:
                        placed = True
                        for i in range(size):
                            if self.board[x, y + i] == 1:
                                placed = False
                                continue
                            self.board[x, y + i] = 1
        
    def get_board(self):
        return self.board
    
    def get_size(self):
        return self.size


if __name__ == "__main__":
    board = Board(10)
    print(board.get_board())
    print(board.get_size())
