import numpy as np

class Board():
    def __init__(self, size: int, ships: list[int] = [2, 3, 3, 4, 5]):
        self.board = np.zeros((size, size))
        self.size = size
        self.ships = ships
        for ship_size in ships:
            placed = False
            while not placed:
                x = np.random.randint(0, self.size)
                y = np.random.randint(0, self.size)
                orientation = np.random.randint(0, 2)
                if orientation == 0:
                    if x + ship_size < self.size:
                        if np.all(self.board[x:x+ship_size, y] == 0):
                            self.board[x:x+ship_size, y] = np.ones_like(self.board[x:x+ship_size, y])
                            placed = True
                else:
                    if y + ship_size < self.size:
                        if np.all(self.board[x, y:y+ship_size] == 0):
                            self.board[x, y:y+ship_size] = np.ones_like(self.board[x, y:y+ship_size])
                            placed = True
        
    def get_board(self):
        return self.board
    
    def get_size(self):
        return self.size
    
    def get_ships(self):
        return self.ships
    
    def attack(self, x: int, y: int):
        # print(x,y)
        # print(self.get_hidden_board())
        # assert x >= 0 and x < self.size
        # assert y >= 0 and y < self.size
        # assert self.board[x, y] != -1
        # assert self.board[x, y] != 2
        if self.board[x, y] == 1:
            self.board[x, y] = 2
            
        else:
            self.board[x, y] = -1
        
        if np.all(self.board != 1):
            return True
    
    def get_hidden_board(self):
        return np.where(self.board == 1, 0, self.board)
    
    def check_if_game_over(self):
        return np.all(self.board != 1)

if __name__ == "__main__":
    board = Board(10)
    print(board.get_board())
    print(board.get_hidden_board())
    print("-------------------")
    board.attack(0, 0)
    print(board.get_board())
    print(board.get_hidden_board())
    print("-------------------")
    board.attack(6, 7)
    print(board.get_board())
    print(board.get_hidden_board())
    print("-------------------")
