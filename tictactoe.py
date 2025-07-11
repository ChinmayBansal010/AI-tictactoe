import random

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
        
    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('|' + '|'.join(row) + '|')
            
    def empty_squares(self):
        return ' ' in self.board
    
    def available_moves(self):
        return [i for i , x in enumerate(self.board) if x == ' ']
    
    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False
    
    def winner(self,square,letter):
        row_ind = square//3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0,4,8]]
            diagonal2 = [self.board[i] for i in [2,4,8]]
            
            if all([s == letter for s in diagonal1]) or all([s == letter for s in diagonal2]):
                return True
            
        return False
    
    def clone(self):
        clone_game = TicTacToe()
        clone_game.board = self.board[:]
        clone_game.current_winner = self.current_winner
        return clone_game         