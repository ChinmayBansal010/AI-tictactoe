import tkinter as tk
import neat
import pickle
import random

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward.txt'
)

with open('best_ai1.pkl', 'rb') as f:
    winner1 = pickle.load(f)

with open('best_ai2.pkl', 'rb') as f:
    winner2 = pickle.load(f)

net1 = neat.nn.FeedForwardNetwork.create(winner1, config)
net2 = neat.nn.FeedForwardNetwork.create(winner2, config)

class TicTacToeGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe - AI & You")
        self.root.configure(bg="#f0f8ff")
        self.buttons = []
        self.board = [0] * 9
        self.turn = 1  # 1 means X, -1 means O
        self.ai_vs_ai = False
        self.current_ai = None  # will hold net1 or net2 for AI vs Human
        self.ai1_turn = True   # in AI vs AI mode, True means AI1's turn, False AI2's turn

        self.status = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f0f8ff")
        self.status.grid(row=3, column=0, columnspan=3, pady=10)

        self.create_buttons()

        self.toggle_button = tk.Button(root, text="Switch to AI vs AI", font=("Arial", 12),
                                       bg="#ffd700", command=self.toggle_mode)
        self.toggle_button.grid(row=4, column=0, columnspan=3, pady=5)

        self.play_again_button = tk.Button(root, text="Play Again", font=("Arial", 12),
                                           bg="#d3d3d3", command=self.reset_game)
        self.play_again_button.grid(row=5, column=0, columnspan=3, pady=10)

        self.reset_game()  # starts game with AI turn or random AI turn

    def create_buttons(self):
        for i in range(9):
            button = tk.Button(self.root, text=" ", font=("Arial", 28, "bold"),
                               width=5, height=2, bg="white",
                               command=lambda i=i: self.player_move(i))
            button.grid(row=i//3, column=i%3, padx=2, pady=2)
            self.buttons.append(button)

    def toggle_mode(self):
        self.ai_vs_ai = not self.ai_vs_ai
        self.toggle_button.config(text="Switch to Human vs AI" if self.ai_vs_ai else "Switch to AI vs AI")
        self.reset_game()

    def player_move(self, index):
        if self.board[index] == 0 and self.turn == -1 and not self.ai_vs_ai:
            self.board[index] = -1
            self.buttons[index].config(text="O", fg="blue", state="disabled")
            if self.check_winner(-1):
                self.status.config(text="You win!")
                self.disable_all()
            elif self.is_draw():
                self.status.config(text="It's a draw!")
            else:
                self.turn = 1
                self.status.config(text="AI's turn (X)")
                self.root.after(500, self.ai_move)

    def ai_move(self):
        valid = [i for i, x in enumerate(self.board) if x == 0]
        if not valid:
            return

        if self.ai_vs_ai:
            # AI vs AI mode
            current_net = net1 if self.ai1_turn else net2
            output = current_net.activate(self.board)
            move = max(((i, output[i]) for i in valid), key=lambda x: x[1])[0]
            self.board[move] = 1 if self.ai1_turn else -1
            symbol = "X" if self.ai1_turn else "O"
            color = "red" if self.ai1_turn else "blue"
            self.buttons[move].config(text=symbol, fg=color, state="disabled")

            if self.check_winner(1 if self.ai1_turn else -1):
                self.status.config(text="AI 1 wins!" if self.ai1_turn else "AI 2 wins!")
                self.disable_all()
                return
            elif self.is_draw():
                self.status.config(text="It's a draw!")
                return

            self.ai1_turn = not self.ai1_turn
            self.status.config(text="AI 1's turn (X)" if self.ai1_turn else "AI 2's turn (O)")
            self.root.after(500, self.ai_move)

        else:
            # Human vs AI mode
            output = self.current_ai.activate(self.board)
            move = max(((i, output[i]) for i in valid), key=lambda x: x[1])[0]
            self.board[move] = 1
            self.buttons[move].config(text="X", fg="red", state="disabled")

            if self.check_winner(1):
                self.status.config(text="AI wins!")
                self.disable_all()
                return
            elif self.is_draw():
                self.status.config(text="It's a draw!")
                return

            self.turn = -1
            self.status.config(text="Your turn (O)")

    def check_winner(self, player):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        return any(all(self.board[i] == player for i in line) for line in wins)

    def is_draw(self):
        return all(cell != 0 for cell in self.board)

    def disable_all(self):
        for b in self.buttons:
            b.config(state="disabled")

    def reset_game(self):
        self.board = [0] * 9
        for b in self.buttons:
            b.config(text=" ", state="normal", bg="white")

        if self.ai_vs_ai:
            self.ai1_turn = random.choice([True, False])
            self.status.config(text="AI 1's turn (X)" if self.ai1_turn else "AI 2's turn (O)")
            self.root.after(500, self.ai_move)
            self.turn = 1 if self.ai1_turn else -1

        else:
            self.current_ai = random.choice([net1, net2])
            self.turn = 1
            self.status.config(text="AI's turn (X)")
            self.root.after(500, self.ai_move)

root = tk.Tk()
game = TicTacToeGame(root)
root.mainloop()
