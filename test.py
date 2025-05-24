import neat
import pickle

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'config-feedforward.txt'
)

with open('best_genome.pkl', 'rb') as f:
    winner = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(winner, config)

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("\nCurrent Board:")
    for i in range(3):
        row = ' | '.join(symbols[board[j]] for j in range(i * 3, i * 3 + 3))
        print(' ' + row)
        if i < 2:
            print("---+---+---")
    print()

def available_moves(board):
    return [i for i, x in enumerate(board) if x == 0]

def is_winner(board, player):
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    return any(all(board[i] == player for i in line) for line in wins)

def is_draw(board):
    return all(cell != 0 for cell in board)

def get_ai_move(board, net):
    output = net.activate(board)
    valid = available_moves(board)
    move = max(((i, output[i]) for i in valid), key=lambda x: x[1])[0]
    return move

def play_game(net):
    board = [0] * 9
    turn = 1

    print("Welcome to Tic-Tac-Toe! You are O. AI is X.\n")
    print("Positions: ")
    print(" 0 | 1 | 2\n---+---+---\n 3 | 4 | 5\n---+---+---\n 6 | 7 | 8\n")

    while True:
        print_board(board)

        if turn == 1:
            print("AI is thinking...")
            move = get_ai_move(board, net)
            print(f"AI chooses position {move}")
        else:
            try:
                move = int(input("Your move (0–8): "))
                while move not in available_moves(board):
                    move = int(input("Invalid move. Try again (0–8): "))
            except:
                print("Please enter a valid number.")
                continue

        board[move] = turn

        if is_winner(board, turn):
            print_board(board)
            print("AI wins!" if turn == 1 else "You win!")
            break
        elif is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        turn *= -1

play_game(net)
