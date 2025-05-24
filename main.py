import neat
import os
import pickle
from tictactoe import TicTacToe
import random

def smart_opponent_move(game, player):
    opponent = 'X' if player == 'O' else 'O'
    for move in game.available_moves():
        game_copy = game.clone()
        game_copy.make_move(move, player)
        if game_copy.current_winner == player:
            return move
    for move in game.available_moves():
        game_copy = game.clone()
        game_copy.make_move(move, opponent)
        if game_copy.current_winner == opponent:
            return move
    return random.choice(game.available_moves())

def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    for _ in range(50):  # more games
        game = TicTacToe()
        player = 'X'
        invalid_move = False
        while game.empty_squares():
            board_input = [1 if x == 'X' else -1 if x == 'O' else 0 for x in game.board]
            output = net.activate(board_input)
            move = output.index(max(output))
            if move not in game.available_moves():
                fitness -= 5  # reduce penalty a bit
                invalid_move = True
                break
            fitness += 0.1  # small reward for valid move
            game.make_move(move, player)
            if game.current_winner == 'X':
                fitness += 30  # bigger reward for win
                break
            elif game.current_winner == 'O':
                fitness -= 20  # bigger penalty for loss
                break

            if not game.available_moves():
                break

            move_opp = smart_opponent_move(game, 'O')
            game.make_move(move_opp, 'O')
            if game.current_winner == 'O':
                fitness -= 20
                break
        if not invalid_move and not game.current_winner:
            fitness += 10  # reward draw higher
    return fitness

def eval_genomes(genomes, config):
    for _, genome in genomes:
        genome.fitness = evaluate_genome(genome, config)

def train_one_population(config_path, save_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(eval_genomes, 50)
    with open(save_path, 'wb') as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    path = os.path.dirname(__file__)
    config_file = os.path.join(path, 'config-feedforward.txt')

    train_one_population(config_file, 'best_ai1.pkl')
    train_one_population(config_file, 'best_ai2.pkl')
