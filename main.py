import neat
import os
import random
import pickle
from tictactoe import TicTacToe

def evaluate_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0

        for _ in range(50):
            game = TicTacToe()
            player = 'X'
            invalid_move = False
            while game.empty_squares():
                if player == 'X':
                    board_input = [1 if x == 'X' else -1 if x == 'O' else 0 for x in game.board]
                    output = net.activate(board_input)
                    move = output.index(max(output))
                    if move not in game.available_moves():
                        fitness -= 15
                        invalid_move = True
                        break
                    game.make_move(move, player)
                else:
                    move = smart_opponent_move(game, 'O')
                    game.make_move(move, player)

                if game.current_winner == 'X':
                    fitness += 20
                    break
                elif game.current_winner == 'O':
                    fitness -= 10
                    break
                player = 'O' if player == 'X' else 'X'
            if not invalid_move and not game.current_winner:
                fitness += 5 
        genome.fitness = fitness
    
def smart_opponent_move(game, player):
    opponent = 'X' if player == 'O' else 'O'
    # First try to win
    for move in game.available_moves():
        game_copy = game.clone()
        game_copy.make_move(move, player)
        if game_copy.current_winner == player:
            return move
    # Then try to block opponent
    for move in game.available_moves():
        game_copy = game.clone()
        game_copy.make_move(move, opponent)
        if game_copy.current_winner == opponent:
            return move
    # Else random
    return random.choice(game.available_moves())
def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(evaluate_genomes, 50)
    print('\n Best genome:\n{!s}'.format(winner))
    
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    
if __name__ == '__main__':
    path = os.path.dirname(__file__)
    config_file = os.path.join(path, 'config-feedforward.txt')
    run(config_file)