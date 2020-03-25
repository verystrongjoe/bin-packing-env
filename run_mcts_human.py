# The following panels are not involved in the learning process
### Play matches between versions (use -1 for human player)
# %%
from game import Game
from funcs import playMatchesBetweenVersions
import loggers as lg
from game import Game, GameState
from model import GenModel
import mcts_config as config
import numpy as np
from agent import Agent
from settings import run_archive_folder, run_folder
import loggers as logging

env = Game()

# Load model if necessary and create an untrained neural network objects from the config file
current_NN = GenModel(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape, env.action_size)
best_NN = GenModel(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape, env.action_size)

current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)

env = Game()
playMatchesBetweenVersions(env, 1, 1, 1, 10, logging.logger_tourney, 0)

# Pass a particular game state through the neural network (setup below for Connect4)
gs = GameState(np.array([
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0
]), 1)

preds = current_player.get_preds(gs)

print(preds)

### See the layers of the current neural network
current_player.model.viewLayers()

### Output a diagram of the neural network architecture
from keras.utils import plot_model
plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes=True)

