"""
https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188


http://www.arxiv-vanity.com/papers/1807.01672/
https://arxiv.org/pdf/1708.05930.pdf


MCTS
"""

import numpy as np
np.set_printoptions(suppress=True)
from shutil import copyfile
import random
from importlib import reload
from game import Game, GameState
from agent import Agent
from memory import Memory
from model import GenModel
from funcs import playMatches, playMatchesBetweenVersions
import pickle
from settings import run_archive_folder,run_folder
import loggers as logging
import torch
import os
import mcts_config as config
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

INITIAL_RUN_NUMBER = None
INITIAL_MODEL_VERSION = None
INITIAL_MEMORY_VERSION = None

env = Game()

# If loading an existing neural network, copy the config file to root
if INITIAL_RUN_NUMBER != None:
    copyfile(run_archive_folder + env.name + '/run' + str(INITIAL_RUN_NUMBER).zfill(4) + '/mcts_config.py', './mcts_config.py')

# Load memories if necessary
if INITIAL_MEMORY_VERSION == None:
    memory = Memory(config.MEMORY_SIZE)
else:
    print('LOADING MEMORY VERSION ' + str(INITIAL_MEMORY_VERSION) + '...')
    memory = pickle.load(open(run_archive_folder + env.name + '/run' + str(INITIAL_RUN_NUMBER).zfill(4) + "/memory/memory" + str(INITIAL_MEMORY_VERSION).zfill(4) + ".p",   "rb"))

# Load model if necessary and create an untrained neural network objects from the config file
current_NN = GenModel(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape, env.action_size)
best_NN = GenModel(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape, env.action_size)

# If loading an existing neural network, set the weights from that model
if INITIAL_MODEL_VERSION != None:
    best_player_version = INITIAL_MODEL_VERSION
    print('LOADING MODEL VERSION ' + str(INITIAL_MODEL_VERSION) + '...')
    m_tmp = best_NN.read(env.name, INITIAL_RUN_NUMBER, best_player_version)
    current_NN.model.load_state_dict(m_tmp.state_dict())
    best_NN.model.load_state_dict(m_tmp.state_dict())

# otherwise just ensure the weights on the two players are the same
else:
    best_player_version = 0
    best_NN.model.load_state_dict(current_NN.model.state_dict())

# copy the config file to the run folder
copyfile('./config.py', run_folder + 'config.py')
# todo : uncomment plot_model after fixing bug
# plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes = True)

# Create the players
current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)
#user_player = User('player1', env.state_size, env.action_size)
iteration = 0

while 1:

    iteration += 1
    reload(logging)
    reload(config)

    print('ITERATION NUMBER ' + str(iteration))

    logging.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
    print('BEST PLAYER VERSION ' + str(best_player_version))

    # self play
    print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, logging.logger_main,
                                  turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=memory)
    print('\n')

    memory.clear_stmemory() # todo : what does stmemory stand for?

    if len(memory.ltmemory) >= config.MEMORY_SIZE:

        print('Retraining......')
        current_player.replay(memory.ltmemory)

        if iteration % 5 == 0:
            pickle.dump(memory, open(run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb"))

        logging.logger_memory.info('NEW MEMORIES')

        memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))

        for s in memory_samp:
            current_value, current_probs, _ = current_player.get_preds(s['state'])
            best_value, best_probs, _ = best_player.get_preds(s['state'])

            logging.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
            logging.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
            logging.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
            logging.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']])
            logging.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in current_probs])
            logging.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in best_probs])
            logging.logger_memory.info('ID: %s', s['state'].id)
            logging.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convert_to_model_input(s['state']))

            s['state'].render(logging.logger_memory)

        print('TOURNAMENT...')
        scores, _, points, sp_scores = playMatches(best_player, current_player, config.EVAL_EPISODES, logging.logger_tourney,
                                                   turns_until_tau0=0, memory=None)
        print('\nSCORES')
        print(scores)
        print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
        print(sp_scores)
        # print(points)

        print('\n\n')

        if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            best_player_version = best_player_version + 1
            best_NN.model.load_state_dict(current_NN.model.state_dict())
            torch.save(best_NN.model.state_dict(), run_folder + 'models/version' + "{0:0>4}".format(best_player_version) + '.h5')

    else:
        print('MEMORY SIZE: ' + str(len(memory.ltmemory)))

