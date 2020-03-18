from modules import *
import logging

class PalletAgent:

    def __init__(self):
        logging.debug('random agent initialized..')

    def get_action(self, state):
        return np.random.choice(ENV.BIN_MAX_COUNT)

    # 메모리에 샘플을 추가
    def save_sample(self, state, action, next_state, reward):
        logging.debug('random agent ignored saving sample..')

    def optimize_model(self):
        logging.debug('random agent ignored optimizing model..')

    def synchronize_model(self):
        logging.debug('random agent ignored synchronizing model..')