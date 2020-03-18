from environment import PalleteWorld
from sample_agent.agent import *
from config import *
import logging
import pygame
import os

# import torch
# import torchvision.utils as vutils
# import numpy as np
# import torchvision.models as models
# from torchvision import datasets
from tensorboardX import SummaryWriter


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

writer = SummaryWriter()


if __name__ == '__main__':

    env = PalleteWorld()
    agent = PalletAgent()

    if ENV.RENDER:
        clock = pygame.time.Clock()

    total_step = 0

    for e in range(ENV.N_EPISODES):

        state = env.reset()
        d = False
        step = 0

        while True:

            # while True:
            #     action = agent.get_action(state)
            #     if action not in env.previous_actions:
            #         break
            action = agent.get_action(state)

            a = Action(bin_index=action[0], priority=action[1], rotate=action[2])
            next_state, reward, done, _ = env.step(a)

            # logging.debug('step : {}, action : {}, reward : {}, next state : {}'.format(step, action, reward, next_state))
            logging.debug(
                '{} step : action : {}, reward : {}'.format(step, action, reward))


            if ENV.RENDER:
                env.render()
                clock.tick(ENV.TICK_INTERVAL)

            if not done:
                state = next_state
            else:
                next_state = None

            agent.save_sample(state, action, next_state, reward)

            state = next_state

            agent.optimize_model()

            step += 1
            total_step += 1
            if done or step == ENV.EPISODE_MAX_STEP:
                logging.debug('episode {} done..'.format(e))
                writer.add_scalar('data/final_reward', reward, e)

                break

        if e % AGENT.TARGET_UPDATE_INTERVAL == 0:
            agent.synchronize_model()

    logging.debug('Complete')


# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()