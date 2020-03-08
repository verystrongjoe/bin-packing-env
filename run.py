from environment import PalleteWorld
# from agent import *
# from random_agent import *
from agent import *
from config import *
import logging

if ENV.RENDER:
    import pygame

if __name__ == '__main__':

    env = PalleteWorld()
    agent = PalletAgent()

    if ENV.RENDER:
        clock = pygame.time.Clock()

    for e in range(ENV.N_EPISODES):

        state = env.reset()
        d = False
        step = 0

        while True:

            while True:
                action = agent.get_action(state)
                if action not in env.previous_actions:
                    break

            logging.debug('trying {} step with {} action'.format(step, action))

            a = Action(bin_index=action, priority=np.random.choice(2), rotate=1)
            next_state, reward, done, _ = env.step(a)

            if ENV.RENDER:
                env.render()
                clock.tick(ENV.TICK_INTERVAL)

            if not done:
                state = next_state
            else:
                next_state = None

            # Store the transition in memory
            agent.save_sample(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_model()

            step += 1
            if done or step == ENV.EPISODE_MAX_STEP:
                logging.debug('episode {} done..'.format(e))
                break

        #Update the target network, copying all weights and biases in DQN
        if e % AGENT.TARGET_UPDATE_INTERVAL == 0:
            agent.synchronize_model()

    logging.debug('Complete')
