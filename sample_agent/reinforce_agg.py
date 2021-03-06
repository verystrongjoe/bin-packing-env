import argparse
import gym
from torch.distributions import Categorical
import logging
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import env_config as env_cfg
# env_cfg.ENV.RENDER = True

from environment import PalleteWorld
import pygame
from tensorboardX import SummaryWriter

writer = SummaryWriter()

# logger = logging.getLogger('uk')
# logger.setLevel(logging.DEBUG)


# env.seed(args.seed)
# torch.manual_seed(args.seed)


class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()
        self.conv1 = nn.Conv2d(1, env_cfg.ENV.BIN_MAX_COUNT, 3, 1)
        self.conv2 = nn.Conv2d(env_cfg.ENV.BIN_MAX_COUNT, 30, 3, 1)

    def forward(self, x):
        bin = x.permute(0, 3, 1, 2)
        bin = F.relu(self.conv1(bin))
        bin = F.max_pool2d(bin, 2, 2)
        bin = F.relu(self.conv2(bin))
        bin = F.max_pool2d(bin, 2, 2)
        return bin.flatten(start_dim=1)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.conv_item = Convolution()
        self.conv_bin = Convolution()
        self.fc1 = nn.Linear(60, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, env_cfg.ENV.BIN_MAX_COUNT)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, mask):
        batch_dim = x.shape[0]
        n_item = x.shape[3] - 1

        bin = x[:, :, :, :1]
        item = x[:, :, :, 1:]

        bin = self.conv_bin(bin)
        item = item.permute(0, 3, 1, 2).reshape(-1, 10, 10, 1)
        item = self.conv_item(item)
        item = item.reshape(batch_dim, n_item, -1)
        item = torch.sum(item, axis=1)

        concat = torch.cat([item, bin], dim=1)

        # x = x.view(-1, 50)
        x = F.relu(self.fc1(concat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        for a in mask:
            x[0][a] = -np.inf

        return F.softmax(x, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state, previous_actions):
    # this is for visual representation.
    boxes = torch.zeros(10, 10, env_cfg.ENV.BIN_MAX_COUNT)  # x, y, box count
    for i, box in enumerate(state[0]):  # box = x,y
        boxes[-1*box[1]:, 0:box[0], i] = 1.

    state = np.concatenate((np.expand_dims(state[1][:,:,0], axis=-1), boxes), axis=-1)
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state, previous_actions)

    # m = Categorical(probs)
    # p = probs[0].detach().numpy()
    p = probs[0].data.numpy()
    # p[previous_actions] = -100
    # odds = np.exp(p)
    # act_probs = odds / np.sum(odds)

    action = np.random.choice(env_cfg.ENV.BIN_MAX_COUNT, 1, p=p)[0]
    # policy.saved_log_probs.append(m.log_prob(probs[0][action]))
    policy.saved_log_probs.append(torch.log(probs[0][action]))
    return action


from collections import deque
last_rewards = deque(maxlen=100)


def finish_episode():
    R = 0
    policy_loss = []
    returns = []

    last_rewards.append(policy.rewards[-1])
    # policy.rewards[-1] -= np.mean(last_rewards)

    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main(env):

    env = env
    ep_rewards = []

    if env_cfg.ENV.RENDER:
        clock = pygame.time.Clock()

    for i_episode in count(1):
        state, ep_reward, previous_actions = env.reset(), 0, []
        n_placed_items = 0

        for t in range(0, env_cfg.ENV.BIN_MAX_COUNT):  # Don't infinite loop while learning
            action = select_action(state, previous_actions)
            # print('{} step , {} action'.format(t ,action))
            previous_actions.append(action)
            a = env_cfg.Action(bin_index=action, priority=1, rotate=0)
            state, reward, done, info = env.step(a)

            if env_cfg.ENV.RENDER:
                env.render()
                clock.tick(env_cfg.ENV.TICK_INTERVAL)

            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                # print('episode end')
                # print('len of action : {}, episode actions : {}'.format(len(env.previous_actions), env.previous_actions))
                ep_rewards.append(ep_reward)
                n_placed_items = len(info['placed_items'])
                break

        finish_episode()
        if i_episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}'.format(i_episode, ep_reward))
        writer.add_scalar('data/final_reward', ep_reward, i_episode)
        writer.add_scalar('data/n_placed_items', n_placed_items, i_episode)

        def checkEqual(iterator):
            return len(set(iterator)) <= 1

        if checkEqual(ep_rewards[-20:]) and len(ep_rewards) > 100:
            print("Solved! last reward is {}".format(ep_rewards[-1]))
            return ep_rewards[-1]

if __name__ == '__main__':
    env = PalleteWorld(n_random_fixed=1)
    main()
