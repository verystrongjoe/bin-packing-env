import math
import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

env = gym.make('CartPole-v0')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    p = probs[0].data.numpy()
    action = np.random.choice(2, 1, p=p)[0]
    policy.saved_log_probs.append(torch.log(probs[0][action]))
    return action


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


for i_episode in count(1):
    state, ep_reward, previous_actions = env.reset(), 0, []
    done = False

    while not done:
        action = select_action(state)
        state, reward, done, info = env.step(action)
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    finish_episode()
    if i_episode % 10 == 0:
        print('Episode {}\tLast reward: {:.2f}'.format(i_episode, ep_reward))

