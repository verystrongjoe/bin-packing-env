import argparse
import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import env_config as env_cfg
from environment import PalleteWorld
from tensorboardX import SummaryWriter

num_frames = 1000
num_steps = 10
num_processes = 10
seed = 0
num_updates = int(num_frames) // num_steps // num_processes
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class MultiEnv:
    def __init__(self, num_env):
        self.envs = []
        for _ in range(num_env):
            self.envs.append(PalleteWorld(n_random_fixed=1))

    def reset(self):
        os = []
        for env in self.envs:
            os.append(env.reset())
        return os

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
            if done:
                env.reset()
        return obs, rewards, dones, infos


envs = MultiEnv(num_env=num_processes)
writer = SummaryWriter()


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


class A2C(nn.Module):

    def __init__(self):
        super(A2C, self).__init__()
        self.conv_item = Convolution()
        self.conv_bin = Convolution()
        self.fc1 = nn.Linear(60, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, env_cfg.ENV.BIN_MAX_COUNT)
        self.fc4 = nn.Linear(200, 1)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, masks, step):
        assert masks.shape == (num_processes, num_steps)

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
        action_probs = self.fc3(x)
        value = self.fc4(x)

        for i, p in enumerate(masks):
            for m in p[:step]:
                action_probs[i][m] = -np.inf

        return F.softmax(x, dim=1), value[1]


policy = A2C()
policy.cuda()
eps = np.finfo(np.float32).eps.item()
optimizer = optim.RMSprop(policy.parameters(), lr=1e-2, eps=eps , alpha=0.99)


def select_action(state, previous_action, step):
    # this is for visual representation.
    boxes = torch.zeros(num_processes, 10, 10, env_cfg.ENV.BIN_MAX_COUNT)  # x, y, box count
    for i, p in enumerate(state):  # box = x,y
        for j, box in enumerate(p[0]):
            boxes[i][-1*box[1]:, 0:box[0], j] = 1.

    state = np.concatenate((
    np.asarray([state[i][1] for i in range(num_steps)])[:,:,:,0:1],
    boxes), axis=-1)

    state = torch.from_numpy(state).float().cuda()
    values, probs = policy(state, previous_action, step)
    ps = probs.cpu().data.numpy()
    actions = []
    action_probs = []
    for p in ps:
        action = np.random.choice(env_cfg.ENV.BIN_MAX_COUNT, 1, p=p)[0]
        actions.append(action)
        action_prob = torch.log(probs[0][action])
        action_probs.append(action_prob)

    return actions, action_probs, values


def main():
    obs_shape = (10, 10, env_cfg.ENV.BIN_MAX_COUNT+1)
    states = torch.zeros(num_steps + 1, num_processes, *obs_shape)
    current_state = torch.zeros(num_processes, *obs_shape)

    state = envs.reset()

    rewards = torch.zeros(num_steps, num_processes, 1)
    value_preds = torch.zeros(num_steps + 1, num_processes, 1)
    returns = torch.zeros(num_steps + 1, num_processes, 1)
    actions = torch.LongTensor(num_steps, num_processes)
    masks = torch.zeros(num_steps, num_processes, 1)
    old_log_probs = torch.zeros(num_steps, num_processes, num_steps)

    episode_rewards = torch.zeros([num_processes, 1])
    final_rewards = torch.zeros([num_processes, 1])

    states = states.cuda()
    current_state = current_state.cuda()
    rewards = rewards.cuda()
    value_preds = value_preds.cuda()
    old_log_probs = old_log_probs.cuda()
    returns = returns.cuda()
    actions = actions.cuda()
    masks = masks.cuda()

    """
    ****************************에피소드 실행 메인**************************************
    """
    for j in range(num_updates):  # total episode

        previous_action = np.zeros((num_processes, num_steps))

        for step in range(num_steps):  # episode step

            # Sample actions
            # value, logits = policy(torch.tensor(states[step]), previous_actions, step)
            # probs = F.softmax(logits)
            # log_probs = F.log_softmax(logits).data
            # actions[step] = probs.multinomial().data  # todo : multinomial sampling

            actions, action_probs, values = select_action(state, previous_action, step)
            log_probs = action_probs

            cpu_actions = actions[step].cpu()
            cpu_actions = cpu_actions.numpy()

            actions = [env_cfg.Action(bin_index=a, priority=1, rotate=0) for a in cpu_actions]

            state, reward, done, info = envs.step(actions)

            for i, a in enumerate(cpu_actions):
                previous_action[i][step] = a

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            np_masks = np.array([0.0 if done_ else 1.0 for done_ in done])  # (process, done)
            n_placed_items = len(info['placed_items'])

            # If done then clean the history of observations.
            pt_masks = torch.from_numpy(np_masks.reshape(np_masks.shape[0], 1, 1, 1)).float()
            pt_masks = pt_masks.cuda()
            current_state *= pt_masks

            states[step + 1].copy_(current_state)
            value_preds[step].copy_(values.data)
            old_log_probs[step].copy_(log_probs)
            rewards[step].copy_(reward)
            masks[step].copy_(torch.from_numpy(np_masks).unsqueeze(1))

            final_rewards *= masks[step].cpu()
            final_rewards += (1 - masks[step].cpu()) * episode_rewards

            episode_rewards *= masks[step].cpu()

            # writer.add_scalar('data/final_reward', ep_reward, i_episode)
            # writer.add_scalar('data/n_placed_items', n_placed_items, i_episode)

if __name__ == '__main__':
    main()
