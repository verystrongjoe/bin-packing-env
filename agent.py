from config import *
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from networks import *
from modules import *


class PalletAgent:

    def __init__(self):
        self.replay_memory = ReplayMemory(capacity=AGENT.REPLAY_MEMORY_SIZE)
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    # return epsilon-greedy based action
    def get_action(self, state):

        s = (
            torch.tensor(state[0], dtype=torch.float).to(device),
            torch.tensor(state[1], dtype=torch.float).to(device)
        )

        if np.random.rand() > AGENT.EPSILON:
            action = np.random.choice(ENV.BIN_MAX_COUNT)
        else:
            state_action = self.policy_net(s[0], s[1])[0]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def save_sample(self, state, action, next_state, reward):
        s = (
            torch.tensor(state[0], dtype=torch.float).to(device),
            torch.tensor(state[1], dtype=torch.float).to(device)
        )

        ns = (
            torch.tensor(next_state[0], dtype=torch.float).to(device),
            torch.tensor(next_state[1], dtype=torch.float).to(device)
        )

        self.replay_memory.push(s, action, ns, reward)

    def optimize_model(self):
        if len(self.replay_memory) < AGENT.BATCH_SIZE:
            return
        transitions = self.replay_memory.sample(AGENT.BATCH_SIZE)
        # https://stackoverflow.com/a/19343/3343043
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states_0 = torch.cat([s[0] for s in batch.next_state if s is not None], dim=0).reshape(AGENT.BATCH_SIZE, ENV.BIN_MAX_COUNT, 3)
        non_final_next_states_1 = torch.cat([s[1] for s in batch.next_state if s is not None], dim=0).reshape(AGENT.BATCH_SIZE, ENV.ROW_COUNT, ENV.COL_COUNT, 2)

        state_batch_0 = torch.cat([s[0] for s in batch.state], dim=0).reshape(AGENT.BATCH_SIZE, ENV.BIN_MAX_COUNT, 3)
        state_batch_1 = torch.cat([s[1] for s in batch.state], dim=0).reshape(AGENT.BATCH_SIZE, ENV.ROW_COUNT, ENV.COL_COUNT, 2)

        action_batch = torch.tensor(batch.action, dtype=torch.float32, device=device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

        # (batch, state-action values)
        state_action_values = self.policy_net(state_batch_0, state_batch_1).gather(1, action_batch.reshape(AGENT.BATCH_SIZE, 1).long())

        next_state_values = torch.zeros(AGENT.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states_0, non_final_next_states_1).max(1)[0].detach()
        expected_state_action_values = (next_state_values * AGENT.GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    def synchronize_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())