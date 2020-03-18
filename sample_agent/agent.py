"""
Action Branching Architectures for Deep Reinforcement Learning
https://arxiv.org/pdf/1711.08946.pdf


if there is other continuous action, we can consider a idea below.
Parametrized Deep Q-Networks Learning: Reinforcement Learning with Discrete-Continuous Hybrid Action Space
https://arxiv.org/pdf/1810.06394.pdf
"""
from sample_agent.networks import *
from modules import *
from config import *

class PalletAgent:

    def __init__(self):
        self.replay_memory = ReplayMemory(capacity=AGENT.REPLAY_MEMORY_SIZE)

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)

        self.policy_net = BranchingDQN(ENV.BIN_MAX_COUNT * 3, 3, [ENV.BIN_MAX_COUNT, 2, 2]).to(device)
        self.target_net = BranchingDQN(ENV.BIN_MAX_COUNT * 3, 3, [ENV.BIN_MAX_COUNT, 2, 2]).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    # return epsilon-greedy based action
    def get_action(self, state):
        s = (
            torch.tensor(state[0], dtype=torch.float).to(device),
            torch.tensor(state[1], dtype=torch.float).to(device)
        )

        action = [0, 0, 0]
        if np.random.rand() > AGENT.EPSILON:
            action[0] = np.random.randint(0, ENV.ACTION_SPACE[0])
            action[1] = np.random.randint(0, ENV.ACTION_SPACE[1])
            action[2] = np.random.randint(0, ENV.ACTION_SPACE[2])
        else:
            list_state_action = self.policy_net(s[0], s[1])
            action = [self.arg_max(state_action.squeeze(0)) for state_action in list_state_action]
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

        if next_state is not None:
            ns = (
                torch.tensor(next_state[0], dtype=torch.float).to(device),
                torch.tensor(next_state[1], dtype=torch.float).to(device)
            )
        else:
            ns = None

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
        non_final_next_states_0 = torch.cat([s[0] for s in batch.next_state if s is not None], dim=0).reshape(-1, ENV.BIN_MAX_COUNT, 3)
        non_final_next_states_1 = torch.cat([s[1] for s in batch.next_state if s is not None], dim=0).reshape(-1, ENV.ROW_COUNT, ENV.COL_COUNT, 2)

        state_batch_0 = torch.cat([s[0] for s in batch.state], dim=0).reshape(AGENT.BATCH_SIZE, ENV.BIN_MAX_COUNT, 3)
        state_batch_1 = torch.cat([s[1] for s in batch.state], dim=0).reshape(AGENT.BATCH_SIZE, ENV.ROW_COUNT, ENV.COL_COUNT, 2)

        action_batch = torch.tensor(batch.action, dtype=torch.float32, device=device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

        # (batch, state-action values)
        list_of_state_action_values = self.policy_net(state_batch_0, state_batch_1)
        state_action_values = []
        for i in range(ENV.ACTION_SIZE):
            state_action_values.append(
                list_of_state_action_values[i].gather(1, action_batch[:,i].reshape(AGENT.BATCH_SIZE).unsqueeze(-1).long())
            )

        next_state_values = [
            torch.zeros(AGENT.BATCH_SIZE, device=device),
            torch.zeros(AGENT.BATCH_SIZE, device=device),
            torch.zeros(AGENT.BATCH_SIZE, device=device)
        ]

        for i in range(ENV.ACTION_SIZE):
            # action_index, non final mask over batch
            next_state_values[i][non_final_mask] = self.target_net(non_final_next_states_0, non_final_next_states_1)[i].max(1)[0].detach()

        # average operator for reduction
        # avg_next_state_values = torch.mean(
        #     torch.stack(
        #         [next_state_values[a].max(1) for a in range(ENV.ACTION_SIZE)], dim=-1), dim=-1)
        # avg_next_state_values = next_state_values[a].max(1) for a in range(ENV.ACTION_SIZE)

        expected_state_action_values = []
        for i in range(ENV.ACTION_SIZE):
            expected_state_action_values.append((next_state_values[i] * AGENT.GAMMA) + reward_batch)

        loss = torch.tensor(0.).to(device)
        for i in range(ENV.ACTION_SIZE):
            loss += F.smooth_l1_loss(state_action_values[i], expected_state_action_values[i].unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    def synchronize_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())