import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from env_config import *


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DQN(nn.Module):
    """
    Ranked Reward처럼 채널 bin의 정보들이 채워져 온다고 하고 문제 풀자
    BIN 10 x 10
    ITEM 5 (size : 0~5 x 0~5)
    BIN ROW * BIN HEIGHT * (BIN STATE, BIN INFO)
    """
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DQN(nn.Module):

    def __init__(self, ):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(ENV.BIN_MAX_COUNT*3, AGENT.EMBEDDING_DIM)
        self.linear2 = nn.Linear(AGENT.EMBEDDING_DIM, AGENT.EMBEDDING_DIM)

        self.conv1 = nn.Conv2d(2, 4, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(8)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(ENV.ROW_COUNT, 5, 2), 3, 2), 2, 2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(ENV.COL_COUNT, 5, 2), 3, 2), 2, 2)

        linear_input_size = convw * convh * 8 + AGENT.EMBEDDING_DIM

        self.linear3 = nn.Linear(linear_input_size, ENV.BIN_MAX_COUNT)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, bin_info, pallet_info):

        assert type(bin_info) == torch.Tensor
        assert type(pallet_info) == torch.Tensor

        if len(bin_info.shape) != 3:
            bin_info = bin_info.flatten()
            bin_info.unsqueeze_(0)
        else:
            bin_info = bin_info.reshape(AGENT.BATCH_SIZE, ENV.BIN_MAX_COUNT * 3)

        if len(pallet_info.shape) != 4:
            pallet_info = pallet_info.unsqueeze(0)
        pallet_info = pallet_info.permute(0, 3, 1, 2)

        x1 = F.relu(self.linear1(bin_info))
        x1 = F.relu(self.linear2(x1))

        x2 = F.relu(self.bn1(self.conv1(pallet_info)))
        x2 = F.relu(self.bn2(self.conv2(x2)))
        x2 = F.relu(self.bn3(self.conv3(x2)))

        x2 = torch.flatten(x2, start_dim=1)

        x = torch.cat([x1, x2], dim=1)
        y = self.linear3(x)

        return y


class BDQN(nn.Module):

    def __init__(self, observation, action_dim, n):
        super(BDQN, self).__init__()
        self.action_dim = action_dim
        self.n = n
        self.model = nn.Sequential(nn.Linear(observation, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU())
        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(action_dim)])

    def forward(self, x):
        out = self.model(x)
        v = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim=1)  # (batch, stacked_advs)
        q_val = v.unsqueeze(2) + advs - advs.mean(2, keepdim=True)


class BranchingDQN(nn.Module):

    def __init__(self, observation, action_dim, action_n_list):
        """
        :param observation: ENV.BIN_MAX_COUNT*3 (1d) + (ENV.ROW_COUNT, ENV.COL_COUNT, 2) (2d)
        :param action_dim:
        :param action_n_list:
        """
        super(BranchingDQN, self).__init__()

        self.action_dim = action_dim
        self.action_n_list = action_n_list

        self.conv1d_1 = nn.Conv1d(ENV.BIN_N_STATE, 4, kernel_size=4)
        self.conv1d_2 = nn.Conv1d(4, 2, kernel_size=2)
        self.conv1d_3 = nn.Conv1d(2, 1, kernel_size=1)

        self.linear1 = nn.Linear(16, AGENT.EMBEDDING_DIM)
        self.linear2 = nn.Linear(AGENT.EMBEDDING_DIM, AGENT.EMBEDDING_DIM)

        self.conv1 = nn.Conv2d(2, 4, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(8)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(ENV.ROW_COUNT, 4, 2), 2, 1)
        convh = conv2d_size_out(conv2d_size_out(ENV.COL_COUNT, 4, 2), 2, 1)

        linear_input_size = convw * convh * 8 + AGENT.EMBEDDING_DIM

        self.shared = nn.Linear(linear_input_size, AGENT.EMBEDDING_DIM)
        self.v_head = nn.Linear(AGENT.EMBEDDING_DIM, 1)

        # self.adv_heads = nn.ModuleList([nn.Linear(AGENT.EMBEDDING_DIM, action_n_list[a]) for a in range(action_dim)])

        self.adv_heads = []
        for a in range(action_dim):
            self.adv_heads.append(nn.Linear(AGENT.EMBEDDING_DIM, action_n_list[a]).to(device))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, bin_info, pallet_info):

        assert type(bin_info) == torch.Tensor
        assert type(pallet_info) == torch.Tensor

        # w/o batch size
        if len(bin_info.shape) != 3:
            assert bin_info.shape == (ENV.BIN_MAX_COUNT, ENV.BIN_N_STATE)
            # bin_info = bin_info.flatten()
            bin_info.unsqueeze_(0)
        else:
            assert bin_info.shape[1:] == (ENV.BIN_MAX_COUNT, ENV.BIN_N_STATE)
            # bin_info = bin_info.reshape(-1, ENV.BIN_MAX_COUNT * ENV.BIN_N_STATE)
            pass
        bin_info = bin_info.permute(0,2,1)

        bin_info = self.conv1d_1(bin_info)
        bin_info = self.conv1d_2(bin_info)
        bin_info = self.conv1d_3(bin_info)
        bin_info = bin_info.squeeze(-2)

        if len(pallet_info.shape) != 4:
            pallet_info = pallet_info.unsqueeze(0)

        pallet_info = pallet_info.permute(0, 3, 1, 2)

        x1 = F.relu(self.linear1(bin_info))  # 1d
        x1 = F.relu(self.linear2(x1))

        x2 = F.relu(self.bn1(self.conv1(pallet_info)))  # 2d
        x2 = F.relu(self.bn2(self.conv2(x2)))

        x2 = torch.flatten(x2, start_dim=1)

        x = torch.cat([x1, x2], dim=1)
        shared = self.shared(x)

        v = self.v_head(shared)  # (batch, value)
        self.action_values = []

        for i in range(self.action_dim):
            self.action_values.append(self.adv_heads[i](shared))

        # (batch, action_index, action_values)
        q_v = []
        for i in range(self.action_dim):
            q_v.append(v + self.action_values[i] - self.action_values[i].mean(1, keepdim=True))  # Eq 1

        return q_v


if __name__ == '__main__':
    # b = BDQN(5,4,6) # o = 5, a = 4, n = 6
    # b(torch.randn(10, 5))

    b = BranchingDQN(ENV.BIN_MAX_COUNT * ENV.BIN_N_STATE, 2, [10, 2]).to(device)

    l = b(
            torch.randn(10, ENV.BIN_MAX_COUNT, ENV.BIN_N_STATE).to(device),
        torch.randn(10, ENV.ROW_COUNT, ENV.COL_COUNT, ENV.BIN_N_STATE).to(device)
    )
