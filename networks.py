import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from config import *


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(ENV.BIN_MAX_COUNT*3, AGENT.EMBEDDING_DIM)
        self.linear2 = nn.Linear(AGENT.EMBEDDING_DIM, AGENT.EMBEDDING_DIM)

        self.conv1 = nn.Conv2d(2, 4, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(8)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
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
