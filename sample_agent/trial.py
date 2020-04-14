import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Convolution(nn.Module):

    def __init__(self):
        super(Convolution, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 30, 3, 1)

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
        self.fc3 = nn.Linear(200, 20)
        
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):

        batch_dim = x.shape[0]
        n_item = x.shape[3]-1

        bin = x[:, :, :, :1]
        item = x[:, :, :, 1:]


        bin = self.conv_bin(bin)
        item = item.permute(0,3,1,2).reshape(-1, 10, 10, 1)
        item = self.conv_item(item)
        item = item.reshape(batch_dim, n_item, -1)
        item = torch.sum(item, axis=1)
        
        concat = torch.cat([item,bin], dim=1)

        # x = x.view(-1, 50)
        x = F.relu(self.fc1(concat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


p = Policy()

t = torch.randn(10,10,10,21)
x = p(t)
print(x.shape)