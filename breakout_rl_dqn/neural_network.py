import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_neural_network(nn.Module):
    def __init__(self,num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8,4)
        self.conv2 = nn.Conv2d(32, 64, 4,2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)  # This size depends on the output size of your conv layers

        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)

