import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from functools import reduce
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        if self.position%1000 == 0:
            print(f"Memory saving at index {self.position}")

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size,    so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h))))
        linear_input_size = convw * convh * 128
        self.head_1 = nn.Linear(linear_input_size, 16)
        self.head_2 = nn.Linear(16, 5)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.sigmoid(x)
        x = self.head_1(x.view(x.size(0), -1))
        x = self.head_2(x)
        return x


class Player(object):
    def __init__(self, env, policy_net, target_net, optimizer, scheduler, memory, fake_memory, state=None):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.memory = memory
        self.fake_memory = fake_memory
        self.state = state

        self.consecutive_noreward = 0
        self.total_reward = 0
        self.screen = None
        self.screen_tensor = None


class DQNUser(nn.Module):
    def __init__(self, h, w, outputs, ksize, n_layers):
        super().__init__()
        self.n_layers = n_layers

        n_filters = 16
        self.conv = [nn.Conv2d(3, n_filters, kernel_size=ksize, stride=2)]
        self.bn = [nn.BatchNorm2d(n_filters)]

        for i in range(1, n_layers):
            in_filters = n_filters * 2**(i-1)
            out_filters = n_filters * 2**i
            self.conv.append(nn.Conv2d(in_filters, out_filters,
                                       kernel_size=ksize, stride=2))
            self.bn.append(nn.BatchNorm2d(out_filters))

        self.conv = nn.ModuleList(self.conv)
        self.bn = nn.ModuleList(self.bn)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size,    so compute it.
        def conv2d_size_out(size, kernel_size=ksize, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = reduce(lambda convw, _: conv2d_size_out(convw), range(n_layers), w)
        convh = reduce(lambda convh, _: conv2d_size_out(convh), range(n_layers), h)
        linear_input_size = convw * convh * n_filters * 2**(n_layers-1)
        self.head_1 = nn.Linear(linear_input_size, 16)
        self.head_2 = nn.Linear(16, outputs)

    def forward(self, x):
        for i, (conv, bn) in enumerate(zip(self.conv, self.bn)):
            if i == self.n_layers-1:
                x = bn(conv(x))
            else:
                x = F.relu(bn(conv(x)))

        x = torch.sigmoid(x)
        x = self.head_1(x.view(x.size(0), -1))
        x = self.head_2(x)
        return x
