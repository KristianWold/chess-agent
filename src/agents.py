import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import numpy as np
import math
import cmath
import config


class NN(nn.Module):

    def __init__(self, state_dim, action_dim, num_layers, scale):
        super(NN, self).__init__()

        self.conv1 = nn.Conv2d(13, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(128*8*8, 2048)
        self.fc2 = nn.Linear(2048, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Agent:

    def __init__(self,
                 board_to_state=None,
                 action_dim=None,
                 num_layers=None,
                 scale = None):

        self.board_to_state = board_to_state
        self.state_dim = board_to_state.state_dim
        self.action_dim = action_dim
        
        self.scale = scale

        self.online_net1 = NN(self.state_dim, action_dim, num_layers, scale).to(config.device)
        self.online_net2 = NN(self.state_dim, action_dim, num_layers, scale).to(config.device)

        self.target_net1 = NN(self.state_dim, action_dim, num_layers, scale).to(config.device)
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net1.eval()

        self.target_net2 = NN(self.state_dim, action_dim, num_layers, scale).to(config.device)
        self.target_net2.load_state_dict(self.online_net2.state_dict())
        self.target_net2.eval()

    def forward(self, board):
        state = self.board_to_state(board)
        Q1 = self.online_net1(state)
        Q2 = self.online_net2(state)
        return (Q1 + Q2) / 2
    
    def apply_rules(self, logits, board):
        return logits

    def select_action(self, state, temp=0, greedy=True):

        with torch.no_grad():
            Q = self.forward(state)
            adv = Q - Q.mean(dim=1, keepdim=True)
            logits = adv - adv.max(dim=1, keepdim=True).values
            
            logits = self.apply_rules(Q, state)

            if greedy:
                return logits.argmax(1, keepdim=True)      
            else:
                dist = F.softmax(logits/temp, dim=1)
                return torch.multinomial(dist, num_samples=1)
            

    def update_target_net(self):
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net2.load_state_dict(self.online_net2.state_dict())
