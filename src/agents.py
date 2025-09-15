import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import numpy as np
import math
import cmath
import config


class DQN(nn.Module):

    def __init__(self,  state_dim, action_dim, num_layers, scale):
        super(DQN, self).__init__()

        self.ln = nn.LayerNorm(state_dim)
        self.layer_up = nn.Linear(state_dim, num_layers[0])
        self.layers_list = nn.ModuleList([nn.Linear(dim1, dim2) for dim1, dim2 in zip(num_layers[:-1], num_layers[1:])])
        self.head = nn.Linear(num_layers[-1], action_dim)
        self.scale = scale

    def forward(self, x):
        x = F.relu(self.layer_up(x/self.scale))
        for layer in self.layers_list:
            x = F.relu(layer(x))
        value = self.head(x)
        return value


class Agent:

    def __init__(self,
                 state_dim=None,
                 action_dim=None,
                 num_layers=None,
                 scale = None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.scale = scale

        self.online_net1 = DQN(state_dim, action_dim, num_layers, scale).to(config.device)
        self.online_net2 = DQN(state_dim, action_dim, num_layers, scale).to(config.device)

        self.target_net1 = DQN(state_dim, action_dim, num_layers, scale).to(config.device)
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net1.eval()

        self.target_net2 = DQN(state_dim, action_dim, num_layers, scale).to(config.device)
        self.target_net2.load_state_dict(self.online_net2.state_dict())
        self.target_net2.eval()

    def forward(self, state):
        Q1 = self.online_net1(state)
        Q2 = self.online_net2(state)
        return (Q1 + Q2) / 2

    def select_action(self, state, temp=0, greedy=True):

        with torch.no_grad():
            Q = self.forward(state)

            if greedy:
                return Q.argmax(1, keepdim=True)      
            else:
                adv = Q - Q.mean(dim=1, keepdim=True)
                logits = adv 
                logits = logits - logits.max(dim=1, keepdim=True).values
                dist = F.softmax(logits/temp, dim=1)
                return torch.multinomial(dist, num_samples=1)

    def update_target_net(self):
        self.target_net1.load_state_dict(self.online_net1.state_dict())
        self.target_net2.load_state_dict(self.online_net2.state_dict())
