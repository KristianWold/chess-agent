import numpy as np
import random
import torch
import config
from collections import namedtuple, deque
from itertools import count
from torch import nn, optim
from tqdm.notebook import tqdm
from environments import *
from copy import deepcopy
from collections import deque
import random



class ReplayMemory():
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size

        self.states = torch.zeros(capacity, 12, 8, 8, dtype = torch.float32, device=config.device)
        self.actions = torch.zeros(capacity, 1, dtype = torch.int64, device=config.device)
        self.next_states = torch.zeros(capacity, 12, 8, 8, dtype = torch.float32, device=config.device)
        self.mask_legal = torch.zeros(capacity, 64*76, dtype = torch.bool, device=config.device)
        self.rewards = torch.zeros(capacity, 1, dtype = torch.float32, device=config.device)
        self.done = torch.zeros(capacity, dtype = torch.bool, device=config.device)

        self.data = torch.utils.data.TensorDataset(self.states, self.actions, self.next_states, self.mask_legal, self.rewards, self.done)

        self.index = 0
        self.num_samples = 0

    def push(self, transition):
        """Save a transition"""

        state, action, next_state, mask_legal, reward, done = transition
        self.states[self.index] = state
        self.actions[self.index] = action
        self.next_states[self.index] = next_state
        self.mask_legal[self.index] = mask_legal
        self.rewards[self.index] = reward
        self.done[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.num_samples = min(self.num_samples + 1, self.capacity)

    def sample(self):
        idx = torch.randperm(self.num_samples, device=config.device)[:self.batch_size]
        batch = self.data[idx]
        return batch


class Logger:
    def __init__(self, sample_freq=1):
        self.sample_freq = sample_freq

        self.fidelity = []
        self.length = []
        self.solved = []
        self.loss = []

    def __call__(self, fidelity, length, solved, loss):
        self.fidelity.append(fidelity)
        self.length.append(length)
        self.solved.append(solved)
        self.loss.append(loss)


class Model:

    def __init__(self,
                 agent=None,
                 environment=None,
                 mem_capacity=None,
                 batch_size=None,
                 policy_update=None,
                 target_update=None,
                 temp_constants=None,
                 opt_list = None,
                 scaler = None,
                 criterion = nn.SmoothL1Loss(),
                 ):

        self.agent = agent
        self.environment = environment
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.policy_update = policy_update
        self.target_update = target_update

        self.temp_start = temp_constants[0]
        self.temp_end = temp_constants[1]
        self.temp_min = temp_constants[2]
        self.temp_decay = temp_constants[3]

        a = environment.max_num_moves
        self.gamma = 1 - 1/a
        self.gamma = 1
        self.criterion = criterion

        self.counter_episode = 0
        
        self.memory = ReplayMemory(mem_capacity, batch_size)

        self.opt_list = opt_list
        self.scaler = scaler

    #@profile
    def train(self, num_episodes, logger=None):
        counter = 0
        loss = 0
        for i_episode in tqdm(range(num_episodes)):
            board = self.environment.reset()
            state = self.agent.board_to_state([board])
            state_prev = None

            temp_max = self.temp_start + (self.temp_end - self.temp_start) * self.counter_episode/ self.temp_decay
            temp_max = min(self.temp_end, temp_max)

            eps = random.uniform(self.temp_min, temp_max)
            self.counter_episode += 1

            for i in range(self.environment.max_num_moves):
                action = self.agent.select_action(board, eps=eps, greedy=False)
                move = self.agent.action_to_move(action)[0]
                board_next, (reward, done) = self.environment.step(
                    move)
                
                state_next = self.agent.board_to_state([board_next])
                legal_mask = self.agent.get_mask_legal([board_next])

                done_tensor = torch.tensor(done, dtype=torch.bool, device=config.device)

                if state_prev is not None:
                    self.memory.push([state_prev, action_prev, state_next, legal_mask, -reward, done_tensor])

                if done:
                   self.memory.push([state, action, state_next, legal_mask, reward, done_tensor])

                board = board_next
                state_prev = state
                action_prev = action
                state = state_next

                if counter % self.policy_update == 0:
                    loss = self.optimize_agent()

                if counter % self.target_update == 0:
                    self.agent.update_target_net()

                counter += 1

                if done:
                    if reward.item() == 1:
                        print(i_episode, "checkmate!", i, eps)
                    else:
                        print(i_episode, "draw!", i, eps)
                    break
            if i_episode % 100 == 0:
                print(loss)

    @torch.compile
    def compute_loss(self):
        if self.memory.num_samples < self.batch_size:
            return 0, None
        state_batch, action_batch, next_state_batch, mask_legal_batch, reward_batch, done_batch = self.memory.sample()

        if random.random() < 0.5:
            online_net = self.agent.online_net1
            target_net = self.agent.target_net2
            opt = self.opt_list[0]
        else:
            online_net = self.agent.online_net2
            target_net = self.agent.target_net1
            opt = self.opt_list[1]

        Q = online_net(state_batch)
        state_action_values = Q.gather(1, action_batch)

        with torch.no_grad():
            Q_next = online_net(next_state_batch)
            Q_next = Q_next.masked_fill(~mask_legal_batch, -1e-9)
            action_star = Q_next.argmax(1, keepdim=True)

            next_state_values = target_net(next_state_batch).gather(1, action_star)
            next_state_values[done_batch] = 0

            expected_state_action_values = reward_batch + self.gamma*next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)
        return loss, opt

    
    def optimize_agent(self):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, opt = self.compute_loss()
        if loss == 0:
            return 0

        # Optimize the model
        self.scaler.scale(loss).backward()

        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad(set_to_none=True)

        return loss.detach()


    def set_temp(self, temp_constants):
        self.temp_start = temp_constants[0]
        self.temp_end = temp_constants[1]
        self.temp_min = temp_constants[2]
        self.temp_decay = temp_constants[3]


        
