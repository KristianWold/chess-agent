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



class ReplayMemory():
    def __init__(self, capacity, batch_size, state_dim):
        self.capacity = capacity
        self.batch_size = batch_size

        self.states = torch.zeros(capacity, state_dim, dtype = torch.float32, device=config.device)
        self.actions = torch.zeros(capacity, 1, dtype = torch.int64, device=config.device)
        self.next_states = torch.zeros(capacity, state_dim, dtype = torch.float32, device=config.device)
        self.rewards = torch.zeros(capacity, 1, dtype = torch.float32, device=config.device)
        self.done = torch.zeros(capacity, dtype = torch.bool, device=config.device)

        self.data = torch.utils.data.TensorDataset(self.states, self.actions, self.next_states, self.rewards, self.done)

        self.index = 0
        self.num_samples = 0

    def push(self, state, action, next_state, reward, done):
        """Save a transition"""
        self.states[self.index] = state
        self.actions[self.index] = action
        self.next_states[self.index] = next_state
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
                 batch_size_min=None,
                 policy_update=None,
                 target_update=None,
                 temp_constants=None,
                 num_val=None,
                 opt_list = None,
                 scaler = None,
                 criterion = nn.SmoothL1Loss(),
                 split_memory = True
                 ):

        self.agent = agent
        self.environment = environment
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.batch_size_min = batch_size_min
        self.policy_update = policy_update
        self.target_update = target_update

        self.temp_start = temp_constants[0]
        self.temp_end = temp_constants[1]
        self.temp_min = temp_constants[2]
        self.temp_decay = temp_constants[3]

        a = environment.max_num_actions
        self.gamma = 1 - 1/a

        self.num_val = num_val
        self.criterion = criterion
        self.split_memory = split_memory

        self.counter_episode = 0

        self.fidelity = environment.fidelity
        self.val_set = [self.environment.data_generator(
            environment, seed = 100000 + i, train=False)[0] for i in range(num_val)]
        
        self.memory_pos = ReplayMemory(mem_capacity, batch_size, self.agent.state_dim)
        self.memory_neg = ReplayMemory(mem_capacity, batch_size, self.agent.state_dim)

        self.opt_list = opt_list
        self.scaler = scaler

    #@profile
    def train(self, num_episodes, logger=None):
        counter = 0
        
        loss = 0
        for i_episode in tqdm(range(num_episodes)):
            state_list = []
            action_list = []
            next_state_list = []
            reward_list = []
            done_list = []
            state, done = self.environment.reset(seed=i_episode)
            temp_max = self.temp_start + (self.temp_end - self.temp_start) * self.counter_episode/ self.temp_decay

            temp = random.uniform(self.temp_min, temp_max)
            self.counter_episode += 1

            for _ in range(self.environment.max_num_actions):
                action = self.agent.select_action(state, temp=temp, greedy=False)
                next_state, reward, done = self.environment.step(
                    action.item())

                done_tensor = torch.tensor(done, dtype=torch.bool, device=config.device)
                state_list.append(state)
                action_list.append(action)
                next_state_list.append(next_state)
                reward_list.append(reward)
                done_list.append(done_tensor)

                if counter % self.policy_update == 0:
                    loss = self.optimize_agent()

                if counter % self.target_update == 0:
                    self.agent.update_target_net()

                counter += 1

                if done:
                    break

                state = next_state
            
            self.push_memory([state_list, action_list, next_state_list, reward_list, done_list], done)

            if (i_episode + 1) % round(logger.sample_freq) == 0:
                self.stats(logger, i_episode, loss, temp)

    @torch.compile
    def compute_loss(self):
        if self.memory_pos.num_samples < self.batch_size_min:
            return 0, None
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.memory_pos.sample()
        
        if self.split_memory and self.memory_neg.num_samples >= self.batch_size_min:
            state_batch2, action_batch2, next_state_batch2, reward_batch2, done_batch2 = self.memory_neg.sample()

            state_batch = torch.cat([state_batch, state_batch2])
            action_batch = torch.cat([action_batch, action_batch2])
            next_state_batch = torch.cat([next_state_batch, next_state_batch2])
            reward_batch = torch.cat([reward_batch, reward_batch2])
            done_batch = torch.cat([done_batch, done_batch2])

        if random.random() < 0.5:
            online_net = self.agent.online_net1
            target_net = self.agent.target_net2
            opt = self.opt_list[0]
        else:
            online_net = self.agent.online_net2
            target_net = self.agent.target_net1
            opt = self.opt_list[1]

        state_action_values = online_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            action_star = online_net(next_state_batch).argmax(1, keepdim=True)
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

    def push_memory(self, transition_list, done):
        if self.split_memory:
            if done:
                for state, action, next_state, reward, done_tensor in zip(*transition_list):
                    self.memory_pos.push(state, action, next_state, reward, done_tensor)
            else:
                for state, action, next_state, reward, done_tensor in zip(*transition_list):
                    self.memory_neg.push(state, action, next_state, reward, done_tensor)
        else:
            for state, action, next_state, reward, done_tensor in zip(*transition_list):
                self.memory_pos.push(state, action, next_state, reward, done_tensor)


    def stats(self, logger, i_episode, loss, temp):
        if loss != 0:
            loss = loss.item()

            avg_fidelity = 0
            avg_length = 0
            avg_solved = 0

            _, _, _, action_count_list, fidelity_list = self.evaluate(self.val_set)
            action_count_list = np.array(action_count_list)
            avg_fidelity = np.mean(fidelity_list)
            avg_length = np.mean(action_count_list[action_count_list < self.environment.max_num_actions])
            avg_solved = np.mean(action_count_list < self.environment.max_num_actions)

            if not logger is None:
                logger(avg_fidelity, avg_length, avg_solved, loss)
            print(
                f"episode: {i_episode+1}, fidelity: {avg_fidelity: .4f}, solved length: {avg_length: .4f}, solved: {avg_solved: .4f}, loss: {loss: .4f}, temp: {temp: .4f}")

    def evaluate(self, input_state_list):
        environment_list = []
        state_list = []
        done_list = []
        U_target_list = []
        U_current_list = []
        action_history_list = []
        action_count_list = []
        for input_state in input_state_list:
            environment_list.append(deepcopy(self.environment))


        for environment, input_state in zip(environment_list, input_state_list):
            state, done = environment.reset(input_state=input_state)
            state_list.append(state)
            done_list.append(done)

        counter = 0
        for _ in range(self.environment.max_num_actions):
            counter += 1
            for i, done in reversed(list(enumerate(done_list))):
                if done:
                    U_target_list.append(environment_list[i].U_target)
                    U_current_list.append(environment_list[i].U_current)
                    action_history_list.append(environment_list[i].action_history)
                    action_count_list.append(environment_list[i].action_count)

                    environment_list.pop(i)
                    state_list.pop(i)
                    done_list.pop(i)

            if len(done_list)==0:
                break

            action_list = self.agent.select_action(torch.cat(state_list)).to("cpu").numpy()

            state_list = []
            done_list = []
            for environment, action in zip(environment_list, action_list):
                next_state, reward, done = environment.step(action[0])
                state_list.append(next_state)
                done_list.append(done)

        for environment in environment_list:
            U_target_list.append(environment.U_target)
            U_current_list.append(environment.U_current)
            action_history_list.append(environment.action_history)
            action_count_list.append(environment.action_count)

        action_count_list = np.array(action_count_list)

        fidelity_list = []
        for U_target, U_current in zip(U_target_list, U_current_list):
            fidelity_list.append(self.fidelity(U_target, U_current))
        fidelity_list = np.array(fidelity_list)

        return U_target_list, U_current_list, action_history_list, action_count_list, fidelity_list


    def set_data_generator(self, data_generator):
        self.environment.data_generator = data_generator
        self.val_set = [self.environment.data_generator(
            self.environment, seed = 1000000 + i, train=False)[0] for i in range(self.num_val)]

    def set_temp(self, temp_constants):
        self.temp_start = temp_constants[0]
        self.temp_end = temp_constants[1]
        self.temp_min = temp_constants[2]
        self.temp_decay = temp_constants[3]


        
