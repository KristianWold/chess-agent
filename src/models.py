from mimetypes import init
import numpy as np
import random
import torch
import config
from collections import namedtuple, deque
from itertools import count
from torch import nn, optim
from environments import *
from copy import deepcopy
from collections import deque
import random
from IPython.display import display, update_display

from tqdm.notebook import tqdm


class ReplayMemory():
    def __init__(self, capacity, batch_size, init_mem = True):
        self.capacity = capacity
        self.batch_size = batch_size

        self.states = torch.zeros(capacity, 20, 8, 8, dtype = torch.float32) if init_mem else None
        self.actions = torch.zeros(capacity, 1, dtype = torch.int64) if init_mem else None
        self.next_states = torch.zeros(capacity, 20, 8, 8, dtype = torch.float32) if init_mem else None
        self.mask_legal = torch.zeros(capacity, 64*76, dtype = torch.bool) if init_mem else None
        self.rewards = torch.zeros(capacity, 1, dtype = torch.float32) if init_mem else None
        self.done = torch.zeros(capacity, 1, dtype = torch.float32) if init_mem else None

        if init_mem:
            self.init()

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
        idx = torch.randperm(self.num_samples)[:self.batch_size]
        batch = [b.to(config.device) for b in self.data[idx]]
        return batch
    
    def init(self):
        self.data = torch.utils.data.TensorDataset(self.states, self.actions, self.next_states, self.mask_legal, self.rewards, self.done)

class TemperatureScaler:
    def __init__(self, temp_start, temp_end, temp_min, episode_decay, transition_decay):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_min = temp_min
        self.episode_decay = episode_decay
        self.transition_decay = transition_decay
        self.counter_episode = -1
        self.step_episode()

        self.counter_transition = 0

    def step_episode(self):
        self.counter_episode += 1
        self.counter_transition = 0
        self.transition_decay_current = random.uniform(self.transition_decay, 1.0)

        temp_max = self.temp_start + (self.temp_end - self.temp_start) * self.counter_episode/ self.episode_decay
        temp_max = max(self.temp_end, temp_max)

        self.temp_current = random.uniform(self.temp_min, temp_max)

    def step_transition(self):
        self.counter_transition += 1

    def get_temp(self):
        temp = self.temp_current*self.transition_decay_current ** self.counter_transition
        return max(self.temp_min, temp)


class Model:

    def __init__(self,
                 agent=None,
                 environment=None,
                 mem_capacity=None,
                 init_mem = True,
                 batch_size=None,
                 num_warmup=None,
                 policy_update=None,
                 tau = None,
                 temp_scaler=None,
                 opt_list = None,
                 scaler = None,
                 criterion = nn.SmoothL1Loss(),
                 ):

        self.agent = agent
        self.environment = environment
        self.mem_capacity = mem_capacity
        self.batch_size = batch_size
        self.num_warmup = num_warmup
        self.policy_update = policy_update
        self.tau = tau
        self.temp_scaler = temp_scaler

        a = environment.max_num_moves
        self.gamma = 1 - 1/a
        self.criterion = criterion
        
        self.counter_episode = 0
        self.memory = ReplayMemory(mem_capacity, batch_size, init_mem=init_mem)

        self.opt_list = opt_list
        self.scaler = scaler

    #@profile
    def train(self, num_episodes, depth=1, breadth=1, evaluate_agents=None, freq=1000):
        counter = 0
        loss = 0

        for i_episode in tqdm(range(num_episodes)):
            if (i_episode +1) % freq == 0:
                self.stats(evaluate_agents, loss)

            board = self.environment.reset()
            state = self.agent.board_to_state(board)
            self.counter_episode += 1

            done = False
            while not done:
                temp = self.temp_scaler.get_temp()
                _, action = self.agent.q_expand(self.environment,
                                                 temp=temp,
                                                    depth=depth,
                                                    breadth=breadth
                                                )

                move = self.agent.action_to_move(action)
                board_next, (reward, done) = self.environment.step(
                    move) # step also flips the board/state

                state_next = self.agent.board_to_state(board_next)
                mask_legal = self.agent.get_mask_legal(self.environment.get_legal_moves())

                done_tensor = torch.tensor(done, dtype=torch.float32, device=config.device).view(1,1)
                
                self.memory.push([state, action, state_next, mask_legal, reward, done_tensor])
                state = state_next

                if counter % self.policy_update == 0:
                    loss = self.optimize_agent()

                counter += 1
                self.temp_scaler.step_transition()

            self.temp_scaler.step_episode()
                

    @torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
    def compute_loss(self, 
                     online_net, 
                     target_net, 
                     state_batch, 
                     action_batch, 
                     next_state_batch, 
                     mask_legal_batch, 
                     reward_batch, 
                     done_batch):

        Q = online_net(state_batch)
        state_action_values = Q.gather(1, action_batch)

        with torch.no_grad():
            Q_next = online_net(next_state_batch)
            Q_next = Q_next.masked_fill(~mask_legal_batch, -1e4)
            action_star = Q_next.argmax(1, keepdim=True)
        
            next_state_values = target_net(next_state_batch).gather(1, action_star)
            next_state_values = (1 - done_batch)*next_state_values

            # negative next_state_values since chess is zero-sum
            expected_state_action_values = reward_batch - self.gamma*next_state_values

        loss = self.criterion(state_action_values, expected_state_action_values)
        return loss

    
    def optimize_agent(self):

        if self.memory.num_samples < self.num_warmup:
            return None

        if random.random() < 0.5:
            online_net = self.agent.online_net1
            target_net = self.agent.target_net2
            opt = self.opt_list[0]
        else:
            online_net = self.agent.online_net2
            target_net = self.agent.target_net1
            opt = self.opt_list[1]
        
        state_batch, action_batch, next_state_batch, mask_legal_batch, reward_batch, done_batch = self.memory.sample()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = self.compute_loss(online_net, 
                                     target_net, 
                                     state_batch, 
                                     action_batch, 
                                     next_state_batch, 
                                     mask_legal_batch, 
                                     reward_batch, 
                                     done_batch)

        # Optimize the model
        self.scaler.scale(loss).backward()
       
        self.scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g['params'] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad(set_to_none=True)

        self.soft_update(target_net, online_net, self.tau)

        return loss.detach()
    
    def soft_update(self, target, online, tau: float):
        for p_t, p_o in zip(target.parameters(), online.parameters()):
            p_t.data.lerp_(p_o.data, tau) # p_t = (1 - tau) * p_t + tau * p_o

    def stats(self, evaluate_agents, loss):
        results = evaluate_agents.evaluate(verbose=False)
        if self.memory.num_samples < self.num_warmup:
            print(results, loss, 0)
            return None

        transitions = self.memory.sample()
        _, _, next_state_batch, mask_legal_batch, _, _ = transitions
        state_next_batch = next_state_batch
        mask_legal_batch = mask_legal_batch
        diff_list = []
        for state_next, mask_legal in zip(state_next_batch, mask_legal_batch):
            if mask_legal.sum() != 0:    
                diff = self.agent.get_diff_Q(state_next.unsqueeze(0), mask_legal.unsqueeze(0))
                diff_list.append(diff)
        diff = np.mean(diff_list)

        print(results, loss, diff)
        

def group_decay_parameters(model, weight_decay=0.01, no_decay=['bias', 'GroupNorm.weight']):
    """
    Groups parameters for optimizer with weight decay and no weight decay.
    """
    param_optimizer = list(model.named_parameters())
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},  # Apply weight decay to these parameters
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}  # No weight decay for these parameters
    ]
    
    return optimizer_grouped_parameters